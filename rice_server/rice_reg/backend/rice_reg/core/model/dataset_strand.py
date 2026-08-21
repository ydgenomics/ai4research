"""Genomic datasets for training and inference."""

import traceback

import numpy as np
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import Dataset

from model.distributed import dist_print, is_main_process
from model.scaling import LabelScaler, cap_threshold_from_index_row


class LazyGenomicDataset(Dataset):
    """Training dataset: one window row becomes two samples (+ strand and - strand).

    Output dict matches the expected keys for the 1-channel predictor head:
    - `input_ids`: tokenized sequence (forward or reverse-complement, depending on strand)
    - `atac_signal`: ATAC values (forward or reversed order)
    - `labels`: scaled RNA values with shape `[1, L]` (single strand per sample)
    """

    def __init__(self, index_df, tokenizer, max_length=32000):
        self.index_df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

        # Use the shared base to implement all I/O and strand tensor construction.
        self._base = _GenomicDatasetBase(
            self.index_df,
            self.tokenizer,
            max_length=self.max_length,
            strict_bw_open=False,
        )

    def __len__(self):
        return 2 * len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx // 2]
        is_minus = bool(idx % 2)

        try:
            return self._base._build_strand_tensors(row=row, is_minus=is_minus)
        except Exception as e:
            if is_main_process():
                print(
                    "[CRITICAL WARNING] Failed to load sample "
                    f"idx={idx} (row={idx // 2}, strand={'-' if is_minus else '+'}) "
                    f"| type={type(e).__name__} | error={repr(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )

            L = self._base.target_len
            seq = "N" * L
            encodings = self.tokenizer(
                seq,
                padding="max_length",
                max_length=L,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=False,
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "atac_signal": torch.zeros(L, dtype=torch.bfloat16),
                "labels": torch.zeros(1, L, dtype=torch.float32),
            }

    def close(self):
        self._base.close()


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    """Reverse-complement a DNA string (supports A/C/G/T/N and lowercase)."""
    return seq.translate(_RC_TABLE)[::-1]


def _process_signal(raw_vals, target_len):
    vals = np.nan_to_num(raw_vals, nan=0.0)
    if len(vals) > target_len:
        vals = vals[:target_len]
    elif len(vals) < target_len:
        vals = np.pad(vals, (0, target_len - len(vals)), constant_values=0.0)
    return vals


class _GenomicDatasetBase:
    """Shared BigWig/FASTA loading + strand tensor building.

    This does not implement Dataset indexing; subclasses handle __len__/__getitem__.
    """

    def __init__(
        self,
        index_df,
        tokenizer,
        *,
        max_length: int,
        strict_bw_open: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.target_len = int(max_length)
        self.strict_bw_open = bool(strict_bw_open)

        self._fasta_cache = {}
        self._bw_cache = {}

        # Open all unique BigWig files referenced by the index rows.
        unique_paths = set()
        for _, row in index_df.iterrows():
            unique_paths.update([row["rna_path_plus"], row["rna_path_minus"], row["atac_path"]])

        for path in unique_paths:
            try:
                self._bw_cache[path] = pyBigWig.open(path, "r")
            except Exception as e:
                if self.strict_bw_open:
                    raise
                dist_print(f"[WARNING] Cannot open {path}: {e}")
                self._bw_cache[path] = None

    def _get_fasta(self, fasta_path: str):
        fasta_path = str(fasta_path)
        if fasta_path not in self._fasta_cache:
            self._fasta_cache[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_cache[fasta_path]

    def close(self):
        for fa in self._fasta_cache.values():
            try:
                fa.close()
            except Exception:
                pass
        self._fasta_cache.clear()

        for bw in self._bw_cache.values():
            if bw is not None:
                try:
                    bw.close()
                except Exception:
                    pass
        self._bw_cache.clear()

    def _load_raw_window(self, row: dict):
        """Load forward-oriented DNA+tracks for one window."""
        chrom = row["chromosome"]
        start = row["start"]
        end = row["end"]

        fasta = self._get_fasta(row["fasta_path"])

        raw_seq = str(fasta[chrom][start:end])
        if len(raw_seq) > self.target_len:
            seq_fwd = raw_seq[: self.target_len]
        elif len(raw_seq) < self.target_len:
            seq_fwd = raw_seq + "N" * (self.target_len - len(raw_seq))
        else:
            seq_fwd = raw_seq

        bw_plus = self._bw_cache[row["rna_path_plus"]]
        bw_minus = self._bw_cache[row["rna_path_minus"]]
        bw_atac = self._bw_cache[row["atac_path"]]
        if not all([bw_plus, bw_minus, bw_atac]):
            raise ValueError("One or more BigWig handles not loaded")

        raw_plus = np.array(bw_plus.values(chrom, start, end))
        raw_minus = np.array(bw_minus.values(chrom, start, end))
        raw_atac = np.array(bw_atac.values(chrom, start, end))

        plus_vals = _process_signal(raw_plus, self.target_len)
        minus_vals = _process_signal(raw_minus, self.target_len)

        atac_raw = np.nan_to_num(raw_atac, nan=0.0)
        if atac_raw.size > 0:
            atac_clipped = np.clip(atac_raw, 0, np.percentile(atac_raw, 99))
        else:
            atac_clipped = np.zeros(self.target_len)
        atac_clipped = _process_signal(atac_clipped, self.target_len)

        return seq_fwd, plus_vals, minus_vals, atac_clipped

    def _build_strand_tensors(
        self,
        *,
        row: dict,
        is_minus: bool,
    ) -> dict:
        """Return model inputs for one strand sample.

        - Plus sample: forward DNA, forward ATAC, `rna_path_plus` values
        - Minus sample: reverse-complement DNA, reversed ATAC, reversed `rna_path_minus` values

        Output label shape is `[1, L]` to match `predictor_ss` (1-channel head).
        """
        seq_fwd, plus_vals, minus_vals, atac_clipped = self._load_raw_window(row)

        if is_minus:
            seq = _reverse_complement(seq_fwd)
            atac = atac_clipped[::-1].copy()
            rna_raw = minus_vals[::-1].copy()
            tmean = float(row["track_mean_minus"])
            strand = "-"
            cap_col = "cap_threshold_minus"
        else:
            seq = seq_fwd
            atac = atac_clipped
            rna_raw = plus_vals
            tmean = float(row["track_mean_plus"])
            strand = "+"
            cap_col = "cap_threshold_plus"

        scaler = LabelScaler(tmean, cap_threshold_from_index_row(row, cap_col))
        scaled = scaler.transform(rna_raw)[np.newaxis, :]

        encodings = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "atac_signal": torch.tensor(atac, dtype=torch.bfloat16),
            "labels": torch.tensor(scaled, dtype=torch.float32),  # [1, L]
            # Inference only: metadata (ignored by CustomTrainer collate_fn).
            "sequence": seq,
            "track_mean": tmean,
            "strand": strand,
            "raw_rna": rna_raw.astype(np.float32, copy=False),
        }


class InferenceDataset:
    """Inference dataset: each window row becomes two samples (+ strand and - strand).

    Compared to the original two-channel-per-window dataset, this returns:
    - `labels`: shape `[1, L]` (single strand per sample)
    - metadata keys for strand routing in the inference loop:
      `strand`, `position`, `sequence`, `track_mean`, `batch_name`
    """

    def __init__(self, index_df, tokenizer, max_length=32000):
        self.df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.target_len = int(max_length)

        # Shared I/O + strand tensor builder.
        self._base = _GenomicDatasetBase(
            self.df,
            self.tokenizer,
            max_length=self.target_len,
            strict_bw_open=True,
        )

    def __len__(self):
        return 2 * len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx // 2]
        is_minus = bool(idx % 2)

        chrom, start, end = row["chromosome"], row["start"], row["end"]
        position = (chrom, int(start), int(end))
        batch_name = row["batch_name_minus"] if is_minus else row["batch_name_plus"]

        try:
            sample = self._base._build_strand_tensors(row=row, is_minus=is_minus)
            sample.update({"position": position, "batch_name": batch_name})
            return sample
        except Exception as e:
            if is_main_process():
                print(
                    "[CRITICAL WARNING] Failed to load inference sample "
                    f"idx={idx} (row={idx // 2}, strand={'-' if is_minus else '+'}) "
                    f"| type={type(e).__name__} | error={repr(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )

            L = self.target_len
            seq = "N" * L
            encodings = self.tokenizer(
                seq,
                padding="max_length",
                max_length=L,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=False,
            )
            strand = "-" if is_minus else "+"
            tmean = float(row["track_mean_minus"] if is_minus else row["track_mean_plus"])
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "atac_signal": torch.zeros(L, dtype=torch.bfloat16),
                "labels": torch.zeros(1, L, dtype=torch.float32),
                "sequence": seq,
                "track_mean": tmean,
                "strand": strand,
                "raw_rna": np.zeros(L, dtype=np.float32),
                "position": position,
                "batch_name": batch_name,
            }

    def close(self):
        self._base.close()
