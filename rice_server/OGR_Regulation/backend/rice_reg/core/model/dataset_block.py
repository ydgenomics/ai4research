"""Datasets for predictor.type=block (region-masked training/inference).

This module is intentionally isolated from `model/dataset.py` to avoid changing
the batch contract for existing predictor types.
"""

from __future__ import annotations

import traceback

import numpy as np
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import Dataset

from model.distributed import dist_print, is_main_process
from model.scaling import LabelScaler, cap_threshold_from_index_row


def _process_signal(raw_vals, target_len: int) -> np.ndarray:
    vals = np.nan_to_num(raw_vals, nan=0.0)
    if len(vals) > target_len:
        vals = vals[:target_len]
    elif len(vals) < target_len:
        vals = np.pad(vals, (0, target_len - len(vals)), constant_values=0.0)
    return vals


def _process_region_mask(raw_vals, target_len: int) -> np.ndarray:
    """Convert BigWig values to a strict 0/1 float mask of length target_len."""
    vals = _process_signal(raw_vals, target_len).astype(np.float32, copy=False)
    # Region files are expected to be 0/1 but may contain NaNs or other floats.
    vals = np.nan_to_num(vals, nan=0.0)
    return (vals > 0.5).astype(np.float32, copy=False)


class LazyGenomicDatasetBlock(Dataset):
    """Paired-window dataset with an additional region mask track.

    Output keys (training):
    - input_ids: [L]
    - atac_signal: [L] (bfloat16)
    - region_signal: [L] (bfloat16)  (additional model input)
    - region_mask: [L] (float32)     (loss mask)
    - labels: [2, L] (float32)
    """

    def __init__(self, index_df, tokenizer, max_length: int = 32000):
        self.index_df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._fasta_cache: dict[str, pyfaidx.Fasta] = {}
        self._bw_cache: dict[str, pyBigWig.pyBigWig | None] = {}

        unique_paths: set[str] = set()
        for _, row in index_df.iterrows():
            unique_paths.update(
                [
                    row["rna_path_plus"],
                    row["rna_path_minus"],
                    row["atac_path"],
                    row.get("region_path"),
                ]
            )
        unique_paths.discard(None)
        unique_paths.discard("")

        for path in unique_paths:
            try:
                self._bw_cache[str(path)] = pyBigWig.open(str(path), "r")
            except Exception as e:
                dist_print(f"[WARNING] Cannot open {path}: {e}")
                self._bw_cache[str(path)] = None

    def _get_fasta(self, fasta_path: str):
        fasta_path = str(fasta_path)
        if fasta_path not in self._fasta_cache:
            self._fasta_cache[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_cache[fasta_path]

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        fasta = self._get_fasta(row["fasta_path"])
        L = int(self.max_length)

        try:
            bw_plus = self._bw_cache[str(row["rna_path_plus"])]
            bw_minus = self._bw_cache[str(row["rna_path_minus"])]
            bw_atac = self._bw_cache[str(row["atac_path"])]
            bw_region = self._bw_cache.get(str(row.get("region_path", "")))
            if not all([bw_plus, bw_minus, bw_atac, bw_region]):
                raise ValueError("One or more BigWig handles not loaded (including region_path)")

            chrom = row["chromosome"]
            start, end = int(row["start"]), int(row["end"])

            raw_seq = str(fasta[chrom][start:end])
            raw_plus = np.array(bw_plus.values(chrom, start, end))
            raw_minus = np.array(bw_minus.values(chrom, start, end))
            raw_atac = np.array(bw_atac.values(chrom, start, end))
            raw_region = np.array(bw_region.values(chrom, start, end))

            seq = raw_seq[:L] if len(raw_seq) > L else raw_seq + "N" * (L - len(raw_seq))

            plus_vals = _process_signal(raw_plus, L)
            minus_vals = _process_signal(raw_minus, L)

            atac_raw = np.nan_to_num(raw_atac, nan=0.0)
            if atac_raw.size > 0:
                atac_clipped = np.clip(atac_raw, 0, np.percentile(atac_raw, 99))
            else:
                atac_clipped = np.zeros(L)
            atac_vals = _process_signal(atac_clipped, L)

            region_mask = _process_region_mask(raw_region, L)
            region_signal = region_mask  # for now: same 0/1 track as model input

            scaler_p = LabelScaler(
                float(row["track_mean_plus"]),
                cap_threshold_from_index_row(row, "cap_threshold_plus"),
            )
            scaler_m = LabelScaler(
                float(row["track_mean_minus"]),
                cap_threshold_from_index_row(row, "cap_threshold_minus"),
            )
            scaled_labels = np.stack(
                [scaler_p.transform(plus_vals), scaler_m.transform(minus_vals)], axis=0
            )

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
                "atac_signal": torch.tensor(atac_vals, dtype=torch.bfloat16),
                "region_signal": torch.tensor(region_signal, dtype=torch.bfloat16),
                "region_mask": torch.tensor(region_mask, dtype=torch.float32),
                "labels": torch.tensor(scaled_labels, dtype=torch.float32),
            }

        except Exception as e:
            if is_main_process():
                print(
                    "[CRITICAL WARNING] Failed to load block sample "
                    f"idx={idx} | type={type(e).__name__} | error={repr(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
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
                "region_signal": torch.zeros(L, dtype=torch.bfloat16),
                "region_mask": torch.zeros(L, dtype=torch.float32),
                "labels": torch.zeros(2, L, dtype=torch.float32),
            }

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


class InferenceDatasetBlock:
    """Inference dataset for predictor.type=block.

    Adds:
    - region_mask: numpy float32 array (length L) for output serialization/metrics
    - region_signal: tensor used for the model input
    """

    def __init__(self, index_df, tokenizer, max_length: int = 32000):
        self.df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.target_len = int(max_length)
        self._fasta_cache: dict[str, pyfaidx.Fasta] = {}
        self._bw_cache: dict[str, pyBigWig.pyBigWig] = {}

        paths: set[str] = set()
        for _, row in self.df.iterrows():
            paths.update(
                [
                    row["rna_path_plus"],
                    row["rna_path_minus"],
                    row["atac_path"],
                    row.get("region_path"),
                ]
            )
        paths.discard(None)
        paths.discard("")
        for path in paths:
            self._bw_cache[str(path)] = pyBigWig.open(str(path), "r")

    def _get_fasta(self, fasta_path: str):
        fasta_path = str(fasta_path)
        if fasta_path not in self._fasta_cache:
            self._fasta_cache[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_cache[fasta_path]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        L = self.target_len
        fasta = self._get_fasta(row["fasta_path"])

        chrom, start, end = row["chromosome"], int(row["start"]), int(row["end"])

        raw_seq = str(fasta[chrom][start:end])
        raw_plus = np.array(self._bw_cache[str(row["rna_path_plus"])].values(chrom, start, end))
        raw_minus = np.array(self._bw_cache[str(row["rna_path_minus"])].values(chrom, start, end))
        raw_atac = np.array(self._bw_cache[str(row["atac_path"])].values(chrom, start, end))
        raw_region = np.array(self._bw_cache[str(row["region_path"])].values(chrom, start, end))

        seq = raw_seq[:L] if len(raw_seq) > L else raw_seq + "N" * (L - len(raw_seq))

        plus_vals = _process_signal(raw_plus, L)
        minus_vals = _process_signal(raw_minus, L)

        atac_raw = np.nan_to_num(raw_atac, nan=0.0)
        atac_clipped = np.clip(
            atac_raw,
            0,
            np.percentile(atac_raw, 99) if atac_raw.size > 0 else 0.0,
        )
        atac_vals = _process_signal(atac_clipped, L)

        region_mask = _process_region_mask(raw_region, L)
        region_signal = region_mask

        scaler_p = LabelScaler(
            float(row["track_mean_plus"]),
            cap_threshold_from_index_row(row, "cap_threshold_plus"),
        )
        scaler_m = LabelScaler(
            float(row["track_mean_minus"]),
            cap_threshold_from_index_row(row, "cap_threshold_minus"),
        )
        scaled_labels = np.stack(
            [scaler_p.transform(plus_vals), scaler_m.transform(minus_vals)], axis=0
        )

        enc = self.tokenizer(
            seq,
            padding="max_length",
            max_length=L,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "atac_signal": torch.tensor(atac_vals, dtype=torch.bfloat16),
            "region_signal": torch.tensor(region_signal, dtype=torch.bfloat16),
            "region_mask": region_mask.astype(np.float32, copy=False),  # numpy for serialization
            "labels": torch.tensor(scaled_labels, dtype=torch.float32),
            "track_mean_plus": row["track_mean_plus"],
            "track_mean_minus": row["track_mean_minus"],
            "raw_rna_plus": plus_vals.astype(np.float32, copy=False),
            "raw_rna_minus": minus_vals.astype(np.float32, copy=False),
            "sequence": seq,
            "position": (chrom, start, end),
            "batch_name_plus": row["batch_name_plus"],
            "batch_name_minus": row["batch_name_minus"],
        }

    def close(self):
        for fa in self._fasta_cache.values():
            try:
                fa.close()
            except Exception:
                pass
        self._fasta_cache.clear()
        for bw in self._bw_cache.values():
            try:
                bw.close()
            except Exception:
                pass
        self._bw_cache.clear()

