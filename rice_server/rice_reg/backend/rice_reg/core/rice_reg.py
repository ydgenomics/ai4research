"""RiceRegPredictor — ATAC-conditioned RNA-seq expression prediction.

Ported from inference2.ipynb (Cells 1-10).  Encapsulates model loading,
genomic window indexing, ATAC+DNA feature extraction, and PyTorch inference
into a single ``RiceRegPredictor`` class with a ``predict()`` interface.
"""

from __future__ import annotations

import logging
import os
import gc
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import torch

from model.load_pretrained import load_model_and_tokenizer
from model.pipeline import (
    build_multimodal_model,
    _normalize_predictor_type,
    _resolve_dataset_type,
    _validate_dataset_type_match,
)
from model.scaling import LabelScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Constants (mirror inference2.ipynb Cell 1 defaults)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_LEN = 32678
DEFAULT_OVERLAP_LEN = 16000
DEFAULT_TRACK_MEAN_PLUS = 2.83
DEFAULT_TRACK_MEAN_MINUS = 2.85
DEFAULT_PREDICTOR_TYPE = "fusion"
DEFAULT_DATASET_TYPE = "v0"
DEFAULT_ATAC_ENCODER_OUTPUT_DIM = 1024
DEFAULT_MODEL_TORCH_DTYPE = "bfloat16"
DEFAULT_INFERENCE_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
#  InferenceOnlyDataset  (from inference2.ipynb Cell 7)
# ---------------------------------------------------------------------------
class InferenceOnlyDataset(torch.utils.data.Dataset):
    """Loads only DNA sequence + ATAC signal for inference (no RNA BigWig needed).

    Mirrors ``inference2.ipynb`` Cell 7.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        tokenizer,
        max_length: int = DEFAULT_TARGET_LEN,
        track_mean_plus: float = DEFAULT_TRACK_MEAN_PLUS,
        track_mean_minus: float = DEFAULT_TRACK_MEAN_MINUS,
    ):
        self.index_df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._track_mean_plus = float(track_mean_plus)
        self._track_mean_minus = float(track_mean_minus)
        self._fasta_cache: Dict[str, pyfaidx.Fasta] = {}
        self._atac_bw_cache: Dict[str, pyBigWig.pyBigWig] = {}

        for path in index_df["atac_path"].unique():
            try:
                self._atac_bw_cache[path] = pyBigWig.open(path, "r")
            except Exception as e:
                logger.warning("Cannot open ATAC %s: %s", path, e)
                self._atac_bw_cache[path] = None

    def _get_fasta(self, fasta_path: str) -> pyfaidx.Fasta:
        if fasta_path not in self._fasta_cache:
            self._fasta_cache[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_cache[fasta_path]

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.index_df.iloc[idx]
        fasta = self._get_fasta(row["fasta_path"])
        target_len = self.max_length

        # --- DNA sequence ---
        raw_seq = str(fasta[row["chromosome"]][row["start"] : row["end"]])
        if len(raw_seq) > target_len:
            seq = raw_seq[:target_len]
        elif len(raw_seq) < target_len:
            seq = raw_seq + "N" * (target_len - len(raw_seq))
        else:
            seq = raw_seq

        # --- ATAC signal ---
        bw_atac = self._atac_bw_cache[row["atac_path"]]
        raw_atac = np.array(
            bw_atac.values(row["chromosome"], row["start"], row["end"])
        )
        atac_vals = np.nan_to_num(raw_atac, nan=0.0)
        if atac_vals.size > 0:
            clip_hi = np.percentile(atac_vals, 99)
            atac_vals = np.clip(atac_vals, 0, clip_hi)
        if len(atac_vals) > target_len:
            atac_vals = atac_vals[:target_len]
        elif len(atac_vals) < target_len:
            atac_vals = np.pad(
                atac_vals, (0, target_len - len(atac_vals)), constant_values=0.0
            )

        # --- Tokenize ---
        enc = self.tokenizer(
            seq,
            padding="max_length",
            max_length=target_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "atac_signal": torch.tensor(atac_vals, dtype=torch.float32),
            "sequence": seq,
            "position": (row["chromosome"], row["start"], row["end"]),
            "track_mean_plus": float(
                row.get("track_mean_plus", self._track_mean_plus)
            ),
            "track_mean_minus": float(
                row.get("track_mean_minus", self._track_mean_minus)
            ),
            "batch_name_plus": f"{row['cell_type']}_plus",
            "batch_name_minus": f"{row['cell_type']}_minus",
        }

    def close(self):
        """Release BigWig file handles."""
        for bw in self._atac_bw_cache.values():
            if bw is not None:
                try:
                    bw.close()
                except Exception:
                    pass
        self._atac_bw_cache.clear()
        self._fasta_cache.clear()


# ---------------------------------------------------------------------------
#  RiceRegPredictor
# ---------------------------------------------------------------------------
class RiceRegPredictor:
    """High-level predictor wrapping model loading, indexing, and inference.

    Usage::

        predictor = RiceRegPredictor(
            base_model_path="/path/to/rice_1B_32k_hf",
            checkpoint_path="/path/to/model.safetensors",
        )
        result = predictor.predict(
            chrom="chr1", start=0, end=32000,
            atac_path="/path/to/ATAC.bw",
            fasta_path="/path/to/genome.fa",
        )
        predictor.release()   # free GPU memory
    """

    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        *,
        predictor_type: str = DEFAULT_PREDICTOR_TYPE,
        dataset_type: str = DEFAULT_DATASET_TYPE,
        atac_encoder_output_dim: int = DEFAULT_ATAC_ENCODER_OUTPUT_DIM,
        target_len: int = DEFAULT_TARGET_LEN,
        overlap_len: int = DEFAULT_OVERLAP_LEN,
        track_mean_plus: float = DEFAULT_TRACK_MEAN_PLUS,
        track_mean_minus: float = DEFAULT_TRACK_MEAN_MINUS,
        model_torch_dtype: str = DEFAULT_MODEL_TORCH_DTYPE,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        device: Optional[torch.device] = None,
    ):
        self.base_model_path = base_model_path
        self.checkpoint_path = checkpoint_path
        self.predictor_type = predictor_type
        self.dataset_type = dataset_type
        self.atac_encoder_output_dim = atac_encoder_output_dim
        self.target_len = target_len
        self.overlap_len = overlap_len
        self.track_mean_plus = track_mean_plus
        self.track_mean_minus = track_mean_minus
        self.model_torch_dtype = model_torch_dtype
        self.inference_batch_size = inference_batch_size

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            # device 允许传 torch.device 或字符串(如 "cuda:0"),统一转成 torch.device
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        self._model = None
        self._tokenizer = None
        self._initialized = False
        # Serialise GPU inference across concurrent requests (single model).
        self._infer_lock = threading.Lock()

    # ---- public API -------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self):
        """Load base model, tokenizer, checkpoint, and build multimodal model.

        Idempotent — safe to call multiple times.
        """
        if self._initialized:
            logger.info("RiceRegPredictor already initialized, skipping.")
            return

        logger.info("Initializing RiceRegPredictor ...")
        logger.info("  base_model_path = %s", self.base_model_path)
        logger.info("  checkpoint_path = %s", self.checkpoint_path)
        logger.info("  predictor_type  = %s", self.predictor_type)
        logger.info("  device          = %s", self.device)

        # Build config dict (mirrors inference2.ipynb Cell 3)
        cfg = {
            "model_path": self.base_model_path,
            "model_torch_dtype": self.model_torch_dtype,
            "ckpt_path": self.checkpoint_path,
            "predictor": {
                "type": self.predictor_type,
                "fusion_gate_entropy_frac": 0.0,
                "skip_gate_entropy_frac": 0.0,
                "unfreeze_base_last_layer": False,
            },
            "dataset_type": self.dataset_type,
            "atac_encoder_output_dim": self.atac_encoder_output_dim,
            "target_len": self.target_len,
        }

        # 1) Load base model + tokenizer
        logger.info("Loading base model and tokenizer ...")
        base_model, tokenizer = load_model_and_tokenizer(cfg)
        self._tokenizer = tokenizer

        # 2) Build multimodal predictor
        logger.info("Building multimodal predictor (%s) ...", self.predictor_type)
        model = build_multimodal_model(cfg, base_model)
        ptype = _normalize_predictor_type(cfg)
        _validate_dataset_type_match(self.dataset_type, ptype, type(model))

        model = model.to(self.device)

        # 3) Load checkpoint
        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        from safetensors.torch import load_file as safe_load

        state_dict = safe_load(self.checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info("Checkpoint loaded (non-strict mode).")

        self._model = model
        self._initialized = True
        logger.info("RiceRegPredictor initialized successfully.")

    def predict(
        self,
        chrom: str,
        start: int,
        end: int,
        atac_path: str,
        fasta_path: str,
        cell_type: str = "sample",
    ) -> Dict[str, Any]:
        """Run inference on a single genomic window.

        Args:
            chrom: Chromosome name (e.g. ``"chr1"``).
            start: 0-based start coordinate.
            end: End coordinate (exclusive).  If ``end - start != target_len``,
                the window is centered and padded/truncated to ``target_len``.
            atac_path: Path to ATAC bigWig file.
            fasta_path: Path to reference genome FASTA file.
            cell_type: Label for the output batch (default ``"sample"``).

        Returns:
            A dict with keys:

            - ``"sequence"``: the DNA sequence string.
            - ``"position"``: ``(chrom, start, end)`` tuple.
            - ``"values"``: dict with ``"RNA-seq_+"`` and ``"RNA-seq_-"``,
              each containing a numpy array of shape ``(target_len,)``.
        """
        if not self._initialized:
            raise RuntimeError(
                "RiceRegPredictor not initialized. Call .initialize() first."
            )

        # Build a single-row index
        index_df = self._build_single_index(
            chrom=chrom,
            start=start,
            end=end,
            atac_path=atac_path,
            fasta_path=fasta_path,
            cell_type=cell_type,
        )

        # Run inference
        plus_results, minus_results = self._run_inference(index_df)

        # Collect results
        plus_key = f"{cell_type}_plus"
        minus_key = f"{cell_type}_minus"

        out = {
            "sequence": None,
            "position": (chrom, start, end),
            "values": {
                "RNA-seq_+": None,
                "RNA-seq_-": None,
            },
        }

        if plus_key in plus_results and len(plus_results[plus_key]) > 0:
            r = plus_results[plus_key][0]
            out["sequence"] = r["sequence"]
            out["position"] = (r["chromosome"], r["start"], r["end"])
            out["values"]["RNA-seq_+"] = r["predicted_expression"]

        if minus_key in minus_results and len(minus_results[minus_key]) > 0:
            r = minus_results[minus_key][0]
            out["values"]["RNA-seq_-"] = r["predicted_expression"]

        return out

    def predict_batch(
        self,
        index_df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """Run inference on a pre-built index DataFrame.

        Args:
            index_df: DataFrame with columns
                ``chromosome, start, end, cell_type, fasta_path, atac_path,
                track_mean_plus, track_mean_minus``.

        Returns:
            ``(plus_results, minus_results)`` dicts keyed by batch name,
            each value a list of result dicts.
        """
        if not self._initialized:
            raise RuntimeError(
                "RiceRegPredictor not initialized. Call .initialize() first."
            )
        return self._run_inference(index_df)

    def release(self):
        """Move model to CPU, clear GPU cache, release file handles."""
        self._initialized = False
        if self._model is not None:
            self._model = self._model.to("cpu")
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("RiceRegPredictor released.")

    # ---- internal helpers -------------------------------------------------

    def _build_single_index(
        self,
        chrom: str,
        start: int,
        end: int,
        atac_path: str,
        fasta_path: str,
        cell_type: str = "sample",
    ) -> pd.DataFrame:
        """Build a single-row index DataFrame for one genomic window."""
        # Adjust window to target_len
        win = self.target_len
        region_len = end - start
        if region_len != win:
            center = (start + end) // 2
            start = max(0, center - win // 2)
            end = start + win

        return pd.DataFrame(
            [
                {
                    "chromosome": chrom,
                    "start": start,
                    "end": end,
                    "cell_type": cell_type,
                    "fasta_path": fasta_path,
                    "atac_path": atac_path,
                    "track_mean_plus": self.track_mean_plus,
                    "track_mean_minus": self.track_mean_minus,
                }
            ]
        )

    def _run_inference(
        self, index_df: pd.DataFrame
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """Core inference loop (mirrors inference2.ipynb Cell 8)."""
        dataset = InferenceOnlyDataset(
            index_df,
            self._tokenizer,
            max_length=self.target_len,
            track_mean_plus=self.track_mean_plus,
            track_mean_minus=self.track_mean_minus,
        )
        n = len(dataset)
        logger.info("Running inference on %d window(s), batch_size=%d", n, self.inference_batch_size)

        plus_results: Dict[str, list] = {}
        minus_results: Dict[str, list] = {}

        for batch_start in range(0, n, self.inference_batch_size):
            idx_chunk = list(
                range(batch_start, min(batch_start + self.inference_batch_size, n))
            )
            samples = [dataset[i] for i in idx_chunk]

            inp = torch.stack([s["input_ids"] for s in samples], dim=0).to(self.device)
            atac = torch.stack([s["atac_signal"] for s in samples], dim=0).to(self.device)

            with torch.no_grad():
                with self._infer_lock:
                    out = self._model(inp, atac)
                logits_b = out["logits"].float().cpu().numpy()

            for sample, logits in zip(samples, logits_b):
                L = len(sample["sequence"])
                scaler_p = LabelScaler(float(sample["track_mean_plus"]), None)
                scaler_m = LabelScaler(float(sample["track_mean_minus"]), None)
                pred_p = scaler_p.inverse_transform(logits[0, :L])
                pred_m = scaler_m.inverse_transform(logits[1, :L])

                chrom, start_pos, end_pos = sample["position"]
                seq = sample["sequence"]
                bp = sample["batch_name_plus"]
                bm = sample["batch_name_minus"]

                plus_results.setdefault(bp, []).append(
                    {
                        "chromosome": chrom,
                        "start": start_pos,
                        "end": end_pos,
                        "sequence": seq,
                        "predicted_expression": np.array(pred_p, copy=True),
                    }
                )
                minus_results.setdefault(bm, []).append(
                    {
                        "chromosome": chrom,
                        "start": start_pos,
                        "end": end_pos,
                        "sequence": seq,
                        "predicted_expression": np.array(pred_m, copy=True),
                    }
                )

        dataset.close()
        logger.info("Inference completed: %d windows", n)
        return plus_results, minus_results
