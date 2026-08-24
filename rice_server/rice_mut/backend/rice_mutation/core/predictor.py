"""RiceMutationPredictor — DNA-only multi-omics expression prediction.

Ported from ``inference.ipynb`` (MultiTrackPredictor class).  Encapsulates
model loading (rice_1B_stage2_8k_hf + GenOmics UNet head), reference-FASTA
sequence fetching, tokenization, and PyTorch inference into a single class.

Key differences vs. rice-reg-server2's predictor:
  - Input is DNA sequence ONLY (no ATAC bigWig conditioning).
  - Output is already inverse-scaled inside ``GenOmics.forward``
    (``predictions_scaling_torch``), i.e. values are in original expression
    scale and can be written to bigWig directly.
  - Output shape: {assay: {biosample: np.ndarray[L]}}.
"""

from __future__ import annotations

import json
import logging
import os
import gc
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pyfaidx
import torch

# ``src`` package lives under backend/ (added to sys.path by main.py / api.py)
from src.model import GenOmics, load_finetuned_model
from src.dataset import load_fasta_sequence

logger = logging.getLogger(__name__)

# Mirror inference.ipynb defaults
DEFAULT_PROJ_DIM = 1024
DEFAULT_NUM_DOWNSAMPLES = 4
DEFAULT_BOTTLENECK_DIM = 1536
DEFAULT_MAX_SEQ_LEN = 32768
DEFAULT_DEVICE = "cuda:0"


class RiceMutationPredictor:
    """High-level predictor wrapping model loading and inference.

    Usage::

        predictor = RiceMutationPredictor(
            base_model_path="/path/to/rice_1B_stage2_8k_hf",
            checkpoint_path="/path/to/model.safetensors",
            index_stat_path="/path/to/index_stat3.json",
            fasta_path="/path/to/genome.fa",
        )
        result = predictor.predict(chrom="Chr1", start=0, end=32000)
        result = predictor.predict_sequence("ACGT...", chrom="Chr1", start=0)
        predictor.release()
    """

    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        index_stat_path: str,
        fasta_path: str,
        *,
        use_flash_attn: bool = True,
        device: str = DEFAULT_DEVICE,
        torch_dtype: str = "bfloat16",
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        proj_dim: int = DEFAULT_PROJ_DIM,
        num_downsamples: int = DEFAULT_NUM_DOWNSAMPLES,
        bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM,
        inference_batch_size: int = 1,
    ):
        self.base_model_path = base_model_path
        self.checkpoint_path = checkpoint_path
        self.index_stat_path = index_stat_path
        self.fasta_path = fasta_path
        self.use_flash_attn = bool(use_flash_attn)
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_seq_len = int(max_seq_len)
        # 单次模型前向处理的序列数(当前 API 单窗口请求, 默认 1)
        self.inference_batch_size = max(1, int(inference_batch_size))

        # GenOmics model init kwargs (mirror inference.ipynb)
        self.model_kwargs = dict(
            proj_dim=int(proj_dim),
            num_downsamples=int(num_downsamples),
            bottleneck_dim=int(bottleneck_dim),
        )

        self._model = None
        self._tokenizer = None
        self._fasta: Optional[pyfaidx.Fasta] = None
        self._index_stat: Dict[str, Any] = {}
        self._initialized = False
        # Serialises GPU inference: only one forward at a time.  The model is a
        # single resident instance, so concurrent threads must not touch it.
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------
    #  Metadata (from index_stat)
    # ------------------------------------------------------------------
    @property
    def assay_titles(self) -> List[str]:
        return list(self._index_stat.get("counts", {}).get("heads", []))

    @property
    def biosample_order(self) -> List[str]:
        return list(self._index_stat.get("counts", {}).get("biosample_order", []))

    @property
    def display_assay_titles(self) -> List[str]:
        """Display assay names (from env ``DISPLAY_HEADS``), fallback to index_stat.

        These names are used for frontend / IGV labels only; the model output
        heads still use the index_stat (checkpoint-matching) names internally.
        """
        env = os.getenv("DISPLAY_HEADS", "")
        if env.strip():
            return [x.strip() for x in env.split(",") if x.strip()]
        return self.assay_titles

    @property
    def display_biosample_order(self) -> List[str]:
        """Display biosample names (from env ``DISPLAY_BIOSAMPLES``), fallback to index_stat."""
        env = os.getenv("DISPLAY_BIOSAMPLES", "")
        if env.strip():
            return [x.strip() for x in env.split(",") if x.strip()]
        return self.biosample_order

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    def initialize(self):
        """Load index_stat, FASTA, base model + GenOmics head + checkpoint.

        Idempotent — safe to call multiple times.
        """
        if self._initialized:
            logger.info("RiceMutationPredictor already initialized, skipping.")
            return

        logger.info("Initializing RiceMutationPredictor ...")
        logger.info("  base_model_path = %s", self.base_model_path)
        logger.info("  checkpoint_path = %s", self.checkpoint_path)
        logger.info("  index_stat_path = %s", self.index_stat_path)
        logger.info("  fasta_path      = %s", self.fasta_path)
        logger.info("  device          = %s", self.device)
        logger.info("  inference_batch_size = %d", self.inference_batch_size)

        # 1) index_stat
        with open(self.index_stat_path, "r", encoding="utf-8") as f:
            self._index_stat = json.load(f)
        logger.info(
            "index_stat: assays=%s biosamples=%s",
            self.assay_titles,
            self.biosample_order,
        )

        # 2) FASTA
        self._fasta = pyfaidx.Fasta(self.fasta_path)

        # 3) Build model (directly on target device)
        from transformers import AutoTokenizer

        logger.info("Loading GenOmics model ...")
        torch_dtype_obj = getattr(torch, self.torch_dtype, torch.bfloat16)
        model = load_finetuned_model(
            model_class=GenOmics,
            model_path=self.base_model_path,
            ckpt_path=self.checkpoint_path,
            use_flash_attn=self.use_flash_attn,
            device=self.device,
            torch_dtype=torch_dtype_obj,
            model_init_kwargs={
                "index_stat": self._index_stat,
                **self.model_kwargs,
            },
        )
        model.eval()
        self._model = model

        # 4) Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            revision="main",
        )

        self._initialized = True
        logger.info("RiceMutationPredictor initialized successfully.")

    def release(self):
        """Move model to CPU, clear GPU cache, release file handles."""
        self._initialized = False
        if self._model is not None:
            self._model = self._model.to("cpu")
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._tokenizer = None
        self._fasta = None
        gc.collect()
        logger.info("RiceMutationPredictor released.")

    # ------------------------------------------------------------------
    #  Public inference API
    # ------------------------------------------------------------------
    def predict(
        self,
        chrom: str,
        start: int,
        end: int,
        biosample_names: Optional[List[str]] = None,
        fasta: Optional[pyfaidx.Fasta] = None,
    ) -> Dict[str, Any]:
        """Predict expression for a genomic window of a reference genome.

        Args:
            chrom: Chromosome name (e.g. ``"Chr1"``).
            start: 0-based start coordinate.
            end: End coordinate (exclusive).  Longer than ``max_seq_len``
                  is truncated (mirrors notebook behavior).
            biosample_names: Optional subset of biosamples.  ``None`` = all.
            fasta: Optional ``pyfaidx.Fasta`` to read the sequence from.
                  When given, it overrides the predictor's built-in FASTA
                  (used for uploaded custom genomes).

        Returns:
            Dict ``{assay: {biosample: np.ndarray[L]}}`` in original scale.
        """
        if not self._initialized:
            raise RuntimeError("RiceMutationPredictor not initialized.")
        fa = fasta if fasta is not None else self._fasta
        if fa is None:
            raise RuntimeError("No FASTA available.")

        seq = load_fasta_sequence(
            fa, chrom, start, end, max_length=self.max_seq_len
        )
        return self._run(seq, chrom=chrom, start=start, end=end,
                         biosample_names=biosample_names)

    def predict_sequence(
        self,
        sequence: str,
        chrom: str = "Chr1",
        start: int = 0,
        end: Optional[int] = None,
        biosample_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Predict expression from a raw DNA sequence string (e.g. mutant seq).

        Mirrors ``MultiTrackPredictor.predict2`` in inference.ipynb.
        Sequence is truncated to ``max_seq_len`` if longer.
        """
        if not self._initialized:
            raise RuntimeError("RiceMutationPredictor not initialized.")

        seq = str(sequence).upper().replace("\n", "").strip()
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        if end is None:
            end = start + len(seq)
        return self._run(seq, chrom=chrom, start=start, end=end,
                         biosample_names=biosample_names)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _run(
        self,
        seq: str,
        chrom: str,
        start: int,
        end: int,
        biosample_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        inputs = self._tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        ).to(self.device)

        # The GPU forward is serialised with a lock: the single resident model
        # must not be called from concurrent threads (GPU contention / OOM).
        # ``empty_cache`` is intentionally NOT called per request — it is a
        # global GPU sync that hurts throughput badly under concurrency.
        with torch.no_grad():
            with self._infer_lock:
                t0 = time.time()
                result = self._model.predict(
                    inputs["input_ids"],
                    biosample_names=biosample_names,
                )
                elapsed = time.time() - t0

        logger.info(
            "Inference on %s:%d-%d (seq_len=%d) done in %.2f s",
            chrom, start, end, len(seq), elapsed,
        )

        # Convert GPU tensors {assay: {biosample: tensor[1, L, 1]}} to
        # {assay: {biosample: np.ndarray[L]}} — already original scale.
        # Internal names (checkpoint-matching) are mapped to display names
        # configured via env (DISPLAY_HEADS / DISPLAY_BIOSAMPLES).
        assay_display = dict(zip(self.assay_titles, self.display_assay_titles))
        bios_display = dict(zip(self.biosample_order, self.display_biosample_order))
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for assay, bios_map in result.items():
            d_assay = assay_display.get(assay, assay)
            out[d_assay] = {}
            for bios, tensor in bios_map.items():
                d_bios = bios_display.get(bios, bios)
                arr = tensor.detach().float().cpu().numpy()
                # tensor shape [1, L, 1] -> flatten to [L]
                out[d_assay][d_bios] = arr.reshape(-1)
        return out

    def get_position(self) -> tuple:
        """Placeholder for API compatibility (position tracked by caller)."""
        return None
