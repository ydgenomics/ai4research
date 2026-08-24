"""Prediction service — singleton predictor lifecycle and core prediction logic.

References:
- genos-reg-server/backend/genos_reg/prediction_service.py
- inference2.ipynb Cells 5-9
"""

from __future__ import annotations

import logging
import os
import re
import gc
import time
import threading
from typing import Any, Dict, Optional

import numpy as np
import pyBigWig
import torch

from rice_reg.core import RiceRegPredictor
from rice_reg.igv_payload import build_prediction_payload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Singleton predictor store
# ---------------------------------------------------------------------------
_PREDICTOR: Dict[str, Any] = {"instance": None}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


# ---------------------------------------------------------------------------
#  Chromosome normalization (chr01..chr12 <-> real FASTA names like chr1)
# ---------------------------------------------------------------------------
# genome -> {input_alias: real_fasta_chrom}
_CHROM_ALIAS_CACHE: Dict[str, Dict[str, str]] = {}


def _chrom_key(chrom: str) -> str:
    """Canonical numeric key for a chromosome alias.

    'chr01' / 'Chr1' / '1' / 'chr1' / 'CHR01' -> '1'.
    Non-numeric names (e.g. 'chrMT', 'chrPltd') are returned lower-cased.
    """
    if chrom is None:
        return ""
    s = str(chrom).strip()
    m = re.fullmatch(r"(?:chr|Chr|CHR)?0*([1-9]\d*)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    return s.lower()


def _load_fai_chroms(fai_path: str) -> list:
    """Read the real chromosome names from a .fai file (first column)."""
    if not fai_path or not os.path.isfile(fai_path):
        return []
    names = []
    try:
        with open(fai_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                col = line.split("\t", 1)[0].strip()
                if col:
                    names.append(col)
    except Exception as e:
        logger.warning("Failed to read %s: %s", fai_path, e)
    return names


def build_chrom_alias_map(
    genome: str,
    genome_config: Optional[dict] = None,
) -> Dict[str, str]:
    """Build a map from every plausible chromosome alias to the real FASTA name.

    Handles the common mismatch where the front-end uses ``chr01``-``chr12``
    while the FASTA / bigWig use ``chr1``, ``Chr1``, ``1``, ``Chr01`` etc.
    """
    if genome in _CHROM_ALIAS_CACHE:
        return _CHROM_ALIAS_CACHE[genome]

    fai = (genome_config or {}).get("fai", "") or _env_str(
        f"GENOME_{genome}_FAI", ""
    )
    real_names = _load_fai_chroms(fai)

    alias_map: Dict[str, str] = {}
    # 1) exact names always map to themselves
    for name in real_names:
        alias_map[name] = name
    # 2) aliases by canonical numeric key
    by_key: Dict[str, str] = {}
    for name in real_names:
        by_key.setdefault(_chrom_key(name), name)
    for key, real in by_key.items():
        if key:
            alias_map[f"chr{key}"] = real
            alias_map[f"Chr{key}"] = real
            alias_map[f"CHR{key}"] = real
            alias_map[key] = real
    # 3) allow front-end zero-padded chr01..chr12
    for key, real in by_key.items():
        if key.isdigit() and len(key) == 1:
            alias_map[f"chr0{key}"] = real
            alias_map[f"Chr0{key}"] = real

    _CHROM_ALIAS_CACHE[genome] = alias_map
    logger.debug("Chromosome alias map for %s: %s", genome, alias_map)
    return alias_map


def normalize_chromosome(
    genome: str,
    chromosome: str,
    genome_config: Optional[dict] = None,
) -> str:
    """Resolve a user-supplied chromosome to the real FASTA chromosome name.

    Raises ValueError when the chromosome cannot be resolved.
    """
    if not chromosome:
        raise ValueError("chromosome is required")
    alias_map = build_chrom_alias_map(genome, genome_config)
    chrom = str(chromosome).strip()
    if chrom in alias_map:
        return alias_map[chrom]
    # fallback: try canonical key lookup even if the map came from a partial fai
    key = _chrom_key(chrom)
    for real in alias_map.values():
        if _chrom_key(real) == key:
            return real
    raise ValueError(
        f"Chromosome '{chromosome}' not found in genome '{genome}' "
        f"(available: {sorted(set(alias_map.values()))})"
    )


# ---------------------------------------------------------------------------
#  Uploaded (custom) genomes — in-memory registry for /uploadFasta + /uploadGff
# ---------------------------------------------------------------------------
# genome_id -> {"fasta": abs_path, "fai": abs_path, "gff": "", "last_used": epoch}
_UPLOADED_GENOMES: Dict[str, Dict[str, Any]] = {}
# Guard concurrent access to the registry (id generation + read-modify-write).
_REGISTRY_LOCK = threading.Lock()


def _touch_uploaded_genome(genome: str) -> None:
    """Refresh the last-used timestamp of an uploaded genome.

    Called whenever a request actually uses an uploaded genome (prediction,
    chromosome listing, IGV reference).  Built-in genomes are no-ops.
    """
    with _REGISTRY_LOCK:
        cfg = _UPLOADED_GENOMES.get(genome)
        if cfg is not None:
            cfg["last_used"] = time.time()
            _UPLOADED_GENOMES[genome] = cfg


def cleanup_expired_uploaded_genomes(ttl_hours: float) -> int:
    """Delete uploaded genomes whose last use is older than ``ttl_hours``.

    Removes the in-memory registry entry together with the on-disk FASTA /
    .fai / GFF files (all are per-upload temporary files).  Returns the number
    of genomes removed.
    """
    cutoff = time.time() - float(ttl_hours) * 3600
    expired: list = []
    with _REGISTRY_LOCK:
        for gid, cfg in list(_UPLOADED_GENOMES.items()):
            if float(cfg.get("last_used", 0)) < cutoff:
                expired.append((gid, cfg))
        for gid, _ in expired:
            _UPLOADED_GENOMES.pop(gid, None)

    for gid, cfg in expired:
        for key in ("fasta", "fai", "gff"):
            path = cfg.get(key) or ""
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", path, e)
        logger.info(
            "Cleaned up expired uploaded genome '%s' (TTL %.1fh)", gid, ttl_hours
        )
    return len(expired)


def start_uploaded_genome_cleanup(ttl_hours: float = 0.5,
                                  interval_seconds: int = 300) -> None:
    """Start a daemon thread that periodically removes idle uploaded genomes."""
    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            try:
                n = cleanup_expired_uploaded_genomes(ttl_hours)
                if n:
                    logger.info("[upload] cleaned %d expired uploaded genome(s)", n)
            except Exception as e:  # never let cleanup take down the server
                logger.warning("[upload] uploaded-genome cleanup error: %s", e)

    threading.Thread(
        target=_loop, daemon=True, name="uploaded-genome-cleanup"
    ).start()
    logger.info(
        "Started uploaded-genome cleanup thread (ttl=%.1fh, interval=%ds)",
        ttl_hours, interval_seconds,
    )


def _builtin_genome_ids() -> list:
    return [
        key[len("GENOME_"):-len("_FASTA")]
        for key, val in sorted(os.environ.items())
        if key.startswith("GENOME_") and key.endswith("_FASTA") and val
    ]


def list_genomes() -> list:
    """All configured genome IDs (built-in from env + uploaded)."""
    ids = _builtin_genome_ids()
    for gid in _UPLOADED_GENOMES:
        if gid not in ids:
            ids.append(gid)
    return ids


def display_chromosome(name: str) -> str:
    """Display name for a real FASTA chromosome (chr01..chr12 style)."""
    m = re.fullmatch(r"(?:chr|Chr|CHR)?0*([1-9]\d*)", str(name).strip(), re.IGNORECASE)
    if m:
        return f"chr{int(m.group(1)):02d}"
    return str(name)


def get_genome_chromosomes(genome: str, genome_config: Optional[dict] = None) -> list:
    """Display chromosome names for a genome (chrNN style where possible)."""
    cfg = genome_config or resolve_genome_config(genome)
    real_names = _load_fai_chroms(cfg.get("fai", ""))
    if not real_names and cfg.get("fasta"):
        try:
            import pyfaidx
            fa = pyfaidx.Fasta(cfg["fasta"])
            real_names = list(fa.keys())
        except Exception as e:
            logger.warning("Could not index FASTA %s: %s", cfg.get("fasta"), e)
    return [display_chromosome(n) for n in real_names]


def resolve_genome_config(genome: str) -> dict:
    """Resolve a genome's FASTA / FAI / GFF config.

    Uploaded (custom) genomes take priority over the built-in ones from env.
    """
    uploaded = _UPLOADED_GENOMES.get(genome)
    if uploaded is not None:
        _touch_uploaded_genome(genome)
        return dict(uploaded)

    fasta = _env_str(f"GENOME_{genome}_FASTA", "")
    if not fasta:
        raise ValueError(
            f"Unknown genome '{genome}'.  Set GENOME_{genome}_FASTA in .env "
            "or upload a custom FASTA file first."
        )
    return {
        "fasta": fasta,
        "fai": _env_str(f"GENOME_{genome}_FAI", ""),
        "gff": _env_str(f"GENOME_{genome}_GFF", ""),
    }


def register_uploaded_genome(fasta_path: str, genome_id: str = "") -> Dict[str, str]:
    """Register an uploaded FASTA as a usable genome (priority over built-in).

    - Builds the .fai index automatically if missing (pyfaidx).
    - Stores the config in the in-memory registry.
    """
    fasta_path = os.path.abspath(fasta_path)
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    fai_path = fasta_path + ".fai"
    if not os.path.isfile(fai_path):
        import pyfaidx
        fa = pyfaidx.Fasta(fasta_path)
        names = list(fa.keys())
    else:
        names = _load_fai_chroms(fai_path)
    if not names:
        raise ValueError(f"No sequence records found in FASTA: {fasta_path}")

    with _REGISTRY_LOCK:
        if not genome_id:
            genome_id = f"custom_{int(time.time())}"
        # avoid clobbering an existing id
        while genome_id in _UPLOADED_GENOMES or genome_id in _builtin_genome_ids():
            genome_id = f"custom_{int(time.time())}_{len(_UPLOADED_GENOMES)}"

        cfg = {"fasta": fasta_path, "fai": fai_path, "gff": "", "last_used": time.time()}
        _UPLOADED_GENOMES[genome_id] = cfg
    logger.info(
        "Registered uploaded genome '%s' (%d chromosomes): %s",
        genome_id, len(names), fasta_path,
    )
    return cfg


def attach_uploaded_gff(genome_id: str, gff_path: str) -> Dict[str, str]:
    """Attach an annotation GFF to an uploaded (custom) genome."""
    gff_path = os.path.abspath(gff_path)
    if not os.path.isfile(gff_path):
        raise FileNotFoundError(f"GFF not found: {gff_path}")
    with _REGISTRY_LOCK:
        cfg = _UPLOADED_GENOMES.get(genome_id)
        if cfg is None:
            raise ValueError(
                f"Genome '{genome_id}' is not an uploaded custom genome. "
                "Upload a FASTA first, then attach a GFF."
            )
        cfg["gff"] = gff_path
        cfg["last_used"] = time.time()
        _UPLOADED_GENOMES[genome_id] = cfg
    logger.info("Attached GFF to uploaded genome '%s': %s", genome_id, gff_path)
    return dict(cfg)


# ---------------------------------------------------------------------------
#  Public lifecycle
# ---------------------------------------------------------------------------
def init_predictor():
    """Initialise the singleton RiceRegPredictor from environment variables."""
    if _PREDICTOR.get("instance") is not None:
        logger.info("Predictor already initialised, skipping.")
        return

    base_model_path = _env_str("BASE_MODEL_PATH", "")
    checkpoint_path = _env_str("CHECKPOINT_PATH", "")
    if not base_model_path or not checkpoint_path:
        logger.warning(
            "BASE_MODEL_PATH or CHECKPOINT_PATH not set — predictor will not be "
            "available until .env is configured."
        )
        return

    predictor = RiceRegPredictor(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        predictor_type=_env_str("PREDICTOR_TYPE", "fusion"),
        dataset_type=_env_str("DATASET_TYPE", "v0"),
        atac_encoder_output_dim=_env_int("ATAC_ENCODER_OUTPUT_DIM", 1024),
        target_len=_env_int("TARGET_LEN", 32000),
        overlap_len=_env_int("OVERLAP_LEN", 16000),
        track_mean_plus=_env_float("FIXED_TRACK_MEAN_PLUS", 2.83),
        track_mean_minus=_env_float("FIXED_TRACK_MEAN_MINUS", 2.85),
        model_torch_dtype=_env_str("MODEL_TORCH_DTYPE", "bfloat16"),
        inference_batch_size=_env_int("INFERENCE_BATCH_SIZE", 4),
        # 机器配置: DEVICE 留空则自动选择 cuda/cpu
        device=_env_str("DEVICE", "") or None,
    )
    predictor.initialize()
    _PREDICTOR["instance"] = predictor
    logger.info("RiceRegPredictor initialised and ready.")


def require_predictor() -> RiceRegPredictor:
    """Return the singleton predictor, or raise if not initialised."""
    inst = _PREDICTOR.get("instance")
    if inst is None:
        raise RuntimeError(
            "Predictor not initialised.  Ensure .env is configured and "
            "init_predictor() has been called."
        )
    return inst


def release_predictor():
    """Release the predictor and free GPU memory."""
    inst = _PREDICTOR.get("instance")
    if inst is not None:
        inst.release()
    _PREDICTOR["instance"] = None
    gc.collect()
    logger.info("Predictor released.")


# ---------------------------------------------------------------------------
#  ATAC path resolution
# ---------------------------------------------------------------------------
def resolve_atac_path(
    atac_source: Optional[str],
    uploaded_atac: Optional[str],
) -> str:
    """Resolve the ATAC bigWig path from either a built-in source or uploaded file.

    Priority: ``uploaded_atac`` > ``atac_source``.
    """
    if uploaded_atac:
        if not os.path.exists(uploaded_atac):
            raise FileNotFoundError(f"Uploaded ATAC file not found: {uploaded_atac}")
        return uploaded_atac

    if atac_source:
        env_key = f"ATAC_PATH_{atac_source}"
        path = os.getenv(env_key)
        if not path:
            # Try direct lookup from ATAC_SIGNAL_PATHS (frontend config)
            try:
                from frontend.config import ATAC_SIGNAL_PATHS
                path = ATAC_SIGNAL_PATHS.get(atac_source)
            except Exception:
                pass
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"Built-in ATAC source '{atac_source}' not found. "
                f"Set {env_key} in .env"
            )
        return path

    raise ValueError(
        "No ATAC source provided.  Set either 'atac_source' (built-in) "
        "or 'uploaded_atac' (user-uploaded file path)."
    )


# ---------------------------------------------------------------------------
#  Core prediction
# ---------------------------------------------------------------------------
def adjust_window(start: int, end: int, target_len: int) -> tuple:
    """Center-align a window to ``target_len`` (deterministic).

    Exported for the API layer so the cache key can be built from the same
    normalised window that ``run_prediction_core`` actually predicts.
    """
    if end is None or end - start != target_len:
        if end is None:
            end = start + target_len
        center = (start + end) // 2
        start = max(0, center - target_len // 2)
        end = start + target_len
    return start, end


def run_prediction_core(
    genome: str,
    chromosome: str,
    start: int,
    end: int,
    atac_path: str,
    fasta_path: str,
    genome_config: dict,
) -> Dict[str, Any]:
    """Run the full prediction pipeline for a single genomic window.

    Steps:
        1. Validate ATAC bigWig chromosome compatibility.
        2. Call ``RiceRegPredictor.predict()``.
        3. Build IGV payload from predicted values.
        4. Save prediction bigWig files to cache.
        5. Return IGV payload dict.

    Returns:
        Dict with keys ``igv_payload``, ``plus_bw_path``, ``minus_bw_path``.
    """
    predictor = require_predictor()
    target_len = predictor.target_len

    # --- Normalize chromosome alias (e.g. chr01 -> chr1) to the real FASTA name ---
    chromosome = normalize_chromosome(genome, chromosome, genome_config)

    # --- Validate ATAC bigWig ---
    _validate_atac_bw(atac_path, chromosome)

    # --- Adjust window ---
    start, end = adjust_window(start, end, target_len)

    # --- Run inference ---
    result = predictor.predict(
        chrom=chromosome,
        start=start,
        end=end,
        atac_path=atac_path,
        fasta_path=fasta_path,
        cell_type="sample",
    )

    pred_plus = result["values"]["RNA-seq_+"]
    pred_minus = result["values"]["RNA-seq_-"]
    pos_chrom, pos_start, pos_end = result["position"]

    # --- Save prediction bigWigs ---
    cache_dir = os.getenv(
        "BACKEND_PREDICTION_CACHE",
        os.path.join(os.path.dirname(__file__), "..", "..", "cache", "predictions"),
    )
    os.makedirs(cache_dir, exist_ok=True)

    safe_chrom = chromosome.replace(" ", "_")
    timestamp = time.time_ns()
    plus_bw_path = os.path.join(cache_dir, f"{safe_chrom}_{pos_start}_{pos_end}_plus_{timestamp}.bw")
    minus_bw_path = os.path.join(cache_dir, f"{safe_chrom}_{pos_start}_{pos_end}_minus_{timestamp}.bw")

    _array_to_bigwig(pred_plus, chromosome, pos_start, pos_end, plus_bw_path)
    _array_to_bigwig(pred_minus, chromosome, pos_start, pos_end, minus_bw_path)

    # --- Build IGV payload ---
    igv_payload = build_prediction_payload(
        genome=genome,
        chromosome=chromosome,
        start=pos_start,
        end=pos_end,
        atac_path=atac_path,
        plus_bw_path=plus_bw_path,
        minus_bw_path=minus_bw_path,
        genome_config=genome_config,
    )

    return {
        "igv_payload": igv_payload,
        "plus_bw_path": plus_bw_path,
        "minus_bw_path": minus_bw_path,
    }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------
def _validate_atac_bw(atac_path: str, chromosome: str):
    """Check that the ATAC bigWig exists and contains the target chromosome."""
    if not os.path.isfile(atac_path):
        raise FileNotFoundError(f"ATAC bigWig not found: {atac_path}")
    try:
        bw = pyBigWig.open(atac_path)
        chroms = bw.chroms()
        bw.close()
    except Exception as e:
        raise RuntimeError(f"Failed to open ATAC bigWig {atac_path}: {e}")

    if chromosome not in chroms:
        raise ValueError(
            f"Chromosome '{chromosome}' not found in ATAC bigWig '{atac_path}'. "
            f"Available: {list(chroms.keys())[:20]}"
        )


def _array_to_bigwig(
    values: np.ndarray,
    chromosome: str,
    start: int,
    end: int,
    output_path: str,
):
    """Write a 1-D numpy array as a single-interval bigWig file."""
    import pyBigWig as _bw

    n = len(values)
    step = (end - start) / n if n > 0 else 1
    with _bw.open(output_path, "w") as bw:
        bw.addHeader([(chromosome, max(end, 1))], maxZooms=0)
        bw.addEntries(
            [chromosome] * n,
            [int(start + i * step) for i in range(n)],
            ends=[int(start + (i + 1) * step) for i in range(n)],
            values=[float(v) for v in values],
        )
