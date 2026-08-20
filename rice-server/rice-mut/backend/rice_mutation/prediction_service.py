"""Prediction service — singleton predictor lifecycle and core prediction logic.

References:
- rice-reg-server2/backend/rice_reg/prediction_service.py
- inference.ipynb (MultiTrackPredictor)
"""

from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import pyBigWig
import pyfaidx

from rice_mutation.core.predictor import RiceMutationPredictor
from src.dataset import load_fasta_sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Chromosome normalization cache
# ---------------------------------------------------------------------------
# genome -> {input_alias: real_fasta_chrom}
_CHROM_ALIAS_CACHE: Dict[str, Dict[str, str]] = {}


def _chrom_key(chrom: str) -> str:
    """Canonical numeric key for a chromosome alias.

    'chr01' / 'Chr1' / '1' / 'chr1' / 'CHR01' -> '1'  (also strips 'chr'/'Chr'/'CHR').
    Non-numeric names (e.g. 'ChrUn', 'Mt', 'Pt') are returned lower-cased.
    """
    if chrom is None:
        return ""
    s = str(chrom).strip()
    m = re.fullmatch(r"(?:chr|Chr|CHR)?0*([1-9]\d*)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    return s.lower()


def _load_fai_chroms(fai_path: str) -> list:
    """Read the actual chromosome names from a .fai file (first column)."""
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


def build_chrom_alias_map(genome: str, genome_config: Optional[dict] = None) -> Dict[str, str]:
    """Build a map from every plausible chromosome alias to the real FASTA name.

    Handles the common mismatch where the front-end uses ``chr01``-``chr12``
    while the FASTA uses ``Chr1``, ``chr1``, ``1``, ``Chr01`` etc.
    """
    if genome in _CHROM_ALIAS_CACHE:
        return _CHROM_ALIAS_CACHE[genome]

    fai = (genome_config or {}).get("fai", "")
    if not fai:
        fai = _env_str(f"GENOME_{genome}_FAI", "")
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
    # fallback: try canonical key lookup even if map was built from a partial fai
    key = _chrom_key(chrom)
    for real in alias_map.values():
        if _chrom_key(real) == key:
            return real
    raise ValueError(
        f"Chromosome '{chromosome}' not found in genome '{genome}' "
        f"(available: {sorted(set(alias_map.values()))})"
    )


# ---------------------------------------------------------------------------
#  Uploaded (custom) genome registry + FASTA cache
# ---------------------------------------------------------------------------
# genome_id -> {"fasta": abs_path, "fai": abs_path, "gff": "", "last_used": epoch}
_UPLOADED_GENOMES: Dict[str, Dict[str, Any]] = {}
# genome_id -> pyfaidx.Fasta (opened lazily, reused across requests)
_FASTA_CACHE: Dict[str, pyfaidx.Fasta] = {}
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

    Removes the in-memory registry entry, closes the cached ``pyfaidx.Fasta``
    handle, and deletes the on-disk FASTA / .fai / GFF files (all are
    per-upload temporary files).  Returns the number of genomes removed.
    """
    cutoff = time.time() - float(ttl_hours) * 3600
    expired: list = []
    with _REGISTRY_LOCK:
        for gid, cfg in list(_UPLOADED_GENOMES.items()):
            if float(cfg.get("last_used", 0)) < cutoff:
                expired.append((gid, cfg))
        for gid, _ in expired:
            _UPLOADED_GENOMES.pop(gid, None)
            inst = _FASTA_CACHE.pop(gid, None)
            if inst is not None:
                try:
                    inst.close()
                except Exception:
                    pass

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


def display_chromosome(name: str) -> str:
    """Standard display name for a real FASTA chromosome.

    Numeric chromosomes are shown as ``chrNN`` (chr01..chr12), everything
    else (e.g. ChrUn, scaffold_1) keeps its original name.  ``normalize_``
    ``chromosome`` maps these back to the real name.
    """
    m = re.fullmatch(r"(?:chr|Chr|CHR)?0*([1-9]\d*)", str(name).strip(), re.IGNORECASE)
    if m:
        return f"chr{int(m.group(1)):02d}"
    return str(name)


def get_genome_chromosomes(genome: str, genome_config: Optional[dict] = None) -> list:
    """Return display chromosome names for a genome (real FASTA names mapped
    to the unified chrNN style where possible)."""
    cfg = genome_config or resolve_genome_config(genome)
    real_names = _load_fai_chroms(cfg.get("fai", ""))
    if not real_names and cfg.get("fasta"):
        # No .fai yet — build it via pyfaidx (also writes the .fai file).
        try:
            fa = pyfaidx.Fasta(cfg["fasta"])
            real_names = list(fa.keys())
        except Exception as e:
            logger.warning("Could not index FASTA %s: %s", cfg.get("fasta"), e)
    return [display_chromosome(n) for n in real_names]


def register_uploaded_genome(fasta_path: str, genome_id: str = "") -> Dict[str, str]:
    """Register an uploaded FASTA as a usable genome (priority over built-in).

    - Builds the .fai index automatically if missing (pyfaidx).
    - Validates the file and reads chromosome names.
    - Stores the config in the in-memory registry.

    Returns the genome config dict.
    """
    fasta_path = os.path.abspath(fasta_path)
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    fai_path = fasta_path + ".fai"
    if not os.path.isfile(fai_path):
        # Build the index (pyfaidx writes the .fai next to the file).
        fa = pyfaidx.Fasta(fasta_path)
        names = list(fa.keys())
    else:
        names = _load_fai_chroms(fai_path)
    if not names:
        raise ValueError(f"No sequence records found in FASTA: {fasta_path}")

    if not genome_id:
        genome_id = f"custom_{int(time.time())}"
    # avoid clobbering an existing id
    while genome_id in _UPLOADED_GENOMES or genome_id in _builtin_genome_ids():
        genome_id = f"custom_{int(time.time())}_{len(_UPLOADED_GENOMES)}"

    with _REGISTRY_LOCK:
        cfg = {"fasta": fasta_path, "fai": fai_path, "gff": "", "last_used": time.time()}
        _UPLOADED_GENOMES[genome_id] = cfg
    logger.info(
        "Registered uploaded genome '%s' (%d chromosomes): %s",
        genome_id, len(names), fasta_path,
    )
    return cfg


def attach_uploaded_gff(genome_id: str, gff_path: str) -> Dict[str, str]:
    """Attach an annotation GFF to an uploaded (custom) genome.

    Only genomes in the in-memory upload registry can be annotated.  The GFF
    path is stored in the registry so IGV payloads include a Genes track for
    the custom genome.  Returns the updated genome config.
    """
    gff_path = os.path.abspath(gff_path)
    if not os.path.isfile(gff_path):
        raise FileNotFoundError(f"GFF not found: {gff_path}")
    cfg = _UPLOADED_GENOMES.get(genome_id)
    if cfg is None:
        raise ValueError(
            f"Genome '{genome_id}' is not an uploaded custom genome. "
            "Upload a FASTA first, then attach a GFF."
        )
    with _REGISTRY_LOCK:
        cfg["gff"] = gff_path
        cfg["last_used"] = time.time()
        _UPLOADED_GENOMES[genome_id] = cfg
    logger.info("Attached GFF to uploaded genome '%s': %s", genome_id, gff_path)
    return dict(cfg)


def _builtin_genome_ids() -> list:
    return [
        key[len("GENOME_"):-len("_FASTA")]
        for key, val in sorted(os.environ.items())
        if key.startswith("GENOME_") and key.endswith("_FASTA") and val
    ]


def get_fasta(genome: str, genome_config: Optional[dict] = None) -> pyfaidx.Fasta:
    """Return the (cached) pyfaidx.Fasta for a genome."""
    _touch_uploaded_genome(genome)
    inst = _FASTA_CACHE.get(genome)
    if inst is not None:
        return inst
    cfg = genome_config or resolve_genome_config(genome)
    fa = pyfaidx.Fasta(cfg["fasta"])
    _FASTA_CACHE[genome] = fa
    return fa


# ---------------------------------------------------------------------------
#  SNV (single-nucleotide variant) prediction
# ---------------------------------------------------------------------------
_VALID_BASES = set("ACGTN")


def run_snv_core(
    genome: str,
    chromosome: str,
    start: int,
    end: int,
    snv_index: int,
    snv_base: str,
    biosample_names: Optional[list] = None,
    genome_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run single-nucleotide-variant prediction: reference vs one-base-mutated.

    The reference window is center-aligned to ``max_seq_len``; ``snv_index``
    is the **absolute genomic position** (0-based bp coordinate) of the
    variant.  It is converted to a window-relative index internally.  The base
    at that position is replaced by ``snv_base`` (A/C/G/T/N) and both
    sequences are predicted.  Returns two value sets plus the original base.

    Returns dict with keys ``ref_values``, ``mut_values``, ``position``
    (real FASTA name + aligned start/end), ``ref_base``, ``snv_index``
    (absolute), ``snv_rel_index`` (window-relative), ``snv_base``.
    """
    predictor = require_predictor()
    max_len = predictor.max_seq_len

    chromosome = normalize_chromosome(genome, chromosome, genome_config)
    start, end = _adjust_window(start, end, max_len)

    snv_index = int(snv_index)
    snv_base = str(snv_base or "").strip().upper()
    if snv_base not in _VALID_BASES:
        raise ValueError(f"Invalid SNV base '{snv_base}'. Use one of A/C/G/T/N.")

    # snv_index is the absolute genomic position; map it into the window
    rel_index = snv_index - start
    if rel_index < 0 or rel_index >= (end - start):
        raise ValueError(
            f"SNV position {snv_index + 1} is outside the prediction window "
            f"{chromosome}:{start + 1}-{end}. Choose a position inside the window."
        )

    fasta = get_fasta(genome, genome_config)
    # Fetch the aligned window sequence (same as reference prediction path)
    seq = str(load_fasta_sequence(
        fasta, chromosome, start, end, max_length=max_len
    ))
    ref_base = seq[rel_index]
    mut_seq = seq[:rel_index] + snv_base + seq[rel_index + 1:]

    # result1 = reference (unchanged) ; result2 = SNV-mutated
    ref_result = predictor.predict(
        chrom=chromosome, start=start, end=end,
        biosample_names=biosample_names, fasta=fasta,
    )
    mut_result = predictor.predict_sequence(
        sequence=mut_seq,
        chrom=chromosome, start=start, end=end,
        biosample_names=biosample_names,
    )

    return {
        "ref_values": ref_result,
        "mut_values": mut_result,
        "position": (chromosome, start, end),
        "genome": genome,
        "chromosome": chromosome,
        "ref_base": ref_base,
        "snv_index": snv_index,
        "snv_rel_index": rel_index,
        "snv_base": snv_base,
        "window_len": end - start,
    }


def compute_region_stats(
    ref_values: Dict[str, Dict[str, np.ndarray]],
    mut_values: Dict[str, Dict[str, np.ndarray]],
    start_idx: int,
    end_idx: int,
) -> Dict[str, Any]:
    """Compute difference stats between result1 (ref) and result2 (mut) arrays
    over the region [start_idx, end_idx) (0-based, relative to the window).

    Returns per (assay, biosample): total-sum diff %, mean diff %, max-diff
    point, region length.  Uses (result1 - result2) / result1.
    """
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx <= start_idx:
        raise ValueError("Region end must be greater than region start.")

    stats: Dict[str, Any] = {}
    for assay, bios_map in ref_values.items():
        mut_map = mut_values.get(assay, {})
        stats[assay] = {}
        for bios, r1 in bios_map.items():
            r2 = mut_map.get(bios)
            if r2 is None or len(r2) != len(r1):
                raise ValueError(
                    f"Mismatched result1/result2 length for {assay}/{bios}"
                )
            a = max(0, min(start_idx, len(r1)))
            b = max(0, min(end_idx, len(r1)))
            if b <= a:
                raise ValueError("Region out of range for track array.")
            x1 = np.asarray(r1[a:b], dtype=np.float64)
            x2 = np.asarray(r2[a:b], dtype=np.float64)

            sum1 = float(x1.sum())
            sum2 = float(x2.sum())
            total_pct = ((sum1 - sum2) / sum1 * 100.0) if abs(sum1) > 1e-12 else None

            diffs = x1 - x2
            mean_pct = (float(diffs.mean()) / float(x1.mean()) * 100.0) \
                if abs(x1.mean()) > 1e-12 else None

            # max absolute diff point (relative to window start)
            max_idx = int(np.argmax(np.abs(diffs)))
            abs_diff = float(diffs[max_idx])
            max_pct = (abs_diff / float(x1[max_idx]) * 100.0) \
                if abs(float(x1[max_idx])) > 1e-12 else None

            stats[assay][bios] = {
                "region": [a, b],
                "region_len": b - a,
                "sum_result1": round(sum1, 6),
                "sum_result2": round(sum2, 6),
                "total_diff_pct": (
                    round(total_pct, 6) if total_pct is not None else None
                ),
                "mean_diff_pct": (
                    round(mean_pct, 6) if mean_pct is not None else None
                ),
                "max_diff_index": a + max_idx,
                "max_abs_diff": round(abs_diff, 6),
                "max_diff_pct": (
                    round(max_pct, 6) if max_pct is not None else None
                ),
            }
    return stats


def compute_region_means(
    values: Dict[str, Dict[str, np.ndarray]],
    start_idx: int,
    end_idx: int,
) -> Dict[str, Dict[str, float]]:
    """Compute the mean expression of each (assay, biosample) track over the
    window-index region [start_idx, end_idx).

    Used by the frontend bar plot to summarise result1/result2 over the
    current IGV viewport.  Returns ``{assay: {biosample: mean}}``.
    """
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx <= start_idx:
        raise ValueError("Region end must be greater than region start.")

    means: Dict[str, Dict[str, float]] = {}
    for assay, bios_map in values.items():
        means[assay] = {}
        for bios, arr in bios_map.items():
            a = max(0, min(start_idx, len(arr)))
            b = max(0, min(end_idx, len(arr)))
            if b <= a:
                raise ValueError(
                    f"Region out of range for track array ({assay}/{bios})."
                )
            x = np.asarray(arr[a:b], dtype=np.float64)
            means[assay][bios] = round(float(x.mean()), 6)
    return means


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


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# ---------------------------------------------------------------------------
#  Public lifecycle
# ---------------------------------------------------------------------------
def init_predictor():
    """Initialise the singleton RiceMutationPredictor from environment variables."""
    if _PREDICTOR.get("instance") is not None:
        logger.info("Predictor already initialised, skipping.")
        return

    base_model_path = _env_str("BASE_MODEL_PATH", "")
    checkpoint_path = _env_str("CHECKPOINT_PATH", "")
    index_stat_path = _env_str("INDEX_STAT_PATH", "")
    fasta_path = _env_str("GENOME_osa1_r7_FASTA", _env_str("GENOME_FASTA", ""))
    if not (base_model_path and checkpoint_path and index_stat_path and fasta_path):
        logger.warning(
            "BASE_MODEL_PATH / CHECKPOINT_PATH / INDEX_STAT_PATH / GENOME_*_FASTA "
            "not set — predictor will not be available until .env is configured."
        )
        return

    predictor = RiceMutationPredictor(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        index_stat_path=index_stat_path,
        fasta_path=fasta_path,
        use_flash_attn=_env_bool("USE_FLASH_ATTN", True),
        device=_env_str("DEVICE", "cuda:0"),
        torch_dtype=_env_str("MODEL_TORCH_DTYPE", "bfloat16"),
        max_seq_len=_env_int("MAX_SEQ_LEN", 32768),
        proj_dim=_env_int("PROJ_DIM", 1024),
        num_downsamples=_env_int("NUM_DOWNSAMPLES", 4),
        bottleneck_dim=_env_int("BOTTLENECK_DIM", 1536),
        inference_batch_size=_env_int("INFERENCE_BATCH_SIZE", 1),
    )
    predictor.initialize()
    _PREDICTOR["instance"] = predictor
    logger.info("RiceMutationPredictor initialised and ready.")


def require_predictor() -> RiceMutationPredictor:
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
#  Genome config resolution
# ---------------------------------------------------------------------------
def resolve_genome_config(genome: str) -> dict:
    """Resolve a genome's FASTA / FAI / GFF config.

    Uploaded (custom) genomes take priority over the built-in ones from
    environment variables.
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


def list_genomes() -> list:
    """All configured genome IDs (built-in from env + uploaded)."""
    ids = _builtin_genome_ids()
    for gid in _UPLOADED_GENOMES:
        if gid not in ids:
            ids.append(gid)
    return ids


# ---------------------------------------------------------------------------
#  Core prediction
# ---------------------------------------------------------------------------
def _adjust_window(start: int, end: int, target_len: int) -> tuple:
    """Center-align window to ``target_len`` (mirrors rice-reg-server2)."""
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
    biosample_names: Optional[list] = None,
    genome_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run reference-sequence prediction for a single genomic window.

    The FASTA is chosen per-genome: an uploaded custom genome overrides the
    built-in one.  Returns dict with keys ``values``
    (``{assay: {biosample: np.ndarray[L]}}``), ``position`` and
    ``genome``/``chromosome`` (real FASTA name).
    """
    predictor = require_predictor()
    max_len = predictor.max_seq_len

    # Normalize user-supplied chromosome (chr01 -> Chr1 etc.) against the .fai
    chromosome = normalize_chromosome(genome, chromosome, genome_config)
    start, end = _adjust_window(start, end, max_len)

    fasta = get_fasta(genome, genome_config)
    result = predictor.predict(
        chrom=chromosome,
        start=start,
        end=end,
        biosample_names=biosample_names,
        fasta=fasta,
    )
    return {
        "values": result,
        "position": (chromosome, start, end),
        "genome": genome,
        "chromosome": chromosome,
    }


# ---------------------------------------------------------------------------
#  bigWig helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  bigWig helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """Sanitise track/biosample names for file names."""
    return re_sub_safe(name)


def re_sub_safe(name: str) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)


def _array_to_bigwig(
    values: np.ndarray,
    chromosome: str,
    start: int,
    end: int,
    output_path: str,
):
    """Write a 1-D numpy array as a single-interval bigWig file."""
    n = len(values)
    if n <= 0:
        raise ValueError("Empty values array — cannot write bigWig.")
    step = (end - start) / n
    with pyBigWig.open(output_path, "w") as bw:
        bw.addHeader([(chromosome, max(end, 1))], maxZooms=0)
        bw.addEntries(
            [chromosome] * n,
            [int(start + i * step) for i in range(n)],
            ends=[int(start + (i + 1) * step) for i in range(n)],
            values=[float(v) for v in values],
        )


def save_prediction_bigwigs(
    values: Dict[str, Dict[str, np.ndarray]],
    chromosome: str,
    start: int,
    end: int,
    tag: str = "ref",
) -> Dict[str, str]:
    """Write all (assay, biosample) tracks to bigWig files.

    Returns a dict ``{(assay, biosample, tag): path}``.
    """
    cache_dir = os.path.abspath(os.getenv(
        "BACKEND_PREDICTION_CACHE",
        os.path.join(os.path.dirname(__file__), "..", "..", "cache", "predictions"),
    ))
    os.makedirs(cache_dir, exist_ok=True)

    timestamp = int(time.time())
    safe_chrom = _safe_name(chromosome)
    paths: Dict[str, str] = {}
    for assay, bios_map in values.items():
        for bios, arr in bios_map.items():
            fname = (
                f"{safe_chrom}_{start}_{end}_"
                f"{_safe_name(assay)}_{_safe_name(bios)}_{tag}_{timestamp}.bw"
            )
            out_path = os.path.join(cache_dir, fname)
            _array_to_bigwig(arr, chromosome, start, end, out_path)
            paths[f"{assay}|{bios}|{tag}"] = out_path
    return paths
