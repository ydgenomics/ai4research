#!/usr/bin/env python3
"""Pre-generate genome-wide RNA-seq (+/-) bigWig tracks for RiceReg.

Motivation
----------
rice_reg predicts RNA expression from a (reference genome FASTA + ATAC bigWig)
pair.  For a FIXED (genome, ATAC) the per-window model output is deterministic,
so instead of running GPU inference on every /predict request we run a
full-genome sliding-window pass ONCE and cache the merged genome-wide signal.

Strategy B (this script) — no per-window files
----------------------------------------------
rice_mut writes one .bw per sliding window (tens of thousands) and merges them
afterwards; that intermediate file explosion is what we avoid here.

* Each worker accumulates per-base ``sum``/``count`` for a WHOLE chromosome
  IN MEMORY (window overlaps are averaged with the same semantics as rice_mut).
* On chromosome completion it writes ONE compact per-chromosome .npz per strand:
      parts/<Genome>__<Atac>/<chrom>_plus.npz
      parts/<Genome>__<Atac>/<chrom>_minus.npz
  (checkpoint granularity = chromosome, not window).
* A final merge stage concatenates the ~24 chromosome parts into exactly TWO
  genome-wide files per (genome, ATAC):
      <out-dir>/<Genome>__<Atac>_plus.bw
      <out-dir>/<Genome>__<Atac>_minus.bw

Output is two files per (genome, ATAC) because rice_reg predicts the plus and
minus strand separately (RNA-seq_+ / RNA-seq_-).

Usage
-----
    # whole genome, single GPU (paths read from rice_reg/.env)
    bash scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1

    # resume a partial run (keeps finished chromosomes)
    python scripts/pregen_bigwigs.py --genome MH63RS3 --atac SAM2_MH63_1 --resume

    # merge only (chromosome parts already computed)
    python scripts/pregen_bigwigs.py --genome MH63RS3 --atac SAM2_MH63_1 --merge-only

    # restrict to a subset (e.g. verify with chr1)
    python scripts/pregen_bigwigs.py --genome MH63RS3 --atac SAM2_MH63_1 --chrom Chr1
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyBigWig

# ---------------------------------------------------------------------------
#  Path setup — make `from rice_reg.core import RiceRegPredictor` importable.
#  core/__init__ injects its own directory so `from model.*` works.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # rice_reg/scripts/
_ROOT = _HERE.parent                             # rice_reg/
_BACKEND = str(_ROOT / "backend")                # rice_reg/backend/
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# NOTE: `from rice_reg.core import RiceRegPredictor` is deliberately deferred
# into build_predictor(). With multiprocessing 'spawn', each worker re-imports
# this __main__ script (runpy under __mp_main__); a module-level
# `from rice_reg.core import ...` fails there with "ModuleNotFoundError:
# 'rice_reg' is not a package" because the child's sys.path / package lookup
# is not settled while the script is being re-run. Importing lazily inside
# build_predictor() (once the worker has finished re-importing this module)
# avoids that entirely.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pregen_bigwigs")

STRANDS = ("plus", "minus")
CELL_TYPE = "sample"          # batch key prefix used by RiceRegPredictor
_PLUS_KEY = f"{CELL_TYPE}_plus"
_MINUS_KEY = f"{CELL_TYPE}_minus"
DEFAULT_OUT_DIR = _ROOT / "cache" / "pregen"
CHUNK_BP = 5_000_000          # bigWig addEntries block to bound memory

# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", str(name))


def _sha256_of_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _load_env_keyvals() -> Dict[str, str]:
    """Load <root>/.env as a plain dict; existing env vars win."""
    out: Dict[str, str] = {}
    env_file = _ROOT / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and os.environ.get(k) is None:
                out[k] = v
    return out


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
#  Env -> genome / ATAC resolution
# ---------------------------------------------------------------------------
def builtin_genomes() -> List[str]:
    return sorted(
        key[len("GENOME_"):-len("_FASTA")]
        for key in os.environ
        if key.startswith("GENOME_") and key.endswith("_FASTA") and os.environ[key]
    )


def builtin_atacs() -> List[str]:
    return sorted(
        key[len("ATAC_PATH_"):]
        for key in os.environ
        if key.startswith("ATAC_PATH_") and os.environ[key]
    )


def resolve_genome(genome: str) -> Dict[str, str]:
    fasta = _env_str(f"GENOME_{genome}_FASTA", "")
    if not fasta or not os.path.isfile(fasta):
        raise SystemExit(
            f"Genome '{genome}' FASTA not found. Set GENOME_{genome}_FASTA in .env "
            f"(available: {builtin_genomes()})."
        )
    cfg = {
        "fasta": fasta,
        "fai": _env_str(f"GENOME_{genome}_FAI", "") or fasta + ".fai",
        "gff": _env_str(f"GENOME_{genome}_GFF", ""),
    }
    if not os.path.isfile(cfg["fai"]):
        raise SystemExit(f"FAI not found: {cfg['fai']} (set GENOME_{genome}_FAI).")
    return cfg


def resolve_atac(atac_id: str) -> str:
    path = _env_str(f"ATAC_PATH_{atac_id}", "")
    if not path or not os.path.isfile(path):
        raise SystemExit(
            f"ATAC '{atac_id}' not found. Set ATAC_PATH_{atac_id} in .env "
            f"(available: {builtin_atacs()})."
        )
    return path


# ---------------------------------------------------------------------------
#  Sliding window positions (mirrors rice_mut pregen semantics)
# ---------------------------------------------------------------------------
def window_starts(chrom_len: int, window: int, hop: int) -> List[int]:
    """Window start positions covering [0, chrom_len).

    Fixed-hop windows (0, hop, 2*hop, ...) plus an end-aligned tail window so
    the last ``window`` bp are covered exactly.  Every window lies fully inside
    the chromosome, so no padding / cropping is needed during accumulation.
    """
    if chrom_len <= 0 or chrom_len < window:
        return []
    starts = list(range(0, chrom_len - window + 1, hop))
    tail = chrom_len - window
    if tail not in starts:
        starts.append(tail)
    return sorted(set(starts))


def _load_fai_chroms(fai_path: str) -> List[Tuple[str, int]]:
    """Return [(name, length)] from a .fai file (real FASTA names)."""
    if not fai_path or not os.path.isfile(fai_path):
        return []
    out: List[Tuple[str, int]] = []
    with open(fai_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                try:
                    out.append((parts[0], int(parts[1])))
                except ValueError:
                    continue
    return out


# ---------------------------------------------------------------------------
#  Predictor construction (same params as prediction_service.init_predictor)
# ---------------------------------------------------------------------------
def build_predictor(device: Optional[str]) -> "RiceRegPredictor":
    # Deferred import — see the NOTE near the top: importing rice_reg.core at
    # module level breaks spawn workers (they re-import __main__); here the
    # worker process has finished re-importing, so sys.path is stable.
    from rice_reg.core import RiceRegPredictor

    base_model_path = _env_str("BASE_MODEL_PATH", "")
    checkpoint_path = _env_str("CHECKPOINT_PATH", "")
    if not (base_model_path and checkpoint_path):
        raise SystemExit(
            "BASE_MODEL_PATH / CHECKPOINT_PATH not set in .env — cannot build predictor."
        )
    predictor = RiceRegPredictor(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        predictor_type=_env_str("PREDICTOR_TYPE", "fusion"),
        dataset_type=_env_str("DATASET_TYPE", "v0"),
        atac_encoder_output_dim=_env_int("ATAC_ENCODER_OUTPUT_DIM", 1024),
        target_len=_env_int("TARGET_LEN", 32678),
        overlap_len=_env_int("OVERLAP_LEN", 16000),
        track_mean_plus=_env_float("FIXED_TRACK_MEAN_PLUS", 2.83),
        track_mean_minus=_env_float("FIXED_TRACK_MEAN_MINUS", 2.85),
        model_torch_dtype=_env_str("MODEL_TORCH_DTYPE", "bfloat16"),
        inference_batch_size=_env_int("INFERENCE_BATCH_SIZE", 4),
        device=device,
    )
    predictor.initialize()
    logger.info("Predictor ready on %s (target_len=%d)", predictor.device, predictor.target_len)
    return predictor


# ---------------------------------------------------------------------------
#  Part (chromosome-level checkpoint) I/O
# ---------------------------------------------------------------------------
def parts_dir_for(args) -> Path:
    return Path(args.out_dir) / "parts" / f"{_safe_name(args.genome)}__{_safe_name(args.atac)}"


def part_path(args, chromosome: str, strand: str) -> Path:
    return parts_dir_for(args) / f"{_safe_name(chromosome)}_{strand}.npz"


def save_part(args, chromosome: str, strand: str, mean: np.ndarray) -> None:
    parts_dir_for(args).mkdir(parents=True, exist_ok=True)
    np.savez(part_path(args, chromosome, strand), values=np.asarray(mean, dtype=np.float32))


def load_part(args, chromosome: str, strand: str) -> Optional[np.ndarray]:
    p = part_path(args, chromosome, strand)
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["values"], dtype=np.float32)


# ---------------------------------------------------------------------------
#  Per-chromosome inference + in-memory overlap averaging
# ---------------------------------------------------------------------------
def _run_chromosome(args, meta: Tuple[str, int], predictor: RiceRegPredictor) -> Tuple[int, int]:
    """Run all windows of one chromosome; write plus/minus .npz parts.

    Returns (n_windows_done, n_windows_skipped).  No per-window files are ever
    written — results accumulate into per-base sums in memory.
    """
    chrom, chrom_len = meta
    window = predictor.target_len
    starts = window_starts(chrom_len, window, args.hop)
    if not starts:
        logger.info("[%s] len=%d < window=%d — skip", chrom, chrom_len, window)
        return 0, 0

    if args.resume and (
        part_path(args, chrom, "plus").is_file() and part_path(args, chrom, "minus").is_file()
    ):
        logger.info("[%s] parts exist — skip (resume)", chrom)
        return 0, len(starts)

    plus_sum = np.zeros(chrom_len, dtype=np.float64)
    plus_cnt = np.zeros(chrom_len, dtype=np.float64)
    minus_sum = np.zeros(chrom_len, dtype=np.float64)
    minus_cnt = np.zeros(chrom_len, dtype=np.float64)

    n = len(starts)
    done = 0
    t0 = time.time()

    def _rows(b0: int, b1: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "chromosome": chrom,
                    "start": starts[i],
                    "end": starts[i] + window,
                    "cell_type": CELL_TYPE,
                    "fasta_path": args.fasta,
                    "atac_path": args.atac_path,
                    "track_mean_plus": predictor.track_mean_plus,
                    "track_mean_minus": predictor.track_mean_minus,
                }
                for i in range(b0, b1)
            ]
        )

    for b0 in range(0, n, args.chunk_windows):
        b1 = min(b0 + args.chunk_windows, n)
        df = _rows(b0, b1)
        plus_results, minus_results = predictor.predict_batch(df)
        plus_list = plus_results.get(_PLUS_KEY, [])
        minus_list = minus_results.get(_MINUS_KEY, [])
        if len(plus_list) != (b1 - b0) or len(minus_list) != (b1 - b0):
            raise RuntimeError(
                f"[{chrom}] expected {b1 - b0} window results, got "
                f"plus={len(plus_list)} minus={len(minus_list)}"
            )
        for i, s in enumerate(starts[b0:b1]):
            arr_p = np.asarray(plus_list[i]["predicted_expression"], dtype=np.float64).reshape(-1)
            arr_m = np.asarray(minus_list[i]["predicted_expression"], dtype=np.float64).reshape(-1)
            e = min(s + max(len(arr_p), len(arr_m)), chrom_len)  # defensive crop
            lp, lm = e - s, e - s
            plus_sum[s:e] += arr_p[:lp]
            plus_cnt[s:e] += 1.0
            minus_sum[s:e] += arr_m[:lm]
            minus_cnt[s:e] += 1.0
            done += 1
        # let GC reclaim per-window arrays before the next chunk
        del plus_results, minus_results
        gc.collect()
        if done % 200 == 0:
            logger.info(
                "  [%s] %d/%d windows (%.1f/s)",
                chrom, done, n, done / max(time.time() - t0, 1e-6),
            )

    # mean over overlaps; NaN marks bases covered by no window
    def _mean(sums: np.ndarray, cnts: np.ndarray) -> np.ndarray:
        out = np.full(chrom_len, np.nan, dtype=np.float32)
        valid = cnts > 0
        out[valid] = (sums[valid] / cnts[valid]).astype(np.float32)
        return out

    save_part(args, chrom, "plus", _mean(plus_sum, plus_cnt))
    save_part(args, chrom, "minus", _mean(minus_sum, minus_cnt))
    logger.info("[%s] done: %d windows (%ds) -> parts saved", chrom, n, time.time() - t0)
    return n, 0


def _run_windows_for_chromosomes(args, metas: List[Tuple[str, int]], device: str) -> Tuple[int, int]:
    predictor = build_predictor(device)
    total_done = total_skipped = 0
    try:
        for meta in metas:
            try:
                done, skipped = _run_chromosome(args, meta, predictor)
            except Exception:
                logger.exception("[%s] failed", meta[0])
                raise
            total_done += done
            total_skipped += skipped
    finally:
        try:
            predictor.release()
        except Exception:
            pass
    return total_done, total_skipped


# ---------------------------------------------------------------------------
#  Merge: chromosome parts -> two genome-wide .bw
# ---------------------------------------------------------------------------
def write_genomewide_bw(
    chroms: List[Tuple[str, int]],
    strand: str,
    out_path: str,
    args,
    tracks_by_chrom: Dict[str, np.ndarray],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pyBigWig.open(out_path, "w") as bw:
        bw.addHeader(chroms, maxZooms=0)
        for chrom, chrom_len in chroms:
            arr = tracks_by_chrom.get(chrom)
            if arr is None or len(arr) != chrom_len:
                logger.warning("  [%s] missing part for %s strand — skipped", chrom, strand)
                continue
            n = chrom_len
            for c0 in range(0, n, CHUNK_BP):
                c1 = min(c0 + CHUNK_BP, n)
                m = c1 - c0
                bw.addEntries(
                    [chrom] * m,
                    list(range(c0, c1)),
                    ends=list(range(c0 + 1, c1 + 1)),
                    values=[float(v) for v in arr[c0:c1]],
                )
    logger.info("  wrote %s (%d chromosomes)", out_path, len(chroms))


def run_merge(args, genome_cfg: Dict[str, str], chroms_meta: List[Tuple[str, int]]) -> int:
    stem = f"{_safe_name(args.genome)}__{_safe_name(args.atac)}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    merged_files: Dict[str, str] = {}
    manifest: dict = {
        "genome": args.genome,
        "atac": args.atac,
        "fasta": genome_cfg["fasta"],
        "fasta_sha256": _sha256_of_file(genome_cfg["fasta"]),
        "atac_path": args.atac_path,
        "atac_sha256": _sha256_of_file(args.atac_path),
        "checkpoint": _env_str("CHECKPOINT_PATH", ""),
        "checkpoint_sha256": (
            _sha256_of_file(_env_str("CHECKPOINT_PATH"))
            if os.path.isfile(_env_str("CHECKPOINT_PATH", ""))
            else ""
        ),
        "window": args.window,
        "hop": args.hop,
        "tracks": list(STRANDS),
        "chromosomes": {},
        "parts": {},
        "merged_files": {},
    }

    for meta in chroms_meta:
        chrom, chrom_len = meta
        manifest["chromosomes"][chrom] = {
            "length": chrom_len,
            "window": args.window,
            "hop": args.hop,
            "windows": [
                {"start": s, "end": s + args.window}
                for s in window_starts(chrom_len, args.window, args.hop)
            ],
        }

    for strand in STRANDS:
        tracks_by_chrom: Dict[str, np.ndarray] = {}
        for chrom, chrom_len in chroms_meta:
            arr = load_part(args, chrom, strand)
            if arr is not None and len(arr) == chrom_len:
                tracks_by_chrom[chrom] = arr
            else:
                logger.warning("  [%s] no part for %s — merge will leave NaN bases", chrom, strand)
        out_path = str(out_dir / f"{stem}_{strand}.bw")
        write_genomewide_bw(chroms_meta, strand, out_path, args, tracks_by_chrom)
        merged_files[strand] = out_path
        manifest["merged_files"][strand] = out_path

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest written: %s", manifest_path)
    logger.info("Merge done in %.1fs", time.time() - t0)

    if args.cleanup:
        parts_dir = parts_dir_for(args)
        if parts_dir.is_dir():
            import shutil
            shutil.rmtree(parts_dir, ignore_errors=True)
            logger.info("Cleaned up parts: %s", parts_dir)
    return 0


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome", default="",
                        help="built-in genome id (GENOME_<id>_FASTA in .env)")
    parser.add_argument("--atac", default="",
                        help="built-in ATAC id (ATAC_PATH_<id> in .env)")
    parser.add_argument("--chrom", nargs="*", default=None,
                        help="real FASTA chromosome names to process (default: all in .fai)")
    parser.add_argument("--hop", type=int, default=0,
                        help="sliding hop (default: target_len // 2)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="output root (default: <rice_reg>/cache/pregen)")
    parser.add_argument("--chunk-windows", type=int, default=256,
                        help="windows per predict_batch call (memory/throughput trade-off)")
    parser.add_argument("--device", default="",
                        help="GPU for THIS worker (single-process mode); default from .env")
    parser.add_argument("--gpus", type=int, default=0,
                        help="number of GPUs when --workers > 1 (default = --workers)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes, one predictor/GPU each")
    parser.add_argument("--resume", action="store_true",
                        help="skip chromosomes whose parts already exist")
    parser.add_argument("--merge-only", action="store_true",
                        help="only merge existing chromosome parts into final .bw")
    parser.add_argument("--no-merge", action="store_true",
                        help="only run inference to parts; skip final merge")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete parts dir after a successful merge")
    args = parser.parse_args()

    # Load <root>/.env into os.environ (existing env vars win)
    os.environ.update(_load_env_keyvals())

    # --- Resolve genome + ATAC --------------------------------------------
    genomes = builtin_genomes()
    if not args.genome:
        args.genome = genomes[0] if len(genomes) == 1 else ""
    if not args.genome:
        raise SystemExit(f"Pass --genome. Available: {genomes}")
    if args.genome not in genomes:
        raise SystemExit(f"Unknown genome '{args.genome}'. Available: {genomes}")

    atacs = builtin_atacs()
    if not args.atac:
        args.atac = atacs[0] if len(atacs) == 1 else ""
    if not args.atac:
        raise SystemExit(f"Pass --atac. Available: {atacs}")
    if args.atac not in atacs:
        raise SystemExit(f"Unknown ATAC '{args.atac}'. Available: {atacs}")

    genome_cfg = resolve_genome(args.genome)
    args.fasta = genome_cfg["fasta"]
    args.atac_path = resolve_atac(args.atac)

    # window is fixed to the model's target_len (32678) — a different value
    # would misalign per-base output with window coordinates.
    args.window = _env_int("TARGET_LEN", 32678)
    if args.hop <= 0:
        args.hop = args.window // 2

    chroms_meta = _load_fai_chroms(genome_cfg["fai"])
    if args.chrom:
        wanted = set(args.chrom)
        chroms_meta = [c for c in chroms_meta if c[0] in wanted]
    if not chroms_meta:
        raise SystemExit("No chromosomes selected / found in .fai")

    if not args.merge_only:
        # --- Validate ATAC chromosome naming against the FASTA -------------
        try:
            bw = pyBigWig.open(args.atac_path)
            atac_chroms = set(bw.chroms().keys())
            bw.close()
        except Exception as e:
            raise SystemExit(f"Cannot open ATAC bigWig {args.atac_path}: {e}")
        missing = [c for c, _ in chroms_meta if c not in atac_chroms]
        if missing:
            logger.warning(
                "Chromosome names in FASTA missing from ATAC bigWig: %s\n"
                "  ATAC chroms: %s\n"
                "  If the ATAC bigWig uses a different naming (e.g. chr1 vs Chr1), "
                "create a renamed copy first — this script reads ATAC with the "
                "FASTA chromosome names.",
                missing[:20], sorted(atac_chroms)[:20],
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Multiprocessing mode ---------------------------------------------
    if args.workers > 1 and not args.merge_only:
        import multiprocessing as mp

        n_gpus = args.gpus or args.workers
        workers = args.workers if args.workers % n_gpus == 0 else n_gpus
        per_worker: List[List[Tuple[str, int]]] = [[] for _ in range(workers)]
        for i, meta in enumerate(chroms_meta):
            per_worker[i % workers].append(meta)
        if not any(per_worker):
            raise SystemExit("No chromosomes selected (worker subset empty)")

        from argparse import Namespace

        ctx = mp.get_context("spawn")
        procs = []
        for w in range(workers):
            if not per_worker[w]:
                continue
            wargs = Namespace(**vars(args))
            wargs.workers = workers
            wargs.device = f"cuda:{w % n_gpus}"
            wargs.chrom = [m[0] for m in per_worker[w]]
            wargs.merge_only = False
            wargs.no_merge = True
            p = ctx.Process(target=_run_worker, args=(wargs, per_worker[w]), name=f"pregen-w{w}")
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = [p for p in procs if p.exitcode != 0]
        if failed:
            logger.error("%d worker(s) failed; see above", len(failed))
            return 1
        logger.info("All workers finished. Merging ...")

    # --- Single-process inference (also used after multi-worker inference) --
    elif not args.merge_only:
        device = args.device or _env_str("DEVICE", "")
        done, skipped = _run_windows_for_chromosomes(args, chroms_meta, device or None)
        logger.info("Inference done: %d computed, %d skipped (resume)", done, skipped)

    # --- Merge --------------------------------------------------------------
    if not args.no_merge:
        return run_merge(args, genome_cfg, chroms_meta)
    return 0


def _run_worker(wargs, metas):
    """Top-level worker entry (required for spawn)."""
    try:
        _run_windows_for_chromosomes(wargs, metas, wargs.device)
    except Exception:
        logger.exception("Worker %s failed", wargs.device)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
