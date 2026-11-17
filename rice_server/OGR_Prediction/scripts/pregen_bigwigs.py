#!/usr/bin/env python3
"""Pre-generate genome-wide bigWig tracks for the built-in osa1_r7 genome.

The model output for a fixed reference genome is deterministic, so instead of
running GPU inference on every /predict request we run a full-genome sliding
window pass ONCE (32768 bp windows, 50% overlap), keep the raw per-window
tracks, and merge them into a single genome-wide .bw per (assay, biosample).

At query time the backend reads the region directly from the merged .bw
(see backend/rice_mutation/pregen_bigwigs.py) — NO model call.  SNV prediction
and uploaded custom genomes still run the model as before.

Layout (under <pregen-dir>, default rice_mut/cache/pregen/osa1_r7):

    manifest.json                       # index + genome FASTA / ckpt hashes
    <Track>.bw                          # merged genome-wide file (query target)
    windows/<chrom>/
        <start>_<end>__<Track>.bw       # per-window track (resume-friendly)

Multiprocessing: each worker process owns its own predictor instance on its
own GPU (one GPU per worker), and processes a disjoint set of chromosomes.
The final merge happens on the main process after all workers finish.

Usage:
    # full run, one worker per detected GPU (e.g. 2 A40 -> 2 workers)
    bash scripts/pregen_bigwigs.sh

    # resume a partial run (keeps finished windows)
    python scripts/pregen_bigwigs.py --resume

    # merge only (windows already computed)
    python scripts/pregen_bigwigs.py --merge-only

    # run only a subset (e.g. verify with Chr1)
    python scripts/pregen_bigwigs.py --chrom Chr1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyBigWig

# backend/ holds the `src` package (from src.model import GenOmics) and the
# rice_mutation package; make it importable from any cwd.
_HERE = Path(__file__).resolve().parent                # scripts/
_ROOT = _HERE.parent                                   # rice_mut/
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from rice_mutation.core.predictor import RiceMutationPredictor  # noqa: E402
from rice_mutation.prediction_service import (                    # noqa: E402
    _env_bool,
    _env_int,
    _env_str,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pregen_bigwigs")

DEFAULT_WINDOW = 32768
DEFAULT_HOP = 16384  # 50% overlap
DEFAULT_PREGEN_DIR = _ROOT / "cache" / "pregen" / "osa1_r7"


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


def _short_hash(path: str) -> str:
    try:
        return _sha256_of_file(path)[:8]
    except OSError:
        return "unknown"


def _load_env_keyvals() -> Dict[str, str]:
    """Load .env as a plain dict (defaults come from .env)."""
    env_file = _ROOT / ".env"
    out: Dict[str, str] = {}
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def track_key(assay: str, biosample: str) -> str:
    """Track key used for file naming: ``<Assay>__<Biosample>`` (sanitised)."""
    return f"{_safe_name(assay)}__{_safe_name(biosample)}"


# ---------------------------------------------------------------------------
#  Sliding window positions
# ---------------------------------------------------------------------------
def window_starts(chrom_len: int, window: int, hop: int) -> List[int]:
    """Window start positions covering [0, chrom_len).

    Fixed-hop windows (0, hop, 2*hop, ...) plus an end-aligned tail window so
    the last `window` bp are covered exactly.
    """
    if chrom_len <= 0:
        return []
    starts = list(range(0, chrom_len - window + 1, hop))
    if chrom_len > window:
        tail = chrom_len - window
        if tail not in starts:
            starts.append(tail)
    return sorted(set(starts))


def _load_fai_chroms(fai_path: str) -> List[Dict[str, int]]:
    """Return [{name, length}] from a .fai file."""
    if not fai_path or not os.path.isfile(fai_path):
        return []
    out = []
    with open(fai_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                try:
                    out.append({"name": parts[0], "length": int(parts[1])})
                except ValueError:
                    continue
    return out


# ---------------------------------------------------------------------------
#  bigWig I/O
# ---------------------------------------------------------------------------
def write_window_bw(
    track_arrays: Dict[str, np.ndarray],
    chromosome: str,
    start: int,
    end: int,
    windows_dir: str,
) -> None:
    """Write one fixed-step .bw per track for a window.

    File name: ``<start>_<end>__<track_key>.bw``, where ``track_key`` is the
    display-name key from ``track_key(assay, biosample)`` (e.g.
    ``RNA-seq__Leaf``) — matches how the backend names prediction tracks.
    """
    os.makedirs(windows_dir, exist_ok=True)
    for track_key_name, arr in track_arrays.items():
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        n = len(arr)
        if n <= 0:
            logger.warning("Empty array for %s @ %s:%d-%d — skip", track_key_name, chromosome, start, end)
            continue
        fname = f"{start}_{end}__{track_key_name}.bw"
        path = os.path.join(windows_dir, fname)
        with pyBigWig.open(path, "w") as bw:
            bw.addHeader([(chromosome, max(end, 1))], maxZooms=0)
            bw.addEntries(
                [chromosome] * n,
                [start + i for i in range(n)],
                ends=[start + i + 1 for i in range(n)],
                values=[float(v) for v in arr],
            )


def iter_window_files(windows_dir: str) -> List[Tuple[int, int, str, str]]:
    """Parse all window .bw files as (start, end, track, path)."""
    out: List[Tuple[int, int, str, str]] = []
    if not os.path.isdir(windows_dir):
        return out
    for fname in sorted(os.listdir(windows_dir)):
        if not fname.endswith(".bw"):
            continue
        m = re.match(r"^(\d+)_(\d+)(?:__(.+))?\.bw$", fname)
        if not m:
            continue
        track = m.group(3) if m.group(3) is not None else ""
        out.append((int(m.group(1)), int(m.group(2)), track, os.path.join(windows_dir, fname)))
    return out


def collect_chromosome_tracks(
    chromosome: str,
    chrom_len: int,
    windows_dir: str,
) -> Dict[str, np.ndarray]:
    """Merge per-window tracks of ONE chromosome into per-base arrays.

    Overlapping windows are averaged (sum / count per base).  Returns
    ``{track: np.ndarray[chrom_len]}`` — no file is written here; the caller
    accumulates all chromosomes and writes one genome-wide .bw per track.
    """
    files = iter_window_files(windows_dir)
    if not files:
        logger.info("[%s] no window files under %s", chromosome, windows_dir)
        return {}

    # group by track (display-name track_key) -> [(start, end, path)]
    by_track: Dict[str, List[Tuple[int, int, str]]] = {}
    for start, end, track, path in files:
        if not track:
            logger.warning("[%s] unparseable window file (no track suffix): %s", chromosome, path)
            continue
        by_track.setdefault(track, []).append((start, end, path))

    out: Dict[str, np.ndarray] = {}
    for track, items in by_track.items():
        sums = np.zeros(chrom_len, dtype=np.float64)
        counts = np.zeros(chrom_len, dtype=np.float64)
        for start, end, path in items:
            try:
                bw = pyBigWig.open(path)
                try:
                    if chromosome not in bw.chroms():
                        continue
                    end = min(end, chrom_len)
                    vals = np.nan_to_num(
                        np.array(bw.values(chromosome, start, end), dtype=np.float64), nan=0.0
                    )
                    if len(vals) == 0:
                        continue
                    stop = min(start + len(vals), chrom_len)
                    slice_len = stop - start
                    sums[start:stop] += vals[:slice_len]
                    counts[start:stop] += 1.0
                finally:
                    bw.close()
            except Exception as e:
                logger.warning("Cannot read %s: %s", path, e)

        valid = counts > 0
        if not valid.any():
            logger.info("[%s] no valid data for track %s", chromosome, track)
            continue
        # NaN marks bases not covered by any pre-generated window — the backend
        # detects NaN on query and falls back to model inference for that region.
        merged_arr = np.full(chrom_len, np.nan, dtype=np.float64)
        merged_arr[valid] = sums[valid] / counts[valid]
        out[track] = merged_arr
        logger.info("  [%s] accumulated track %s (%d bp)", chromosome, track, chrom_len)
    return out


def write_genomewide_bw(
    tracks_by_chrom: Dict[str, Dict[str, np.ndarray]],
    chroms: List[Tuple[str, int]],
    out_path: str,
    chunk_bp: int = 5_000_000,
) -> None:
    """Write ONE .bw containing all chromosomes for each track.

    ``tracks_by_chrom`` maps ``chrom -> {track: per-base array}``; every track
    present in the first chromosome is written.  Entries are added in
    ``chunk_bp`` blocks to bound memory for the long rice chromosomes.
    NaN values are allowed (pyBigWig supports them) and mark bases outside
    any pre-generated window.

    Returns nothing; raises on empty input.
    """
    if not chroms or not tracks_by_chrom:
        raise ValueError("No data to write genome-wide bigWig")
    first_chrom = next(iter(tracks_by_chrom))
    track_names = list(tracks_by_chrom[first_chrom].keys())
    if not track_names:
        raise ValueError("No tracks accumulated")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for track in track_names:
        with pyBigWig.open(out_path, "w") as bw:
            bw.addHeader(chroms, maxZooms=0)
            for chrom, chrom_len in chroms:
                arr = tracks_by_chrom.get(chrom, {}).get(track)
                if arr is None or len(arr) != chrom_len:
                    logger.warning("  track %s missing chrom %s — skipped", track, chrom)
                    continue
                n = chrom_len
                for c0 in range(0, n, chunk_bp):
                    c1 = min(c0 + chunk_bp, n)
                    m = c1 - c0
                    bw.addEntries(
                        [chrom] * m,
                        list(range(c0, c1)),
                        ends=list(range(c0 + 1, c1 + 1)),
                        values=[float(v) for v in arr[c0:c1]],
                    )
        logger.info("  wrote genome-wide %s (%d chromosomes)", out_path, len(chroms))


# ---------------------------------------------------------------------------
#  Predictor (each worker has its own instance on its own GPU)
# ---------------------------------------------------------------------------
def build_predictor(args, device: str) -> RiceMutationPredictor:
    env = _load_env_keyvals()

    def _pick(flag_val, env_key) -> str:
        return flag_val if flag_val else env.get(env_key, "")

    base_model_path = _pick(args.base_model_path, "BASE_MODEL_PATH")
    checkpoint_path = _pick(args.checkpoint_path, "CHECKPOINT_PATH")
    index_stat_path = _pick(args.index_stat_path, "INDEX_STAT_PATH")
    fasta_path = _pick(args.fasta, "GENOME_osa1_r7_FASTA") or _pick("", "GENOME_FASTA")
    if not (base_model_path and checkpoint_path and index_stat_path and fasta_path):
        raise SystemExit(
            "Missing model/genome paths. Set BASE_MODEL_PATH / CHECKPOINT_PATH / "
            "INDEX_STAT_PATH / GENOME_osa1_r7_FASTA in .env or pass --base-model-path "
            "--checkpoint-path --index-stat-path --fasta."
        )
    if not os.path.isfile(fasta_path):
        raise SystemExit(f"FASTA not found: {fasta_path}")

    # The predictor maps internal -> display names via DISPLAY_HEADS /
    # DISPLAY_BIOSAMPLES from the environment.  Make sure they're visible so
    # window files / merged files are named with the DISPLAY names (e.g.
    # RNA-seq__Leaf) — matching how the backend builds IGV payloads.
    if "DISPLAY_HEADS" not in os.environ and env.get("DISPLAY_HEADS"):
        os.environ["DISPLAY_HEADS"] = env["DISPLAY_HEADS"]
    if "DISPLAY_BIOSAMPLES" not in os.environ and env.get("DISPLAY_BIOSAMPLES"):
        os.environ["DISPLAY_BIOSAMPLES"] = env["DISPLAY_BIOSAMPLES"]

    predictor = RiceMutationPredictor(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        index_stat_path=index_stat_path,
        fasta_path=fasta_path,
        use_flash_attn=_env_bool("USE_FLASH_ATTN", True),
        device=device,
        torch_dtype=_env_str("MODEL_TORCH_DTYPE", "bfloat16"),
        max_seq_len=_env_int("MAX_SEQ_LEN", DEFAULT_WINDOW),
        proj_dim=_env_int("PROJ_DIM", 1024),
        num_downsamples=_env_int("NUM_DOWNSAMPLES", 4),
        bottleneck_dim=_env_int("BOTTLENECK_DIM", 1536),
        inference_batch_size=_env_int("INFERENCE_BATCH_SIZE", 1),
    )
    predictor.initialize()
    logger.info(
        "Track pairs for pre-gen: %s (assay %s x biosample %s)",
        [track_key(a, b) for a in predictor.display_assay_titles for b in predictor.display_biosample_order],
        predictor.display_assay_titles, predictor.display_biosample_order,
    )
    return predictor


def _run_windows_for_chromosomes(
    args,
    chroms_meta: List[Dict[str, int]],
    device: str,
) -> Tuple[int, int]:
    """Run inference for all windows in the given chromosomes. Returns
    (n_windows_done, n_windows_skipped)."""
    predictor = build_predictor(args, device)
    total_done = 0
    total_skipped = 0
    try:
        for meta in chroms_meta:
            chrom = meta["name"]
            chrom_len = meta["length"]
            starts = window_starts(chrom_len, args.window, args.hop)
            if not starts:
                logger.info("[%s] len=%d — no windows", chrom, chrom_len)
                continue
            win_dir = Path(args.pregen_dir) / "windows" / chrom
            win_dir.mkdir(parents=True, exist_ok=True)
            existing = {f for f in os.listdir(win_dir) if f.endswith(".bw")}
            t0 = time.time()
            for s in starts:
                e = min(s + args.window, chrom_len)
                # window files are named <start>_<end>__<track>.bw (one per track)
                probe_prefix = f"{s}_{e}__"
                if args.resume and any(fname.startswith(probe_prefix) for fname in existing):
                    total_skipped += 1
                    continue
                # Single forward per window (batch covered by per-worker GPUs).
                vals = predictor.predict(
                    chrom=chrom, start=s, end=e,  # {assay:{bios: arr}} (display names)
                )
                track_arrays: Dict[str, np.ndarray] = {}
                for assay, bios_map in vals.items():
                    for bios, arr in bios_map.items():
                        track_arrays[track_key(assay, bios)] = np.asarray(arr, dtype=np.float64)
                write_window_bw(track_arrays, chrom, s, e, str(win_dir))
                total_done += 1
                if total_done % 50 == 0:
                    logger.info(
                        "  [%s] %d/%d windows done on %s (%.1f/s)",
                        chrom, total_done, len(starts), device,
                        total_done / max(time.time() - t0, 1e-6),
                    )
            logger.info("[%s] %s done: %d windows (%ds)", chrom, device, len(starts), time.time() - t0)
    finally:
        try:
            predictor.release()
        except Exception:
            pass
    return total_done, total_skipped


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", nargs="*", default=None,
                        help="chromosomes to process (default: all in .fai)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP)
    parser.add_argument("--pregen-dir", default=str(DEFAULT_PREGEN_DIR))
    parser.add_argument("--fai", default="")
    parser.add_argument("--fasta", default="")
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--index-stat-path", default="")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU device for THIS worker (used directly when --workers=1)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="number of GPUs to use when --workers > 1 (default = --workers)")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers (each own predictor/GPU)")
    parser.add_argument("--worker", type=int, default=-1,
                        help="(internal) which worker this process is (0-based). "
                             "When >= 0 only this worker's chromosome subset is computed.")
    parser.add_argument("--resume", action="store_true",
                        help="skip windows whose .bw already exists")
    parser.add_argument("--merge-only", action="store_true",
                        help="only (re)merge existing window .bw into genome-wide files")
    parser.add_argument("--no-merge", action="store_true",
                        help="do not merge; only write per-window .bw")
    args = parser.parse_args()

    env = _load_env_keyvals()
    fasta_path = args.fasta or env.get("GENOME_osa1_r7_FASTA") or env.get("GENOME_FASTA", "")
    fai_path = args.fai or env.get("GENOME_osa1_r7_FAI") or (fasta_path + ".fai" if fasta_path else "")
    if not fai_path or not os.path.isfile(fai_path):
        raise SystemExit(f"FAI not found: {fai_path}")

    chroms_meta = _load_fai_chroms(fai_path)
    if args.chrom:
        wanted = set(args.chrom)
        chroms_meta = [c for c in chroms_meta if c["name"] in wanted]
    if not chroms_meta:
        raise SystemExit("No chromosomes selected / found in .fai")

    pregen_dir = Path(args.pregen_dir)
    pregen_dir.mkdir(parents=True, exist_ok=True)

    # --- Multiprocessing mode -------------------------------------------------
    if args.workers > 1:
        import multiprocessing as mp

        n_gpus = args.gpus or args.workers
        workers = args.workers
        if workers % n_gpus != 0 and n_gpus != workers:
            logger.warning("--workers not divisible by --gpus; rounding workers=%d", n_gpus)
            workers = n_gpus

        # split chromosomes across workers (round-robin by index)
        per_worker: List[List[Dict[str, int]]] = [[] for _ in range(workers)]
        for i, meta in enumerate(chroms_meta):
            per_worker[i % workers].append(meta)

        if not any(per_worker):
            raise SystemExit("No chromosomes selected (workers subset empty?)")

        # worker args need the raw device index; assign gpu = worker % n_gpus
        from argparse import Namespace

        ctx = mp.get_context("spawn")  # spawn: clean env, no fork deadlocks with CUDA
        procs = []
        for w in range(workers):
            if not per_worker[w]:
                continue
            wargs = Namespace(**vars(args))
            wargs.worker = w
            wargs.workers = workers
            wargs.device = f"cuda:{w % n_gpus}"
            wargs.chrom = [m["name"] for m in per_worker[w]]
            wargs.merge_only = False
            wargs.no_merge = True  # merge once at the end (rank 0)
            p = ctx.Process(
                target=_run_worker,
                args=(wargs, per_worker[w]),
                name=f"pregen-w{w}",
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        failed = [p for p in procs if p.exitcode != 0]
        if failed:
            logger.error("%d worker(s) failed; see above", len(failed))
            return 1

        logger.info("All workers finished. Merging ...")
        # fall through to merge using this process (no predictor)
        args.merge_only = True
        args.no_merge = False

    # --- Single-process inference (workers=1) ----------------------------------
    # inference runs unless --merge-only (which only merges existing windows)
    if not args.merge_only:
        done, skipped = _run_windows_for_chromosomes(args, chroms_meta, args.device)
        logger.info("Inference done: %d computed, %d skipped (resume)", done, skipped)

    # --- Merge -----------------------------------------------------------------
    # Collect per-chromosome arrays for every track, then write ONE
    # genome-wide .bw per track (all chromosomes in a single file).
    if not args.no_merge:
        t0 = time.time()
        _ckpt_path = args.checkpoint_path or env.get("CHECKPOINT_PATH", "")
        manifest: dict = {
            "genome_id": "osa1_r7",
            "fasta": fasta_path,
            "fasta_sha256": _sha256_of_file(fasta_path),
            "checkpoint": _ckpt_path,
            "checkpoint_sha256": _sha256_of_file(_ckpt_path) if _ckpt_path and os.path.isfile(_ckpt_path) else "",
            "window": args.window,
            "hop": args.hop,
            "chromosomes": {},
            "tracks": [],
            "merged_files": {},
        }
        tracks_by_chrom: Dict[str, Dict[str, np.ndarray]] = {}
        chroms_used: List[Tuple[str, int]] = []
        for meta in chroms_meta:
            chrom = meta["name"]
            chrom_len = meta["length"]
            win_dir = pregen_dir / "windows" / chrom
            entries = []
            for s in window_starts(chrom_len, args.window, args.hop):
                e = min(s + args.window, chrom_len)
                entries.append({"start": s, "end": e})
            manifest["chromosomes"][chrom] = {
                "length": chrom_len,
                "window": args.window,
                "hop": args.hop,
                "windows": entries,
            }
            if not win_dir.is_dir():
                continue
            tracks = collect_chromosome_tracks(chrom, chrom_len, str(win_dir))
            if tracks:
                tracks_by_chrom[chrom] = tracks
                chroms_used.append((chrom, chrom_len))

        if chroms_used:
            # Track names come from the window file suffixes (display-name keys).
            first_tracks = next(iter(tracks_by_chrom.values()))
            track_set = set(first_tracks.keys())
            for track in sorted(track_set):
                # one genome-wide .bw per track covering ALL chromosomes
                per_track_by_chrom: Dict[str, Dict[str, np.ndarray]] = {}
                for chrom, tmap in tracks_by_chrom.items():
                    arr = tmap.get(track)
                    if arr is not None:
                        per_track_by_chrom[chrom] = {track: arr}
                if not per_track_by_chrom:
                    continue
                out_path = str(pregen_dir / f"{track}.bw")
                write_genomewide_bw(per_track_by_chrom, chroms_used, out_path)
                if track not in manifest["tracks"]:
                    manifest["tracks"].append(track)
                manifest["merged_files"][track] = out_path
        else:
            logger.warning("No window data found — nothing to merge")

        manifest_path = pregen_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Manifest written: %s", manifest_path)
        logger.info("Merge done in %.1fs", time.time() - t0)

    return 0


def _run_worker(wargs, chroms_meta):
    """Top-level worker entry (required for spawn)."""
    try:
        _run_windows_for_chromosomes(wargs, chroms_meta, wargs.device)
    except Exception:
        logger.exception("Worker %s failed", wargs.device)
        raise


if __name__ == "__main__":
    raise SystemExit(main())