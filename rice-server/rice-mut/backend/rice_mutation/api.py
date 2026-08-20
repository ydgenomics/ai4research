"""Rice-Mutation API routes.

Endpoints:
- GET  /health                       — health check + metadata
- GET  /genomes                      — list configured genomes
- GET  /genomes/{id}/chromosomes     — chromosome names of a genome
- GET  /assays                       — assay titles from index_stat
- GET  /biosamples                   — biosample order from index_stat
- POST /uploadFasta                  — upload a custom genome FASTA (auto .fai)
- POST /predict                      — reference-sequence expression prediction
- POST /predict/snv                  — single-nucleotide variant (result1/result2)
- POST /predict/snv/stat             — region diff stats between result1/result2
- POST /predict/bar                  — per (assay, biosample) mean expression in a region (for the frontend bar plot)
"""

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rice_mutation.cache_service import prediction_cache, start_bigwig_cleanup
from rice_mutation.igv_payload import set_static_base_url
from rice_mutation.prediction_service import (
    _adjust_window,
    attach_uploaded_gff,
    compute_region_means,
    compute_region_stats,
    get_genome_chromosomes,
    init_predictor,
    list_genomes,
    register_uploaded_genome,
    release_predictor,
    require_predictor,
    resolve_genome_config,
    run_prediction_core,
    run_snv_core,
    save_prediction_bigwigs,
    start_uploaded_genome_cleanup,
)

ROOT_DIR = Path(__file__).resolve().parents[2]  # rice-mutation/
PROJECT_DIR = ROOT_DIR
FASTA_CACHE_DIR_ABS = os.path.abspath(os.getenv(
    "BACKEND_UPLOADED_FASTA",
    str(PROJECT_DIR / "cache" / "uploaded_fasta"),
))
PREDICTION_CACHE_DIR_ABS = os.path.abspath(os.getenv(
    "BACKEND_PREDICTION_CACHE",
    str(PROJECT_DIR / "cache" / "predictions"),
))
UPLOADED_GENOMES_DIR_ABS = os.path.abspath(os.getenv(
    "BACKEND_UPLOADED_GENOMES",
    str(PROJECT_DIR / "cache" / "uploaded_genomes"),
))
# Max upload size for custom genome FASTA / GFF — configurable via
# MAX_UPLOAD_MB in .env (unit: MB), default 640 MB.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "640"))
MAX_FASTA_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# In-memory cache of SNV results: snv_id -> {ref_values, mut_values, meta}
_SNV_CACHE: dict = {}
# In-memory cache of reference predictions: ref_id -> {ref_values, mut_values: None, meta}
_REF_CACHE: dict = {}
# Value-array caches keep this many entries (>= the content cache so cached
# prediction_ids stay valid for /predict/bar and /predict/snv/stat).
_CACHE_ARRAY_MAX = 256

# 优先使用平台注入的 PORT(API 模式),回退 BACKEND_PORT(本地/网页版)
BACKEND_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8001")))

app = FastAPI(title="Rice-Mutation Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    genome: str
    chromosome: str
    start: int
    end: Optional[int] = None
    biosample_names: Optional[list] = None
    # Optional matplotlib figure: png | svg | html | none (default none = skip)
    figure_format: str = "none"


class SnvRequest(BaseModel):
    genome: str
    chromosome: str
    start: int
    end: Optional[int] = None
    biosample_names: Optional[list] = None
    snv_index: int
    snv_base: str


class SnvStatRequest(BaseModel):
    snv_id: str
    region_start: int
    region_end: int


class BarRequest(BaseModel):
    prediction_id: str
    region_start: int
    region_end: int


class PredictResponse(BaseModel):
    success: bool
    message: str = ""
    igv_payload: Optional[dict] = None
    figure_url: Optional[str] = None
    figure_format: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
#  Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    # Clean caches from previous runs
    for d in (PREDICTION_CACHE_DIR_ABS, FASTA_CACHE_DIR_ABS, UPLOADED_GENOMES_DIR_ABS):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"[startup] Cleaned cache: {d}")
        os.makedirs(d, exist_ok=True)

    # Mount root filesystem so IGV.js can load local files via HTTP.
    static_mount_path = "/static-files"
    app.mount(static_mount_path, StaticFiles(directory="/"), name="static-files")
    static_base_url = f"http://127.0.0.1:{BACKEND_PORT}{static_mount_path}"
    set_static_base_url(static_base_url)

    init_predictor()

    # Background cleanup of stale prediction bigWig files.
    start_bigwig_cleanup(PREDICTION_CACHE_DIR_ABS)

    # Runtime TTL cleanup of idle uploaded custom genomes (FASTA + .fai + GFF).
    _genome_ttl = float(os.getenv("UPLOADED_GENOMES_TTL_HOURS", "0.5"))
    _genome_interval = int(os.getenv("UPLOADED_GENOMES_CLEANUP_INTERVAL", "300"))
    start_uploaded_genome_cleanup(
        ttl_hours=_genome_ttl, interval_seconds=_genome_interval,
    )


@app.on_event("shutdown")
def on_shutdown():
    release_predictor()


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    from rice_mutation.prediction_service import _PREDICTOR

    inst = _PREDICTOR.get("instance")
    return {
        "status": "ok",
        "predictor_initialized": inst is not None,
        "genomes": list_genomes(),
        "assays": inst.display_assay_titles if inst else [],
        "biosamples": inst.display_biosample_order if inst else [],
    }


@app.get("/genomes")
def genomes():
    return {"genomes": list_genomes()}


@app.get("/genomes/{genome_id}/chromosomes")
def genome_chromosomes(genome_id: str):
    """Return display chromosome names for a genome (chrNN style for numeric)."""
    genome_config = resolve_genome_config(genome_id)
    return {"genome": genome_id, "chromosomes": get_genome_chromosomes(genome_id, genome_config)}


@app.get("/assays")
def assays():
    inst = require_predictor()
    return {"assays": inst.display_assay_titles}


@app.get("/biosamples")
def biosamples():
    inst = require_predictor()
    return {"biosamples": inst.display_biosample_order}


@app.post("/uploadFasta")
async def upload_fasta(file: UploadFile = File(...)):
    """Upload a custom genome FASTA file.  Automatically builds the .fai index
    and registers it as a new genome (priority over built-in genomes).

    Returns the new genome id, its config paths and chromosome names.
    """
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".fa", ".fasta", ".fna"):
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Only .fa / .fasta accepted.",
        )

    content = await file.read()
    if len(content) > MAX_FASTA_UPLOAD_BYTES:
        raise HTTPException(400, f"File exceeds maximum upload size ({MAX_UPLOAD_MB} MB).")

    safe_name = f"{time.time_ns()}_{Path(file.filename).name}"
    dest = Path(UPLOADED_GENOMES_DIR_ABS) / safe_name
    with open(dest, "wb") as f:
        f.write(content)

    try:
        # Build .fai if missing, read chromosomes, register as genome
        cfg = register_uploaded_genome(str(dest))
        genome_id = _find_uploaded_genome_id(cfg["fasta"])
        chromosomes = get_genome_chromosomes(genome_id, cfg)
    except Exception as e:
        # clean up the file so a bad upload doesn't linger
        try:
            os.remove(dest)
        except OSError:
            pass
        raise HTTPException(400, f"Failed to index FASTA: {e}")

    return {
        "success": True,
        "genome": genome_id,
        "file_path": str(dest),
        "file_name": file.filename,
        "size_bytes": len(content),
        "chromosomes": chromosomes,
    }


@app.post("/uploadGff")
async def upload_gff(
    genome: str = Form(...),
    file: UploadFile = File(...),
):
    """Attach an annotation GFF to an uploaded custom genome.

    - ``genome``: the custom genome id returned by ``/uploadFasta``.
    - ``file``:  .gff / .gff3 / .gtf (optionally gzipped).

    Updates the in-memory registry so IGV payloads include a Genes track for
    the custom genome.  Only uploaded (custom) genomes can be annotated.
    """
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    lower_name = file.filename.lower()
    if not (
        lower_name.endswith((".gff", ".gff3", ".gtf"))
        or lower_name.endswith((".gff.gz", ".gff3.gz", ".gtf.gz"))
    ):
        raise HTTPException(
            400,
            "Unsupported file type. Only .gff / .gff3 / .gtf (optionally .gz) accepted.",
        )

    content = await file.read()
    if len(content) > MAX_FASTA_UPLOAD_BYTES:
        raise HTTPException(400, f"File exceeds maximum upload size ({MAX_UPLOAD_MB} MB).")

    safe_name = f"{time.time_ns()}_{Path(file.filename).name}"
    dest = Path(UPLOADED_GENOMES_DIR_ABS) / safe_name
    with open(dest, "wb") as f:
        f.write(content)

    try:
        cfg = attach_uploaded_gff(genome, str(dest))
    except (FileNotFoundError, ValueError) as e:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise HTTPException(400, str(e))

    return {
        "success": True,
        "genome": genome,
        "gff_path": cfg["gff"],
        "file_name": file.filename,
        "size_bytes": len(content),
    }


def _find_uploaded_genome_id(fasta_path: str) -> str:
    from rice_mutation.prediction_service import _UPLOADED_GENOMES

    for gid, cfg in _UPLOADED_GENOMES.items():
        if os.path.abspath(cfg["fasta"]) == os.path.abspath(fasta_path):
            return gid
    raise ValueError("Uploaded genome not found in registry")



@app.get("/download")
def download(fname: str):
    """Force-download a generated figure file (png/svg/html).

    Only files inside the prediction cache directory are served, to avoid
    arbitrary file disclosure via path traversal.
    """
    name = os.path.basename(fname)  # strip any path components
    if not name:
        raise HTTPException(400, "fname is required")
    path = os.path.abspath(os.path.join(PREDICTION_CACHE_DIR_ABS, name))
    if not path.startswith(os.path.abspath(PREDICTION_CACHE_DIR_ABS) + os.sep):
        raise HTTPException(400, "Invalid file name")
    if not os.path.isfile(path):
        raise HTTPException(404, f"File not found: {name}")
    media_type = {
        "png": "image/png",
        "svg": "image/svg+xml",
        "html": "text/html",
    }.get(os.path.splitext(name)[1].lstrip("."), "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        filename=name,
        content_disposition_type="attachment",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run reference-sequence expression prediction (content-cached)."""
    t0 = time.time()
    try:
        predictor = require_predictor()

        # Deterministic window normalisation (mirrors run_prediction_core) so
        # equivalent requests share a single cache entry.
        norm_start, norm_end = _adjust_window(req.start, req.end, predictor.max_seq_len)
        cache_key = prediction_cache.build_key(
            "ref",
            genome=req.genome,
            chromosome=req.chromosome,
            start=norm_start,
            end=norm_end,
            biosample=",".join(req.biosample_names) if req.biosample_names else "*",
        )
        hit = prediction_cache.get(cache_key)
        if hit is not None:
            elapsed = time.time() - t0
            return PredictResponse(
                success=True,
                message=f"cached ({elapsed:.1f}s)",
                igv_payload=hit["igv_payload"],
                elapsed_seconds=elapsed,
                metadata=hit["metadata"],
            )

        genome_config = resolve_genome_config(req.genome)
        result = run_prediction_core(
            genome=req.genome,
            chromosome=req.chromosome,
            start=req.start,
            end=req.end,
            biosample_names=req.biosample_names,
            genome_config=genome_config,
        )
        chromosome, start, end = result["position"]

        track_paths = save_prediction_bigwigs(
            result["values"], chromosome, start, end, tag="ref"
        )
        igv_payload = _build_payload(
            req.genome, chromosome, start, end,
            track_paths=track_paths, genome_config=genome_config,
        )

        # Cache value arrays so the frontend bar plot can query per-track means
        # over the current IGV viewport without re-running the model.
        ref_id = f"ref_{int(time.time() * 1000)}"
        _REF_CACHE[ref_id] = {
            "ref_values": result["values"],
            "mut_values": None,
            "meta": {
                "kind": "ref",
                "genome": req.genome,
                "chromosome": chromosome,
                "start": start,
                "end": end,
                "window_len": end - start,
            },
        }
        if len(_REF_CACHE) > _CACHE_ARRAY_MAX:
            for k in list(_REF_CACHE)[: len(_REF_CACHE) - _CACHE_ARRAY_MAX]:
                _REF_CACHE.pop(k, None)

        figure_url = _render_figure(
            ref_values=result["values"],
            chromosome=chromosome, start=start, end=end,
            genome_config=genome_config,
            fmt=req.figure_format,
        )

        metadata = {
            "ref_id": ref_id,
            "chromosome": chromosome,
            "window_start": start,
            "window_len": end - start,
        }
        prediction_cache.put(cache_key, ref_id, igv_payload, metadata, "ref")

        elapsed = time.time() - t0
        return PredictResponse(
            success=True,
            message=f"Prediction completed in {elapsed:.1f}s",
            igv_payload=igv_payload,
            figure_url=figure_url,
            figure_format=req.figure_format,
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected error: {e}")


@app.post("/predict/snv", response_model=PredictResponse)
def predict_snv(req: SnvRequest):
    """Single-nucleotide variant prediction — result1 (ref) vs result2 (mut).

    Content-cached: identical (genome, window, snv, base) requests skip the GPU.
    """
    t0 = time.time()
    try:
        predictor = require_predictor()

        # Deterministic window normalisation (mirrors run_snv_core).
        norm_start, norm_end = _adjust_window(req.start, req.end, predictor.max_seq_len)
        cache_key = prediction_cache.build_key(
            "snv",
            genome=req.genome,
            chromosome=req.chromosome,
            start=norm_start,
            end=norm_end,
            snv_index=req.snv_index,
            snv_base=req.snv_base,
            biosample=",".join(req.biosample_names) if req.biosample_names else "*",
        )
        hit = prediction_cache.get(cache_key)
        if hit is not None:
            elapsed = time.time() - t0
            return PredictResponse(
                success=True,
                message=f"cached ({elapsed:.1f}s)",
                igv_payload=hit["igv_payload"],
                elapsed_seconds=elapsed,
                metadata=hit["metadata"],
            )

        genome_config = resolve_genome_config(req.genome)
        result = run_snv_core(
            genome=req.genome,
            chromosome=req.chromosome,
            start=req.start,
            end=req.end,
            snv_index=req.snv_index,
            snv_base=req.snv_base,
            biosample_names=req.biosample_names,
            genome_config=genome_config,
        )
        chromosome, start, end = result["position"]

        ref_track_paths = save_prediction_bigwigs(
            result["ref_values"], chromosome, start, end, tag="result1"
        )
        mut_track_paths = save_prediction_bigwigs(
            result["mut_values"], chromosome, start, end, tag="result2"
        )
        # 1-based display in the IGV track label (internal snv_index is 0-based)
        snv_note = f"@{result['ref_base']}{result['snv_index'] + 1}>{result['snv_base']}"
        igv_payload = _build_payload(
            req.genome, chromosome, start, end,
            ref_track_paths=ref_track_paths,
            mut_track_paths=mut_track_paths,
            genome_config=genome_config,
            ref_label_fmt="{bios} {assay} result1 (ref)",
            mut_label_fmt=("{bios} {assay} result2 (mut " + snv_note + ")"),
        )

        # Cache arrays for the region-stat endpoint
        snv_id = f"snv_{int(time.time() * 1000)}"
        _SNV_CACHE[snv_id] = {
            "ref_values": result["ref_values"],
            "mut_values": result["mut_values"],
            "meta": {
                "genome": req.genome,
                "chromosome": chromosome,
                "start": start,
                "end": end,
                "window_len": result["window_len"],
                "ref_base": result["ref_base"],
                "snv_index": result["snv_index"],
                "snv_base": result["snv_base"],
            },
        }
        # Bound cache size (safety)
        if len(_SNV_CACHE) > _CACHE_ARRAY_MAX:
            for k in list(_SNV_CACHE)[: len(_SNV_CACHE) - _CACHE_ARRAY_MAX]:
                _SNV_CACHE.pop(k, None)

        metadata = {
            "snv_id": snv_id,
            "ref_base": result["ref_base"],
            "snv_base": result["snv_base"],
            "snv_index": result["snv_index"],
            "window_len": result["window_len"],
            "window_start": start,
        }
        prediction_cache.put(cache_key, snv_id, igv_payload, metadata, "snv")

        elapsed = time.time() - t0
        return PredictResponse(
            success=True,
            message=f"SNV prediction completed in {elapsed:.1f}s",
            igv_payload=igv_payload,
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected error: {e}")


@app.post("/predict/snv/stat")
def predict_snv_stat(req: SnvStatRequest):
    """Compute region diff stats between result1 (ref) and result2 (mut)."""
    entry = _SNV_CACHE.get(req.snv_id)
    if entry is None:
        raise HTTPException(404, f"snv_id '{req.snv_id}' not found (expired?)")

    try:
        stats = compute_region_stats(
            entry["ref_values"],
            entry["mut_values"],
            req.region_start,
            req.region_end,
        )
        return {
            "success": True,
            "snv_id": req.snv_id,
            "region": [req.region_start, req.region_end],
            "stats": stats,
            "meta": entry["meta"],
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/predict/bar")
def predict_bar(req: BarRequest):
    """Per-(assay, biosample) mean expression over a genomic region — feeds the
    frontend bar plot.

    ``prediction_id`` is a ``ref_id`` (only result1) or an ``snv_id``
    (result1 + result2).  The requested region is clipped to the prediction
    window; an empty overlap returns ``overlap: False`` so the frontend can
    skip redrawing.
    """
    entry = _SNV_CACHE.get(req.prediction_id)
    kind = "snv"
    if entry is None:
        entry = _REF_CACHE.get(req.prediction_id)
        kind = "ref"
    if entry is None:
        raise HTTPException(
            404, f"prediction_id '{req.prediction_id}' not found (expired?)"
        )

    meta = entry["meta"]
    win_start = int(meta["start"])
    win_end = int(meta["end"])
    region_start = int(req.region_start)
    region_end = int(req.region_end)

    # Clip the requested region to the prediction window.
    ov_start = max(region_start, win_start)
    ov_end = min(region_end, win_end)
    if ov_end <= ov_start:
        return {
            "success": True,
            "overlap": False,
            "kind": kind,
            "window": [win_start, win_end],
            "region": [region_start, region_end],
            "values": [],
        }

    start_idx = ov_start - win_start
    end_idx = ov_end - win_start
    try:
        r1 = compute_region_means(entry["ref_values"], start_idx, end_idx)
        r2 = (
            compute_region_means(entry["mut_values"], start_idx, end_idx)
            if entry.get("mut_values") is not None else None
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    values = []
    for assay, bios_map in r1.items():
        for bios, m1 in bios_map.items():
            values.append({
                "assay": assay,
                "biosample": bios,
                "result1": m1,
                "result2": r2[assay][bios] if r2 is not None else None,
            })

    return {
        "success": True,
        "overlap": True,
        "kind": kind,
        "window": [win_start, win_end],
        "overlap_region": [ov_start, ov_end],
        "region": [region_start, region_end],
        "values": values,
    }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------
def _build_payload(genome, chromosome, start, end, genome_config, **kwargs):
    from rice_mutation.igv_payload import build_prediction_payload

    return build_prediction_payload(
        genome=genome,
        chromosome=chromosome,
        start=start,
        end=end,
        genome_config=genome_config,
        **kwargs,
    )


def _render_figure(
    ref_values: dict,
    chromosome: str,
    start: int,
    end: int,
    genome_config: dict,
    fmt: str = "none",
    mut_values: Optional[dict] = None,
) -> Optional[str]:
    """Render a summary figure (png/svg/html) and return its URL (or None).

    Best-effort: any rendering error is logged and returns None so that the
    prediction API itself never fails because of plotting. fmt="none" skips.
    """
    fmt = (fmt or "none").lower().lstrip(".")
    if fmt == "none" or fmt not in ("png", "svg", "html"):
        return None
    from rice_mutation.igv_payload import _path_to_url
    from rice_mutation.plot_service import render_prediction_figure

    try:
        annotation_path = genome_config.get("gff", "") or ""
        fig_path = render_prediction_figure(
            ref_values=ref_values,
            mut_values=mut_values,
            chromosome=chromosome,
            start=start,
            end=end,
            annotation_path=annotation_path,
            out_dir=PREDICTION_CACHE_DIR_ABS,
            fmt=fmt,
        )
        if not fig_path:
            return None
        return _path_to_url(fig_path)
    except Exception as e:
        traceback.print_exc()
        return None
