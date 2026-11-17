"""Rice-Reg API routes.

Endpoints:
- GET  /health              — health check
- POST /uploadFile          — upload ATAC bigWig file
- POST /predict/rice-reg    — run prediction
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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rice_reg.cache_service import (
    cached_or_compute,
    prediction_cache,
    start_bigwig_cleanup,
)
from rice_reg.prediction_service import (
    adjust_window,
    init_predictor,
    release_predictor,
    require_predictor,
    run_prediction_core,
    resolve_atac_path,
    list_genomes,
    get_genome_chromosomes,
    register_uploaded_genome,
    attach_uploaded_gff,
    resolve_genome_config,
    start_uploaded_genome_cleanup,
)
from rice_reg.igv_payload import set_static_base_url

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR
FRONTEND_DIR = PROJECT_DIR / "frontend"
ATAC_CACHE_DIR_ABS = os.getenv(
    "BACKEND_UPLOADED_ATAC",
    str(PROJECT_DIR / "cache" / "uploaded_atac"),
)
PREDICTION_CACHE_DIR_ABS = os.getenv(
    "BACKEND_PREDICTION_CACHE",
    str(PROJECT_DIR / "cache" / "predictions"),
)
MAX_ATAC_UPLOAD_MB = int(os.getenv("MAX_ATAC_UPLOAD_MB", "10240"))  # default 10 GB
MAX_ATAC_UPLOAD_BYTES = MAX_ATAC_UPLOAD_MB * 1024 * 1024

UPLOADED_GENOMES_DIR_ABS = os.getenv(
    "BACKEND_UPLOADED_GENOMES",
    str(PROJECT_DIR / "cache" / "uploaded_genomes"),
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "640"))
MAX_FASTA_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "7001"))

app = FastAPI(title="Rice-Reg Server", version="0.1.0")

# Allow cross-origin requests from the Gradio frontend (different port)
# so that IGV.js in an about:srcdoc iframe can load scripts and data
# from the backend static file server.
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
    atac_source: Optional[str] = None
    uploaded_atac: Optional[str] = None


class PredictResponse(BaseModel):
    success: bool
    message: str = ""
    igv_payload: Optional[dict] = None
    elapsed_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
#  Startup event — initialise predictor
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    # Clean prediction cache from previous runs
    if os.path.isdir(PREDICTION_CACHE_DIR_ABS):
        shutil.rmtree(PREDICTION_CACHE_DIR_ABS)
        print(f"[startup] Cleaned prediction cache: {PREDICTION_CACHE_DIR_ABS}")

    # Clean uploaded ATAC cache from previous runs
    if os.path.isdir(ATAC_CACHE_DIR_ABS):
        shutil.rmtree(ATAC_CACHE_DIR_ABS)
        print(f"[startup] Cleaned upload cache: {ATAC_CACHE_DIR_ABS}")

    # Clean uploaded custom genomes from previous runs (registry is in-memory)
    if os.path.isdir(UPLOADED_GENOMES_DIR_ABS):
        shutil.rmtree(UPLOADED_GENOMES_DIR_ABS)
        print(f"[startup] Cleaned uploaded genomes: {UPLOADED_GENOMES_DIR_ABS}")

    # Recreate cache directories
    os.makedirs(ATAC_CACHE_DIR_ABS, exist_ok=True)
    os.makedirs(PREDICTION_CACHE_DIR_ABS, exist_ok=True)
    os.makedirs(UPLOADED_GENOMES_DIR_ABS, exist_ok=True)

    # Mount static file serving for IGV.js to access local files via HTTP.
    # We mount the root filesystem at /static-files/ so that any absolute path
    # like /mnt/rice/... becomes /static-files/mnt/rice/...
    static_mount_path = "/static-files"
    app.mount(static_mount_path, StaticFiles(directory="/"), name="static-files")
    static_base_url = f"http://127.0.0.1:{BACKEND_PORT}{static_mount_path}"
    set_static_base_url(static_base_url)

    init_predictor()

    # Background cleanup of stale prediction bigWig files (shared across users).
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
    from rice_reg.prediction_service import _PREDICTOR

    return {
        "status": "ok",
        "predictor_initialized": _PREDICTOR.get("instance") is not None,
    }


@app.post("/uploadFile")
async def upload_file(file: UploadFile = File(...)):
    """Upload an ATAC bigWig file.  Returns the server-side path."""
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".bw", ".bigwig"):
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Only .bw / .bigWig files are accepted.",
        )

    # Sanitise filename
    safe_name = f"{int(time.time())}_{file.filename}"
    dest = Path(ATAC_CACHE_DIR_ABS) / safe_name

    content = await file.read()
    if len(content) > MAX_ATAC_UPLOAD_BYTES:
        raise HTTPException(400, f"File exceeds maximum upload size ({MAX_ATAC_UPLOAD_MB} MB).")

    with open(dest, "wb") as f:
        f.write(content)

    return {
        "success": True,
        "file_path": str(dest),
        "file_name": file.filename,
        "size_bytes": len(content),
    }


@app.post("/uploadFasta")
async def upload_fasta(file: UploadFile = File(...)):
    """Upload a custom genome FASTA file.  Automatically builds the .fai index
    and registers it as a new genome (priority over built-in genomes).
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
        cfg = register_uploaded_genome(str(dest))
        genome_id = _find_uploaded_genome_id(cfg["fasta"])
        chromosomes = get_genome_chromosomes(genome_id, cfg)
    except Exception as e:
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
    """Attach an annotation GFF to an uploaded custom genome."""
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


@app.get("/genomes")
def genomes():
    return {"genomes": list_genomes()}


@app.get("/genomes/{genome_id}/chromosomes")
def genome_chromosomes(genome_id: str):
    """Return display chromosome names for a genome (chrNN style)."""
    genome_config = resolve_genome_config(genome_id)
    return {
        "genome": genome_id,
        "chromosomes": get_genome_chromosomes(genome_id, genome_config),
    }


def _find_uploaded_genome_id(fasta_path: str) -> str:
    from rice_reg.prediction_service import _UPLOADED_GENOMES

    for gid, cfg in _UPLOADED_GENOMES.items():
        if os.path.abspath(cfg["fasta"]) == os.path.abspath(fasta_path):
            return gid
    raise ValueError("Uploaded genome not found in registry")


@app.post("/predict/rice-reg", response_model=PredictResponse)
def predict_rice_reg(req: PredictRequest):
    """Run ATAC → RNA-seq expression prediction (content-cached).

    Multi-user concurrency:
    - Identical (genome, chromosome, window, atac) requests reuse a shared
      content-addressed cache entry — GPU inference runs once per locus.
    - When several users hit the same cache-missed locus at the same time,
      only the first request runs inference; the others wait for its result
      (in-flight merge).
    """
    t0 = time.time()
    try:
        predictor = require_predictor()

        # Deterministic window normalisation so equivalent requests share a
        # single cache entry (same normalisation as run_prediction_core).
        norm_start, norm_end = adjust_window(req.start, req.end, predictor.target_len)

        # Resolve ATAC path (part of the cache key — built-in or user upload).
        atac_path = resolve_atac_path(
            atac_source=req.atac_source,
            uploaded_atac=req.uploaded_atac,
        )

        cache_key = prediction_cache.build_key(
            "rice-reg",
            genome=req.genome,
            chromosome=req.chromosome,
            start=norm_start,
            end=norm_end,
            atac=atac_path,
        )

        def _compute() -> dict:
            """Full prediction pipeline (called only on cache miss)."""
            genome_config = _resolve_genome_config(req.genome)
            result = run_prediction_core(
                genome=req.genome,
                chromosome=req.chromosome,
                start=req.start,
                end=req.end,
                atac_path=atac_path,
                fasta_path=genome_config["fasta"],
                genome_config=genome_config,
            )
            return result["igv_payload"]

        igv_payload = cached_or_compute(cache_key, _compute)

        elapsed = time.time() - t0
        return PredictResponse(
            success=True,
            message=f"Prediction completed in {elapsed:.1f}s",
            igv_payload=igv_payload,
            elapsed_seconds=elapsed,
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Internal error: {e}")


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------
def _resolve_genome_config(genome: str) -> dict:
    """Read genome config (uploaded custom genome takes priority over env)."""
    return resolve_genome_config(genome)
