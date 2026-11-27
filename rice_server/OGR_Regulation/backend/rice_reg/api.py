"""Rice-Reg API routes.

Endpoints:
- GET  /health              — health check
- POST /uploadFile          — upload ATAC bigWig file
- POST /predict/rice-reg    — run prediction
- POST /api/aigress/openai/OGR-Regulation/Genome        — genome list (DCS)
- POST /api/aigress/openai/OGR-Regulation/Chromosome    — chromosome list (DCS)
- POST /api/aigress/openai/OGR-Regulation/Prediction    — ATAC→RNA-seq prediction (DCS)
- POST /api/aigress/openai/OGR-Regulation/RegisterGenome — register custom genome (DCS)

All DCS sub-paths share one process/port with the web routes, so the web
frontend and pure-API (curl) callers use exactly the same dialect.
"""

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
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
    _validate_atac_bw,
    adjust_window,
    init_predictor,
    normalize_chromosome,
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

# DCS 计费口径与可选 API Key 鉴权（空 = 免鉴权）
PROMPT_TOKEN_MULTIPLIER = float(os.getenv("DCS_PROMPT_TOKEN_MULTIPLIER", "1"))
COMPLETION_TOKEN_MULTIPLIER = float(os.getenv("DCS_COMPLETION_TOKEN_MULTIPLIER", "1"))
DCS_API_KEY = os.getenv("DCS_API_KEY", "").strip()

# CACHE_MODE only controls the content-addressed prediction cache + lazy loading.
# false = lazy-load deployment: no content cache (re-run inference each request);
#         custom genomes are registered by path only (managed=False).
# In BOTH modes intermediate prediction bigWigs live in BACKEND_PREDICTION_CACHE
# and the dir is wiped on server start (see on_startup).
CACHE_MODE = os.getenv("CACHE_MODE", "true").strip().lower() in ("1", "true", "yes", "on")

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
    # --- unified DCS-style envelope (also returned by every prediction route) ---
    model_name: Optional[str] = "OGR"
    usage: Optional[dict] = None
    token_cnt: Optional[int] = 0
    status: Optional[int] = 200
    result: Optional[dict] = None


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
    safe_name = f"{time.time_ns()}_{file.filename}"
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
        genome_id = str(cfg.get("id") or "")
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


# ===========================================================================
#  DCS-style sub-path routes (camelCase keys, 1-based Start)
# ===========================================================================
# The web frontend and pure-API (curl) callers share the exact same dialect:
#   Genome / Chromosome / Start(1-based) / ATAC-seq_Source / uploaded_atac / Format
# All 1-based → 0-based conversion happens here in the backend.
#  - /api/aigress/openai/OGR-Regulation/Genome          → genome list
#  - /api/aigress/openai/OGR-Regulation/Chromosome      → chromosome list
#  - /api/aigress/openai/OGR-Regulation/Prediction      → ATAC→RNA-seq prediction
#  - /api/aigress/openai/OGR-Regulation/RegisterGenome  → register fasta_path
class RequestError(ValueError):
    """Request parameter error (→ 400), distinct from execution errors (→ 500)."""


_DCS_USAGE_TEXT = (
    "Usage: POST /api/aigress/openai/OGR-Regulation/Prediction\n"
    "Body (JSON, keys are case-sensitive):\n"
    '  "Genome": string, optional — genome id (default first configured)\n'
    '  "Chromosome": string, optional — chromosome (e.g. "chr01".."chr12"), default "chr01"\n'
    '  "Start": int, required — 1-based inclusive start coordinate\n'
    '  "ATAC-seq_Source": string, optional — built-in ATAC id (e.g. "SAM2_MH63_1")\n'
    '  "uploaded_atac": string, optional — local path to an ATAC bigWig (takes priority)\n'
    '  "Format": string, optional — "Full" (default) or "Mean"\n'
    '  "fasta_path"/"gff_path": optional — lazy-register a custom genome in place\n'
)


def _format_array(arr, fmt: str):
    """把单个 numpy 数组按 Format 转为 JSON 可序列化结构(Full/Mean)。"""
    arr = np.asarray(arr, dtype=np.float64)
    if fmt == "mean":
        return round(float(arr.mean()), 6)
    return [round(float(x), 6) for x in arr]


def _count_elements(values: dict) -> int:
    """统计输出数组元素总数(用于 completion_tokens)。"""
    total = 0
    for arr in values.values():
        if arr is not None:
            total += int(np.asarray(arr).size)
    return total


def _dcs_usage(prompt_tokens: int, completion_tokens: int) -> dict:
    pt = int(prompt_tokens * PROMPT_TOKEN_MULTIPLIER)
    ct = int(completion_tokens * COMPLETION_TOKEN_MULTIPLIER)
    return {"prompt_tokens": pt, "completion_tokens": ct}


def _dcs_ok(usage: dict, message: str, result: dict) -> dict:
    """Unified success envelope shared by web and API routes."""
    return {
        "model_name": "OGR",
        "usage": usage,
        "token_cnt": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        "status": 200,
        "message": message,
        "result": result,
    }


def _dcs_err(message: str, status: int = 400, detail: dict | None = None) -> dict:
    """Unified error envelope (matches the DCS error dialect)."""
    resp = {
        "model_name": "OGR",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "token_cnt": 0,
        "status": status,
        "message": message,
        "result": None,
    }
    if detail:
        resp["detail"] = detail
    return resp


def _unauthorized(message: str = "Invalid or missing API Key") -> dict:
    return _dcs_err(message, 401)


def _check_api_key(authorization: str | None = None, x_api_key: str | None = None) -> None:
    """Validate API key; no-op when DCS_API_KEY is unset (backward compatible)."""
    if not DCS_API_KEY:
        return
    key = ""
    if authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    if not key and x_api_key:
        key = x_api_key.strip()
    if key != DCS_API_KEY:
        raise RequestError("Invalid or missing API Key")


def _summarize_body(body: dict) -> dict:
    """Request-body digest: only fields needed to locate / reproduce an error."""
    keys = (
        "model", "Genome", "Chromosome", "Start",
        "ATAC-seq_Source", "uploaded_atac", "Format",
        "fasta_path", "genome_id", "gff_path",
    )
    return {k: body[k] for k in keys if k in body}


def _parse_common(body: dict):
    """Parse Genome / Chromosome / Start / ATAC input / Format.

    Returns (genome, chromosome, start_1, atac_source, uploaded_atac, output_format).
    Coordinates are 1-based inclusive; the caller converts to 0-based.
    """
    genome = str(body.get("Genome", "") or "")
    if not genome:
        genomes = list_genomes()
        genome = genomes[0] if genomes else ""
    if not genome:
        raise RequestError(
            "Missing required parameter 'Genome' and no default genome available. " + _DCS_USAGE_TEXT
        )
    chromosome = str(body.get("Chromosome", "") or "chr01")
    if "Start" not in body or body.get("Start") is None or str(body.get("Start", "")).strip() == "":
        raise RequestError("Missing required parameter 'Start' (1-based int). " + _DCS_USAGE_TEXT)
    try:
        start_1 = int(body["Start"])
    except (TypeError, ValueError):
        raise RequestError(
            f"Invalid 'Start' value '{body.get('Start')}': must be an int. " + _DCS_USAGE_TEXT
        )
    if start_1 < 1:
        raise RequestError(f"Parameter 'Start' must be >= 1 (1-based). " + _DCS_USAGE_TEXT)
    atac_source = str(body.get("ATAC-seq_Source", "") or "").strip() or None
    uploaded_atac = str(body.get("uploaded_atac", "") or "").strip() or None
    if not atac_source and not uploaded_atac:
        raise RequestError(
            "Missing ATAC input: provide 'ATAC-seq_Source' (built-in) or "
            "'uploaded_atac' (file path). " + _DCS_USAGE_TEXT
        )
    output_format = str(body.get("Format", "Full") or "Full").strip().lower()
    if output_format not in ("full", "mean"):
        raise RequestError(
            f"Invalid 'Format' '{body.get('Format')}': must be Full or Mean. " + _DCS_USAGE_TEXT
        )
    return genome, chromosome, start_1, atac_source, uploaded_atac, output_format


def _lazy_register_genome(body: dict) -> tuple:
    """If body carries fasta_path, register it in place (managed=False, zero-copy)
    and attach optional gff.  Returns (genome_id, cfg-or-None)."""
    fasta_path = str(body.get("fasta_path", "") or "").strip()
    if not fasta_path:
        return "", None
    gff_path = str(body.get("gff_path", "") or "").strip() or ""
    fname = os.path.basename(fasta_path) or "custom.fa"
    genome_id = f"{time.time_ns()}_{fname}"
    cfg = register_uploaded_genome(fasta_path, genome_id=genome_id, managed=False)
    gid = str(cfg.get("id") or genome_id)
    if gff_path:
        cfg = attach_uploaded_gff(gid, gff_path)
    return gid, cfg


@app.post("/api/aigress/openai/OGR-Regulation/Genome")
@app.post("/OGR-Regulation/Genome")
async def dcs_genomes(authorization: str | None = Header(default=None),
                      x_api_key: str | None = Header(default=None)):
    """List configured genomes (DCS-style sub-path)."""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), "Genome list",
        {"model": "OGR-Regulation", "Genome": list_genomes()},
    ))


@app.post("/api/aigress/openai/OGR-Regulation/Chromosome")
@app.post("/OGR-Regulation/Chromosome")
async def dcs_chromosomes(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """List chromosomes of a genome (DCS-style sub-path)."""
    try:
        body = await req.json()
    except Exception:
        body = {}
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)
    genome = str((body or {}).get("Genome", "") or "") or (list_genomes() or [""])[0]
    try:
        cfg = resolve_genome_config(genome)
        chrs = get_genome_chromosomes(genome, cfg)
    except Exception as e:
        return JSONResponse(_dcs_err(f"Chromosome list failed: {e}", 400), status_code=400)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), "Chromosome list",
        {"model": "OGR-Regulation", "Genome": genome, "Chromosome": chrs},
    ))


@app.post("/api/aigress/openai/OGR-Regulation/Prediction")
@app.post("/OGR-Regulation/Prediction")
async def dcs_predict(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """ATAC→RNA-seq expression prediction (DCS-style sub-path)."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _DCS_USAGE_TEXT, 400), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _DCS_USAGE_TEXT, 400), status_code=400)
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)

    try:
        # Lazy register fasta_path → replaces Genome key.
        lazy_genome, lazy_cfg = _lazy_register_genome(body)
        if lazy_genome:
            body["Genome"] = lazy_genome

        genome, chromosome, start_1, atac_source, uploaded_atac, fmt = _parse_common(body)
        genome_config = resolve_genome_config(genome)
        atac_path = resolve_atac_path(atac_source=atac_source, uploaded_atac=uploaded_atac)

        start_0 = max(0, start_1 - 1)
        core = run_prediction_core(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=None,
            atac_path=atac_path,
            fasta_path=genome_config["fasta"],
            genome_config=genome_config,
        )
    except RequestError as e:
        return JSONResponse(
            _dcs_err(f"Prediction failed: {e}", 400, detail={"request": _summarize_body(body)}),
            status_code=400,
        )
    except FileNotFoundError as e:
        return JSONResponse(
            _dcs_err(f"Prediction failed: {e}", 404, detail={"request": _summarize_body(body)}),
            status_code=404,
        )
    except ValueError as e:
        return JSONResponse(
            _dcs_err(f"Prediction failed: {e}", 400, detail={"request": _summarize_body(body)}),
            status_code=400,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            _dcs_err(f"Prediction failed: {e}", 500, detail={"request": _summarize_body(body)}),
            status_code=500,
        )

    chrom, pos_start, pos_end = core["position"]
    pos_start, pos_end = int(pos_start), int(pos_end)
    values = core["values"]
    plus = _format_array(values["RNA-seq_+"], fmt)
    minus = _format_array(values["RNA-seq_-"], fmt)
    prompt_tokens = pos_end - pos_start
    completion_tokens = _count_elements(values)

    result = {
        "model": "OGR-Regulation",
        "Genome": genome,
        "fasta_path": (lazy_cfg or {}).get("fasta", "") if lazy_cfg else "",
        "gff_path": (lazy_cfg or {}).get("gff", "") if lazy_cfg else "",
        "Chromosome": chrom,
        "Start": pos_start + 1,
        "window_start": pos_start,
        "window_len": pos_end - pos_start,
        "ATAC-seq_Source": atac_source or "",
        "atac_path": atac_path,
        "Format": "Full" if fmt == "full" else "Mean",
        "values": {"RNA-seq_+": plus, "RNA-seq_-": minus},
        "igv_payload": core.get("igv_payload"),
    }
    return JSONResponse(_dcs_ok(
        _dcs_usage(prompt_tokens, completion_tokens),
        "ATAC→RNA-seq expression prediction succeeded",
        result,
    ))


@app.post("/api/aigress/openai/OGR-Regulation/RegisterGenome")
@app.post("/OGR-Regulation/RegisterGenome")
async def dcs_register_genome(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Register a custom genome by local path (DCS-style sub-path).

    Two modes:
      - fasta_path present                    → register a new genome (auto .fai)
      - fasta_path empty + genome_id + gff_path → attach a GFF to an existing genome
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _DCS_USAGE_TEXT, 400), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _DCS_USAGE_TEXT, 400), status_code=400)
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)

    try:
        fasta_path = str(body.get("fasta_path", "") or "").strip()
        gff_path = str(body.get("gff_path", "") or "").strip()
        genome_id = str(body.get("genome_id", "") or "").strip()

        if not fasta_path:
            # Attach-GFF mode: the genome must already be registered.
            if not genome_id or not gff_path:
                raise RequestError(
                    "RegisterGenome needs 'fasta_path' (register) or "
                    "'genome_id' + 'gff_path' (attach annotation). " + _DCS_USAGE_TEXT
                )
            cfg = attach_uploaded_gff(genome_id, gff_path)
            gid = genome_id
            message = "annotation attached"
        else:
            if not genome_id:
                genome_id = f"{time.time_ns()}_{os.path.basename(fasta_path) or 'custom.fa'}"
            cfg = register_uploaded_genome(fasta_path, genome_id=genome_id, managed=False)
            gid = str(cfg.get("id") or genome_id)
            if gff_path:
                cfg = attach_uploaded_gff(gid, gff_path)
            message = "genome registered"

        chroms = get_genome_chromosomes(gid, cfg)
    except RequestError as e:
        return JSONResponse(_dcs_err(f"RegisterGenome failed: {e}", 400, detail={"request": _summarize_body(body)}), status_code=400)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse(_dcs_err(f"RegisterGenome failed: {e}", 400, detail={"request": _summarize_body(body)}), status_code=400)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(_dcs_err(f"RegisterGenome failed: {e}", 500, detail={"request": _summarize_body(body)}), status_code=500)

    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), message,
        {
            "model": "OGR-Regulation",
            "Genome": gid,
            "fasta_path": (cfg or {}).get("fasta", ""),
            "gff_path": (cfg or {}).get("gff", ""),
            "chromosomes": chroms,
        },
    ))


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

        # CACHE_MODE=false (lazy-load deployment): skip the content-addressed
        # cache and re-run inference; intermediate bigWigs still go to the
        # prediction cache dir and are cleared on restart.
        igv_payload = cached_or_compute(cache_key, _compute) if CACHE_MODE else _compute()

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
