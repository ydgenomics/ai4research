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

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from rice_mutation.cache_service import prediction_cache, start_bigwig_cleanup
from rice_mutation.igv_payload import set_static_base_url
from rice_mutation.prediction_service import (
    CACHE_MODE,
    _adjust_window,
    attach_uploaded_gff,
    compute_region_means,
    compute_region_stats,
    get_bigwig_cleanup_settings,
    get_genome_chromosomes,
    get_prediction_cache_dir,
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
# bigWig output dir follows cache mode (persistent vs temp)
PREDICTION_BW_DIR_ABS = get_prediction_cache_dir()
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
#  Static-file caching
# ---------------------------------------------------------------------------
# plotly.min.js (~3.5 MB) and igv.min.js (多 MB) are served through
# /static-files; without Cache-Control the browser re-validates (304
# round-trip) on every page load / iframe rebuild, which makes the bar-plot
# iframe show its "Plotly chart library is loading" placeholder.  Everything
# under /static-files is content-addressed (prediction_id) or a fixed filename
# (plotly.min.js / igv.min.js), so a one-year immutable cache is safe: the
# browser loads it once, then never re-requests it for the same container
# run.  New predictions get new prediction_id URLs, so a stale URL can never
# hit an old cached body.
@app.middleware("http")
async def add_static_cache_headers(request, call_next):
    response = await call_next(request)
    if response.status_code < 400 and request.url.path.startswith("/static-files/"):
        if not response.headers.get("Cache-Control"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


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
    # When False (default) skip writing intermediate bigWig / figure files —
    # used by pure-API (curl) callers that only need numeric values.  The web
    # frontend always passes True so IGV tracks are rendered.
    save_intermediate: bool = False


class SnvRequest(BaseModel):
    genome: str
    chromosome: str
    start: int
    end: Optional[int] = None
    biosample_names: Optional[list] = None
    snv_index: int
    snv_base: str
    # Same semantics as PredictRequest.save_intermediate.
    save_intermediate: bool = False


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
    # --- unified DCS-style envelope (also returned by every prediction route) ---
    model_name: Optional[str] = "OGR"
    usage: Optional[dict] = None
    token_cnt: Optional[int] = 0
    status: Optional[int] = 200
    result: Optional[dict] = None


# ---------------------------------------------------------------------------
#  Unified DCS-style response wrapper
# ---------------------------------------------------------------------------
# All prediction endpoints return the same outer envelope (whether they are
# called from the web front end via the snake_case paths or by pure API /
# gateway clients via the DCS-style camelCase sub-paths):
#
#     {
#       "model_name": "OGR",
#       "usage": {"prompt_tokens": N, "completion_tokens": M},
#       "token_cnt": N + M,
#       "status": 200,
#       "message": "...",
#       "result": {
#         "model": "OGR-Prediction",
#         "Genome": "...", "Chromosome": "...", "Start": N, "window_len": N,
#         "Format": "Full|Mean",
#         "values": {...},          # numeric result (ref prediction)
#         "igv_payload": {...}      # visualisation URLs (fa/gff/bw + locus + tracks)
#         # SNV additionally: ref_values / mut_values
#       }
#     }
#
# ``metadata`` (ref_id / snv_id / window info) is merged into ``result`` so a
# pure API caller has everything the web front end needs to render IGV.
PROMPT_TOKEN_MULTIPLIER = float(os.getenv("DCS_PROMPT_TOKEN_MULTIPLIER", "1"))
COMPLETION_TOKEN_MULTIPLIER = float(os.getenv("DCS_COMPLETION_TOKEN_MULTIPLIER", "1"))
# Optional API-key auth for the DCS-style sub-paths (empty = disabled).
DCS_API_KEY = os.getenv("DCS_API_KEY", "").strip()


def _count_elements(values: dict) -> int:
    """Count total output array elements (for completion_tokens)."""
    total = 0
    for bios_map in values.values():
        for arr in bios_map.values():
            total += int(np.asarray(arr).size)
    return total


def _format_values(values: dict, fmt: str) -> dict:
    """Convert {assay: {biosample: np.ndarray}} to JSON-serialisable values.

    full — full arrays (rounded to 6 dp); mean — per-track window mean scalar.
    """
    out: dict = {}
    for assay, bios_map in values.items():
        out[assay] = {}
        for bios, arr in bios_map.items():
            arr = np.asarray(arr, dtype=float)
            if fmt == "mean":
                out[assay][bios] = round(float(arr.mean()), 6)
            else:  # full
                out[assay][bios] = [round(float(x), 6) for x in arr]
    return out


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


# ---------------------------------------------------------------------------
#  DCS-style parameter parsing (camelCase, 1-based Start)
# ---------------------------------------------------------------------------
_USAGE = (
    "Usage: POST /api/aigress/openai/OGR-Prediction/Prediction "
    "or .../OGR-Prediction/SNV. "
    "Body keys: model (OGR-Prediction, optional), Genome (optional, default first), "
    "Chromosome (optional, default chr01), Start (required, 1-based int), "
    "Format (optional, Full|Mean, default Full), "
    "biosample_names (optional, list or comma-separated string); "
    "SNV additionally: SNV_position (required, 1-based int), "
    "SNV_base (required, A/C/G/T). No mode/end/max_points keys."
)


class RequestError(ValueError):
    """Request parameter error (→ 400), distinct from execution errors (→ 500)."""


def _parse_common(body: dict):
    """Parse Genome / Chromosome / Start / Format / biosample_names.

    Returns (genome, chromosome, start_1, biosample_names, output_format).
    Coordinates are 0-based half-open internally (window 32768 centre-aligned).
    """
    genome = str(body.get("Genome", "") or "")
    if not genome:
        genomes = list_genomes()
        genome = genomes[0] if genomes else "osa1_r7"
    chromosome = str(body.get("Chromosome", "") or "chr01")
    if "Start" not in body or body.get("Start") is None or str(body.get("Start", "")).strip() == "":
        raise RequestError("Missing required parameter 'Start' (1-based int). " + _USAGE)
    try:
        start_1 = int(body["Start"])
    except (TypeError, ValueError):
        raise RequestError(f"Invalid 'Start' value '{body.get('Start')}': must be an int. " + _USAGE)
    biosample_names = body.get("biosample_names")
    if isinstance(biosample_names, str) and biosample_names.strip():
        biosample_names = [s.strip() for s in biosample_names.split(",") if s.strip()]
    if not isinstance(biosample_names, list):
        biosample_names = None
    output_format = str(body.get("Format", "Full") or "Full").strip().lower()
    if output_format not in ("full", "mean"):
        raise RequestError(
            f"Invalid 'Format' '{body.get('Format')}': must be Full or Mean. " + _USAGE
        )
    return genome, chromosome, start_1, biosample_names, output_format


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


def _unauthorized(message: str = "Invalid or missing API Key") -> dict:
    return {
        "model_name": "OGR",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "token_cnt": 0,
        "status": 401,
        "message": message,
        "result": None,
    }


def _lazy_register_genome(body: dict) -> tuple:
    """If body carries fasta_path, register it in place (managed=False) and
    attach optional gff_path.  Returns (genome_id, cfg-or-None)."""
    fasta_path = str(body.get("fasta_path", "") or "").strip()
    if not fasta_path:
        return "", None
    gff_path = str(body.get("gff_path", "") or "").strip() or ""
    fname = os.path.basename(fasta_path) or "custom.fa"
    genome_id = f"{time.time_ns()}_{fname}"
    cfg = register_uploaded_genome(fasta_path, genome_id=genome_id, managed=False)
    genome_id = str(cfg.get("id") or genome_id)
    if gff_path:
        cfg = attach_uploaded_gff(genome_id, gff_path)
    return genome_id, cfg


def _summarize_body(body: dict) -> dict:
    """Request-body digest: only fields needed to locate / reproduce an error."""
    keys = (
        "model", "Genome", "Chromosome", "Start",
        "SNV_position", "SNV_base", "Format", "biosample_names",
        "fasta_path", "genome_id", "gff_path",
    )
    return {k: body[k] for k in keys if k in body}


# ---------------------------------------------------------------------------
#  Startup / shutdown
# ---------------------------------------------------------------------------
def _startup_clean_enabled() -> bool:
    """startup 缓存清理开关。

    仅当 start_all.sh 以 web / monitor-web(ENABLE_WEB=true)启动时后端才会
    执行 startup 清理(rmtree + makedirs + 后台 TTL 清理线程),此时注入
    OGR_ENABLE_STARTUP_CLEAN=1。纯 API 模式(start / monitor)注入 =0,
    后端跳过所有删除与目录重建,以规避低权限部署下缓存目录不可删/只读导致
    启动失败(纯 API 调用方默认不保存中间文件,见 save_intermediate)。
    """
    return str(os.getenv("OGR_ENABLE_STARTUP_CLEAN", "0")).strip() in ("1", "true", "yes", "on")


@app.on_event("startup")
def on_startup():
    # web 模式才清理旧缓存目录;纯 API 模式不删除、不重建(低权限部署兜底)。
    if _startup_clean_enabled():
        # Clean caches from previous runs: uploaded dirs are always service-owned,
        # and the prediction bigWig dir is wiped on every start (both cache modes).
        for d in (FASTA_CACHE_DIR_ABS, UPLOADED_GENOMES_DIR_ABS, PREDICTION_BW_DIR_ABS):
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"[startup] Cleaned cache: {d}")
            os.makedirs(d, exist_ok=True)

    # Mount root filesystem so IGV.js can load local files via HTTP.
    static_mount_path = "/static-files"
    app.mount(static_mount_path, StaticFiles(directory="/"), name="static-files")

    # IGV track URLs are generated as **frontend same-origin relative paths**
    # (/backend/static-files/<abs path>).  The browser always talks to the
    # frontend port (:8000), which reverse-proxies /backend/* to this backend,
    # so a relative URL is directly consumable by the front end without any
    # host rewriting.  (An absolute http://127.0.0.1:8001/... URL would force
    # the front end to rewrite it, and would break when the browser cannot
    # reach the backend port directly.)
    set_static_base_url("/backend/static-files")

    init_predictor()

    # Background cleanup of stale prediction bigWig files (interval/TTL follow
    # the cache mode; lightweight mode cleans the temp dir aggressively).
    # 清理线程同样只在 web 模式下启动(纯 API 不写 bigWig,无需清理)。
    if _startup_clean_enabled():
        bw_interval, bw_ttl = get_bigwig_cleanup_settings()
        start_bigwig_cleanup(PREDICTION_BW_DIR_ABS,
                             interval_seconds=bw_interval, ttl_seconds=bw_ttl)

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
async def upload_fasta(
    file: UploadFile = File(...),
    path: str = Form(default=""),
):
    """Upload a custom genome FASTA file.  Automatically builds the .fai index
    and registers it as a new genome (priority over built-in genomes).

    Lightweight cache mode (``CACHE_MODE=false``): the front end sends the local
    ``path`` of the file (front/back end share the machine); the backend
    registers that path in place without copying the file.  Otherwise the file
    content is stored into the upload cache directory as before.

    Returns the new genome id, its config paths and chromosome names.
    """
    # Zero-copy registration path (lightweight mode): validate + register the
    # given local path directly.  The multipart content is still transmitted
    # (kept for the full-cache mode), but we prefer the path when available.
    local_path = str(path or "").strip()
    if local_path and not CACHE_MODE:
        if not os.path.isfile(local_path):
            raise HTTPException(400, f"Path not found on server: {local_path}")
        suffix = Path(local_path).suffix.lower()
        if suffix not in (".fa", ".fasta", ".fna"):
            raise HTTPException(
                400,
                f"Unsupported file type '{suffix}'. Only .fa / .fasta accepted.",
            )
        try:
            cfg = register_uploaded_genome(local_path, managed=False)
            genome_id = str(cfg.get("id") or "")
            chromosomes = get_genome_chromosomes(genome_id, cfg)
        except Exception as e:
            raise HTTPException(400, f"Failed to index FASTA: {e}")
        return {
            "success": True,
            "genome": genome_id,
            "file_path": cfg["fasta"],
            "file_name": Path(local_path).name,
            "size_bytes": os.path.getsize(local_path),
            "chromosomes": chromosomes,
            "registered_in_place": True,
        }

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
        genome_id = str(cfg.get("id") or "")
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
    path: str = Form(default=""),
):
    """Attach an annotation GFF to an uploaded custom genome.

    - ``genome``: the custom genome id returned by ``/uploadFasta``.
    - ``file``:  .gff / .gff3 / .gtf (optionally gzipped).
    - ``path``:  optional local path (lightweight cache mode) — when set and
      ``CACHE_MODE=false`` the GFF is registered in place without copying.

    Updates the in-memory registry so IGV payloads include a Genes track for
    the custom genome.  Only uploaded (custom) genomes can be annotated.
    """
    # Zero-copy registration path (lightweight mode).
    local_path = str(path or "").strip()
    if local_path and not CACHE_MODE:
        if not os.path.isfile(local_path):
            raise HTTPException(400, f"Path not found on server: {local_path}")
        lower_name = local_path.lower()
        if not (
            lower_name.endswith((".gff", ".gff3", ".gtf"))
            or lower_name.endswith((".gff.gz", ".gff3.gz", ".gtf.gz"))
        ):
            raise HTTPException(
                400,
                "Unsupported file type. Only .gff / .gff3 / .gtf (optionally .gz) accepted.",
            )
        try:
            cfg = attach_uploaded_gff(genome, local_path)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {
            "success": True,
            "genome": genome,
            "gff_path": cfg["gff"],
            "file_name": Path(local_path).name,
            "size_bytes": os.path.getsize(local_path),
            "registered_in_place": True,
        }

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
    """Run reference-sequence expression prediction (content-cached).

    Returns the unified DCS-style envelope; ``result`` carries both the
    numeric values (Full/Mean) and the igv_payload (visualisation URLs).
    """
    t0 = time.time()
    try:
        predictor = require_predictor()

        # Content-addressed cache is only used in full cache mode; lightweight
        # mode re-runs every request (temp bigWigs are still short-TTL cleaned).
        if CACHE_MODE:
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
                # Reuse cached payload + metadata (values arrays stay in _REF_CACHE).
                meta = hit["metadata"] or {}
                ref_id = meta.get("ref_id", "")
                prompt_tokens = int(meta.get("window_len", 0))
                # Cached hit: values arrays are still in _REF_CACHE under ref_id.
                ref_entry = _REF_CACHE.get(ref_id)
                fmt = "full"
                return PredictResponse(
                    success=True,
                    message=f"cached ({elapsed:.1f}s)",
                    igv_payload=hit["igv_payload"],
                    elapsed_seconds=elapsed,
                    metadata=meta,
                    # extra unified envelope fields (consumed by pure API callers)
                    model_name="OGR",
                    usage=_dcs_usage(prompt_tokens, 0),
                    token_cnt=prompt_tokens,
                    status=200,
                    result={
                        "model": "OGR-Prediction",
                        "Genome": req.genome,
                        "Chromosome": meta.get("chromosome", req.chromosome),
                        "Start": int(meta.get("window_start", 0)) + 1,
                        "window_start": int(meta.get("window_start", 0)),
                        "window_len": int(meta.get("window_len", 0)),
                        "Format": fmt,
                        "values": _format_values(ref_entry["ref_values"], fmt) if ref_entry else None,
                        "mean_values": _format_values(ref_entry["ref_values"], "mean") if ref_entry else None,
                        "igv_payload": hit["igv_payload"],
                        "ref_id": ref_id,
                    },
                )
        else:
            hit = None
            cache_key = None

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

        # Pure-API (curl) callers default to save_intermediate=False → skip
        # writing bigWig / figure files (values are still returned in full).
        if req.save_intermediate:
            track_paths = save_prediction_bigwigs(
                result["values"], chromosome, start, end, tag="ref"
            )
        else:
            track_paths = {}
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

        figure_url = None
        if req.save_intermediate:
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
        # Only persist the content-addressed cache when intermediate files were
        # actually written (an empty-payload entry would poison later web hits).
        if CACHE_MODE and cache_key is not None and req.save_intermediate:
            prediction_cache.put(cache_key, ref_id, igv_payload, metadata, "ref")

        elapsed = time.time() - t0
        window_len = end - start
        # Numeric result — default Full arrays (web front end consumes igv_payload;
        # pure API callers get both values and visualisation URLs).
        return PredictResponse(
            success=True,
            message=f"Prediction completed in {elapsed:.1f}s",
            igv_payload=igv_payload,
            figure_url=figure_url,
            figure_format=req.figure_format,
            elapsed_seconds=elapsed,
            metadata=metadata,
            model_name="OGR",
            usage=_dcs_usage(window_len, _count_elements(result["values"])),
            token_cnt=_dcs_usage(window_len, _count_elements(result["values"]))["prompt_tokens"]
            + _dcs_usage(window_len, _count_elements(result["values"]))["completion_tokens"],
            status=200,
            result={
                "model": "OGR-Prediction",
                "Genome": req.genome,
                "Chromosome": chromosome,
                "Start": start + 1,
                "window_start": start,
                "window_len": window_len,
                "Format": "full",
                "values": _format_values(result["values"], "full"),
                "mean_values": _format_values(result["values"], "mean"),
                "igv_payload": igv_payload,
                "figure_url": figure_url,
                "figure_format": req.figure_format,
                "ref_id": ref_id,
            },
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

    Returns the unified DCS-style envelope; ``result`` carries both the
    numeric ref/mut values (Full/Mean) and the igv_payload (dual tracks).
    """
    t0 = time.time()
    try:
        predictor = require_predictor()

        # Content-addressed cache is only used in full cache mode; lightweight
        # mode re-runs every request.
        if CACHE_MODE:
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
                meta = hit["metadata"] or {}
                snv_id = meta.get("snv_id", "")
                entry = _SNV_CACHE.get(snv_id)
                fmt = "full"
                return PredictResponse(
                    success=True,
                    message=f"cached ({elapsed:.1f}s)",
                    igv_payload=hit["igv_payload"],
                    elapsed_seconds=elapsed,
                    metadata=meta,
                    model_name="OGR",
                    usage=_dcs_usage(int(meta.get("window_len", 0)), 0),
                    token_cnt=int(meta.get("window_len", 0)),
                    status=200,
                    result={
                        "model": "OGR-Prediction",
                        "Genome": req.genome,
                        "Chromosome": meta.get("chromosome", req.chromosome),
                        "Start": int(meta.get("window_start", 0)) + 1,
                        "window_start": int(meta.get("window_start", 0)),
                        "window_len": int(meta.get("window_len", 0)),
                        "SNV_position": int(meta.get("snv_index", 0)) + 1,
                        "ref_base": meta.get("ref_base", ""),
                        "SNV_base": meta.get("snv_base", ""),
                        "Format": fmt,
                        "ref_values": _format_values(entry["ref_values"], fmt) if entry else None,
                        "mut_values": _format_values(entry["mut_values"], fmt) if entry else None,
                        "ref_mean_values": _format_values(entry["ref_values"], "mean") if entry else None,
                        "mut_mean_values": _format_values(entry["mut_values"], "mean") if entry else None,
                        "igv_payload": hit["igv_payload"],
                        "snv_id": snv_id,
                    },
                )
        else:
            hit = None
            cache_key = None

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

        # Pure-API (curl) callers default to save_intermediate=False → skip
        # writing the result1/result2 bigWig files (values still returned).
        if req.save_intermediate:
            ref_track_paths = save_prediction_bigwigs(
                result["ref_values"], chromosome, start, end, tag="result1"
            )
            mut_track_paths = save_prediction_bigwigs(
                result["mut_values"], chromosome, start, end, tag="result2"
            )
        else:
            ref_track_paths = {}
            mut_track_paths = {}
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
        # Only persist the content-addressed cache when intermediate files were
        # actually written (an empty-payload entry would poison later web hits).
        if CACHE_MODE and cache_key is not None and req.save_intermediate:
            prediction_cache.put(cache_key, snv_id, igv_payload, metadata, "snv")

        elapsed = time.time() - t0
        window_len = result["window_len"]
        completion = _count_elements(result["ref_values"]) + _count_elements(result["mut_values"])
        usage = _dcs_usage(window_len, completion)
        return PredictResponse(
            success=True,
            message=f"SNV prediction completed in {elapsed:.1f}s",
            igv_payload=igv_payload,
            elapsed_seconds=elapsed,
            metadata=metadata,
            model_name="OGR",
            usage=usage,
            token_cnt=usage["prompt_tokens"] + usage["completion_tokens"],
            status=200,
            result={
                "model": "OGR-Prediction",
                "Genome": req.genome,
                "Chromosome": chromosome,
                "Start": start + 1,
                "window_start": start,
                "window_len": window_len,
                "SNV_position": result["snv_index"] + 1,
                "ref_base": result["ref_base"],
                "SNV_base": result["snv_base"],
                "Format": "full",
                "ref_values": _format_values(result["ref_values"], "full"),
                "mut_values": _format_values(result["mut_values"], "full"),
                "ref_mean_values": _format_values(result["ref_values"], "mean"),
                "mut_mean_values": _format_values(result["mut_values"], "mean"),
                "igv_payload": igv_payload,
                "snv_id": snv_id,
            },
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
        return _dcs_ok(
            _dcs_usage(0, 0),
            "snv stat computed",
            {
                "success": True,
                "snv_id": req.snv_id,
                "region": [req.region_start, req.region_end],
                "stats": stats,
                "meta": entry["meta"],
            },
        )
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
        return _dcs_ok(
            _dcs_usage(0, 0),
            "no overlap",
            {
                "success": True,
                "overlap": False,
                "kind": kind,
                "window": [win_start, win_end],
                "region": [region_start, region_end],
                "values": [],
            },
        )

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

    data = {
        "success": True,
        "overlap": True,
        "kind": kind,
        "window": [win_start, win_end],
        "overlap_region": [ov_start, ov_end],
        "region": [region_start, region_end],
        "values": values,
    }
    return _dcs_ok(
        _dcs_usage(len(values), 0),
        "bar data computed",
        data,
    )


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
            # Figures follow the same cache-mode directory as bigWigs (persistent
            # cache or temp dir); they are served via /static-files either way.
            out_dir=PREDICTION_BW_DIR_ABS,
            fmt=fmt,
        )
        if not fig_path:
            return None
        return _path_to_url(fig_path)
    except Exception as e:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
#  DCS-style sub-path routes (camelCase params, 1-based Start)
# ---------------------------------------------------------------------------
# These are the same backend process as the web routes: a pure API caller
# (curl / DCS gateway) hits the same inference core and receives the same
# unified envelope, so the web front end and the API path are one route.
#  - /api/aigress/openai/OGR-Prediction/Prediction   → reference prediction
#  - /api/aigress/openai/OGR-Prediction/SNV          → SNV prediction
#  - /api/aigress/openai/OGR-Prediction/Genome       → genome list
#  - /api/aigress/openai/OGR-Prediction/Chromosome   → chromosome list
#  - /api/aigress/openai/OGR-Prediction/RegisterGenome → register fasta_path
@app.post("/api/aigress/openai/OGR-Prediction/Prediction")
@app.post("/OGR-Prediction/Prediction")
async def dcs_predict_ref(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Reference prediction (DCS-style sub-path)."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _USAGE, 400), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _USAGE, 400), status_code=400)
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)

    try:
        # Lazy register fasta_path (managed=False) → replaces Genome key.
        lazy_genome, lazy_cfg = _lazy_register_genome(body)
        if lazy_genome:
            body["Genome"] = lazy_genome

        genome, chromosome, start_1, biosample_names, fmt = _parse_common(body)
        start_0 = max(0, start_1 - 1)
        # save_intermediate defaults False for pure-API (curl) callers; the web
        # frontend always passes True so IGV bigWig tracks are written.
        save_intermediate = bool(body.get("save_intermediate", False))
        # Reuse the exact same core endpoint → same unified envelope.
        web_req = PredictRequest(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=None,
            biosample_names=biosample_names,
            figure_format="none",
            save_intermediate=save_intermediate,
        )
        resp: PredictResponse = predict(web_req)
    except RequestError as e:
        return JSONResponse(
            _dcs_err(f"Reference prediction failed: {e}", 400,
                     detail={"request": _summarize_body(body)}),
            status_code=400,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            _dcs_err(f"Reference prediction failed: {e}", 500,
                     detail={"request": _summarize_body(body)}),
            status_code=500,
        )

    result = resp.result or {}
    # Override Format per caller's request: Mean collapses the per-base arrays
    # to scalars and drops the Full-only mean_values duplicate.
    if fmt == "mean" and result.get("values"):
        result = {**result, "Format": "mean",
                  "values": _format_values(resp.result["values"] or {}, "mean")}
        result.pop("mean_values", None)
    # Attach fasta_path/gff_path for lazy-registered custom genomes.
    if lazy_cfg:
        result["fasta_path"] = lazy_cfg.get("fasta", "")
        result["gff_path"] = lazy_cfg.get("gff", "")
    payload = {
        "model_name": resp.model_name,
        "usage": resp.usage,
        "token_cnt": resp.token_cnt,
        "status": resp.status,
        "message": resp.message,
        "result": result,
    }
    return JSONResponse(payload)


@app.post("/api/aigress/openai/OGR-Prediction/SNV")
@app.post("/OGR-Prediction/SNV")
async def dcs_predict_snv(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """SNV prediction (DCS-style sub-path)."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _USAGE, 400), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(_dcs_err("Request body is not valid JSON. " + _USAGE, 400), status_code=400)
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)

    try:
        lazy_genome, lazy_cfg = _lazy_register_genome(body)
        if lazy_genome:
            body["Genome"] = lazy_genome

        genome, chromosome, start_1, biosample_names, fmt = _parse_common(body)
        if "SNV_position" not in body or body.get("SNV_position") is None or str(body.get("SNV_position", "")).strip() == "":
            raise RequestError("Missing required parameter 'SNV_position' (1-based int). " + _USAGE)
        try:
            snv_1 = int(body["SNV_position"])
        except (TypeError, ValueError):
            raise RequestError(
                f"Invalid 'SNV_position' value '{body.get('SNV_position')}': must be an int. " + _USAGE
            )
        snv_base = str(body.get("SNV_base", "") or "").strip().upper()
        if snv_base not in ("A", "C", "G", "T"):
            raise RequestError(
                f"Invalid 'SNV_base' '{body.get('SNV_base')}': must be A/C/G/T. " + _USAGE
            )

        start_0 = max(0, start_1 - 1)
        snv_0 = max(0, snv_1 - 1)
        # save_intermediate defaults False for pure-API (curl) callers; the web
        # frontend always passes True so IGV bigWig tracks are written.
        save_intermediate = bool(body.get("save_intermediate", False))
        web_req = SnvRequest(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=None,
            biosample_names=biosample_names,
            snv_index=snv_0,
            snv_base=snv_base,
            save_intermediate=save_intermediate,
        )
        resp: PredictResponse = predict_snv(web_req)
    except RequestError as e:
        return JSONResponse(
            _dcs_err(f"SNV prediction failed: {e}", 400,
                     detail={"request": _summarize_body(body)}),
            status_code=400,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            _dcs_err(f"SNV prediction failed: {e}", 500,
                     detail={"request": _summarize_body(body)}),
            status_code=500,
        )

    result = resp.result or {}
    if fmt == "mean":
        result = {
            **result, "Format": "mean",
            "ref_values": _format_values(resp.result.get("ref_values") or {}, "mean"),
            "mut_values": _format_values(resp.result.get("mut_values") or {}, "mean"),
        }
        # Full-only scalar duplicates are not returned in Mean mode.
        result.pop("ref_mean_values", None)
        result.pop("mut_mean_values", None)
    if lazy_cfg:
        result["fasta_path"] = lazy_cfg.get("fasta", "")
        result["gff_path"] = lazy_cfg.get("gff", "")
    payload = {
        "model_name": resp.model_name,
        "usage": resp.usage,
        "token_cnt": resp.token_cnt,
        "status": resp.status,
        "message": resp.message,
        "result": result,
    }
    return JSONResponse(payload)


@app.post("/api/aigress/openai/OGR-Prediction/Genome")
@app.post("/OGR-Prediction/Genome")
async def dcs_genomes(authorization: str | None = Header(default=None),
                      x_api_key: str | None = Header(default=None)):
    """List configured genomes (DCS-style sub-path)."""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), "genome list",
        {"model": "OGR-Prediction", "Genome": list_genomes()},
    ))


@app.post("/api/aigress/openai/OGR-Prediction/Chromosome")
@app.post("/OGR-Prediction/Chromosome")
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
    genome = str((body or {}).get("Genome", "") or "") or (list_genomes() or ["osa1_r7"])[0]
    try:
        cfg = resolve_genome_config(genome)
        chrs = get_genome_chromosomes(genome, cfg)
    except Exception as e:
        return JSONResponse(_dcs_err(f"Chromosome list failed: {e}", 400), status_code=400)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), "chromosome list",
        {"model": "OGR-Prediction", "Genome": genome, "Chromosome": chrs},
    ))


@app.post("/api/aigress/openai/OGR-Prediction/RegisterGenome")
@app.post("/OGR-Prediction/RegisterGenome")
async def dcs_register_genome(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """Register a custom genome by local path (DCS-style sub-path).

    Body (camelCase, same dialect as curl callers):
      {"fasta_path": "/abs/path/to/xxx.fa",
       "genome_id": "my_id",           # optional, default auto timestamp id
       "gff_path": "/abs/path/xxx.gff3"}  # optional annotation

    Two modes:
      - fasta_path present          → register a new genome (auto .fai index).
      - fasta_path empty + genome_id + gff_path → attach a GFF to an already
        registered custom genome (frontend "attach GFF" action).
    Files must already exist on the backend machine (no file transfer).
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(_dcs_err("Request body is not valid JSON.", 400), status_code=400)
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
                    "'genome_id' + 'gff_path' (attach annotation)."
                )
            cfg = attach_uploaded_gff(genome_id, gff_path)
            message = "annotation attached"
        else:
            if not genome_id:
                genome_id = f"{time.time_ns()}_{os.path.basename(fasta_path) or 'custom.fa'}"
            cfg = register_uploaded_genome(
                fasta_path, genome_id=genome_id, managed=False
            )
            # register_uploaded_genome may suffix the id on collision; the cfg
            # records the final id, so prefer it over the requested one.
            genome_id = str((cfg or {}).get("id") or genome_id)
            if gff_path:
                cfg = attach_uploaded_gff(genome_id, gff_path)
            message = "genome registered"
    except RequestError as e:
        return JSONResponse(_dcs_err(f"Register genome failed: {e}", 400), status_code=400)
    except FileNotFoundError as e:
        return JSONResponse(_dcs_err(f"Register genome failed: {e}", 404), status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(_dcs_err(f"Register genome failed: {e}", 500), status_code=500)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), message,
        {
            "model": "OGR-Prediction",
            "Genome": genome_id,
            "fasta_path": (cfg or {}).get("fasta", ""),
            "gff_path": (cfg or {}).get("gff", ""),
            "chromosomes": get_genome_chromosomes(genome_id, cfg),
        },
    ))
