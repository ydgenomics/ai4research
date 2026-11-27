"""Rice-Introgression API 路由。

Endpoints:
- GET  /health                     — 健康检查 + 模型/基因组元信息
- GET  /genomes                    — 基因组列表
- GET  /genomes/{id}/chromosomes   — 某基因组的染色体列表
- GET  /genomes/{id}/chromosomes/{chrom}/length — 染色体长度
- POST /uploadFasta                — 上传自定义基因组 FASTA（自动建 .fai；轻量模式可带 path 懒注册）
- POST /analyze                    — 渗入分析（单窗 / 整条染色体）
- POST /analyze-genome             — 全基因组渗入分析
- POST /api/aigress/openai/OGR-Introgression/Genome        — genome list (DCS)
- POST /api/aigress/openai/OGR-Introgression/Chromosome    — chromosome list (DCS)
- POST /api/aigress/openai/OGR-Introgression/Prediction    — introgression (DCS)
- POST /api/aigress/openai/OGR-Introgression/RegisterGenome — register custom genome (DCS)

All DCS sub-paths share one process/port with the web routes, so the web
frontend and pure-API (curl) callers use exactly the same dialect.
"""

import logging
import os
import shutil
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

from rice_introgression.cache_service import prediction_cache
from rice_introgression.genome_service import (
    get_chromosome_length,
    get_genome_chromosomes,
    list_genomes,
    register_uploaded_genome,
    resolve_genome_config,
    start_uploaded_genome_cleanup,
)
from rice_introgression.prediction_service import (
    CACHE_MODE,
    analysis_params_from_env,
    inject_genome_context,
    run_genome_introgression,
    run_introgression,
)
from rice_introgression.predictor import init_predictor, release_predictor

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """解析 .env 布尔开关：true/1/yes/on（不区分大小写）→ True，其余 → default。"""
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes", "on")

ROOT_DIR = Path(__file__).resolve().parents[2]  # rice-intro/
UPLOADED_GENOMES_DIR_ABS = os.path.abspath(os.getenv(
    "BACKEND_UPLOADED_FASTA",
    str(ROOT_DIR / "cache" / "uploaded_fasta"),
))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "640"))
MAX_FASTA_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# 优先使用平台注入的 PORT(API 模式),回退 BACKEND_PORT(本地/网页版)
BACKEND_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "5001")))

# DCS 计费口径与可选 API Key 鉴权（空 = 免鉴权）
PROMPT_TOKEN_MULTIPLIER = float(os.getenv("DCS_PROMPT_TOKEN_MULTIPLIER", "1"))
COMPLETION_TOKEN_MULTIPLIER = float(os.getenv("DCS_COMPLETION_TOKEN_MULTIPLIER", "1"))
DCS_API_KEY = os.getenv("DCS_API_KEY", "").strip()

app = FastAPI(title="Rice-Introgression Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载根文件系统：使 iframe 能通过
#   <frontend>/backend/static-files<绝对路径>/plotly.min.js 加载前端静态资源
# （经前端反向代理 -> 后端 -> 静态文件，浏览器只接触前端端口）
app.mount("/static-files", StaticFiles(directory="/"), name="static-files")


# ---------------------------------------------------------------------------
#  DCS 统一返回 envelope 与鉴权（网页端与纯 API 调用共用同一套结构）
# ---------------------------------------------------------------------------
class RequestError(ValueError):
    """请求参数错误（→ 400），与预测执行错误（→ 500）区分开。"""


def _dcs_usage(prompt_tokens: int, completion_tokens: int) -> dict:
    pt = int(prompt_tokens * PROMPT_TOKEN_MULTIPLIER)
    ct = int(completion_tokens * COMPLETION_TOKEN_MULTIPLIER)
    return {"prompt_tokens": pt, "completion_tokens": ct}


def _dcs_ok(usage: dict, message: str, result: dict) -> dict:
    """统一成功 envelope（网页端与 API 路由共用）。"""
    return {
        "model_name": "OGR",
        "usage": usage,
        "token_cnt": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        "status": 200,
        "message": message,
        "result": result,
    }


def _dcs_err(message: str, status: int = 400, detail: dict | None = None) -> dict:
    """统一错误 envelope（与 DCS 错误方言一致）。"""
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
    return {
        "model_name": "OGR",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "token_cnt": 0,
        "status": 401,
        "message": message,
        "result": None,
    }


def _check_api_key(authorization: str | None = None, x_api_key: str | None = None) -> None:
    """校验 API Key；未配置 DCS_API_KEY 时直接放行（向后兼容）。"""
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
    """请求体摘要：只保留定位/复现所需字段，用于出错时的 detail。"""
    keys = ("model", "Genome", "Chromosome", "Start", "End", "fasta_path", "genome_id")
    return {k: body[k] for k in keys if k in body}


# 英文使用说明：参数错误时随错误响应返回
_USAGE_HINT = (
    "Usage: POST /api/aigress/openai/OGR-Introgression/Prediction "
    "or .../OGR-Introgression/RegisterGenome. "
    "Body (JSON object; Start is required):\n"
    '  {"Genome": "YF47", "Chromosome": "Chr01", "Start": 100001, "End": 356001}\n'
    "  Genome     : genome ID (default: first configured GENOME_*_FASTA)\n"
    "  Chromosome : chromosome name as in the FASTA header (default: Chr01)\n"
    "  Start      : REQUIRED, 1-based inclusive start coordinate\n"
    "  End        : optional, 1-based inclusive end coordinate\n"
    "Behavior (start-anchored windows, no grid snapping):\n"
    "  - Start only           -> 1 window [Start, Start+256k)\n"
    "  - Start and End        -> overlapping windows (step 64k) from Start until Start>End;\n"
    "                            the last window whose start <= End is kept even if its\n"
    "                            end > End. Capped at MAX_NUMBER_256W windows.\n"
    "  - First window extends beyond the chromosome end -> empty result (status 200,\n"
    "    n_windows=0, message explains); later windows beyond the chromosome end are dropped.\n"
)


def _param_err(message: str, body: dict) -> dict:
    """参数错误响应：message 附英文使用说明，detail 含请求摘要。"""
    return _dcs_err(
        f"{message} {_USAGE_HINT}",
        400,
        detail={"request": _summarize_body(body), "usage_hint": _USAGE_HINT},
    )


def _lazy_register_genome(body: dict) -> tuple[str, dict | None]:
    """若 body 含 fasta_path，现场注册（managed=False）并返回 (genome_id, cfg)；
    否则返回 ("", None)。轻量模式零拷贝：路径指向外部文件，TTL 不删文件。
    """
    fasta_path = str(body.get("fasta_path", "") or "").strip()
    if not fasta_path:
        return "", None

    fname = os.path.basename(fasta_path) or "custom.fa"
    genome_id = f"{time.time_ns()}_{fname}"
    cfg = register_uploaded_genome(fasta_path, genome_id=genome_id, managed=False)
    return genome_id, cfg


# ---------------------------------------------------------------------------
#  Request / Response schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    genome: str
    chromosome: str
    # 前端 1-based（用户友好）；analyze 端点统一转为内部 0-based half-open：
    # start_0b = start_1b - 1，end_0b = end_1b（闭区间终点即开区间终点）。
    # start 为空 -> 整条染色体；end 为空 -> start+256k
    start: Optional[int] = None
    end: Optional[int] = None


class AnalyzeGenomeRequest(BaseModel):
    genome: str
    # 可选：只分析指定染色体（默认全部）
    chromosomes: Optional[list[str]] = None


class AnalyzeWindowRequest(BaseModel):
    genome: str
    chromosome: str
    start: int  # 窗口起点（1-based，前端展示友好）
    end: Optional[int] = None


# ---------------------------------------------------------------------------
#  Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    upload_dir = UPLOADED_GENOMES_DIR_ABS
    if os.path.isdir(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir, exist_ok=True)

    # 统一语义：服务重启时自动清空预测缓存（内存 + 磁盘），不做预热。
    n = len(prediction_cache)
    prediction_cache.clear()
    logger.info("Startup: cleared prediction cache (%d in-memory entries before)", n)

    init_predictor()

    ttl = float(os.getenv("UPLOADED_GENOMES_TTL_HOURS", "1.0"))
    interval = int(os.getenv("UPLOADED_GENOMES_CLEANUP_INTERVAL", "300"))
    start_uploaded_genome_cleanup(ttl_hours=ttl, interval_seconds=interval)


@app.on_event("shutdown")
def on_shutdown():
    release_predictor()


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    from rice_introgression.predictor import _PREDICTOR

    inst = _PREDICTOR.get("instance")
    return {
        "status": "ok",
        "predictor_initialized": inst is not None,
        "genomes": list_genomes(),
        "params": analysis_params_from_env().__dict__,
    }


@app.get("/progress")
def progress():
    """推理进度（前端轮询，Plan B）。

    返回当前活动任务快照：genome / chromosome / done_segments /
    total_segments / percent / elapsed_seconds / status / message。
    无活动任务时返回 {"status": "idle"}。
    """
    from rice_introgression.progress import progress_tracker

    return progress_tracker.get()


@app.get("/genomes")
def genomes():
    return {"genomes": list_genomes()}


@app.get("/genomes/{genome_id}/chromosomes")
def genome_chromosomes(genome_id: str):
    """返回某基因组的染色体名（FASTA 实际命名）。"""
    config = resolve_genome_config(genome_id)
    return {"genome": genome_id, "chromosomes": get_genome_chromosomes(genome_id, config)}


@app.get("/genomes/{genome_id}/chromosomes/{chrom}/length")
def chromosome_length(genome_id: str, chrom: str):
    config = resolve_genome_config(genome_id)
    length = get_chromosome_length(genome_id, chrom, config)
    return {"genome": genome_id, "chromosome": chrom, "length": length}


@app.post("/uploadFasta")
async def upload_fasta(file: UploadFile = File(...),
                       path: str = Form(default="")):
    """上传自定义基因组 FASTA，自动建 .fai 并注册为新 genome。

    轻量模式（CACHE_MODE=false）：前端/后端同机，multipart 可带 ``path`` 表单字段
    指向本机文件路径，后端原地注册（managed=False，零拷贝，不复制文件）。
    否则走传统字节上传（复制进 cache 目录，managed=True，TTL 可删）。
    """
    # 零拷贝注册路径（轻量模式）：直接校验并注册给定本地路径
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


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """渗入分析：单窗（end 空 -> start+256k）或整条染色体（start/end 都空）。

    前端坐标为 1-based（用户友好，自动减一）：
    start_0b = start_1b - 1；end_0b = end_1b（1-based 闭区间终点 = 0-based 开区间终点）。
    推理只跑请求区域（单次），返回恒为全基因组展示 payload：
    chromosomes / chromosome_lengths（12 条车道骨架）+ 本次 windows/regions。
    """
    t0 = time.time()
    start_0b = max(0, req.start - 1) if req.start is not None else None
    end_0b = req.end
    try:
        if not req.chromosome:
            raise ValueError("chromosome is required")
        payload = run_introgression(
            genome=req.genome,
            chromosome=req.chromosome,
            start=start_0b,
            end=end_0b,
        )
        payload = inject_genome_context(payload)
        payload["elapsed_seconds"] = round(time.time() - t0, 3)
        payload["success"] = True
        return payload
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected error: {e}")


@app.post("/analyze-genome")
def analyze_genome(req: AnalyzeGenomeRequest):
    """全基因组渗入分析（12 条染色体全景）。逐染色体走缓存，未命中才推理。"""
    t0 = time.time()
    try:
        payload = run_genome_introgression(genome=req.genome)
        payload["elapsed_seconds"] = round(time.time() - t0, 3)
        payload["success"] = True
        return payload
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
#  DCS 风格子路径（与网页路由同一进程/端口）
#  网页端与纯 API（curl）调用者共用完全相同的请求方言：camelCase 键、
#  1-based 坐标；1-based → 0-based 转换在后端完成。返回统一 envelope
#  {model_name, usage, token_cnt, status, message, result}。
# ---------------------------------------------------------------------------
def _predict_payload_to_result(payload: dict, lazy_cfg: dict | None = None) -> dict:
    """把 run_introgression 的 payload 转换为 DCS result 方言。

    payload 坐标已是 1-based inclusive（win_start = start_0b + 1，
    win_end = 0-based half-open 末尾 == 1-based 闭区间末尾），直接透传；
    空结果（windows 空）仍 200，message 沿用 payload 说明。
    网页端与纯 API（curl）共用此结构：result 同时含窗口数据与
    全基因组展示骨架（chromosomes / chromosome_lengths / regions.chromosome），
    Plotly 渲染完全来自该返回值（与 /analyze 语义一致）。
    """
    params = payload.get("params", {})
    windows = []
    for w in payload.get("windows", []):
        windows.append({
            "win_start": int(w["win_start"]),      # 1-based inclusive
            "win_end": int(w["win_end"]),          # 1-based inclusive（= 0-based half-open 末尾）
            "center": int(w.get("center", 0)),     # 1-based
            "n_segments": int(w["n_segments"]),
            "topk_mean_jap": float(w["topk_mean_jap"]),
            "topk_mean_ind": float(w["topk_mean_ind"]),
            "group": str(w["group"]),              # Ind | Jap | uncertain
        })

    ws = payload.get("win_start")
    we = payload.get("win_end")
    if ws is None or we is None:
        pos = {"start": 1, "end": int(payload["chrom_len"])}
    else:
        pos = {"start": int(ws), "end": int(we)}

    # 全基因组展示骨架（前端 12 车道绘图必需；由 inject_genome_context 注入）
    chromosomes = payload.get("chromosomes", [])
    chrom_lengths = payload.get("chromosome_lengths", {})
    # regions 带 chromosome 字段（绘图按染色体过滤用）
    regions_out = {}
    for g, recs in payload.get("regions", {}).items():
        out = []
        for r in recs:
            r = dict(r)
            r.setdefault("chromosome", payload.get("chromosome"))
            out.append(r)
        regions_out[g] = out

    return {
        "model": "OGR-Introgression",
        "genome": payload["genome"],
        "fasta_path": (lazy_cfg or {}).get("fasta", ""),
        "chromosome": payload["chromosome"],
        "chrom_len": int(payload["chrom_len"]),
        "mode": str(payload["mode"]),              # window | chromosome | genome
        "position_1based": pos,
        "window_len": int(params.get("window_size", 256000)),
        "n_windows": len(windows),
        "windows": windows,
        "segments": payload.get("segments", []),
        "regions": regions_out,
        "chromosomes": chromosomes,
        "chromosome_lengths": chrom_lengths,
        "message": payload.get("message", ""),
        "cached": bool(payload.get("cached", False)),
        "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
        "params": {
            "segment_size": params.get("segment_size"),
            "window_size": params.get("window_size"),
            "window_step": params.get("window_step"),
            "top_k": params.get("top_k"),
            "threshold_jap": params.get("threshold_jap"),
            "threshold_ind": params.get("threshold_ind"),
        },
        "threshold_rule": payload.get("threshold_rule", ""),
    }


@app.post("/api/aigress/openai/OGR-Introgression/health")
@app.post("/OGR-Introgression/health")
async def dcs_health(req: Request):
    """健康检查（子路径，免鉴权）。"""
    from rice_introgression.predictor import _PREDICTOR

    inst = _PREDICTOR.get("instance")
    return _dcs_ok(
        _dcs_usage(0, 0),
        "health",
        {
            "model": "OGR-Introgression",
            "status": "ok",
            "predictor_initialized": inst is not None,
            "genomes": list_genomes(),
        },
    )


@app.post("/api/aigress/openai/OGR-Introgression/Prediction")
@app.post("/OGR-Introgression/Prediction")
@app.post("/api/aigress/openai/OGR-Introgression/predict")
@app.post("/OGR-Introgression/predict")
async def dcs_predict(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """渗入分析（DCS 风格子路径）。

    Body（camelCase，与 curl 调用者同方言；全部可选）：
      {"Genome": "YF47", "Chromosome": "Chr01", "Start": 100001, "End": 356001}
    fasta_path（可选）：轻量模式懒注册本地 FASTA 路径，替代 Genome 键。
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(_dcs_err("Request body is not valid JSON.", 400), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(_dcs_err("Request body is not valid JSON.", 400), status_code=400)
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)

    lazy_genome, lazy_cfg = "", None
    try:
        lazy_genome, lazy_cfg = _lazy_register_genome(body)
        if lazy_genome:
            body["Genome"] = lazy_genome

        genome = str(body.get("Genome", "") or "")
        if not genome:
            genomes = list_genomes()
            genome = genomes[0] if genomes else "YF47"
        chromosome = str(body.get("Chromosome", "") or "Chr01")
        # Start 必填（1-based inclusive）；永远只对一条染色体的某个区域推理，
        # 不再支持「整条染色体」模式（2026-11 用户确认）。
        start_1 = body.get("Start")
        if start_1 is None or str(start_1).strip() == "":
            raise RequestError("Missing required parameter 'Start' (1-based int).")
        try:
            start_0 = int(start_1) - 1
        except (TypeError, ValueError):
            raise RequestError(f"Invalid 'Start' value '{start_1}': must be an int (1-based).")
        if start_0 < 0:
            raise RequestError("Invalid 'Start': must be >= 1 (1-based)")
        end_1 = body.get("End")
        end_0 = int(end_1) if end_1 is not None else None

        resolve_genome_config(genome)  # 未知基因组尽早报 400
        payload = run_introgression(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=end_0,
        )
    except RequestError as e:
        return JSONResponse(_param_err(f"Introgression prediction failed: {e}", body), status_code=400)
    except FileNotFoundError as e:
        return JSONResponse(
            _dcs_err(f"Introgression prediction failed: {e}", 404,
                     detail={"request": _summarize_body(body)}),
            status_code=404,
        )
    except ValueError as e:
        return JSONResponse(_param_err(f"Introgression prediction failed: {e}", body), status_code=400)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            _dcs_err(f"Introgression prediction failed: {e}", 500,
                     detail={
                         "error_type": e.__class__.__name__,
                         "traceback": traceback.format_exc()[-2000:],
                         "request": _summarize_body(body),
                     }),
            status_code=500,
        )

    # 计费：prompt_tokens = 片段数 × segment_size；completion_tokens = 窗口数
    seg_size = int(payload.get("params", {}).get("segment_size", 8000))
    prompt_tokens = len(payload.get("segments", [])) * seg_size

    # 网页端与纯 API 共用同一返回值：注入全基因组展示骨架（12 车道绘图字段），
    # 并附带 elapsed_seconds / success —— 前端渲染完全来自 curl 返回值。
    t0 = time.time()
    payload = inject_genome_context(payload)
    payload["elapsed_seconds"] = round(time.time() - t0, 3)
    payload["success"] = True

    result = _predict_payload_to_result(payload, lazy_cfg)
    empty_msg = payload.get("message", "")
    return JSONResponse(_dcs_ok(
        _dcs_usage(prompt_tokens, len(result["windows"])),
        empty_msg or "Introgression analysis succeeded",
        result,
    ))


@app.post("/api/aigress/openai/OGR-Introgression/Genome")
@app.post("/OGR-Introgression/Genome")
async def dcs_genomes(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """列出可用基因组（DCS 风格子路径）。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), "genome list",
        {"model": "OGR-Introgression", "Genome": list_genomes()},
    ))


@app.post("/api/aigress/openai/OGR-Introgression/Chromosome")
@app.post("/OGR-Introgression/Chromosome")
async def dcs_chromosomes(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """列出某基因组的染色体（DCS 风格子路径）。"""
    try:
        body = await req.json()
    except Exception:
        body = {}
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return JSONResponse(_unauthorized(str(e)), status_code=401)
    genome = str((body or {}).get("Genome", "") or "") or (list_genomes() or ["YF47"])[0]
    try:
        cfg = resolve_genome_config(genome)
        chrs = get_genome_chromosomes(genome, cfg)
    except Exception as e:
        return JSONResponse(_dcs_err(f"Chromosome list failed: {e}", 400), status_code=400)
    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0), "chromosome list",
        {"model": "OGR-Introgression", "Genome": genome, "Chromosome": chrs},
    ))


@app.post("/api/aigress/openai/OGR-Introgression/RegisterGenome")
@app.post("/OGR-Introgression/RegisterGenome")
async def dcs_register_genome(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """按本地路径注册自定义基因组（DCS 风格子路径，不做文件传输）。

    Body: {"fasta_path": "/abs/path/to/xxx.fa", "genome_id": "optional_id"}
    文件已预先放在本机；后端校验并注册后，预测请求用 Genome 字段引用。
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
        if not fasta_path:
            raise RequestError(
                "Missing required parameter 'fasta_path' (local path to a .fa/.fasta/.fna file). "
                "Usage: POST .../OGR-Introgression/RegisterGenome with "
                '{"fasta_path": "/path/to/genome.fa", "genome_id": "optional_id"}'
            )
        genome_id = str(body.get("genome_id", "") or "").strip() or ""
        cfg = register_uploaded_genome(fasta_path, genome_id=genome_id, managed=False)
        gid = str(cfg.get("id") or genome_id)  # 注册表可能对冲突 id 加了后缀，cfg.id 为最终 id
        chroms = get_genome_chromosomes(gid, cfg)
    except RequestError as e:
        return JSONResponse(_dcs_err(f"RegisterGenome failed: {e}", 400,
                                     detail={"request": _summarize_body(body)}), status_code=400)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse(_dcs_err(f"RegisterGenome failed: {e}", 400,
                                     detail={"request": _summarize_body(body)}), status_code=400)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(_dcs_err(f"RegisterGenome failed: {e}", 500,
                                     detail={"request": _summarize_body(body)}), status_code=500)

    return JSONResponse(_dcs_ok(
        _dcs_usage(0, 0),
        f"Custom genome registered: {gid}",
        {
            "model": "OGR-Introgression",
            "genome": gid,
            "fasta_path": cfg["fasta"],
            "chromosomes": chroms,
            "chromosome_count": len(chroms),
            "note": "Files are referenced by path only (no upload). Use the returned 'genome' as the Genome field in predict requests.",
        },
    ))