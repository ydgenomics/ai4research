"""Rice-Introgression API 路由。

Endpoints:
- GET  /health                     — 健康检查 + 模型/基因组元信息
- GET  /genomes                    — 基因组列表
- GET  /genomes/{id}/chromosomes   — 某基因组的染色体列表
- GET  /genomes/{id}/chromosomes/{chrom}/length — 染色体长度
- POST /uploadFasta                — 上传自定义基因组 FASTA（自动建 .fai）
- POST /analyze                    — 渗入分析（单窗 / 整条染色体）
"""

import logging
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rice_introgression.cache_service import prediction_cache, warm_prediction_cache
from rice_introgression.genome_service import (
    get_chromosome_length,
    get_genome_chromosomes,
    list_genomes,
    register_uploaded_genome,
    resolve_genome_config,
    start_uploaded_genome_cleanup,
)
from rice_introgression.prediction_service import (
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

    # 重启后的预测缓存策略（.env 可配）：
    #   CLEAR_CACHE_ON_STARTUP=true  → 清空内存+磁盘预测缓存（重建）
    #   默认/其他                 → 预热磁盘缓存（跨进程重启保留已推理窗口）
    if _env_flag("CLEAR_CACHE_ON_STARTUP", False):
        n = len(prediction_cache)
        prediction_cache.clear()
        logger.info("CLEAR_CACHE_ON_STARTUP=true: cleared prediction cache (%d in-memory entries before)", n)
    else:
        warm_prediction_cache()

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
async def upload_fasta(file: UploadFile = File(...)):
    """上传自定义基因组 FASTA，自动建 .fai 并注册为新 genome。"""
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


def _find_uploaded_genome_id(fasta_path: str) -> str:
    from rice_introgression.genome_service import _UPLOADED_GENOMES

    for gid, cfg in _UPLOADED_GENOMES.items():
        if os.path.abspath(cfg["fasta"]) == os.path.abspath(fasta_path):
            return gid
    raise ValueError("Uploaded genome not found in registry")


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