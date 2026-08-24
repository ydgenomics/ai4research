"""DCS API 适配层 — rice_mut (DNA → 多组学表达预测)

对外提供 OpenAI 风格的 HTTP API,由 DCS 平台网关转发:

    POST /api/aigress/openai/rice_mut        参考序列表达预测
    POST /api/aigress/openai/rice_mut/snv    单碱基变异 (SNV) 双轨对比预测
    GET  /api/aigress/openai/health          健康检查

坐标约定:请求使用 1-based(与网页版一致),内部转 0-based 后调用现有
``rice_mutation.prediction_service`` 的 ``run_prediction_core`` /
``run_snv_core``(复用染色体别名归一化、窗口中心对齐与缓存)。

返回结构遵循 dcs.md 规范:

    {
      "usage": {"prompt_tokens": N, "completion_tokens": M},
      "status": 200,
      "message": "...",
      "result": {...}
    }

计费口径(可用环境变量调整):
    prompt_tokens      = 输入窗口碱基数 × DCS_PROMPT_TOKEN_MULTIPLIER      (默认 1)
    completion_tokens  = 输出数组元素总数 × DCS_COMPLETION_TOKEN_MULTIPLIER (默认 1)

鉴权(可选):在 rice_mut/.env 配置 DCS_API_KEY 后,POST 路由需要请求头
    Authorization: Bearer <DCS_API_KEY>  (或 X-API-Key: <DCS_API_KEY>)
    留空则不启用鉴权;GET /health 始终免鉴权。

请求体示例(参考预测):

    {
      "model": "rice_mut",
      "genome": "osa1_r7",
      "chromosome": "chr01",
      "start": 20716774,          # 1-based inclusive
      "end": 20749541,            # 1-based inclusive(可省略,默认 32768 窗口)
      "biosample_names": null,    # 可选,缺省全部
      "output_format": "full",    # full | mean | downsample
      "max_points": 1024          # downsample 时的目标点数(默认 1024)
    }

请求体示例(SNV 预测,额外两个字段):

    {
      "model": "rice_mut",
      "genome": "osa1_r7",
      "chromosome": "chr01",
      "start": 20716774,
      "end": 20749541,
      "snv_index": 20731844,      # 1-based
      "snv_base": "T"             # A/C/G/T/N
    }

用法:

    python backend/dcs_adapter.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware


def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


ROOT_DIR = Path(__file__).resolve().parents[1]   # rice_mut/
BACKEND_DIR = Path(__file__).resolve().parent     # rice_mut/backend/
_load_env_file(ROOT_DIR / ".env")

# backend/ 下含 src/ 包 (from src.util import ...),加入 sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rice_mutation.prediction_service import (  # noqa: E402
    _adjust_window,
    init_predictor,
    list_genomes,
    require_predictor,
    resolve_genome_config,
    run_prediction_core,
    run_snv_core,
)

# ---------------------------------------------------------------------------
#  异常与计费口径
# ---------------------------------------------------------------------------
class RequestError(ValueError):
    """请求参数错误(→ HTTP 400)。与预测执行错误(→ 500)区分开。"""


PROMPT_TOKEN_MULTIPLIER = float(os.getenv("DCS_PROMPT_TOKEN_MULTIPLIER", "1"))
COMPLETION_TOKEN_MULTIPLIER = float(os.getenv("DCS_COMPLETION_TOKEN_MULTIPLIER", "1"))

# API Key 鉴权(可选):留空 = 不启用;配置后 POST 路由需带
#   Authorization: Bearer <DCS_API_KEY> 或 X-API-Key: <DCS_API_KEY>
DCS_API_KEY = os.getenv("DCS_API_KEY", "").strip()

# 监听地址/端口:__main__ 与 diagnostics / health 共用同一份,避免两份代码漂移。
# 端口优先级:平台注入的 PORT > BACKEND_PORT(本地/网页版)。
_LISTEN_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
_LISTEN_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8001")))


def _count_elements(values: dict) -> int:
    """统计输出数组元素总数(用于 completion_tokens)。"""
    total = 0
    for bios_map in values.values():
        for arr in bios_map.values():
            total += int(np.asarray(arr).size)
    return total


def _format_values(values: dict, fmt: str, max_points: int) -> dict:
    """按 output_format 把 {assay: {biosample: np.ndarray}} 转为 JSON 可序列化结构。

    full       — 完整数组(round 到 6 位)
    mean       — 每轨道窗口均值(标量)
    downsample — 每轨道均匀降采样到 max_points 点
    """
    out: dict = {}
    for assay, bios_map in values.items():
        out[assay] = {}
        for bios, arr in bios_map.items():
            arr = np.asarray(arr, dtype=np.float64)
            if fmt == "mean":
                out[assay][bios] = round(float(arr.mean()), 6)
            elif fmt == "downsample":
                n = len(arr)
                if n > max_points:
                    idx = np.linspace(0, n - 1, max_points).astype(int)
                    arr = arr[idx]
                out[assay][bios] = [round(float(x), 6) for x in arr]
            else:  # full
                out[assay][bios] = [round(float(x), 6) for x in arr]
    return out


def _parse_common(body: dict):
    """解析公共参数:genome / chromosome / start / end / biosample_names。

    返回 (genome, chromosome, start_0, end_0, biosample_names, output_format,
    max_points)。坐标为 0-based half-open(内部约定)。
    """
    genome = str(body.get("genome", "") or "")
    if not genome:
        genomes = list_genomes()
        genome = genomes[0] if genomes else "osa1_r7"
    chromosome = str(body.get("chromosome", "") or "chr01")
    if "start" not in body:
        raise RequestError("缺少必填参数 'start'(1-based)")
    start_1 = int(body["start"])
    end_1 = body.get("end")
    end_0 = int(end_1) if end_1 is not None else None
    biosample_names = body.get("biosample_names")
    if isinstance(biosample_names, str) and biosample_names.strip():
        biosample_names = [s.strip() for s in biosample_names.split(",") if s.strip()]
    if not isinstance(biosample_names, list):
        biosample_names = None
    output_format = str(body.get("output_format", "full")).lower()
    if output_format not in ("full", "mean", "downsample"):
        raise RequestError(
            f"output_format 必须是 full/mean/downsample,收到 '{output_format}'"
        )
    max_points = int(body.get("max_points", 1024))
    return genome, chromosome, start_1, end_0, biosample_names, output_format, max_points


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": int(prompt_tokens * PROMPT_TOKEN_MULTIPLIER),
        "completion_tokens": int(completion_tokens * COMPLETION_TOKEN_MULTIPLIER),
    }


def _ok(usage: dict, message: str, result: dict):
    return {
        "usage": usage,
        "status": 200,
        "message": message,
        "result": result,
    }


def _err(message: str, status: int = 400, detail: dict | None = None):
    resp = {
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "status": status,
        "message": message,
        "result": None,
    }
    if detail:
        resp["detail"] = detail
    return resp


def _summarize_body(body: dict) -> dict:
    """请求体摘要:只保留定位/复现所需的字段,用于出错时的 detail。"""
    keys = (
        "model", "genome", "chromosome", "start", "end",
        "snv_index", "snv_base", "output_format", "max_points",
    )
    return {k: body[k] for k in keys if k in body}


def _check_api_key(authorization: str | None = None, x_api_key: str | None = None) -> None:
    """校验 API Key;未配置 DCS_API_KEY 时直接放行(向后兼容)。"""
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
        raise RequestError("无效或缺失的 API Key")


def _unauthorized(message: str = "无效或缺失的 API Key"):
    return {
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "status": 401,
        "message": message,
        "result": None,
    }


# ---------------------------------------------------------------------------
#  环境诊断与容错初始化(排查部署问题用):
#   1) init_predictor() 失败不再退出进程 —— 服务照常监听端口,网关可连通;
#   2) 失败原因存入 _INIT_ERROR,由 /health 与推理接口返回,无需 SSH 即可查看。
# ---------------------------------------------------------------------------
_INIT_ERROR: dict | None = None      # init_predictor() 失败时保存的错误信息


def _collect_diagnostics() -> dict:
    """收集部署机环境诊断:python/依赖/GPU/模型文件/关键配置。"""
    import importlib.util

    def _ver(mod: str):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "installed")
        except Exception as e:
            return f"MISSING ({e.__class__.__name__})"

    def _exists(p: str | None):
        return bool(p and Path(p).exists())

    diag: dict = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "env_python_bin": os.getenv("BACKEND_PYTHON_BIN", ""),
        },
        "deps": {
            "torch": _ver("torch"),
            "flash_attn": _ver("flash_attn"),
            "fastapi": _ver("fastapi"),
            "transformers": _ver("transformers"),
        },
        "gpu": {"cuda_available": False, "device_count": 0, "device_name": ""},
        "files": {
            "BASE_MODEL_PATH": os.getenv("BASE_MODEL_PATH", ""),
            "CHECKPOINT_PATH": os.getenv("CHECKPOINT_PATH", ""),
            "INDEX_STAT_PATH": os.getenv("INDEX_STAT_PATH", ""),
            "GENOME_FASTA": os.getenv("GENOME_osa1_r7_FASTA", ""),
            "GENOME_FAI": os.getenv("GENOME_osa1_r7_FAI", ""),
            "base_model_exists": _exists(os.getenv("BASE_MODEL_PATH")),
            "checkpoint_exists": _exists(os.getenv("CHECKPOINT_PATH")),
            "index_stat_exists": _exists(os.getenv("INDEX_STAT_PATH")),
            "genome_fasta_exists": _exists(os.getenv("GENOME_osa1_r7_FASTA")),
            "genome_fai_exists": _exists(os.getenv("GENOME_osa1_r7_FAI")),
        },
        "listen": {
            "BACKEND_HOST": os.getenv("BACKEND_HOST", "0.0.0.0"),
            "BACKEND_PORT": os.getenv("BACKEND_PORT", "8001"),
            "PORT": os.getenv("PORT", ""),
            "actual_host": _LISTEN_HOST,   # 实际监听地址
            "actual_port": _LISTEN_PORT,   # 实际监听端口(与 uvicorn.run 同一变量)
        },
        "init_error": _INIT_ERROR,
    }
    try:
        import torch
        diag["gpu"]["cuda_available"] = bool(torch.cuda.is_available())
        diag["gpu"]["device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            diag["gpu"]["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        diag["gpu"]["torch_error"] = f"{e.__class__.__name__}: {e}"
    return diag


def init_predictor_safe():
    """包装 init_predictor():失败不崩溃,错误保存到 _INIT_ERROR,服务照常监听。"""
    global _INIT_ERROR
    try:
        init_predictor()
        _INIT_ERROR = None
    except Exception as e:
        _INIT_ERROR = {
            "error": f"{e.__class__.__name__}: {e}",
            "traceback": traceback.format_exc()[-4000:],
        }
        print(f"[dcs_adapter] init_predictor FAILED: {_INIT_ERROR['error']}", flush=True)


app = FastAPI(title="DCS Adapter (rice_mut)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------
@app.api_route("/api/aigress/openai/health", methods=["GET", "POST"])
@app.api_route("/health", methods=["GET", "POST"])
async def health(request: Request):
    from rice_mutation.prediction_service import _PREDICTOR

    initialized = _PREDICTOR.get("instance") is not None
    # scope["server"] = 请求实际到达的 socket 地址(uvicorn 注入),即平台转发目标端口
    served = request.scope.get("server") or [None, None]
    served_port = served[1]
    return {
        # status 恒为 ok = uvicorn 在运行(HTTP 服务存活,不破坏平台探活);
        # 模型是否就绪看 predictor_initialized,失败原因看 diagnostics.init_error
        "status": "ok",
        "predictor_initialized": initialized,
        "genomes": list_genomes() if initialized else [],
        "diagnostics": _collect_diagnostics(),
        "gateway": {
            # 平台实际转发进来的路径(排查 404 用)
            "received_path": request.url.path,
            # 容器内实际接收请求的地址 / 端口 = 平台转发目标端口
            "served_host": served[0],
            "served_port": served_port,
            # 外部入口 host[:port](与 served_port 不一致属正常,平台网关会改写)
            "host_header": request.headers.get("host", ""),
            "remote_addr": (
                f"{request.client.host}:{request.client.port}"
                if request.client else None
            ),
            # 便捷布尔:端口/路径是否与预期一致
            "port_matches": served_port == _LISTEN_PORT if served_port else None,
            "path_matches": request.url.path in (
                "/api/aigress/openai/health", "/health",
            ),
        },
    }


@app.post("/api/aigress/openai/rice_mut")
@app.post("/rice_mut")
async def predict_ref(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """参考序列表达预测。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    try:
        body = await req.json()
    except Exception:
        return _err("请求体不是合法 JSON", 400)

    if _INIT_ERROR:
        return _err(
            f"预测器初始化失败,无法推理: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    try:
        genome, chromosome, start_1, end_0, biosample_names, fmt, max_points = (
            _parse_common(body)
        )
        # 1-based inclusive [start_1, end_1] → 0-based half-open [start_1-1, end_1)
        start_0 = max(0, start_1 - 1)

        genome_config = resolve_genome_config(genome)
        result = run_prediction_core(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=end_0,
            biosample_names=biosample_names,
            genome_config=genome_config,
        )
    except RequestError as e:
        return _err(
            f"参考预测失败: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"参考预测失败: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    pos_chrom, pos_start, pos_end = result["position"]
    prompt_tokens = pos_end - pos_start
    completion_tokens = _count_elements(result["values"])
    return _ok(
        usage=_usage(prompt_tokens, completion_tokens),
        message="参考序列表达预测成功",
        result={
            "model": "rice_mut",
            "genome": result["genome"],
            "chromosome": pos_chrom,
            "position_1based": {"start": pos_start + 1, "end": pos_end},
            "window_len": pos_end - pos_start,
            "output_format": fmt,
            "values": _format_values(result["values"], fmt, max_points),
        },
    )


@app.post("/api/aigress/openai/rice_mut/snv")
@app.post("/rice_mut/snv")
async def predict_snv(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """单碱基变异预测:result1 = 参考, result2 = 突变。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    try:
        body = await req.json()
    except Exception:
        return _err("请求体不是合法 JSON", 400)

    if _INIT_ERROR:
        return _err(
            f"预测器初始化失败,无法推理: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    try:
        genome, chromosome, start_1, end_0, biosample_names, fmt, max_points = (
            _parse_common(body)
        )
        if "snv_index" not in body:
            raise RequestError("缺少必填参数 'snv_index'(1-based)")
        snv_1 = int(body["snv_index"])
        snv_base = str(body.get("snv_base", "") or "").strip().upper()
        if snv_base not in ("A", "C", "G", "T", "N"):
            raise RequestError(f"snv_base 必须是 A/C/G/T/N,收到 '{snv_base}'")

        start_0 = max(0, start_1 - 1)
        snv_0 = max(0, snv_1 - 1)

        genome_config = resolve_genome_config(genome)
        result = run_snv_core(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=end_0,
            snv_index=snv_0,
            snv_base=snv_base,
            biosample_names=biosample_names,
            genome_config=genome_config,
        )
    except RequestError as e:
        return _err(
            f"SNV 预测失败: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"SNV 预测失败: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    pos_chrom, pos_start, pos_end = result["position"]
    ref_values = result["ref_values"]
    mut_values = result["mut_values"]
    prompt_tokens = pos_end - pos_start
    completion_tokens = _count_elements(ref_values) + _count_elements(mut_values)
    return _ok(
        usage=_usage(prompt_tokens, completion_tokens),
        message=f"SNV 预测成功 (ref {result['ref_base']} → {result['snv_base']})",
        result={
            "model": "rice_mut",
            "genome": result["genome"],
            "chromosome": pos_chrom,
            "position_1based": {"start": pos_start + 1, "end": pos_end},
            "window_len": pos_end - pos_start,
            "snv_index_1based": result["snv_index"] + 1,
            "ref_base": result["ref_base"],
            "snv_base": result["snv_base"],
            "output_format": fmt,
            "ref_values": _format_values(ref_values, fmt, max_points),
            "mut_values": _format_values(mut_values, fmt, max_points),
        },
    )


# ---------------------------------------------------------------------------
#  调试回显路由(排查用):捕获一切未匹配路径,回显网关实际转发的路径与方法。
#  确认网关转发形态后即可删除。
# ---------------------------------------------------------------------------
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def _debug_echo_path(request: Request, full_path: str):
    return {
        "debug_received_path": f"/{full_path}",
        "method": request.method,
        "note": "请求已到达 dcs_adapter(网关确实把请求转了过来)",
    }


if __name__ == "__main__":
    # 容错初始化:失败不退出,错误写入 _INIT_ERROR 由 /health 返回
    init_predictor_safe()
    # 与 diagnostics/health 共用 _LISTEN_HOST/_LISTEN_PORT(见文件头部定义)
    print(f"[dcs_adapter] starting on {_LISTEN_HOST}:{_LISTEN_PORT} ...")
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)
