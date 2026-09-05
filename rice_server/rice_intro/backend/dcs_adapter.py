"""DCS API 适配层 — rice_intro (DNA → 粳/籼血缘渗入分析)

对外提供 OpenAI 风格的 HTTP API，由 DCS 平台网关转发（独立进程，监听 PORT：
平台注入的 PORT 优先，回退 BACKEND_PORT，DCS 部署约定 5001）:

    POST /api/aigress/openai/rice_intro    单入口:body mode 分发(health/predict)
    GET  /api/aigress/openai/health        健康检查(免鉴权)

坐标约定:请求 1-based inclusive(与 rice_mut/rice_reg 一致)，内部转 0-based
half-open 后调用 ``rice_introgression.prediction_service.run_introgression``
(复用 trim 缓存、标准窗口网格、top-k 聚合、双阈值分组、区域融合与预测缓存)，
因此与网页版 5001 (/analyze) 结果逐位一致。

返回结构遵循 dcs.md / rice_mut API.md 规范:

    {
      "usage": {"prompt_tokens": N, "completion_tokens": M},
      "status": 200,
      "message": "...",
      "result": {...}
    }

计费口径(可用环境变量调整):
    prompt_tokens      = 推理总碱基数(片段数 × segment_size ≈ n_windows×window_size)
                         × DCS_PROMPT_TOKEN_MULTIPLIER      (默认 1)
    completion_tokens  = 返回窗口数 × DCS_COMPLETION_TOKEN_MULTIPLIER (默认 1)

鉴权(可选):在 .env 配置 DCS_API_KEY 后,POST 路由需要请求头
    Authorization: Bearer <DCS_API_KEY>  (或 X-API-Key: <DCS_API_KEY>)
    留空则不启用鉴权;GET /health 始终免鉴权。

请求体(渗入分析，mode=predict 或含 start 自动推断):

    {
      "model": "rice_intro",
      "genome": "YF47",          # 可选,默认第一个 GENOME_*_FASTA
      "chromosome": "Chr01",     # 可选,默认 "Chr01"
      "start": 100001,           # 可选,1-based inclusive
      "end": 356001              # 可选,1-based inclusive
    }

    - start/end 都空 → 整条染色体(受 .env MAX_NUMBER_256W 限制:空=全部窗口,N=前 N 窗)
    - 只填 start      → 1 个覆盖 start 的最大 256k 窗口(忽略 MAX_NUMBER_256W)
    - start/end 都填  → 与该区间覆盖度最大的 ≤MAX_NUMBER_256W 个 256k 窗口

用法:

    python backend/dcs_adapter.py
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse as FastAPIJSONResponse


def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


ROOT_DIR = Path(__file__).resolve().parents[1]    # rice_intro/
BACKEND_DIR = Path(__file__).resolve().parent      # rice_intro/backend/
_load_env_file(ROOT_DIR / ".env")

# backend/ 下含 rice_introgression 包,加入 sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rice_introgression.genome_service import (  # noqa: E402
    list_genomes,
    resolve_genome_config,
)
from rice_introgression.prediction_service import (  # noqa: E402
    run_introgression,
)
from rice_introgression.predictor import (  # noqa: E402
    _PREDICTOR,
    init_predictor,
)

logger = logging.getLogger("dcs_adapter.rice_intro")


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
# DCS 部署约定:平台注入 PORT=5001;本地回退 BACKEND_PORT(网页端同端口,不同机器)。
_LISTEN_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
_LISTEN_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "5001")))


def _count_prompt_tokens(payload: dict) -> int:
    """推理总碱基数 = 片段数 × segment_size(窗口重叠按实际推理量重复计入)。"""
    seg_size = int(payload.get("params", {}).get("segment_size", 8000))
    return len(payload.get("segments", [])) * seg_size


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
    keys = ("model", "genome", "chromosome", "start", "end")
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
        raise RequestError("Invalid or missing API Key")


def _unauthorized(message: str = "Invalid or missing API Key"):
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
            "transformers": _ver("transformers"),
            "fastapi": _ver("fastapi"),
            "pandas": _ver("pandas"),
        },
        "gpu": {"cuda_available": False, "device_count": 0, "device_name": ""},
        "files": {
            "BASE_MODEL_PATH": os.getenv("BASE_MODEL_PATH", ""),
            "CHECKPOINT_PATH": os.getenv("CHECKPOINT_PATH", ""),
            "GENOME_YF47_FASTA": os.getenv("GENOME_YF47_FASTA", ""),
            "base_model_exists": _exists(os.getenv("BASE_MODEL_PATH")),
            "checkpoint_exists": _exists(os.getenv("CHECKPOINT_PATH")),
            "genome_yf47_exists": _exists(os.getenv("GENOME_YF47_FASTA")),
        },
        "listen": {
            "BACKEND_HOST": os.getenv("BACKEND_HOST", "0.0.0.0"),
            "BACKEND_PORT": os.getenv("BACKEND_PORT", "5001"),
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
        print(f"[dcs_adapter.rice_intro] init_predictor FAILED: {_INIT_ERROR['error']}", flush=True)


class _NewlineJSONResponse(FastAPIJSONResponse):
    """默认 JSON 响应类:序列化后追加换行符(避免 curl 输出与 shell 提示符粘连)。"""

    def render(self, content) -> bytes:
        return super().render(content) + b"\n"


app = FastAPI(
    title="DCS Adapter (rice_intro)",
    version="0.1.0",
    default_response_class=_NewlineJSONResponse,
)
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
    """健康检查路由(免鉴权)。body 单入口的 health 模式复用同一响应构造。"""
    return await _health_response(request)


async def _health_response(request: Request) -> dict:
    """构造健康检查响应:含监听端口 / gateway 诊断。

    供两条路径复用:
      1) GET/POST /health、/api/aigress/openai/health(平台探活);
      2) 单入口 POST /rice_intro + body {"mode":"health"}(DCS 单地址转发)。
    """
    initialized = _PREDICTOR.get("instance") is not None
    served = request.scope.get("server") or [None, None]
    served_port = served[1]
    return {
        # status 恒为 ok = uvicorn 在运行(HTTP 服务存活,不破坏平台探活);
        # 模型是否就绪看 predictor_initialized,失败原因看 diagnostics.init_error
        "status": "ok",
        "predictor_initialized": initialized,
        "genomes": list_genomes(),
        "diagnostics": _collect_diagnostics(),
        "gateway": {
            "received_path": request.url.path,
            "served_host": served[0],
            "served_port": served_port,
            "host_header": request.headers.get("host", ""),
            "remote_addr": (
                f"{request.client.host}:{request.client.port}"
                if request.client else None
            ),
            "port_matches": served_port == _LISTEN_PORT if served_port else None,
            "path_matches": request.url.path in (
                "/api/aigress/openai/health", "/health",
            ),
        },
    }


HEALTH_MODES = ("", "health")
PREDICT_MODES = ("predict", "analyze", "intro")


def _mode_from_body(body: dict) -> str:
    """从请求体推断调用模式(单入口 /rice_intro 下按 body 分发)。

    DCS 平台只允许一个转发地址,无法使用 /health 等子路径,
    因此通过请求体字段区分调用模式:
      mode == "health"  → health(健康检查)
      mode == "predict" → predict(渗入分析)
      未指定 mode 时按字段自动推断:
        body 为空      → health
        其余(有 start) → predict
    """
    mode = str(body.get("mode", "") or "").strip().lower()
    if mode:
        if mode in HEALTH_MODES:
            return "health"
        if mode in PREDICT_MODES:
            return "predict"
        return mode  # 未知 mode,交由调用方报错
    if not body:
        return "health"
    return "predict"


@app.post("/api/aigress/openai/rice_intro")
@app.post("/rice_intro")
async def predict_intro(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """单入口统一分发:根据请求体 mode 区分 health / predict。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON", 400)

    mode = _mode_from_body(body)
    if mode == "health":
        return await _health_response(req)
    if mode not in PREDICT_MODES:
        return _err(f"Unknown mode '{mode}', must be health/predict", 400)

    return await _predict_inner(req, body, authorization, x_api_key)


async def _predict_inner(
    req: Request,
    body: dict,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """渗入分析(默认模式):粳/籼血缘渗入窗口聚合。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    if _INIT_ERROR:
        return _err(
            f"Predictor initialization failed, cannot inference: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    try:
        # 基因组默认取第一个已配置 GENOME_*_FASTA
        genome = str(body.get("genome", "") or "")
        if not genome:
            genomes = list_genomes()
            genome = genomes[0] if genomes else "YF47"
        chromosome = str(body.get("chromosome", "") or "Chr01")
        # 坐标:1-based inclusive → 0-based half-open(start_0 = start_1 - 1; end_0 = end_1)
        start_1 = body.get("start")
        end_1 = body.get("end")
        start_0 = int(start_1) - 1 if start_1 is not None else None
        end_0 = int(end_1) if end_1 is not None else None
        if start_0 is not None and start_0 < 0:
            raise RequestError("Invalid 'start': must be >= 1 (1-based)")

        resolve_genome_config(genome)  # 未知基因组尽早报 400
        payload = run_introgression(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=end_0,
        )
    except RequestError as e:
        return _err(
            f"Introgression prediction failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except FileNotFoundError as e:
        return _err(
            f"Introgression prediction failed: {e}", 404,
            detail={"request": _summarize_body(body)},
        )
    except ValueError as e:
        return _err(
            f"Introgression prediction failed: {e}", 400,
            detail={
                "error_type": e.__class__.__name__,
                "request": _summarize_body(body),
            },
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"Introgression prediction failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    params = payload.get("params", {})
    windows = []
    for w in payload.get("windows", []):
        win_start_0 = int(w["win_start"])
        win_end_0 = int(w["win_end"])
        windows.append({
            "win_start": win_start_0 + 1,          # 1-based inclusive
            "win_end": win_end_0,                  # 0-based half-open end == 1-based inclusive end
            "center": int(w.get("center", win_start_0 + 256000 // 2)) + 1,  # 1-based
            "n_segments": int(w["n_segments"]),
            "topk_mean_jap": float(w["topk_mean_jap"]),
            "topk_mean_ind": float(w["topk_mean_ind"]),
            "group": str(w["group"]),              # Ind | Jap | uncertain
        })

    # position_1based:实际匹配窗口覆盖区间(1-based inclusive)
    ws0 = payload.get("win_start")
    we0 = payload.get("win_end")
    if ws0 is None or we0 is None:
        pos = {"start": 1, "end": int(payload["chrom_len"])}
    else:
        pos = {"start": int(ws0) + 1, "end": int(we0)}

    prompt_tokens = _count_prompt_tokens(payload)
    completion_tokens = len(windows)
    return _ok(
        usage=_usage(prompt_tokens, completion_tokens),
        message="Introgression analysis succeeded",
        result={
            "model": "rice_intro",
            "genome": payload["genome"],
            "chromosome": payload["chromosome"],
            "chrom_len": int(payload["chrom_len"]),
            "mode": str(payload["mode"]),          # window | chromosome
            "position_1based": pos,
            "window_len": int(params.get("window_size", 256000)),
            "n_windows": len(windows),
            "windows": windows,
            "params": {
                "segment_size": params.get("segment_size"),
                "window_size": params.get("window_size"),
                "window_step": params.get("window_step"),
                "top_k": params.get("top_k"),
                "threshold_jap": params.get("threshold_jap"),
                "threshold_ind": params.get("threshold_ind"),
            },
            "threshold_rule": payload.get("threshold_rule", ""),
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
        "note": "request reached dcs_adapter (gateway forwarded correctly)",
    }


if __name__ == "__main__":
    # 容错初始化:失败不退出,错误写入 _INIT_ERROR 由 /health 返回
    init_predictor_safe()
    # 与 diagnostics/health 共用 _LISTEN_HOST/_LISTEN_PORT(见文件头部定义)
    print(f"[dcs_adapter.rice_intro] starting on {_LISTEN_HOST}:{_LISTEN_PORT} ...")
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)