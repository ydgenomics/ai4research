"""DCS API 适配层 — OGR-Base (DNA 序列 Embedding / 下游碱基预测)

对外提供 OpenAI 风格的 HTTP API,由 DCS 平台网关转发,全部通过**子路径**调用:

    POST /api/aigress/openai/OGR-Base/Embedding   提取序列 Embedding
    POST /api/aigress/openai/OGR-Base/NTP         下游碱基预测(NTP)
    GET  /api/aigress/openai/health               健康检查(免鉴权)
    GET  /api/aigress/openai/models               模型列表(免鉴权)

复用 `dna_embedding.EmbeddingExtractor`(模型加载、设备分配、embedding 提取与
碱基预测逻辑均不变),仅在最外层加一层 DCS 网关适配:OpenAI 风格请求体、
子路径路由、usage 计费字段、鉴权与 JSON 末尾换行。

返回结构遵循 rice_server/dcs.md 规范:

    {
      "usage": {"prompt_tokens": N, "completion_tokens": M},
      "token_cnt": N + M,
      "status": 200,
      "message": "...",
      "result": {...}
    }

计费口径(可用环境变量调整):
    prompt_tokens      = 输入 token 数 × DCS_PROMPT_TOKEN_MULTIPLIER      (默认 1)
    completion_tokens  = 输出元素总数 × DCS_COMPLETION_TOKEN_MULTIPLIER   (默认 1)
    水稻基模为单碱基编码:1 bp = 1 token,`token_count` 即输入碱基数。

鉴权(可选):在 OGR_Base/.env 配置 DCS_API_KEY 后,POST 路由需要请求头
    Authorization: Bearer <DCS_API_KEY>  (或 X-API-Key: <DCS_API_KEY>)
    留空则不启用鉴权;GET /health、GET /models 始终免鉴权。

模型固定为注册表中的 `1B_8k`(无需也不接受 `model_name` / `model` / `mode` 参数)。

请求体示例(Embedding):

    {
      "sequence": "ACGTTGCATGCAACGT",
      "pooling_method": "mean"          # mean | max | last | none
    }

请求体示例(NTP):

    {
      "sequence": "ACGTTGCATGCAACGT",
      "predict_length": 10
    }

用法:

    python dcs_adapter.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse as FastAPIJSONResponse


# ---------------------------------------------------------------------------
#  环境 / 路径准备:加载 .env 并把 rice_OGR/ 加入 sys.path(导入 dna_embedding)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent   # rice_OGR/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dna_embedding import (  # noqa: E402
    get_env_registry,
    get_or_create_extractor,
    logger,
)


# ---------------------------------------------------------------------------
#  异常与计费口径
# ---------------------------------------------------------------------------
class RequestError(ValueError):
    """请求参数错误(→ HTTP 400)。与推理执行错误(→ 500)区分开。"""


PROMPT_TOKEN_MULTIPLIER = float(os.getenv("DCS_PROMPT_TOKEN_MULTIPLIER", "1"))
COMPLETION_TOKEN_MULTIPLIER = float(os.getenv("DCS_COMPLETION_TOKEN_MULTIPLIER", "1"))

# API Key 鉴权(可选):留空 = 不启用;配置后 POST 路由需带
#   Authorization: Bearer <DCS_API_KEY> 或 X-API-Key: <DCS_API_KEY>
DCS_API_KEY = os.getenv("DCS_API_KEY", "").strip()

# 监听地址/端口:优先级 平台注入的 PORT > BACKEND_PORT > 8001
# (8001 与 rice_mut 的 dcs_adapter 一致;Sanic 原服务默认 8000)
_LISTEN_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
_LISTEN_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8001")))

# 服务名(DCS 网关转发的 path 段,请求体无需再带 model 字段)
SERVICE_NAME = "OGR-Base"

# 固定使用的模型名(.env 注册表 MODEL_<NAME>_* 之一;取消 model_name 参数,统一用 1B_8k)
DEFAULT_MODEL_NAME = "1B_8k"

# 参数错误时返回的英文使用说明(与 OGR_Prediction 适配层风格一致)
USAGE = (
    "Usage: POST /api/aigress/openai/OGR-Base/Embedding or .../OGR-Base/NTP. "
    "Body keys: sequence (required, DNA string), "
    "pooling_method (optional for Embedding, mean|max|last|none, default mean), "
    "predict_length (optional for NTP, positive int <= 1000, default 10). "
    "No mode/model/model_name keys: model is fixed to 1B_8k, "
    "mode is determined by the URL sub-path (Embedding | NTP)."
)


# ---------------------------------------------------------------------------
#  推理入口(复用 dna_embedding.EmbeddingExtractor)
# ---------------------------------------------------------------------------
def _get_extractor(model_name: str):
    """获取提取器实例(复用 get_or_create_extractor 的注册表/缓存逻辑)。"""
    return get_or_create_extractor(model_name)


async def _run_extract(extractor, body: dict, request_id: str | None = None):
    """执行 Embedding 子路径:提取序列 embedding。

    return: dict,结构为 {"usage": {...}, "result": {...}}
    """
    sequence = body.get("sequence")
    if not isinstance(sequence, str) or not sequence.strip():
        raise RequestError("Missing required parameter 'sequence' (DNA string). " + USAGE)

    pooling_method = str(body.get("pooling_method", "mean")).lower()
    if pooling_method not in ("mean", "max", "last", "none"):
        raise RequestError(
            f"Invalid 'pooling_method' '{body.get('pooling_method')}': "
            f"must be mean/max/last/none. " + USAGE
        )

    t0 = time.time()
    result = await extractor.extract_embedding(sequence, pooling_method)
    elapsed = time.time() - t0

    usage = _usage(
        prompt_tokens=result["token_count"],
        completion_tokens=0,
    )
    payload = {
        "model": SERVICE_NAME,
        "model_name": extractor.model_name,
        "sequence": result["sequence"],
        "sequence_length": result["sequence_length"],
        "token_count": result["token_count"],
        "embedding_shape": result["embedding_shape"],
        "embedding_dim": result["embedding_dim"],
        "pooling_method": result["pooling_method"],
        "model_type": result.get("model_type"),
        "device": result.get("device"),
        "embedding": result["embedding"],
        "elapsed_seconds": round(elapsed, 4),
    }
    return usage, payload


async def _run_predict(extractor, body: dict, request_id: str | None = None):
    """执行 NTP 子路径:自回归预测下游碱基。"""
    sequence = body.get("sequence")
    if not isinstance(sequence, str) or not sequence.strip():
        raise RequestError("Missing required parameter 'sequence' (DNA string). " + USAGE)

    try:
        predict_length = int(body.get("predict_length", 10))
    except (TypeError, ValueError):
        raise RequestError(
            f"Invalid 'predict_length' value '{body.get('predict_length')}': "
            f"must be a positive integer. " + USAGE
        )
    if predict_length <= 0:
        raise RequestError("'predict_length' must be a positive integer. " + USAGE)
    if predict_length > 1000:
        raise RequestError("'predict_length' must not exceed 1000. " + USAGE)

    t0 = time.time()
    result = await extractor.predict_next_bases(sequence, predict_length)
    elapsed = time.time() - t0

    usage = _usage(
        prompt_tokens=result["total_length"] - result["predict_length"],
        completion_tokens=result["predict_length"],
    )
    payload = {
        "model": SERVICE_NAME,
        "model_name": extractor.model_name,
        "original_sequence": result["original_sequence"],
        "predicted_sequence": result["predicted_sequence"],
        "predicted_bases": result["predicted_bases"],
        "predict_length": result["predict_length"],
        "total_length": result["total_length"],
        "elapsed_seconds": round(elapsed, 4),
    }
    return usage, payload


# ---------------------------------------------------------------------------
#  请求解析 / 响应构造
# ---------------------------------------------------------------------------
def _parse_model(body: dict) -> str:
    """解析实际模型注册名:固定使用注册表中的 `1B_8k`,不接受 model_name/model 参数。

    与 OGR_Prediction 一致:服务固定一个模型,取消 model_name 参数。
    """
    env = get_env_registry()
    configs = env.get("model_configs", {}) or {}
    if DEFAULT_MODEL_NAME in configs:
        return DEFAULT_MODEL_NAME
    if configs:
        # 注册表无 1B_8k 时回退到第一个已注册模型(部署配置异常时仍可服务)
        return next(iter(configs.keys()))
    raise RequestError("No model configured in registry (.env MODEL_<NAME>_PATH). " + USAGE)


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    pt = int(prompt_tokens * PROMPT_TOKEN_MULTIPLIER)
    ct = int(completion_tokens * COMPLETION_TOKEN_MULTIPLIER)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
    }


def _ok(usage: dict, message: str, result: dict):
    return {
        "model_name": "OGR",
        "usage": usage,
        "token_cnt": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        "status": 200,
        "message": message,
        "result": result,
    }


def _err(message: str, status: int = 400, detail: dict | None = None):
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


def _summarize_body(body: dict) -> dict:
    """请求体摘要:只保留定位/复现所需的字段,用于出错时的 detail。"""
    keys = (
        "sequence", "pooling_method", "predict_length",
    )
    out = {}
    for k in keys:
        if k in body:
            v = body[k]
            if k == "sequence" and isinstance(v, str):
                v = v[:64] + ("..." if len(v) > 64 else "")
            out[k] = v
    return out


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
        "model_name": "OGR",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "token_cnt": 0,
        "status": 401,
        "message": message,
        "result": None,
    }


# ---------------------------------------------------------------------------
#  环境诊断与容错初始化(排查部署问题用):
#   1) 模型加载失败不再退出进程 —— 服务照常监听端口,网关可连通;
#   2) 失败原因存入 _INIT_ERROR,由 /health 与推理接口返回,无需 SSH 即可查看。
# ---------------------------------------------------------------------------
_INIT_ERROR: dict | None = None      # 预加载模型失败时保存的错误信息
_INIT_MODELS: list = []              # 启动时预加载的模型名列表


def _collect_diagnostics() -> dict:
    """收集部署机环境诊断:python/依赖/GPU/模型文件/关键配置。"""
    import importlib.util

    def _ver(mod: str):
        try:
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "installed")
        except Exception as e:
            return f"MISSING ({e.__class__.__name__})"

    def _model_exists(name: str):
        env = get_env_registry()
        cfg = (env.get("model_configs", {}) or {}).get(name, {})
        path = cfg.get("path", "")
        return bool(path and Path(path).exists())

    env = get_env_registry()
    configs = env.get("model_configs", {}) or {}
    cfg = env.get("config", {}) or {}

    diag: dict = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "deps": {
            "torch": _ver("torch"),
            "flash_attn": _ver("flash_attn"),
            "fastapi": _ver("fastapi"),
            "transformers": _ver("transformers"),
        },
        "gpu": {"cuda_available": False, "device_count": 0, "device_name": ""},
        "models": {
            "registered": list(configs.keys()),
            "preloaded": _INIT_MODELS,
            "loaded": list(get_or_create_extractor.__globals__.get("extractors", {}).keys()),
            "path_exists": {name: _model_exists(name) for name in configs},
        },
        "listen": {
            "BACKEND_HOST": os.getenv("BACKEND_HOST", "0.0.0.0"),
            "BACKEND_PORT": os.getenv("BACKEND_PORT", "8000"),
            "PORT": os.getenv("PORT", ""),
            "actual_host": _LISTEN_HOST,
            "actual_port": _LISTEN_PORT,
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


def init_extractors_safe():
    """按 .env 注册表预加载模型;失败不崩溃,错误保存到 _INIT_ERROR。"""
    global _INIT_ERROR, _INIT_MODELS
    env = get_env_registry()
    configs = env.get("model_configs", {}) or {}
    if not configs:
        logger.info("[dcs_adapter] no MODEL_*_PATH configured, skip preloading")
        return
    for name in configs:
        try:
            logger.info("[dcs_adapter] preloading model %s ...", name)
            get_or_create_extractor(name)
            _INIT_MODELS.append(name)
        except Exception as e:
            logger.error("[dcs_adapter] preload model %s failed: %s", name, e)
            _INIT_ERROR = {
                "model": name,
                "error": f"{e.__class__.__name__}: {e}",
                "traceback": traceback.format_exc()[-4000:],
            }


class _NewlineJSONResponse(FastAPIJSONResponse):
    """默认 JSON 响应类:序列化后追加换行符。

    FastAPI 默认返回的 JSON 末尾无换行,curl 输出会和 shell 提示符
    粘连在同一行。覆盖 render() 统一追加 \n,对所有路由生效。
    """

    def render(self, content) -> bytes:
        return super().render(content) + b"\n"


app = FastAPI(
    title="DCS Adapter (OGR-Base)",
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
    """健康检查路由(免鉴权)。"""
    served = request.scope.get("server") or [None, None]
    initialized = bool(_INIT_MODELS or (get_or_create_extractor.__globals__.get("extractors", {})))
    return {
        "status": "ok",
        "extractor_initialized": initialized,
        "models": {
            "registered": (get_env_registry().get("model_configs", {}) or {}).keys()
            | set(_INIT_MODELS),
            "loaded": list(get_or_create_extractor.__globals__.get("extractors", {}).keys()),
        },
        "diagnostics": _collect_diagnostics(),
        "gateway": {
            "received_path": request.url.path,
            "served_host": served[0],
            "served_port": served[1],
            "host_header": request.headers.get("host", ""),
            "remote_addr": (
                f"{request.client.host}:{request.client.port}"
                if request.client else None
            ),
        },
    }


@app.api_route("/api/aigress/openai/models", methods=["GET"])
@app.api_route("/models", methods=["GET"])
async def models():
    """模型列表(免鉴权)。"""
    env = get_env_registry()
    configs = env.get("model_configs", {}) or {}
    loaded = list(get_or_create_extractor.__globals__.get("extractors", {}).keys())
    return {
        "status": "ok",
        "models": list(dict.fromkeys(list(configs.keys()) + loaded)),
        "loaded": loaded,
    }


MODES = ("Embedding", "NTP")


def _mode_from_body(body: dict) -> str:
    """取消 mode 参数:调用模式完全由 URL 子路径决定。

    保留此函数仅为路由显式传入 mode;请求体中的 mode 字段将被忽略。
    """
    return body.get("_mode", "")


async def _dispatch(req: Request, body: dict, mode: str, authorization, x_api_key):
    """子路径统一分发:mode 由路由显式指定(Embedding / NTP)。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    if _INIT_ERROR:
        return _err(
            f"Model preload failed, may be unable to inference: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    if mode not in MODES:
        return _err(f"Unknown mode '{mode}', must be one of {MODES}. " + USAGE, 400)

    try:
        model_name = _parse_model(body)
        extractor = _get_extractor(model_name)
    except RequestError as e:
        return _err(f"Parameter error: {e}", 400, detail={"request": _summarize_body(body)})
    except Exception as e:
        return _err(
            f"Model loading failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    try:
        if mode == "NTP":
            usage, payload = await _run_predict(extractor, body)
            message = "Downstream base (NTP) prediction succeeded"
        else:
            usage, payload = await _run_extract(extractor, body)
            message = "DNA sequence embedding extraction succeeded"
        return _ok(usage=usage, message=message, result=payload)
    except RequestError as e:
        return _err(f"{mode} failed: {e}", 400, detail={"request": _summarize_body(body)})
    except Exception as e:
        logger.exception("[dcs_adapter] %s inference failed", mode)
        return _err(
            f"{mode} failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )


@app.post("/api/aigress/openai/OGR-Base/Embedding")
@app.post("/OGR-Base/Embedding")
async def ogr_base_embedding(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """DNA 序列 Embedding 提取(子路径,无 mode 分发)。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON. " + USAGE, 400)
    if not isinstance(body, dict):
        return _err("Request body must be a JSON object. " + USAGE, 400)
    return await _dispatch(req, body, "Embedding", authorization, x_api_key)


@app.post("/api/aigress/openai/OGR-Base/NTP")
@app.post("/OGR-Base/NTP")
async def ogr_base_ntp(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """下游碱基预测 NTP(子路径,无 mode 分发)。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON. " + USAGE, 400)
    if not isinstance(body, dict):
        return _err("Request body must be a JSON object. " + USAGE, 400)
    return await _dispatch(req, body, "NTP", authorization, x_api_key)


# ---------------------------------------------------------------------------
#  调试回显路由(排查用):捕获一切未匹配路径,回显网关实际转发的路径与方法。
# ---------------------------------------------------------------------------
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def _debug_echo_path(request: Request, full_path: str):
    return {
        "debug_received_path": f"/{full_path}",
        "method": request.method,
        "note": "request reached dcs_adapter (gateway forwarded correctly)",
    }


if __name__ == "__main__":
    # 容错预加载:失败不退出,错误写入 _INIT_ERROR 由 /health 返回
    init_extractors_safe()
    print(f"[dcs_adapter] starting on {_LISTEN_HOST}:{_LISTEN_PORT} ...", flush=True)
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)