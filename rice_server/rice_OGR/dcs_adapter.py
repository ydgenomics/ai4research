"""DCS API 适配层 — rice_OGR (DNA 序列 Embedding / 下游碱基预测)

对外提供 OpenAI 风格的 HTTP API,由 DCS 平台网关转发:

    POST /api/aigress/openai/rice_ogr    单入口(按 body.mode 分发)
    GET  /api/aigress/openai/health      健康检查
    GET  /api/aigress/openai/models      模型列表

复用 `dna_embedding.EmbeddingExtractor`(模型加载、设备分配、embedding 提取与
碱基预测逻辑均不变),仅在最外层加一层 DCS 网关适配:OpenAI 风格请求体、
mode 分发、usage 计费字段、鉴权与 JSON 末尾换行。

返回结构遵循 rice_server/dcs.md 规范:

    {
      "usage": {"prompt_tokens": N, "completion_tokens": M},
      "status": 200,
      "message": "...",
      "result": {...}
    }

计费口径(可用环境变量调整):
    prompt_tokens      = 输入 token 数 × DCS_PROMPT_TOKEN_MULTIPLIER      (默认 1)
    completion_tokens  = 输出元素总数 × DCS_COMPLETION_TOKEN_MULTIPLIER   (默认 1)
    水稻基模为单碱基编码:1 bp = 1 token,`token_count` 即输入碱基数。

鉴权(可选):在 rice_OGR/.env 配置 DCS_API_KEY 后,POST 路由需要请求头
    Authorization: Bearer <DCS_API_KEY>  (或 X-API-Key: <DCS_API_KEY>)
    留空则不启用鉴权;GET /health、GET /models 始终免鉴权。

请求体示例(dna_embedding 模式,默认):

    {
      "model": "rice_ogr",              # 服务名(DCS 网关转发 path 末段)
      "model_name": "1B_8k",            # 实际模型注册名(.env MODEL_<NAME>_*, 如 1B_8k / 1B_32k)
      "mode": "dna_embedding",
      "sequence": "ACGTTGCATGCAACGT",
      "pooling_method": "mean"          # mean | max | last | none
    }

请求体示例(predict 模式):

    {
      "model": "rice_ogr",
      "model_name": "1B_8k",
      "mode": "predict",
      "sequence": "ACGTTGCATGCAACGT",
      "predict_length": 10
    }

`model` 为服务名;实际模型名放在 `model_name`(向后兼容:`model` 不等于 `rice_ogr` 时仍视为模型名)。

`mode` 未指定时按字段自动推断:带 `predict_length` → predict,否则 → dna_embedding。

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

# 服务名(DCS 网关转发的 path 末段,也是请求体 model 字段的推荐值)
SERVICE_NAME = "rice_ogr"


# ---------------------------------------------------------------------------
#  推理入口(复用 dna_embedding.EmbeddingExtractor)
# ---------------------------------------------------------------------------
def _get_extractor(model_name: str):
    """获取提取器实例(复用 get_or_create_extractor 的注册表/缓存逻辑)。"""
    return get_or_create_extractor(model_name)


async def _run_extract(extractor, body: dict, request_id: str | None = None):
    """执行 dna_embedding 模式:提取序列 embedding。

    return: dict,结构为 {"usage": {...}, "result": {...}}
    """
    sequence = body.get("sequence")
    if not isinstance(sequence, str) or not sequence.strip():
        raise RequestError("sequence 必须是非空字符串")

    pooling_method = str(body.get("pooling_method", "mean")).lower()
    if pooling_method not in ("mean", "max", "last", "none"):
        raise RequestError(
            f"pooling_method 必须是 mean/max/last/none,收到 '{pooling_method}'"
        )

    t0 = time.time()
    result = await extractor.extract_embedding(sequence, pooling_method)
    elapsed = time.time() - t0

    usage = _usage(
        prompt_tokens=result["token_count"],
        completion_tokens=0,
    )
    payload = {
        "model": body.get("model", SERVICE_NAME),
        "model_name": extractor.model_name,
        "mode": "dna_embedding",
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
    """执行 predict 模式:自回归预测下游碱基。"""
    sequence = body.get("sequence")
    if not isinstance(sequence, str) or not sequence.strip():
        raise RequestError("sequence 必须是非空字符串")

    try:
        predict_length = int(body.get("predict_length", 10))
    except (TypeError, ValueError):
        raise RequestError("predict_length 必须是正整数")
    if predict_length <= 0:
        raise RequestError("predict_length 必须是正整数")
    if predict_length > 1000:
        raise RequestError("predict_length 不能超过 1000")

    t0 = time.time()
    result = await extractor.predict_next_bases(sequence, predict_length)
    elapsed = time.time() - t0

    usage = _usage(
        prompt_tokens=result["total_length"] - result["predict_length"],
        completion_tokens=result["predict_length"],
    )
    payload = {
        "model": body.get("model", SERVICE_NAME),
        "model_name": extractor.model_name,
        "mode": "predict",
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
    """解析实际模型注册名。

    新规范:请求体 `model` 为服务名(如 rice_ogr),实际模型名放在 `model_name`
    (如 1B_8k / 1B_32k)。为向后兼容,`model` 不等于服务名时仍视为模型名。
    都未指定时取注册表第一个(key 按 .env 扫描顺序)。
    """
    model_name = str(body.get("model_name", "") or "").strip()
    if model_name:
        return model_name
    model = str(body.get("model", "") or "").strip()
    if model and model != SERVICE_NAME:
        return model
    env = get_env_registry()
    configs = env.get("model_configs", {}) or {}
    if not configs:
        raise RequestError("model_name 必填(未配置任何注册表模型)")
    return next(iter(configs.keys()))


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
        "model", "model_name", "mode", "sequence",
        "pooling_method", "predict_length",
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
        logger.info("[dcs_adapter] .env 未配置任何 MODEL_*_PATH,跳过预加载")
        return
    for name in configs:
        try:
            logger.info("[dcs_adapter] 预加载模型 %s ...", name)
            get_or_create_extractor(name)
            _INIT_MODELS.append(name)
        except Exception as e:
            logger.error("[dcs_adapter] 预加载模型 %s 失败: %s", name, e)
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
    title="DCS Adapter (rice_OGR)",
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


MODES = ("dna_embedding", "predict")


def _mode_from_body(body: dict) -> str:
    """从请求体推断调用模式(单入口 /rice_ogr 下按 body 分发)。

    DCS 平台只允许一个转发地址,因此通过请求体字段区分调用模式:
      mode == "dna_embedding" → embedding 提取
      mode == "predict"       → 下游碱基预测
      未指定 mode 时按字段自动推断:
        body 为空           → dna_embedding(报缺 sequence)
        带 predict_length  → predict(现有 predict 调用零改动)
        其余              → dna_embedding(向后兼容,默认行为)
    """
    mode = str(body.get("mode", "") or "").strip().lower()
    if mode:
        if mode in MODES:
            return mode
        return mode  # 未知 mode,交由调用方报错
    # 未指定 mode → 自动推断(向后兼容)
    if "predict_length" in body:
        return "predict"
    return "dna_embedding"


async def _dispatch(req: Request, body: dict, authorization, x_api_key):
    """单入口统一分发:根据请求体 mode 区分 dna_embedding / predict。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    if _INIT_ERROR:
        return _err(
            f"模型预加载失败,可能无法推理: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    mode = _mode_from_body(body)
    if mode not in MODES:
        return _err(f"未知 mode '{mode}',必须是 {MODES}", 400)

    try:
        model_name = _parse_model(body)
        extractor = _get_extractor(model_name)
    except RequestError as e:
        return _err(f"参数错误: {e}", 400, detail={"request": _summarize_body(body)})
    except Exception as e:
        return _err(
            f"模型加载失败: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    try:
        if mode == "predict":
            usage, payload = await _run_predict(extractor, body)
            message = "下游碱基预测成功"
        else:
            usage, payload = await _run_extract(extractor, body)
            message = "DNA sequence embedding 提取成功"
        return _ok(usage=usage, message=message, result=payload)
    except RequestError as e:
        return _err(f"{mode} 失败: {e}", 400, detail={"request": _summarize_body(body)})
    except Exception as e:
        logger.exception("[dcs_adapter] %s 推理失败", mode)
        return _err(
            f"{mode} 失败: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )


@app.post("/api/aigress/openai/rice_ogr")
@app.post("/rice_ogr")
async def rice_ogr(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """单入口统一分发:根据请求体 mode 区分 dna_embedding / predict。"""
    try:
        body = await req.json()
    except Exception:
        return _err("请求体不是合法 JSON", 400)
    if not isinstance(body, dict):
        return _err("请求体必须是 JSON 对象", 400)
    return await _dispatch(req, body, authorization, x_api_key)


# 兼容旧路径:子路径方式(与单入口 body.mode 等价)
@app.post("/api/aigress/openai/rice_ogr/dna_embedding")
@app.post("/rice_ogr/dna_embedding")
async def rice_ogr_dna_embedding(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    try:
        body = await req.json()
    except Exception:
        return _err("请求体不是合法 JSON", 400)
    body.setdefault("mode", "dna_embedding")
    return await _dispatch(req, body, authorization, x_api_key)


@app.post("/api/aigress/openai/rice_ogr/predict")
@app.post("/rice_ogr/predict")
async def rice_ogr_predict(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    try:
        body = await req.json()
    except Exception:
        return _err("请求体不是合法 JSON", 400)
    body.setdefault("mode", "predict")
    return await _dispatch(req, body, authorization, x_api_key)


# ---------------------------------------------------------------------------
#  调试回显路由(排查用):捕获一切未匹配路径,回显网关实际转发的路径与方法。
# ---------------------------------------------------------------------------
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def _debug_echo_path(request: Request, full_path: str):
    return {
        "debug_received_path": f"/{full_path}",
        "method": request.method,
        "note": "请求已到达 dcs_adapter(网关确实把请求转了过来)",
    }


if __name__ == "__main__":
    # 容错预加载:失败不退出,错误写入 _INIT_ERROR 由 /health 返回
    init_extractors_safe()
    print(f"[dcs_adapter] starting on {_LISTEN_HOST}:{_LISTEN_PORT} ...", flush=True)
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)