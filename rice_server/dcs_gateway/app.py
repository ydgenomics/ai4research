"""DCS 统一网关 — 单端口收口 rice_mut / rice_reg / rice_intro / rice_OGR（model_sub 路由版）

外部**统一入口**，两种路由方式（推荐 URL 路径路由）：

    POST .../api/aigress/openai/OGR/{model_sub}[/{mode}]   # URL 路径路由
    POST .../api/aigress/openai/OGR                          # body model_sub 路由

网关优先解析 URL 路径段（model_sub / mode，可选），否则读取请求体 JSON 中
的 `model_sub` 字段路由到三个后端 dcs_adapter 进程，并把路径重写为后端
期望的入口（/api/aigress/openai/rice_mut 等）。

路由表（model_sub 小写匹配，缺省走主服务 rice_ogr）：

    model_sub=rice_mut  → 127.0.0.1:8001  /api/aigress/openai/rice_mut
    model_sub=rice_reg  → 127.0.0.1:7001  /api/aigress/openai/rice_reg
    model_sub=rice_ogr  → 127.0.0.1:6001  /api/aigress/openai/rice_ogr   (缺省)

说明：
- 网关为轻量反向代理：**不加载任何模型、不持有 GPU**。
- 转发前剥离 `model_sub`（仅供路由），其余字段（mode/sequence/start/...）
  与请求头（Authorization / X-API-Key 等）原样透传，后端计费/错误语义不变。
- /health 为聚合健康检查：汇总三个后端的就绪状态（本地合成，不转发）。
- 依赖：仅 Python 标准库（http.client）+ fastapi/uvicorn（与三个后端同栈）。
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
from typing import Dict, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
#  后端地址与入口路径（路径重写目标；.env 可覆盖，同机部署默认 127.0.0.1）
# ---------------------------------------------------------------------------
BACKENDS: Dict[str, dict] = {
    "rice_mut": {
        "host": os.getenv("RICE_MUT_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_MUT_PORT", "8001")),
        "path": "/api/aigress/openai/rice_mut",
    },
    "rice_reg": {
        "host": os.getenv("RICE_REG_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_REG_PORT", "7001")),
        "path": "/api/aigress/openai/rice_reg",
    },
    "rice_intro": {
        "host": os.getenv("RICE_INTRO_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_INTRO_PORT", "5001")),
        "path": "/api/aigress/openai/rice_intro",
    },
    "rice_ogr": {
        "host": os.getenv("RICE_OGR_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_OGR_PORT", "6001")),  # 与 rice_mut 错开
        "path": "/api/aigress/openai/rice_ogr",
    },
}

# model_sub 缺省路由到主服务（外部入口名即 OGR）
DEFAULT_SUB = "rice_ogr"

# 网关自身监听: DCS 平台注入的 PORT 优先级最高
_LISTEN_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
_LISTEN_PORT = int(os.getenv("PORT", os.getenv("GATEWAY_PORT", "9000")))

_HEALTH_PATHS = ("/api/aigress/openai/health", "/health")

# 不允许透传的 hop-by-hop 请求头（Host 由 http.client 按后端地址重建）
_SKIP_REQ_HEADERS = {
    "host", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authorization", "proxy-connection", "te",
}

app = FastAPI(
    title="DCS Gateway (rice_mut + rice_reg + rice_intro + rice_OGR, model_sub routing)",
    version="0.1.0",
)


# URL 路径路由已知 mode 集合（用于校验路径第二段是否为合法 mode）
_KNOWN_MODES = {"predict", "snv", "genomes", "chromosomes", "dna_embedding"}


def _parse_path(full_path: str) -> Tuple[str, str]:
    """从 URL 路径提取路由段 (model_sub, mode)；非入口路径返回 ("", "")。

    支持形式（大小写不敏感）：
      .../api/aigress/openai/OGR/rice_mut/predict → ("rice_mut", "predict")
      .../OGR/rice_ogr/dna_embedding             → ("rice_ogr", "dna_embedding")
      .../OGR/health 或 .../OGR/{sub}/health       → ("", "health")（聚合健康）
      .../OGR                                      → ("", "")（纯 body 路由）
    未找到入口段 OGR / 无后续段时返回 ("", "")，退化为 body model_sub 路由。
    """
    segs = [s for s in str(full_path or "").split("/") if s]
    try:
        idx = next(i for i, s in enumerate(segs) if s.lower() == "ogr")
    except StopIteration:
        return "", ""
    rest = segs[idx + 1:]
    if not rest:
        return "", ""
    if rest[-1].lower() == "health":
        return "", "health"
    sub = rest[0].lower()
    mode = rest[1].lower() if len(rest) > 1 else ""
    if mode not in _KNOWN_MODES:
        mode = ""  # 非合法 mode（多余路径残余）→ 不当作 mode
    return sub, mode


def _forward_sync(method: str, path: str, query: str,
                  headers: Dict[str, str], body: bytes,
                  host: str, port: int, timeout: float) -> Tuple[int, str, bytes]:
    """同步转发单个 HTTP 请求（在线程池中执行，见 _forward）。"""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    target = path + (("?" + query) if query else "")
    conn.request(method, target, body=body or None, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp.getheader("content-type", "application/json"), data


async def _forward(request: Request, full_path: str = "") -> Response:
    """统一转发：URL 路径路由（推荐）+ body model_sub 路由（兼容）→ 后端。"""
    # .../OGR/health 或 .../OGR/{sub}/health → 网关聚合健康检查
    path_sub, path_mode = _parse_path(full_path)
    if path_mode == "health":
        return await aggregated_health(request)

    raw = await request.body()
    try:
        body: dict = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            raise ValueError("body 不是 JSON 对象")
    except Exception:
        return JSONResponse(
            {"status": 400, "message": "请求体不是合法 JSON 对象"}, status_code=400
        )

    # 路由优先级：URL 路径段 > body model_sub > 缺省 rice_ogr
    body_sub = str((body or {}).get("model_sub", "") or "").strip().lower()
    sub = path_sub or body_sub or DEFAULT_SUB
    cfg = BACKENDS.get(sub)
    if cfg is None:
        return JSONResponse(
            {
                "status": 400,
                "message": f"未知 model_sub '{sub}',可选: rice_mut / rice_reg / rice_intro / rice_ogr",
            },
            status_code=400,
        )

    # 路径段 mode 优先注入 body（后端单入口按 body.mode 分发）
    if path_mode:
        body = dict(body or {})
        body["mode"] = path_mode

    # 剥离仅供路由使用的 model_sub，其余字段原样透传
    if body:
        body.pop("model_sub", None)
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")

    # 构造可转发请求头（剔除 hop-by-hop；保留 Authorization / X-API-Key 等）
    fwd_headers: Dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() not in _SKIP_REQ_HEADERS and k.lower() != "content-length":
            fwd_headers[k] = v
    fwd_headers["Content-Length"] = str(len(raw))

    status, content_type, data = await asyncio.to_thread(
        _forward_safe,
        request.method, cfg["path"], request.url.query,
        fwd_headers, raw,
        cfg["host"], cfg["port"],
        timeout=180.0,  # 推理请求耗时可达数十秒
    )
    # 原样透传（后端 dcs_adapter 的 JSON 已带末尾换行，字节不改动）
    return Response(content=data, status_code=status, media_type=content_type or None)


def _forward_safe(method: str, path: str, query: str, headers: Dict[str, str],
                  body: bytes, host: str, port: int, timeout: float):
    """转发并捕获连接层错误：后端不可达/超时 → 502 JSON（统一 DCS 错误方言）。"""
    try:
        return _forward_sync(method, path, query, headers, body, host, port, timeout)
    except Exception as e:
        return (
            502,
            "application/json",
            json.dumps(
                {
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "status": 502,
                    "message": f"gateway: backend {host}:{port} unreachable ({type(e).__name__}: {e})",
                    "result": None,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )


def _probe(sub: str, cfg: dict) -> dict:
    """对单个后端做 /health 探活（免鉴权，短超时）。"""
    try:
        status, content_type, data = _forward_safe(
            "GET", "/health", "", {}, b"", cfg["host"], cfg["port"], timeout=5.0
        )
        payload = json.loads(data)
        return {
            "reachable": True,
            "http_status": status,
            "predictor_initialized": payload.get("predictor_initialized"),
            "extractor_initialized": payload.get("extractor_initialized"),
            "init_error": (
                payload.get("diagnostics", {}).get("init_error")
                or payload.get("init_error")
            ),
        }
    except Exception as e:  # 连接失败 / 超时 / 解析失败
        return {
            "reachable": False,
            "http_status": None,
            "error": f"{type(e).__name__}: {e}",
        }


@app.api_route("/api/aigress/openai/health", methods=["GET", "POST"])
@app.api_route("/health", methods=["GET", "POST"])
async def aggregated_health(request: Request):
    """聚合健康检查：汇总三个后端的就绪状态（本地合成，不转发）。"""
    services: Dict[str, dict] = {}
    for sub, cfg in BACKENDS.items():
        services[sub] = _probe(sub, cfg)
    all_ok = all(
        v.get("reachable") and v.get("http_status") == 200
        for v in services.values()
    )
    return {
        "status": "ok" if all_ok else "degraded",
        "services": services,
    }


# 唯一外部入口 + 其他任何路径：URL 路径路由 / body model_sub 路由
@app.api_route("/{full_path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def gateway(request: Request, full_path: str):
    return await _forward(request, full_path)


if __name__ == "__main__":
    print(
        f"[dcs_gateway] listening on {_LISTEN_HOST}:{_LISTEN_PORT} | "
        f"model_sub routes: mut={BACKENDS['rice_mut']['port']}, "
        f"reg={BACKENDS['rice_reg']['port']}, "
        f"intro={BACKENDS['rice_intro']['port']}, "
        f"ogr={BACKENDS['rice_ogr']['port']} (default)",
        flush=True,
    )
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)