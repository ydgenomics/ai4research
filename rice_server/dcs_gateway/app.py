"""DCS 统一网关 — 单端口收口 rice_mut / rice_reg / rice_OGR（model_sub 路由版）

外部只有**唯一入口**：

    POST https://www.dcs.cloud/api/aigress/openai/OGR

网关**不按 URL 前缀**区分服务（路径恒为 .../OGR），而是读取请求体 JSON 中
的 `model_sub` 字段路由到三个后端 dcs_adapter 进程，并把路径重写为后端
期望的入口（/api/aigress/openai/rice_mut 等）。

路由表（model_sub 小写匹配，缺省走主服务 rice_ogr）：

    model_sub=rice_mut  → 127.0.0.1:8001  /api/aigress/openai/rice_mut
    model_sub=rice_reg  → 127.0.0.1:7001  /api/aigress/openai/rice_reg
    model_sub=rice_ogr  → 127.0.0.1:8003  /api/aigress/openai/rice_ogr   (缺省)

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
from typing import Dict, Optional, Tuple

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
    "rice_ogr": {
        "host": os.getenv("RICE_OGR_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_OGR_PORT", "8003")),  # 与 rice_mut 错开
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
    title="DCS Gateway (rice_mut + rice_reg + rice_OGR, model_sub routing)",
    version="0.1.0",
)


def _route(body: dict) -> Tuple[Optional[str], Optional[dict]]:
    """按请求体 model_sub 选择后端；返回 (sub, cfg) 或 (sub, None)。"""
    sub = str((body or {}).get("model_sub", "") or "").strip().lower()
    if not sub:
        sub = DEFAULT_SUB
    return sub, BACKENDS.get(sub)


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


async def _forward(request: Request) -> Response:
    """统一转发：读 body → model_sub 路由 → 路径重写 → 透传。"""
    raw = await request.body()
    try:
        body: dict = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            raise ValueError("body 不是 JSON 对象")
    except Exception:
        return JSONResponse(
            {"status": 400, "message": "请求体不是合法 JSON 对象"}, status_code=400
        )

    sub, cfg = _route(body)
    if cfg is None:
        return JSONResponse(
            {
                "status": 400,
                "message": f"未知 model_sub '{sub}',可选: rice_mut / rice_reg / rice_ogr",
            },
            status_code=400,
        )

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


# 唯一外部入口 + 其他任何路径：统一交给 model_sub 路由
@app.api_route("/{full_path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def gateway(request: Request, full_path: str):
    return await _forward(request)


if __name__ == "__main__":
    print(
        f"[dcs_gateway] listening on {_LISTEN_HOST}:{_LISTEN_PORT} | "
        f"model_sub routes: mut={BACKENDS['rice_mut']['port']}, "
        f"reg={BACKENDS['rice_reg']['port']}, "
        f"ogr={BACKENDS['rice_ogr']['port']} (default)",
        flush=True,
    )
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)