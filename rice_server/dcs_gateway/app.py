"""DCS 统一网关 — 单端口收口 OGR-Prediction / OGR-Regulation / OGR-Introgression / OGR-Base（model_sub 路由版）

外部**统一入口**，两种路由方式（推荐 URL 路径路由）：

    POST .../api/aigress/openai/OGR/{model_sub}[/{subpath}]   # URL 路径路由
    POST .../api/aigress/openai/OGR                          # body model_sub 路由

网关优先解析 URL 路径段（model_sub / subpath，可选），否则读取请求体 JSON 中
的 `model_sub` 字段路由到各后端统一服务进程（OGR-Prediction/Regulation/Introgression
的网页端与纯 API 同一进程；OGR-Base 仍为独立 dcs_adapter），并把路径重写为后端
期望的入口（/api/aigress/openai/OGR-Prediction/Prediction 等）。

路由表（model_sub 小写匹配，连字符自动归一为下划线，缺省走主服务 ogr_base）：

    model_sub=OGR-Prediction   → 127.0.0.1:8001  /api/aigress/openai/OGR-Prediction/{Prediction,SNV,RegisterGenome}
    model_sub=OGR-Regulation   → 127.0.0.1:7001  /api/aigress/openai/OGR-Regulation/{Genome,Chromosome,Prediction,RegisterGenome}
    model_sub=OGR-Introgression→ 127.0.0.1:5001  /api/aigress/openai/OGR-Introgression/{Genome,Chromosome,Prediction,RegisterGenome}
    model_sub=OGR-Base         → 127.0.0.1:6001  /api/aigress/openai/OGR-Base/{Embedding,NTP}   (缺省)

说明：
- 各后端均为**纯子路径路由（无 body mode）**：URL 路径的第二段
  （Prediction/SNV/Genome/Chromosome/Embedding/NTP/predict/RegisterGenome）映射到后端子路径，
  网关只做路径重写，**不注入任何 mode 字段**；请求体其余字段原样透传。
- RegisterGenome：三个服务（OGR-Prediction/Regulation/Introgression）支持**传本地文件路径**
  注册自定义基因组（FASTA/GFF），文件需已预置于后端同机，不做文件传输；
  注册后预测请求用返回的 genome 字段引用（详见 API.md §4.5）。
- 转发前剥离 `model_sub`（仅供路由），其余字段（Genome/Chromosome/Start/...）
  与请求头（Authorization / X-API-Key 等）原样透传，后端计费/错误语义不变。
- /health 为聚合健康检查：汇总各后端的就绪状态（本地合成，不转发）。

稳定性修复（v0.2.0）——解决统一网关在 DCS 平台下并发不稳定问题：
- **连接复用**：改用 httpx.AsyncClient（keep-alive 连接池），替代原 http.client
  每请求新建连接的写法，避免瞬时新建连接数冲高撞上平台对后端的连接配额。
- **健康检查异步化**：聚合 /health 用 asyncio.gather 并发探活，不再同步阻塞
  事件循环（原来每探活一次冻结最多 4×5=20s，期间所有转发请求被挂起）。
- **精细超时语义**：connect/read 分开设置；连接失败→502，读超时→504（后端
  慢而非不可达），不再用 180s 一刀切把慢推理误杀成 unreachable。
- **响应头透传**：除 hop-by-hop 外原样透传后端响应头，不再只留 content-type；
  并剔除 content-encoding / content-length（httpx 自动解压后 body 已变，需按
  实际字节重算），避免压缩头与解压后 body 错位导致客户端解压失败。
- 依赖：httpx（0.28+）+ fastapi/uvicorn（与各后端同栈；httpx 已在环境内）。
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import pathlib
from datetime import datetime
from typing import Dict, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
#  后端地址与入口路径（路径重写目标；.env 可覆盖，同机部署默认 127.0.0.1）
# ---------------------------------------------------------------------------
BACKENDS: Dict[str, dict] = {
    "ogr_prediction": {
        "host": os.getenv("RICE_MUT_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_MUT_PORT", "8001")),
        "path": "/api/aigress/openai/OGR-Prediction",
        # body model_sub 路由且 URL 无子路径段时的默认子路径（旧式单地址兼容）
        "default_subpath": "/Prediction",
        "subpath_modes": ("prediction", "snv", "registergenome"),   # URL 第二段 → 后端子路径（归一形式）
        "subpath_map": {
            "prediction": "Prediction", "snv": "SNV",
            "registergenome": "RegisterGenome",
        },
    },
    "ogr_regulation": {
        "host": os.getenv("RICE_REG_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_REG_PORT", "7001")),
        "path": "/api/aigress/openai/OGR-Regulation",
        "default_subpath": "/Prediction",
        "subpath_modes": ("genome", "chromosome", "prediction", "registergenome"),
        "subpath_map": {
            "genome": "Genome", "chromosome": "Chromosome", "prediction": "Prediction",
            "registergenome": "RegisterGenome",
        },
    },
    "ogr_introgression": {
        "host": os.getenv("RICE_INTRO_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_INTRO_PORT", "5001")),
        "path": "/api/aigress/openai/OGR-Introgression",
        "default_subpath": "/Prediction",
        "subpath_modes": ("genome", "chromosome", "prediction", "health", "registergenome"),
        "subpath_map": {
            "genome": "Genome", "chromosome": "Chromosome",
            "prediction": "Prediction", "health": "health",
            "registergenome": "RegisterGenome",
        },
    },
    "ogr_base": {
        "host": os.getenv("RICE_OGR_HOST", "127.0.0.1"),
        "port": int(os.getenv("RICE_OGR_PORT", "6001")),  # 与 OGR-Prediction 错开
        "path": "/api/aigress/openai/OGR-Base",
        "default_subpath": "/Embedding",
        "subpath_modes": ("embedding", "ntp"),
        "subpath_map": {"embedding": "Embedding", "ntp": "NTP"},
    },
}

# model_sub 缺省路由到主服务（外部入口名即 OGR）
DEFAULT_SUB = "ogr_base"

# 外部 model_sub 别名 → BACKENDS 键（小写，去掉连字符）
_SUB_ALIAS = {
    "ogr_prediction": "ogr_prediction",
    "ogrprediction": "ogr_prediction",
    "ogr_regulation": "ogr_regulation",
    "ogrregulation": "ogr_regulation",
    "ogr_introgression": "ogr_introgression",
    "ogrintrogression": "ogr_introgression",
    "ogr_base": "ogr_base",
    "ogrbase": "ogr_base",
}

# 网关自身监听: DCS 平台注入的 PORT 优先级最高
_LISTEN_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
_LISTEN_PORT = int(os.getenv("PORT", os.getenv("GATEWAY_PORT", "9000")))

_HEALTH_PATHS = ("/api/aigress/openai/health", "/health")

# 不允许透传的 hop-by-hop 请求头（Host 由 httpx 按后端地址重建；Connection 由客户端管理）
_SKIP_REQ_HEADERS = {
    "host", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authorization", "proxy-connection", "te",
    # accept-encoding 不透传：httpx 自己管理压缩偏好（默认 gzip/deflate/br），
    # 避免把客户端压缩偏好转发给后端导致响应被压缩（见下方 content-encoding 说明）
    "accept-encoding",
}
# 不允许透传的 hop-by-hop 响应头
_SKIP_RESP_HEADERS = {
    "connection", "keep-alive", "transfer-encoding", "upgrade",
    # httpx 默认会自动解压 gzip/deflate/br 响应（resp.content 已是解码后字节），
    # 但 resp.headers 仍保留 content-encoding；若原样透传会发生「已解压 body +
    # gzip 头 + 压缩前 content-length」错位，客户端二次解压失败。
    # 故一并剔除，让 Starlette 按实际 body 重新计算 content-length。
    "content-encoding", "content-length",
}

# 转发超时设置：connect 建连短超时；read 给足长推理时间（predict_length 大时可达分钟级）
_FWD_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=600.0,
    write=30.0,
    pool=5.0,
)

app = FastAPI(
    title="DCS Gateway (OGR-Prediction + OGR-Regulation + OGR-Introgression + OGR-Base, model_sub routing)",
    version="0.4.0",
)


# URL 路径路由已知子路径集合（用于校验路径第二段是否为合法 subpath；
# 全部为归一后的形式——小写、无连字符/下划线）
_KNOWN_MODES = {"prediction", "snv", "genome", "chromosome",
                "predict", "health", "embedding", "ntp",
                "registergenome"}

# 全局共享的异步客户端（keep-alive 连接池；max_connections 放宽以支撑并发）
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """惰性创建全局 httpx.AsyncClient（连接复用，避免每请求新建 TCP 连接）。"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=_FWD_TIMEOUT,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
    return _client


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

    # 路由优先级：URL 路径段 > body model_sub > 缺省 ogr_base
    # （body 的 model_sub 与 URL 路径段一样做去连字符归一，兼容 OGR-Prediction / OGRPrediction / ogr_prediction）
    body_sub = str((body or {}).get("model_sub", "") or "").strip().lower().replace("-", "")
    sub = _SUB_ALIAS.get(path_sub or body_sub or DEFAULT_SUB, "") or DEFAULT_SUB
    cfg = BACKENDS.get(sub)
    if cfg is None:
        return JSONResponse(
            {
                "status": 400,
                "message": f"未知 model_sub '{sub}',可选: OGR-Prediction / OGR-Regulation / OGR-Introgression / OGR-Base",
            },
            status_code=400,
        )

    # 路径第二段 → 后端子路径（四个后端均为纯子路径路由，无 body mode 注入）
    extra_path = ""
    if path_mode:
        subpath_modes = cfg.get("subpath_modes", ())
        if path_mode in subpath_modes:
            extra_path = "/" + cfg.get("subpath_map", {}).get(path_mode, path_mode)
        else:
            # 未知子路径段：透传原样，由后端报错（含英文使用说明）
            extra_path = "/" + path_mode
    else:
        # body model_sub 路由（旧式单地址）且 URL 无子路径段：用该服务的默认子路径
        # （例如 OGR-Prediction → /Prediction），保证与网页端/纯 API 一致。
        extra_path = cfg.get("default_subpath", "")

    # 剥离仅供路由使用的 model_sub，其余字段原样透传
    if body:
        body.pop("model_sub", None)
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")

    # 构造可转发请求头（剔除 hop-by-hop；保留 Authorization / X-API-Key 等）
    # 注意：无效的鉴权头（如 "Authorization: Bearer " 只有前缀无实际 key）会触发
    # httpx 的 header 校验异常（LocalProtocolError: Illegal header value），
    # 这类头无鉴权意义，直接跳过；后端未启用鉴权时本来就放行。
    def _valid_header_value(k: str, v: str) -> bool:
        if v is None or v.strip() == "":
            return False
        k_l = k.lower()
        if k_l in ("authorization", "proxy-authorization"):
            # 只有 "Bearer"/"Basic" 前缀而无实际凭据 → 视为空
            parts = v.strip().split(None, 1)
            return len(parts) >= 2 and parts[1].strip() != ""
        # X-API-Key 为裸 key（无前缀），非空即有效
        return True

    fwd_headers: Dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() not in _SKIP_REQ_HEADERS and k.lower() != "content-length":
            if _valid_header_value(k, v):
                fwd_headers[k] = v
    fwd_headers["Content-Length"] = str(len(raw))

    target = cfg["path"] + extra_path + (("?" + request.url.query) if request.url.query else "")
    try:
        client = get_client()
        resp = await client.request(
            request.method, f"http://{cfg['host']}:{cfg['port']}{target}",
            content=raw, headers=fwd_headers,
        )
    except httpx.ConnectError as e:
        return _gateway_err(
            502, f"gateway: backend {cfg['host']}:{cfg['port']} unreachable "
                 f"({type(e).__name__}: {e})"
        )
    except httpx.ReadTimeout as e:
        return _gateway_err(
            504, f"gateway: backend {cfg['host']}:{cfg['port']} read timeout "
                 f"({type(e).__name__}: {e})"
        )
    except httpx.TimeoutException as e:
        return _gateway_err(
            504, f"gateway: backend {cfg['host']}:{cfg['port']} timeout "
                 f"({type(e).__name__}: {e})"
        )
    except Exception as e:
        return _gateway_err(
            502, f"gateway: backend {cfg['host']}:{cfg['port']} unreachable "
                 f"({type(e).__name__}: {e})"
        )

    # 透传后端响应头（除 hop-by-hop），保留 Content-Encoding/Set-Cookie 等
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _SKIP_RESP_HEADERS
    }
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


def _gateway_err(status: int, message: str) -> JSONResponse:
    """统一 DCS 错误方言（与后端 dcs_adapter 的 _err 一致）。"""
    return JSONResponse(
        {
            "model_name": "OGR",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "token_cnt": 0,
            "status": status,
            "message": message,
            "result": None,
        },
        status_code=status,
    )


async def _probe(sub: str, cfg: dict) -> dict:
    """对单个后端做 /health 探活（免鉴权，短超时；异步非阻塞）。"""
    try:
        client = get_client()
        resp = await client.get(
            f"http://{cfg['host']}:{cfg['port']}/health",
            timeout=httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0),
        )
        payload = resp.json()
        return {
            "reachable": True,
            "http_status": resp.status_code,
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


def _parse_path(full_path: str) -> Tuple[str, str]:
    """从 URL 路径提取路由段 (model_sub, subpath)；非入口路径返回 ("", "")。

    支持形式（大小写不敏感，连字符不敏感）：
      .../api/aigress/openai/OGR/OGR-Prediction/Prediction → ("ogr_prediction", "prediction")
      .../OGR/OGR-Base/Embedding                           → ("ogr_base", "embedding")
      .../OGR/health 或 .../OGR/{sub}/health                → ("", "health")（聚合健康）
      .../OGR                                               → ("", "")（纯 body 路由）
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
    sub = rest[0].lower().replace("-", "")
    mode = rest[1].lower() if len(rest) > 1 else ""
    # mode 同样做归一:去连字符与下划线(RegisterGenome / register_genome /
    # registergenome 均归一为 registergenome),与 sub 的处理保持一致
    mode = mode.replace("-", "").replace("_", "")
    if mode not in _KNOWN_MODES:
        mode = ""  # 非合法子路径（多余路径残余）→ 不当作子路径
    return sub, mode


@app.api_route("/api/aigress/openai/health", methods=["GET", "POST"])
@app.api_route("/health", methods=["GET", "POST"])
async def aggregated_health(request: Request):
    """聚合健康检查：异步并发探活四个后端（不阻塞事件循环）。"""
    results = await asyncio.gather(*(
        _probe(sub, cfg) for sub, cfg in BACKENDS.items()
    ))
    services: Dict[str, dict] = dict(zip(BACKENDS.keys(), results))
    all_ok = all(
        v.get("reachable") and v.get("http_status") == 200
        for v in services.values()
    )
    return {
        "status": "ok" if all_ok else "degraded",
        "services": services,
    }


# ---------------------------------------------------------------------------
#  文件系统探查接口（调试排查用，EXPLORE_ENABLED=true 时开放；生产默认关闭）
#  - 只读：fs/list | fs/stat | fs/find | fs/cat，无任何写/删/改操作
#  - 不限路径（排查阶段按需访问部署机任意绝对路径；用后请立即关闭开关）
#  - 鉴权：沿用 DCS_API_KEY（配置后需 Authorization: Bearer <key> / X-API-Key: <key>）
#  - 限流：cat 默认 64KB、list 默认 200 条、find 默认深度 4 / 200 条
# ---------------------------------------------------------------------------
_EXPLORE_ENABLED = os.getenv("EXPLORE_ENABLED", "false").strip().lower() in (
    "1", "true", "yes", "on",
)
_EXPLORE_MAX_FILE_BYTES = int(os.getenv("EXPLORE_MAX_FILE_BYTES", "65536") or 65536)
_EXPLORE_MAX_LIST_ITEMS = int(os.getenv("EXPLORE_MAX_LIST_ITEMS", "200") or 200)
_EXPLORE_MAX_FIND_DEPTH = int(os.getenv("EXPLORE_MAX_FIND_DEPTH", "4") or 4)


def _fs_auth_error(request: Request) -> str | None:
    """网关本地鉴权（仅探查接口用）：与后端一致，DCS_API_KEY 配置后需匹配。"""
    key = os.getenv("DCS_API_KEY", "").strip()
    if not key:
        return None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and auth[7:].strip() == key:
        return None
    if request.headers.get("x-api-key", "") == key:
        return None
    return "invalid api key (Authorization: Bearer <key> / X-API-Key: <key>)"


def _resolve_abs(raw: object) -> pathlib.Path:
    """校验并解析绝对路径（不支持相对路径 / 空值）。"""
    if not isinstance(raw, str) or not raw.strip() or not raw.strip().startswith("/"):
        raise ValueError("path must be an absolute path (e.g. /data/xxx)")
    return pathlib.Path(raw.strip()).resolve()


def _fs_entry(p: pathlib.Path) -> dict:
    """单个文件/目录详情（等价 ll 单行）。"""
    st = p.stat()
    is_link = p.is_symlink()
    if is_link:
        etype = "symlink"
    elif p.is_dir():
        etype = "dir"
    elif p.is_file():
        etype = "file"
    else:
        etype = "other"
    return {
        "name": p.name,
        "path": str(p),
        "type": etype,
        "size": st.st_size,
        "mode": oct(st.st_mode & 0o7777),
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "target": os.readlink(p) if is_link else None,
    }


def _fs_list(body: dict) -> dict:
    """列目录（等价 ls -lh）。"""
    p = _resolve_abs(body.get("path"))
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise ValueError(f"path is not a directory: {p}")
    entries = []
    with os.scandir(p) as it:
        for e in sorted(it, key=lambda x: x.name):
            try:
                entries.append(_fs_entry(pathlib.Path(e.path)))
            except OSError:
                continue
    total = len(entries)
    truncated = total > _EXPLORE_MAX_LIST_ITEMS
    return {
        "path": str(p),
        "total": total,
        "truncated": truncated,
        "items": entries[:_EXPLORE_MAX_LIST_ITEMS],
    }


def _fs_stat(body: dict) -> dict:
    """单个文件/目录详情（等价 ll /path）。"""
    p = _resolve_abs(body.get("path"))
    if not p.exists():
        raise FileNotFoundError(str(p))
    return {"path": str(p), "entry": _fs_entry(p)}


def _fs_find(body: dict) -> dict:
    """按文件名模式递归搜索（等价 find /path -name PATTERN）。"""
    p = _resolve_abs(body.get("path"))
    pattern = str(body.get("pattern") or "*")
    depth = min(int(body.get("max_depth") or _EXPLORE_MAX_FIND_DEPTH), 20)
    limit = min(int(body.get("limit") or 200), 1000)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise ValueError(f"path is not a directory: {p}")
    matches, matched = [], 0
    for root, dirs, files in os.walk(p):
        rel = pathlib.Path(root).relative_to(p)
        rel_depth = 0 if str(rel) == "." else len(rel.parts)
        if rel_depth >= depth:
            dirs[:] = []
            continue
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(str(pathlib.Path(root) / name))
                matched += 1
                if matched >= limit:
                    break
        if matched >= limit:
            break
    return {
        "path": str(p),
        "pattern": pattern,
        "max_depth": depth,
        "matched": matched,
        "truncated": matched >= limit,
        "matches": matches,
    }


def _fs_cat(body: dict) -> dict:
    """读文件内容（等价 head -c 64K；tail=true 时读末尾）。"""
    p = _resolve_abs(body.get("path"))
    tail = bool(body.get("tail", False))
    limit = min(int(body.get("limit") or _EXPLORE_MAX_FILE_BYTES), _EXPLORE_MAX_FILE_BYTES)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError(f"path is not a file: {p}")
    size = p.stat().st_size
    with open(p, "rb") as f:
        if tail and size > limit:
            f.seek(size - limit)
            data = f.read(limit)
            truncated = True
        else:
            data = f.read(limit)
            truncated = size > len(data)
    return {
        "path": str(p),
        "size": size,
        "read_bytes": len(data),
        "truncated": truncated,
        "tail": tail,
        "content": data.decode("utf-8", errors="replace"),
    }


def _fs_error(status: int, message: str) -> JSONResponse:
    """探查接口统一错误方言（与 _gateway_err / 后端 _err 一致）。"""
    return JSONResponse(
        {
            "model_name": "OGR",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "token_cnt": 0,
            "status": status,
            "message": message,
            "result": None,
        },
        status_code=status,
    )


async def _fs_handle(request: Request, fn) -> dict | JSONResponse:
    """探查接口统一入口：开关检查 → 鉴权 → 参数解析 → 执行 → 错误方言。"""
    if not _EXPLORE_ENABLED:
        return _fs_error(
            403,
            "fs explore disabled (set EXPLORE_ENABLED=true in dcs_gateway/.env)",
        )
    err = _fs_auth_error(request)
    if err:
        return _fs_error(401, err)
    try:
        body = await request.json()
    except Exception:
        return _fs_error(400, "invalid JSON body")
    if not isinstance(body, dict):
        return _fs_error(400, "request body must be a JSON object")
    try:
        result = fn(body)
    except FileNotFoundError as e:
        return _fs_error(404, f"not found: {e}")
    except PermissionError as e:
        return _fs_error(403, f"permission denied: {e}")
    except (ValueError, TypeError) as e:
        return _fs_error(400, str(e))
    except Exception as e:
        return _fs_error(500, f"{type(e).__name__}: {e}")
    return {
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "token_cnt": 0,
        "status": 200,
        "message": "ok",
        "result": result,
    }


@app.api_route("/api/aigress/openai/OGR/fs/list", methods=["POST"])
async def fs_list(request: Request):
    return await _fs_handle(request, _fs_list)


@app.api_route("/api/aigress/openai/OGR/fs/stat", methods=["POST"])
async def fs_stat(request: Request):
    return await _fs_handle(request, _fs_stat)


@app.api_route("/api/aigress/openai/OGR/fs/find", methods=["POST"])
async def fs_find(request: Request):
    return await _fs_handle(request, _fs_find)


@app.api_route("/api/aigress/openai/OGR/fs/cat", methods=["POST"])
async def fs_cat(request: Request):
    return await _fs_handle(request, _fs_cat)


# 唯一外部入口 + 其他任何路径：URL 路径路由 / body model_sub 路由
@app.api_route("/{full_path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def gateway(request: Request, full_path: str):
    return await _forward(request, full_path)


if __name__ == "__main__":
    print(
        f"[dcs_gateway] listening on {_LISTEN_HOST}:{_LISTEN_PORT} | "
        f"model_sub routes: ogr_prediction={BACKENDS['ogr_prediction']['port']}, "
        f"ogr_regulation={BACKENDS['ogr_regulation']['port']}, "
        f"ogr_introgression={BACKENDS['ogr_introgression']['port']}, "
        f"ogr_base={BACKENDS['ogr_base']['port']} (default)",
        flush=True,
    )
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)