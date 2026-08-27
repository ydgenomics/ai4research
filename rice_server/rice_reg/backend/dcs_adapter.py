"""DCS API 适配层 — rice_reg (ATAC → RNA-seq 表达预测)

对外提供 OpenAI 风格的 HTTP API,由 DCS 平台网关转发(单地址,路径前缀被剥离):

    POST /api/aigress/openai/rice_reg    单入口:body mode 分发(health/predict/genomes/chromosomes)
    GET  /api/aigress/openai/health      健康检查

由于 DCS 平台**只允许一个转发地址**,无法使用 /health、/genomes、
/predict/rice-reg 等子路径,因此单入口按请求体 ``mode`` 字段分发:

    {"mode": "health"}        → 健康检查(含监听端口 + 网关诊断)
    {"mode": "genomes"}       → 已配置基因组列表
    {"mode": "chromosomes"}   → 指定基因组的染色体列表(需带 genome)
    {"mode": "predict"}       → ATAC→RNA-seq 表达预测(默认)
    未指定 mode 时自动推断:空 body → health;其余 → predict

坐标约定:与网页版一致,start 为 1-based inclusive;内部经
``adjust_window`` 归一化后送入模型。

ATAC 输入(二选一,``uploaded_atac`` 优先):
    atac_source:    内置源 ID,如 "SAM2_MH63_1" → 查 .env 的 ATAC_PATH_SAM2_MH63_1
    uploaded_atac:  服务器上已上传的 ATAC bigWig 文件路径

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

鉴权(可选):在 rice_reg/.env 配置 DCS_API_KEY 后,POST 路由需要请求头
    Authorization: Bearer <DCS_API_KEY>  (或 X-API-Key: <DCS_API_KEY>)
    留空则不启用鉴权;health 始终免鉴权。

请求体示例(预测):

    {
      "model": "rice_reg",
      "genome": "MH63RS3",
      "chromosome": "chr01",
      "start": 20716774,          # 1-based inclusive
      "end": 20749541,            # 1-based inclusive(可省略,默认 TARGET_LEN 窗口)
      "atac_source": "SAM2_MH63_1",   # 内置 ATAC 源(或改传 uploaded_atac)
      "output_format": "full",    # full | mean | downsample
      "max_points": 1024          # downsample 时的目标点数(默认 1024)
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


ROOT_DIR = Path(__file__).resolve().parents[1]   # rice_reg/
BACKEND_DIR = Path(__file__).resolve().parent     # rice_reg/backend/
_load_env_file(ROOT_DIR / ".env")

# rice_reg 包位于 backend/rice_reg/,把 backend/ 加入 sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rice_reg.prediction_service import (  # noqa: E402
    _PREDICTOR,
    _validate_atac_bw,
    adjust_window,
    get_genome_chromosomes,
    init_predictor,
    list_genomes,
    normalize_chromosome,
    require_predictor,
    resolve_atac_path,
    resolve_genome_config,
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
_LISTEN_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "7001")))


def _format_array(arr, fmt: str, max_points: int):
    """把单个 numpy 数组按 output_format 转为 JSON 可序列化结构(full/mean/downsample)。"""
    arr = np.asarray(arr, dtype=np.float64)
    if fmt == "mean":
        return round(float(arr.mean()), 6)
    if fmt == "downsample":
        n = len(arr)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            arr = arr[idx]
        return [round(float(x), 6) for x in arr]
    return [round(float(x), 6) for x in arr]


def _count_elements(*arrays) -> int:
    """统计输出数组元素总数(用于 completion_tokens)。"""
    total = 0
    for arr in arrays:
        if arr is not None:
            total += int(np.asarray(arr).size)
    return total


def _parse_common(body: dict):
    """解析公共参数:genome / chromosome / start / end / atac / 输出格式。

    返回 (genome, chromosome, start_1, end_1, atac_source, uploaded_atac,
    output_format, max_points)。坐标 1-based inclusive(与网页版一致)。
    """
    genome = str(body.get("genome", "") or "")
    if not genome:
        genomes = list_genomes()
        genome = genomes[0] if genomes else ""
    if not genome:
        raise RequestError("Missing required parameter 'genome' and no default genome available")
    chromosome = str(body.get("chromosome", "") or "chr01")
    if "start" not in body:
        raise RequestError("Missing required parameter 'start' (1-based)")
    start_1 = int(body["start"])
    end_1 = body.get("end")
    end_1 = int(end_1) if end_1 is not None else None
    atac_source = str(body.get("atac_source", "") or "").strip() or None
    uploaded_atac = str(body.get("uploaded_atac", "") or "").strip() or None
    if not atac_source and not uploaded_atac:
        raise RequestError("Missing ATAC input: provide 'atac_source' (built-in) or 'uploaded_atac' (file path)")
    output_format = str(body.get("output_format", "full")).lower()
    if output_format not in ("full", "mean", "downsample"):
        raise RequestError(
            f"output_format must be full/mean/downsample, got '{output_format}'"
        )
    max_points = int(body.get("max_points", 1024))
    return genome, chromosome, start_1, end_1, atac_source, uploaded_atac, output_format, max_points


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
        "atac_source", "uploaded_atac", "output_format", "max_points",
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


def _env_entries(prefix: str, suffix: str) -> dict:
    """收集 .env 中形如 <prefix><ID><suffix> 的配置项(ID → 值是否存在)。

    注意:suffix 为空(如 ATAC_PATH_<ID>)时切片不能用 `-len(suffix)`(= 0),
    Python 的 -0 == 0 会导致 `key[len(prefix):0]` 返回空串。
    """
    out: dict = {}
    key_len = len(prefix)
    for key, val in sorted(os.environ.items()):
        if (
            key.startswith(prefix)
            and key.endswith(suffix)
            and len(key) > key_len + len(suffix)
        ):
            gid = key[key_len:] if not suffix else key[key_len:-len(suffix)]
            out[gid] = bool(val and Path(val).exists())
    return out


def _collect_diagnostics() -> dict:
    """收集部署机环境诊断:python/依赖/GPU/模型文件/基因组/ATAC 源/关键配置。"""
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
            "pyBigWig": _ver("pyBigWig"),
        },
        "gpu": {"cuda_available": False, "device_count": 0, "device_name": ""},
        "files": {
            "BASE_MODEL_PATH": os.getenv("BASE_MODEL_PATH", ""),
            "CHECKPOINT_PATH": os.getenv("CHECKPOINT_PATH", ""),
            "base_model_exists": _exists(os.getenv("BASE_MODEL_PATH")),
            "checkpoint_exists": _exists(os.getenv("CHECKPOINT_PATH")),
        },
        "genomes": _env_entries("GENOME_", "_FASTA"),
        "atac_sources": _env_entries("ATAC_PATH_", ""),
        "listen": {
            "BACKEND_HOST": os.getenv("BACKEND_HOST", "0.0.0.0"),
            "BACKEND_PORT": os.getenv("BACKEND_PORT", "7001"),
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


class _NewlineJSONResponse(FastAPIJSONResponse):
    """默认 JSON 响应类:序列化后追加换行符。

    FastAPI 默认返回的 JSON 末尾无换行,curl 输出会和 shell 提示符
    粘连在同一行。覆盖 render() 统一追加 \n,对所有路由生效。
    """

    def render(self, content) -> bytes:
        return super().render(content) + b"\n"


app = FastAPI(
    title="DCS Adapter (rice_reg)",
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
      2) 单入口 POST /rice_reg + body {"mode":"health"}(DCS 单地址转发)。
    """
    initialized = _PREDICTOR.get("instance") is not None
    # scope["server"] = 请求实际到达的 socket 地址(uvicorn 注入),即平台转发目标端口
    served = request.scope.get("server") or [None, None]
    served_port = served[1]
    return {
        # status 恒为 ok = uvicorn 在运行(HTTP 服务存活,不破坏平台探活);
        # 模型是否就绪看 predictor_initialized,失败原因看 diagnostics.init_error
        "status": "ok",
        "predictor_initialized": initialized,
        # genomes / atac_sources 来自 env 配置,不依赖模型初始化,始终返回(便于排查)
        "genomes": list_genomes(),
        "atac_sources": sorted(
            k for k in os.environ if k.startswith("ATAC_PATH_") and os.environ[k]
        ),
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


HEALTH_MODES = ("", "health")
PREDICT_MODES = ("predict",)
GENOMES_MODES = ("genomes", "genome_list")
CHROMOSOMES_MODES = ("chromosomes", "chromosome_list")


def _mode_from_body(body: dict) -> str:
    """从请求体推断调用模式(单入口 /rice_reg 下按 body 分发)。

    DCS 平台只允许一个转发地址,无法使用 /health、/genomes 等子路径,
    因此通过请求体字段区分调用模式:
      mode == "health"      → health(健康检查)
      mode == "genomes"     → genomes(Genome list)
      mode == "chromosomes" → chromosomes(指定基因组的染色体列表)
      mode == "predict"     → predict(ATAC→RNA-seq 表达预测)
      未指定 mode 时按字段自动推断:
        body 为空        → health
        其余             → predict
    """
    mode = str(body.get("mode", "") or "").strip().lower()
    if mode:
        if mode in HEALTH_MODES:
            return "health"
        if mode in GENOMES_MODES:
            return "genomes"
        if mode in CHROMOSOMES_MODES:
            return "chromosomes"
        if mode in PREDICT_MODES:
            return "predict"
        return mode  # 未知 mode,交由调用方报错
    # 未指定 mode → 自动推断
    if not body:
        return "health"
    return "predict"


@app.post("/api/aigress/openai/rice_reg")
@app.post("/rice_reg")
async def single_entry(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """单入口统一分发:根据请求体 mode 区分 health / genomes / chromosomes / predict。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON", 400)

    mode = _mode_from_body(body)
    if mode == "health":
        return await _health_response(req)
    if mode == "genomes":
        return await _genomes_inner(req, authorization, x_api_key)
    if mode == "chromosomes":
        return await _chromosomes_inner(req, body, authorization, x_api_key)
    if mode not in PREDICT_MODES:
        return _err(
            f"Unknown mode '{mode}', must be health/genomes/chromosomes/predict", 400
        )
    return await _predict_inner(req, body, authorization, x_api_key)


async def _genomes_inner(
    req: Request,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """已配置基因组列表(env GENOME_*_FASTA + 上传的自定义基因组)。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))
    try:
        genomes = list_genomes()
    except Exception:
        traceback.print_exc()
        genomes = []
    return _ok(
        usage=_usage(0, 0),
        message="Genome list",
        result={"model": "rice_reg", "genomes": genomes},
    )


async def _chromosomes_inner(
    req: Request,
    body: dict,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """指定基因组的染色体列表(chrNN 风格,需要 genome 参数)。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))
    try:
        genome = str(body.get("genome", "") or "")
        if not genome:
            raise RequestError("Missing required parameter 'genome'")
        genome_config = resolve_genome_config(genome)
        chroms = get_genome_chromosomes(genome, genome_config)
    except RequestError as e:
        return _err(f"Chromosome query failed: {e}", 400, detail={"request": _summarize_body(body)})
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"Chromosome query failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )
    return _ok(
        usage=_usage(0, 0),
        message="Chromosome list",
        result={"model": "rice_reg", "genome": genome, "chromosomes": chroms},
    )


async def _predict_inner(
    req: Request,
    body: dict,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """ATAC→RNA-seq 表达预测(默认模式)。

    直接调用预测器取数值数组(不写预测 bigWig / 不生成 IGV payload——
    后者依赖本地静态文件服务,对 DCS 调用方无意义)。
    """
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
        genome, chromosome, start_1, end_1, atac_source, uploaded_atac, fmt, max_points = (
            _parse_common(body)
        )
        genome_config = resolve_genome_config(genome)
        # 解析 ATAC 输入路径(uploaded_atac 优先,否则 atac_source 查 ATAC_PATH_<ID>)
        atac_path = resolve_atac_path(atac_source, uploaded_atac)

        predictor = require_predictor()

        # 窗口归一化(center-align 到 target_len,与网页版 /predict/rice-reg 一致)
        norm_start, norm_end = adjust_window(start_1, end_1, predictor.target_len)
        # 染色体别名归一化 + ATAC bigWig 兼容性校验(与 run_prediction_core 一致)
        chrom = normalize_chromosome(genome, chromosome, genome_config)
        _validate_atac_bw(atac_path, chrom)

        result = predictor.predict(
            chrom=chrom,
            start=norm_start,
            end=norm_end,
            atac_path=atac_path,
            fasta_path=genome_config["fasta"],
            cell_type="sample",
        )

        pos_chrom, pos_start, pos_end = result["position"]
        # pandas 行取出的坐标可能是 numpy int64,JSON 无法序列化
        # (FastAPI 在 handler 返回后的序列化阶段抛错 → 纯文本 Internal Server Error),
        # 因此显式转成 Python int。
        pos_start = int(pos_start)
        pos_end = int(pos_end)
        plus = result["values"].get("RNA-seq_+")
        minus = result["values"].get("RNA-seq_-")
        if plus is None or minus is None:
            raise RuntimeError("Predictor did not return RNA-seq +/- values (no valid region in window)")
    except RequestError as e:
        return _err(
            f"Prediction failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except FileNotFoundError as e:
        return _err(
            f"Prediction failed: {e}", 404,
            detail={"request": _summarize_body(body)},
        )
    except ValueError as e:
        return _err(
            f"Prediction failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except RuntimeError as e:
        return _err(
            f"Prediction failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"Prediction failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    prompt_tokens = pos_end - pos_start       # 窗口碱基数
    completion_tokens = _count_elements(plus, minus)
    return _ok(
        usage=_usage(prompt_tokens, completion_tokens),
        message="ATAC→RNA-seq expression prediction succeeded",
        result={
            "model": "rice_reg",
            "genome": genome,
            "chromosome": pos_chrom,
            "position_1based": {"start": pos_start + 1, "end": pos_end},
            "window_len": pos_end - pos_start,
            "atac_source": atac_source,
            "atac_path": atac_path,
            "output_format": fmt,
            "values": {
                "RNA-seq_+": _format_array(plus, fmt, max_points),
                "RNA-seq_-": _format_array(minus, fmt, max_points),
            },
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
    print(f"[dcs_adapter] starting on {_LISTEN_HOST}:{_LISTEN_PORT} ...")
    uvicorn.run(app, host=_LISTEN_HOST, port=_LISTEN_PORT, reload=False)