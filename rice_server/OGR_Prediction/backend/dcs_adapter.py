"""DCS API 适配层 — OGR-Prediction (DNA → 多组学表达预测)

对外提供 OpenAI 风格的 HTTP API,由 DCS 平台网关转发,全部通过子路径路由:

    POST /api/aigress/openai/OGR-Prediction/Prediction  参考序列表达预测
    POST /api/aigress/openai/OGR-Prediction/SNV         单碱基变异 (SNV) 双轨对比预测
    GET  /health                                       健康检查

参数命名(大写驼峰,无 mode / end / max_points):

    {
      "model": "OGR-Prediction",
      "Genome": "osa1_r7",          # 可选,默认唯一基因组
      "Chromosome": "chr01",        # 可选,默认 chr01;chr01/Chr1/1 均自动通配
      "Start": 20716774,            # 必填,1-based inclusive
      "Format": "Full",             # 可选,Full(默认,逐碱基数组) | Mean(每轨道标量均值)
      "biosample_names": null       # 可选,缺省全部
    }

SNV 预测(额外两个字段,大小写不敏感):

    {
      "model": "OGR-Prediction",
      "Genome": "osa1_r7",
      "Chromosome": "chr01",
      "Start": 20716774,
      "SNV_position": 20731844,     # 必填,1-based 变异位点,须落在窗口内
      "SNV_base": "T"               # 必填,A/C/G/T
    }

参数校验失败返回英文使用说明(message 字段)。

坐标约定:请求使用 1-based(与网页版一致),内部转 0-based 后调用现有
``rice_mutation.prediction_service`` 的 ``run_prediction_core`` /
``run_snv_core``(复用染色体别名归一化、窗口中心对齐与缓存)。

返回结构遵循 dcs.md 规范:

    {
      "usage": {"prompt_tokens": N, "completion_tokens": M},
      "token_cnt": N + M,
      "status": 200,
      "message": "...",
      "result": {...}
    }

计费口径(可用环境变量调整):
    prompt_tokens      = 输入窗口碱基数 × DCS_PROMPT_TOKEN_MULTIPLIER      (默认 1)
    completion_tokens  = 输出数组元素总数 × DCS_COMPLETION_TOKEN_MULTIPLIER (默认 1)

鉴权(可选):在 OGR_Prediction/.env 配置 DCS_API_KEY 后,POST 路由需要请求头
    Authorization: Bearer <DCS_API_KEY>  (或 X-API-Key: <DCS_API_KEY>)
    留空则不启用鉴权;GET /health 始终免鉴权。

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


ROOT_DIR = Path(__file__).resolve().parents[1]   # OGR_Prediction/
BACKEND_DIR = Path(__file__).resolve().parent     # OGR_Prediction/backend/
_load_env_file(ROOT_DIR / ".env")

# backend/ 下含 src/ 包 (from src.util import ...),加入 sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rice_mutation.prediction_service import (  # noqa: E402
    _adjust_window,
    attach_uploaded_gff,
    init_predictor,
    list_genomes,
    register_uploaded_genome,
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


def _format_values(values: dict, fmt: str) -> dict:
    """按 Format 把 {assay: {biosample: np.ndarray}} 转为 JSON 可序列化结构。

    full — 完整数组(round 到 6 位)
    mean — 每轨道窗口均值(标量)
    """
    out: dict = {}
    for assay, bios_map in values.items():
        out[assay] = {}
        for bios, arr in bios_map.items():
            arr = np.asarray(arr, dtype=np.float64)
            if fmt == "mean":
                out[assay][bios] = round(float(arr.mean()), 6)
            else:  # full
                out[assay][bios] = [round(float(x), 6) for x in arr]
    return out


# 参数使用说明(校验失败时随 message 返回英文提示)
USAGE = (
    "Usage: POST /api/aigress/openai/OGR-Prediction/Prediction "
    "or .../OGR-Prediction/SNV. "
    "Body keys: model (OGR-Prediction, optional), Genome (optional, default first), "
    "Chromosome (optional, default chr01), Start (required, 1-based int), "
    "Format (optional, Full|Mean, default Full), "
    "biosample_names (optional, list or comma-separated string); "
    "SNV additionally: SNV_position (required, 1-based int), "
    "SNV_base (required, A/C/G/T). No mode/end/max_points keys."
)


def _parse_common(body: dict):
    """解析公共参数:Genome / Chromosome / Start / Format / biosample_names。

    返回 (genome, chromosome, start_1, biosample_names, output_format)。
    坐标为 0-based half-open(内部约定,窗口长 32768 中心对齐)。
    """
    genome = str(body.get("Genome", "") or "")
    if not genome:
        genomes = list_genomes()
        genome = genomes[0] if genomes else "osa1_r7"
    chromosome = str(body.get("Chromosome", "") or "chr01")
    if "Start" not in body or body.get("Start") is None or str(body.get("Start", "")).strip() == "":
        raise RequestError("Missing required parameter 'Start' (1-based int). " + USAGE)
    try:
        start_1 = int(body["Start"])
    except (TypeError, ValueError):
        raise RequestError(f"Invalid 'Start' value '{body.get('Start')}': must be an int. " + USAGE)
    biosample_names = body.get("biosample_names")
    if isinstance(biosample_names, str) and biosample_names.strip():
        biosample_names = [s.strip() for s in biosample_names.split(",") if s.strip()]
    if not isinstance(biosample_names, list):
        biosample_names = None
    output_format = str(body.get("Format", "Full") or "Full").strip().lower()
    if output_format not in ("full", "mean"):
        raise RequestError(
            f"Invalid 'Format' '{body.get('Format')}': must be Full or Mean. " + USAGE
        )
    return genome, chromosome, start_1, biosample_names, output_format


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
        "model", "Genome", "Chromosome", "Start",
        "SNV_position", "SNV_base", "Format", "biosample_names",
        "fasta_path", "genome_id", "gff_path",
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
        "model_name": "OGR",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "token_cnt": 0,
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


class _NewlineJSONResponse(FastAPIJSONResponse):
    """默认 JSON 响应类:序列化后追加换行符。

    FastAPI 默认返回的 JSON 末尾无换行,curl 输出会和 shell 提示符
    粘连在同一行。覆盖 render() 统一追加 \n,对所有路由生效。
    """

    def render(self, content) -> bytes:
        return super().render(content) + b"\n"


app = FastAPI(
    title="DCS Adapter (OGR-Prediction)",
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
    return await _health_response(request)


async def _health_response(request: Request) -> dict:
    """构造健康检查响应:含监听端口 / gateway 诊断。

    GET/POST /health、/api/aigress/openai/health(平台探活)均走此路径。
    """
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


# ---------------------------------------------------------------------------
#  路由(全部子路径,无 mode 分发):
#    POST /api/aigress/openai/OGR-Prediction/Prediction  参考表达预测
#    POST /api/aigress/openai/OGR-Prediction/SNV         SNV 双轨对比
#    GET  /health                                       健康检查(免鉴权)
# ---------------------------------------------------------------------------
@app.post("/api/aigress/openai/OGR-Prediction/Prediction")
@app.post("/OGR-Prediction/Prediction")
async def predict_ref(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """参考序列表达预测(子路径)。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON. " + USAGE, 400)
    return await _predict_ref_inner(req, body, authorization, x_api_key)


# ---------------------------------------------------------------------------
#  自定义基因组懒注册:预测请求 body 直接带 fasta_path(有则替代 Genome 键)
#  生成的 genome id 与网页端同风格:{time_ns}_{原文件名}
# ---------------------------------------------------------------------------
def _lazy_register_genome(body: dict) -> tuple[str, dict | None]:
    """若 body 含 fasta_path,现场注册并返回 (genome_id, cfg);
    否则返回 ("", None)。gff_path 随 fasta_path 一起附加。
    """
    fasta_path = str(body.get("fasta_path", "") or "").strip()
    if not fasta_path:
        return "", None
    gff_path = str(body.get("gff_path", "") or "").strip() or ""

    # genome id = {time_ns}_{原文件名},与网页端落盘文件命名一致
    fname = os.path.basename(fasta_path) or "custom.fa"
    genome_id = f"{time.time_ns()}_{fname}"
    cfg = register_uploaded_genome(fasta_path, genome_id=genome_id)
    if gff_path:
        cfg = attach_uploaded_gff(genome_id, gff_path)
    return genome_id, cfg


async def _predict_ref_inner(
    req: Request,
    body: dict,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """参考序列表达预测。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    if _INIT_ERROR:
        return _err(
            f"Predictor initialization failed, cannot inference: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    lazy_genome, lazy_cfg = "", None
    try:
        # 懒注册:body 有 fasta_path → 现场注册并生成 genome(替代 Genome 键)
        lazy_genome, lazy_cfg = _lazy_register_genome(body)
        if lazy_genome:
            body["Genome"] = lazy_genome
            body["gff_path"] = ""

        genome, chromosome, start_1, biosample_names, fmt = _parse_common(body)
        # 1-based inclusive → 0-based half-open;窗口长 32768 由内部中心对齐
        start_0 = max(0, start_1 - 1)

        genome_config = resolve_genome_config(genome)
        result = run_prediction_core(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=None,
            biosample_names=biosample_names,
            genome_config=genome_config,
        )
    except RequestError as e:
        return _err(
            f"Reference prediction failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"Reference prediction failed: {e}", 500,
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
        message="Reference expression prediction succeeded",
        result={
            "model": "OGR-Prediction",
            "Genome": result["genome"],
            "fasta_path": (lazy_cfg or {}).get("fasta", ""),
            "gff_path": (lazy_cfg or {}).get("gff", ""),
            "Chromosome": pos_chrom,
            "Start": pos_start + 1,
            "window_len": pos_end - pos_start,
            "Format": fmt,
            "values": _format_values(result["values"], fmt),
        },
    )


@app.post("/api/aigress/openai/OGR-Prediction/SNV")
@app.post("/OGR-Prediction/SNV")
async def predict_snv(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """单碱基变异预测(子路径)。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON. " + USAGE, 400)
    return await _predict_snv_inner(req, body, authorization, x_api_key)


async def _predict_snv_inner(
    req: Request,
    body: dict,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """单碱基变异预测:result1 = 参考, result2 = 突变。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    if _INIT_ERROR:
        return _err(
            f"Predictor initialization failed, cannot inference: {_INIT_ERROR['error']}", 503,
            detail={"init_error": _INIT_ERROR, "request": _summarize_body(body)},
        )

    lazy_genome, lazy_cfg = "", None
    try:
        # 懒注册:body 有 fasta_path → 现场注册并生成 genome(替代 Genome 键)
        lazy_genome, lazy_cfg = _lazy_register_genome(body)
        if lazy_genome:
            body["Genome"] = lazy_genome
            body["gff_path"] = ""

        genome, chromosome, start_1, biosample_names, fmt = _parse_common(body)
        if "SNV_position" not in body or body.get("SNV_position") is None or str(body.get("SNV_position", "")).strip() == "":
            raise RequestError("Missing required parameter 'SNV_position' (1-based int). " + USAGE)
        try:
            snv_1 = int(body["SNV_position"])
        except (TypeError, ValueError):
            raise RequestError(
                f"Invalid 'SNV_position' value '{body.get('SNV_position')}': must be an int. " + USAGE
            )
        snv_base = str(body.get("SNV_base", "") or "").strip().upper()
        if snv_base not in ("A", "C", "G", "T"):
            raise RequestError(
                f"Invalid 'SNV_base' '{body.get('SNV_base')}': must be A/C/G/T. " + USAGE
            )

        start_0 = max(0, start_1 - 1)
        snv_0 = max(0, snv_1 - 1)

        genome_config = resolve_genome_config(genome)
        result = run_snv_core(
            genome=genome,
            chromosome=chromosome,
            start=start_0,
            end=None,
            snv_index=snv_0,
            snv_base=snv_base,
            biosample_names=biosample_names,
            genome_config=genome_config,
        )
    except RequestError as e:
        return _err(
            f"SNV prediction failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"SNV prediction failed: {e}", 500,
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
        message=f"SNV prediction succeeded (ref {result['ref_base']} → {result['snv_base']})",
        result={
            "model": "OGR-Prediction",
            "Genome": result["genome"],
            "fasta_path": (lazy_cfg or {}).get("fasta", ""),
            "gff_path": (lazy_cfg or {}).get("gff", ""),
            "Chromosome": pos_chrom,
            "Start": pos_start + 1,
            "window_len": pos_end - pos_start,
            "SNV_position": result["snv_index"] + 1,
            "ref_base": result["ref_base"],
            "SNV_base": result["snv_base"],
            "Format": fmt,
            "ref_values": _format_values(ref_values, fmt),
            "mut_values": _format_values(mut_values, fmt),
        },
    )


# ---------------------------------------------------------------------------
#  自定义基因组路径注册(传文件地址即可,不做文件传输)
#  文件已预先放在本机;调用方只需在 body 中传 fasta_path(必填) +
#  可选 genome_id / gff_path,后端校验并注册后,预测请求用 Genome 字段引用。
# ---------------------------------------------------------------------------
@app.post("/api/aigress/openai/OGR-Prediction/RegisterGenome")
@app.post("/OGR-Prediction/RegisterGenome")
async def register_genome(
    req: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    """注册本地 FASTA 为可用基因组(传路径,不传输文件)。"""
    try:
        body = await req.json()
    except Exception:
        return _err("Request body is not valid JSON. " + USAGE, 400)
    return await _register_genome_inner(req, body, authorization, x_api_key)


async def _register_genome_inner(
    req: Request,
    body: dict,
    authorization: str | None = None,
    x_api_key: str | None = None,
):
    """按本地路径注册自定义基因组(可附带 GFF 注释)。"""
    try:
        _check_api_key(authorization, x_api_key)
    except RequestError as e:
        return _unauthorized(str(e))

    try:
        fasta_path = str(body.get("fasta_path", "") or "").strip()
        if not fasta_path:
            raise RequestError(
                "Missing required parameter 'fasta_path' (local path to a .fa/.fasta/.fna file). " + USAGE
            )
        genome_id = str(body.get("genome_id", "") or "").strip() or ""
        gff_path = str(body.get("gff_path", "") or "").strip() or ""

        cfg = register_uploaded_genome(fasta_path, genome_id=genome_id)
        # 定位刚注册的 genome id(register_uploaded_genome 返回 config 不含 id)
        gid = _find_registered_id(cfg["fasta"])
        if gff_path:
            cfg = attach_uploaded_gff(gid, gff_path)

        # 读取实际染色体名
        from rice_mutation.prediction_service import get_genome_chromosomes
        chroms = get_genome_chromosomes(gid, cfg)
    except RequestError as e:
        return _err(
            f"RegisterGenome failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except (FileNotFoundError, ValueError) as e:
        return _err(
            f"RegisterGenome failed: {e}", 400,
            detail={"request": _summarize_body(body)},
        )
    except Exception as e:
        traceback.print_exc()
        return _err(
            f"RegisterGenome failed: {e}", 500,
            detail={
                "error_type": e.__class__.__name__,
                "traceback": traceback.format_exc()[-2000:],
                "request": _summarize_body(body),
            },
        )

    return _ok(
        usage=_usage(0, 0),
        message=f"Custom genome registered: {gid}",
        result={
            "model": "OGR-Prediction",
            "genome": gid,
            "fasta_path": cfg["fasta"],
            "gff_path": cfg.get("gff") or "",
            "chromosomes": chroms,
            "chromosome_count": len(chroms),
            "note": "Files are referenced by path only (no upload). Use the returned 'genome' as the Genome field in Prediction/SNV requests.",
        },
    )


def _find_registered_id(fasta_path: str) -> str:
    """按 fasta 绝对路径在注册表中反查 genome id(注册时 id 由时间戳生成)。"""
    from rice_mutation.prediction_service import _UPLOADED_GENOMES
    fasta_abs = os.path.abspath(fasta_path)
    for gid, cfg in _UPLOADED_GENOMES.items():
        if os.path.abspath(cfg.get("fasta", "")) == fasta_abs:
            return gid
    raise ValueError(f"Uploaded genome not found in registry for {fasta_path}")


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
