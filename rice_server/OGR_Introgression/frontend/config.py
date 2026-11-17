"""前端配置 —— 读取 .env，扫描 GENOME_*_FASTA 构建基因组列表。

前端只做 UI + HTTP 调用后端 + iframe Plotly 渲染，不加载模型。
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # rice-intro/


def _load_env_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


ENV = _load_env_file(ROOT_DIR / ".env")
# 全量注入 .env 到 os.environ（setdefault：不覆盖进程已有变量），
# 使 GENOME_*_FASTA / BACKEND_API_URL 等对后续扫描可见。
for _k, _v in ENV.items():
    os.environ.setdefault(_k, _v)

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = os.getenv("BACKEND_PORT", "5001")
# 优先使用显式 API URL（127.0.0.1 环回，跨容器场景由部署方配置）
BACKEND_URL = os.getenv(
    "BACKEND_API_URL",
    f"http://{BACKEND_HOST}:{BACKEND_PORT}",
)

FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "5000"))

STATIC_DIR = ROOT_DIR / "frontend" / "static"
PLOTLY_JS = STATIC_DIR / "plotly.min.js"


def get_frontend_host() -> str:
    return FRONTEND_HOST


def _fasta_envs() -> dict:
    """扫描 ``GENOME_*_FASTA`` 环境变量 -> {钥匙: (工具名, 绝对路径)}。"""
    out = {}
    for key in sorted(os.environ):
        if key.startswith("GENOME_") and key.endswith("_FASTA"):
            display = key[len("GENOME_"):-len("_FASTA")]
            path = os.environ[key]
            if path and os.path.isfile(path):
                out[display] = (display, os.path.abspath(path))
    return out


def get_genome_configs() -> dict:
    """返回 {钥匙: {"name": 展示名, "fasta": 路径}} —— 内置基因组列表。"""
    return {
        key: {"name": name, "fasta": path}
        for key, (name, path) in _fasta_envs().items()
    }


def get_backend_base_url() -> str:
    return BACKEND_URL


def get_frontend_port() -> int:
    return FRONTEND_PORT


def get_plotly_js_path() -> Path | None:
    return PLOTLY_JS


def get_plotly_js_local_url() -> str | None:
    """plotly.min.js 的 iframe 可访问 URL。

    经前端反向代理 /backend/* -> 后端 /static-files/ 根文件系统挂载，
    浏览器只接触前端端口（同源），避免 CORS。URL 格式与 rice_mut 一致：
    /backend/static-files{STATIC_DIR_ABS}/plotly.min.js
    """
    if PLOTLY_JS.exists():
        return f"/backend/static-files{STATIC_DIR_ABS}/plotly.min.js"
    return None


# 供 iframe 引用的 static 根（与 rice_mut 一致：STATIC_DIR_ABS）
STATIC_DIR_ABS = str(STATIC_DIR)