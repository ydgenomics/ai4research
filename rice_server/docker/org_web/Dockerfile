# ============================================================================
# org_web — 纯依赖环境镜像
#
# 仅安装 requirements.txt 中的 Python 包（rice-mut 与 rice-reg 共用依赖），
# 不包含任何项目代码 / 模型 / 数据 / 配置文件。
#
# 构建:
#   cd docker/org_web && docker build -t org_web:latest .
#
# 运行 (交互式进入环境):
#   docker run --gpus all -it org_web:latest
# ============================================================================

# 基础镜像: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9 (devel 含 nvcc, 用于编译 flash-attn)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# flash-attn 源码编译所需系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 只拷贝依赖清单(构建上下文中唯一的文件)
COPY requirements.txt /tmp/requirements.txt

# 一次性安装全部依赖
#   ninja/packaging: flash-attn 源码编译所需(--no-build-isolation 模式)
#   MAX_JOBS=4: 限制 flash-attn 并行编译, 避免 OOM
#   注意: packaging 是 transformers 的运行时依赖, 编译完成后不能卸载;
#         ninja 仅编译时需要, 可卸载(基础镜像自带, 卸载以减小体积)。
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ninja packaging \
    && MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation -r /tmp/requirements.txt \
    && pip uninstall -y ninja \
    && rm -rf /tmp/requirements.txt /tmp/pip*

# 构建时验证所有关键包可导入
RUN python -c "import torch, transformers, flash_attn, safetensors, gradio, fastapi, uvicorn, multipart, yaml, matplotlib, numpy, pandas, pyBigWig, pyfaidx, dotenv; print('ALL KEY PACKAGES IMPORT OK')"

CMD ["bash"]
