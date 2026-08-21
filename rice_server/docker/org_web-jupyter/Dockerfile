# ============================================================================
# org_web-jupyter — 在 org_web 依赖环境基础上安装 JupyterLab
#
# 构建:
#   cd docker/org_web-jupyter && docker build -t org_web:jupyter .
#
# 运行 (开发调试用):
#   docker run --gpus all -it --rm -p 8888:8888 \
#       -v $PWD:/workspace \
#       org_web:jupyter
#   浏览器打开 http://localhost:8888  密码/token: orgweb
# ============================================================================

# 基础镜像: 已包含全部运行依赖的 org_web (见 ../org_web/Dockerfile)
FROM org_web:latest

# 安装 JupyterLab + Notebook + IPython 内核
#   ipykernel: 让 notebook 能选到 python3 内核
RUN pip install --no-cache-dir \
        jupyterlab \
        notebook \
        ipykernel \
    && python -m ipykernel install --name python3 --display-name "Python 3 (org_web)" \
    && jupyter lab --version

# Jupyter 访问令牌(默认 orgweb, 可用 --env JUPYTER_TOKEN=xxx 覆盖;
# 生产环境务必用强密码)
ENV JUPYTER_TOKEN=orgweb

# 默认工作目录(挂载宿主代码用)
WORKDIR /workspace

# JupyterLab 端口
EXPOSE 8888

# 启动 JupyterLab: root 用户 + 允许外部访问 + 不自动开浏览器 + 固定 token
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=orgweb", "--ServerApp.root_dir=/workspace"]
