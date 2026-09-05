#!/usr/bin/env bash
# 启动前自检：检查 .env、Python 解释器、模型文件、基因组文件、端口占用
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "[FAIL] .env not found: $ENV_FILE (copy from .env.example)"
    exit 1
fi
set -a; . "$ENV_FILE"; set +a

echo "== 1. Python 解释器 =="
if [ -x "$BACKEND_PYTHON_BIN" ]; then
    echo "[OK] backend  : $BACKEND_PYTHON_BIN"
else
    echo "[FAIL] backend python not found: $BACKEND_PYTHON_BIN"
fi
if [ -x "$FRONTEND_PYTHON_BIN" ]; then
    echo "[OK] frontend : $FRONTEND_PYTHON_BIN"
else
    echo "[FAIL] frontend python not found: $FRONTEND_PYTHON_BIN"
fi

echo "== 2. 模型文件 =="
if [ -d "$BASE_MODEL_PATH" ]; then
    echo "[OK] base model: $BASE_MODEL_PATH"
else
    echo "[FAIL] base model dir missing: $BASE_MODEL_PATH"
fi
CKPT_FILE="$CHECKPOINT_PATH"
if [ -d "$CHECKPOINT_PATH" ]; then
    CKPT_FILE="$CHECKPOINT_PATH/model.safetensors"
fi
if [ -f "$CKPT_FILE" ]; then
    echo "[OK] checkpoint: $CKPT_FILE"
else
    echo "[FAIL] checkpoint file missing: $CKPT_FILE"
fi

echo "== 3. 基因组文件 =="
for key in $(env | grep -oE '^GENOME_[A-Z0-9_]+_FASTA' || true); do
    path="${!key}"
    if [ -f "$path" ]; then
        echo "[OK] $key -> $path"
    else
        echo "[FAIL] $key -> $path (missing)"
    fi
done

echo "== 4. 端口占用 =="
for pair in "BACKEND_PORT:$BACKEND_PORT" "FRONTEND_PORT:$FRONTEND_PORT"; do
    name="${pair%%:*}"
    port="${pair#*:}"
    if (command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ":$port ") || \
       (command -v netstat >/dev/null 2>&1 && netstat -ltn | grep -q ":$port "); then
        echo "[WARN] $name port $port already in use; consider changing in .env"
    else
        echo "[OK] $name port $port free"
    fi
done

echo "== 5. 前端静态资源 =="
if [ -f "$ROOT_DIR/frontend/static/plotly.min.js" ]; then
    echo "[OK] plotly.min.js present"
else
    echo "[FAIL] plotly.min.js missing (前端 iframe 无法离线渲染)"
fi

echo ""
echo "自检完成。可运行: bash $ROOT_DIR/backend/run_backend.sh && bash $ROOT_DIR/frontend/run_frontend.sh"