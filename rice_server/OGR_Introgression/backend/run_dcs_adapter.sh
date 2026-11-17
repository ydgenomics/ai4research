#!/usr/bin/env bash
# DCS 适配层启动（rice_intro） — 与网页版后端独立进程，DCS 部署约定监听 5001
# POSIX 兼容（云平台 /bin/sh=dash，无 pipefail）：set -eu + [ ] + . 替代
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

# 加载 .env
if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
fi

LOG_DIR="$ROOT_DIR/backend/logs"
LOG_FILE="$LOG_DIR/dcs_adapter.nohup.log"
PID_FILE="$LOG_DIR/dcs_adapter.pid"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "DCS adapter already running (PID=$OLD_PID). Log: $LOG_FILE"
        exit 0
    fi
fi

PYTHON_BIN="${BACKEND_PYTHON_BIN:-python}"
nohup "$PYTHON_BIN" backend/dcs_adapter.py >> "$LOG_FILE" 2>&1 &
PID=$!

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[ERROR] DCS adapter failed to start. Check log: $LOG_FILE"
    exit 1
fi

echo "$PID" > "$PID_FILE"
echo "DCS adapter started (PID=$PID). Log: $LOG_FILE"