#!/usr/bin/env bash
# POSIX 兼容（云平台 /bin/sh=dash，无 pipefail）：set -eu + [ ] + . 替代
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
fi

LOG_DIR="$ROOT_DIR/frontend/logs"
LOG_FILE="$LOG_DIR/frontend.nohup.log"
PID_FILE="$LOG_DIR/frontend.pid"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Frontend already running (PID=$OLD_PID). Log: $LOG_FILE"
        exit 0
    fi
fi

nohup "$FRONTEND_PYTHON_BIN" frontend/app.py >> "$LOG_FILE" 2>&1 &
PID=$!

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[ERROR] Frontend failed to start. Check log: $LOG_FILE"
    exit 1
fi

echo "$PID" > "$PID_FILE"
echo "Frontend started (PID=$PID). Log: $LOG_FILE"
