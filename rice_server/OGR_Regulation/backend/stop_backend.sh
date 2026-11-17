#!/usr/bin/env bash
# POSIX 兼容（云平台 /bin/sh=dash，无 pipefail）：set -eu + [ ] + $0；{1..10}→seq 替代
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/backend/logs"
LOG_FILE="$LOG_DIR/backend.nohup.log"
PID_FILE="$LOG_DIR/backend.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Backend not running (no PID file)."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "$PID" ]; then
    rm -f "$PID_FILE"
    exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping backend (PID=$PID)..."
    kill "$PID"
    # 等待优雅退出（最多 10 秒）
    for _ in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Grace period exceeded, sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Backend stopped."
else
    echo "Backend not running (stale PID)."
fi
rm -f "$PID_FILE"
