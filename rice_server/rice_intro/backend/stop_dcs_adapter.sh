#!/usr/bin/env bash
# DCS 适配层停止（rice_intro）
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/backend/logs"
PID_FILE="$LOG_DIR/dcs_adapter.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "DCS adapter not running (no PID file)."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "$PID" ]; then
    rm -f "$PID_FILE"
    echo "DCS adapter not running (stale PID file removed)."
    exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping DCS adapter (PID=$PID)..."
    kill "$PID"
    i=0
    while kill -0 "$PID" 2>/dev/null && [ $i -lt 10 ]; do
        sleep 1
        i=$((i + 1))
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing (PID=$PID)..."
        kill -9 "$PID"
    fi
else
    echo "DCS adapter not running (stale PID)."
fi
rm -f "$PID_FILE"