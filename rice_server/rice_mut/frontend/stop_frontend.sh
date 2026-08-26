#!/usr/bin/env bash
# POSIX 兼容（云平台 /bin/sh=dash，无 pipefail）：set -eu + [ ] + $0 替代
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT_DIR/frontend/logs/frontend.pid"

if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Frontend stopped (PID=$PID)."
    else
        echo "Frontend not running (stale PID file)."
    fi
    rm -f "$PID_FILE"
else
    echo "Frontend not running (no PID file)."
fi
