#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/frontend/logs"
LOG_FILE="$LOG_DIR/frontend.nohup.log"
PID_FILE="$LOG_DIR/frontend.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "Frontend not running (no PID file)."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
    rm -f "$PID_FILE"
    exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping frontend (PID=$PID)..."
    kill "$PID"
    for _ in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Grace period exceeded, sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Frontend stopped."
else
    echo "Frontend not running (stale PID)."
fi
rm -f "$PID_FILE"
