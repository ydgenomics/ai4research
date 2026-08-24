#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT_DIR/backend/logs/backend.pid"

if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Backend stopped (PID=$PID)."
    else
        echo "Backend not running (stale PID file)."
    fi
    rm -f "$PID_FILE"
else
    echo "Backend not running (no PID file)."
fi
