#!/usr/bin/env bash
# ============================================================================
# Pre-generate genome-wide bigWig tracks for the built-in osa1_r7 genome.
#
# Runs scripts/pregen_bigwigs.py: full-genome sliding-window inference
# (32768 bp windows / 16384 bp hop) on GPU, then merges everything into one
# genome-wide .bw per track.  At query time the backend reads these files
# directly — the model is NOT called for reference queries on osa1_r7.
#
# Usage:
#   ./scripts/pregen_bigwigs.sh                      # default: use all GPUs
#   ./scripts/pregen_bigwigs.sh --workers 2          # 2 parallel workers
#   ./scripts/pregen_bigwigs.sh --chrom Chr1         # single chromosome
#   ./scripts/pregen_bigwigs.sh --resume             # keep finished windows
#   ./scripts/pregen_bigwigs.sh --merge-only         # only merge existing windows
#
# Extra flags after the script name are forwarded to pregen_bigwigs.py, e.g.:
#   ./scripts/pregen_bigwigs.sh --workers 4 --hop 8192 --resume
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/pregen_bigwigs.log"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

# --- Load .env for model/genome paths (like run_backend.sh) ---
if [ -f "$ROOT_DIR/.env" ]; then
    set -a; . "$ROOT_DIR/.env"; set +a
fi

PYTHON_BIN="${BACKEND_PYTHON_BIN:-python}"

# --- Auto-detect GPU count (fallback to 1) ---
DETECTED_GPUS=1
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DETECTED_GPUS="$(nvidia-smi -L | wc -l)"
fi

# --- Default workers = number of GPUs (bounded, user can override) ---
WORKERS="${PREGEN_WORKERS:-$DETECTED_GPUS}"

# Collect --workers if given explicitly
for arg in "$@"; do
    case "$arg" in
        --workers=*) WORKERS="${arg#*=}" ;;
        --workers) : ;;  # value is the next arg; handled below
    esac
done
# handle `--workers N` (value as separate arg)
prev=""
for arg in "$@"; do
    if [ "$prev" = "--workers" ]; then
        WORKERS="$arg"
        break
    fi
    prev="$arg"
done

echo "=============================================="
echo "  Pre-gen genome-wide bigWig for osa1_r7"
echo "=============================================="
echo "  GPU(s) detected : $DETECTED_GPUS"
echo "  workers         : $WORKERS"
echo "  python          : $PYTHON_BIN"
echo "  log             : $LOG_FILE"
echo "  extra args      : $*"
echo ""

export PREGEN_LOG_FILE="$LOG_FILE"

if [ "$WORKERS" -gt 1 ] && [[ "$*" != *"--merge-only"* ]]; then
    echo ">> Launching $WORKERS workers (one per GPU / split by chromosome)"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/pregen_bigwigs.py" --workers "$WORKERS" --gpus "$WORKERS" "$@" 2>&1 | tee -a "$LOG_FILE"
    rc="${PIPESTATUS[0]:-0}"
else
    echo ">> Single worker"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/pregen_bigwigs.py" "$@" 2>&1 | tee -a "$LOG_FILE"
    rc="${PIPESTATUS[0]:-0}"
fi

if [ "$rc" -ne 0 ]; then
    echo "!! Pregen failed (exit $rc). Check $LOG_FILE" >&2
    exit "$rc"
fi

echo ""
echo ">> Done. Merged genome-wide files under:"
echo "   $ROOT_DIR/cache/pregen/osa1_r7/"