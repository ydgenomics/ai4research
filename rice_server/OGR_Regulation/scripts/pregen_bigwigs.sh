#!/usr/bin/env bash
# ============================================================================
# Pre-generate genome-wide RNA-seq (+/-) bigWig tracks for RiceReg.
#
# Strategy B: per-chromosome .npz parts are written by GPU workers, then a
# fast merge produces exactly TWO genome-wide files per (genome, ATAC):
#     cache/pregen/<Genome>__<Atac>_plus.bw
#     cache/pregen/<Genome>__<Atac>_minus.bw
# No per-window .bw files are ever produced.
#
# Usage:
#   ./scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1
#   ./scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1 --workers 2
#   ./scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1 --chrom Chr1
#   ./scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1 --resume
#   ./scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1 --merge-only
#
# Extra flags after the script name are forwarded to pregen_bigwigs.py.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/pregen_bigwigs.log"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

# --- Load .env for model/genome/ATAC paths (like run_backend.sh) ---
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
prev=""
for arg in "$@"; do
    if [ "$prev" = "--workers" ]; then
        WORKERS="$arg"
        break
    fi
    case "$arg" in
        --workers=*) WORKERS="${arg#*=}" ;;
    esac
    prev="$arg"
done

echo "=============================================="
echo "  Pre-gen genome-wide bigWig (rice_reg)"
echo "=============================================="
echo "  GPU(s) detected : $DETECTED_GPUS"
echo "  workers         : $WORKERS"
echo "  python          : $PYTHON_BIN"
echo "  log             : $LOG_FILE"
echo "  extra args      : $*"
echo ""

export PREGEN_LOG_FILE="$LOG_FILE"

if [ "$WORKERS" -gt 1 ] && [[ "$*" != *"--merge-only"* ]] && [[ "$*" != *"--no-merge"* ]]; then
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
echo "   $ROOT_DIR/cache/pregen/<Genome>__<Atac>_{plus,minus}.bw"
