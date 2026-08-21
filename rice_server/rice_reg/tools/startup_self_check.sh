#!/usr/bin/env bash
# ============================================================================
# Rice-Reg Server — 启动自检脚本
# 检查端口、模型文件、基因组文件、ATAC 文件、Python 解释器等
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo -e "  [${GREEN}PASS${NC}] $1"
    PASS=$((PASS + 1))
}

check_fail() {
    echo -e "  [${RED}FAIL${NC}] $1"
    FAIL=$((FAIL + 1))
}

check_warn() {
    echo -e "  [${YELLOW}WARN${NC}] $1"
    WARN=$((WARN + 1))
}

echo ""
echo "=========================================="
echo "  Rice-Reg Server — 启动自检"
echo "=========================================="
echo ""

# --- 1. 加载 .env ---
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    check_pass ".env file loaded: $ENV_FILE"
else
    check_fail ".env file not found: $ENV_FILE"
fi

echo ""

# --- 2. 端口检查 ---
echo "--- Port Check ---"
for port_var in "BACKEND_PORT:7001" "FRONTEND_PORT:7000"; do
    var_name="${port_var%%:*}"
    default_port="${port_var##*:}"
    port="${!var_name:-$default_port}"
    if ss -tlnp | grep -q ":$port "; then
        check_fail "Port $port ($var_name) is already in use"
    else
        check_pass "Port $port ($var_name) is available"
    fi
done

echo ""

# --- 3. Python 解释器检查 ---
echo "--- Python Interpreter Check ---"
for py_var in "BACKEND_PYTHON_BIN" "FRONTEND_PYTHON_BIN"; do
    py_path="${!py_var:-}"
    if [[ -z "$py_path" ]]; then
        check_fail "$py_var is not set"
    elif [[ -x "$py_path" ]]; then
        ver=$("$py_path" --version 2>&1)
        check_pass "$py_var: $py_path ($ver)"
    else
        check_fail "$py_var: $py_path not found or not executable"
    fi
done

echo ""

# --- 4. 模型文件检查 ---
echo "--- Model Files Check ---"
if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
    if [[ -d "$BASE_MODEL_PATH" ]]; then
        check_pass "BASE_MODEL_PATH: $BASE_MODEL_PATH"
    else
        check_fail "BASE_MODEL_PATH not found: $BASE_MODEL_PATH"
    fi
else
    check_warn "BASE_MODEL_PATH not set"
fi

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    if [[ -f "$CHECKPOINT_PATH" ]]; then
        check_pass "CHECKPOINT_PATH: $CHECKPOINT_PATH"
    else
        check_fail "CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
    fi
else
    check_warn "CHECKPOINT_PATH not set"
fi

echo ""

# --- 5. 基因组文件检查 ---
echo "--- Genome Files Check ---"
genome_count=0
for key in $(compgen -v | grep "^GENOME_.*_FASTA$"); do
    fasta="${!key}"
    genome_id="${key#GENOME_}"
    genome_id="${genome_id%_FASTA}"
    fai_key="GENOME_${genome_id}_FAI"
    gff_key="GENOME_${genome_id}_GFF"
    fai="${!fai_key:-}"
    gff="${!gff_key:-}"

    if [[ -f "$fasta" ]]; then
        check_pass "[$genome_id] FASTA: $fasta"
    else
        check_fail "[$genome_id] FASTA not found: $fasta"
    fi

    if [[ -n "$fai" ]]; then
        if [[ -f "$fai" ]]; then
            check_pass "[$genome_id] FAI: $fai"
        else
            check_fail "[$genome_id] FAI not found: $fai"
        fi
    fi

    if [[ -n "$gff" ]]; then
        if [[ -f "$gff" ]]; then
            check_pass "[$genome_id] GFF: $gff"
        else
            check_fail "[$genome_id] GFF not found: $gff"
        fi
    fi
    genome_count=$((genome_count + 1))
done

if [[ $genome_count -eq 0 ]]; then
    check_warn "No genome configurations found (GENOME_*_FASTA)"
fi

echo ""

# --- 6. ATAC 文件检查 ---
echo "--- ATAC Files Check ---"
atac_count=0
for key in $(compgen -v | grep "^ATAC_PATH_"); do
    path="${!key}"
    atac_id="${key#ATAC_PATH_}"
    if [[ -f "$path" ]]; then
        check_pass "[$atac_id] $path"
    else
        check_fail "[$atac_id] not found: $path"
    fi
    atac_count=$((atac_count + 1))
done

if [[ $atac_count -eq 0 ]]; then
    check_warn "No ATAC paths configured (ATAC_PATH_*)"
fi

echo ""

# --- 7. 缓存目录检查 ---
echo "--- Cache Directories Check ---"
for cache_var in "BACKEND_UPLOADED_ATAC" "BACKEND_PREDICTION_CACHE"; do
    dir="${!cache_var:-}"
    if [[ -n "$dir" ]]; then
        mkdir -p "$dir"
        check_pass "$cache_var: $dir"
    else
        check_warn "$cache_var not set"
    fi
done

echo ""

# --- Summary ---
echo "=========================================="
echo -e "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$WARN warnings${NC}"
echo "=========================================="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
