#!/usr/bin/env bash
# ============================================================================
# Rice-Reg Server — API 测试脚本
# 测试后端 API 端点的基本功能
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

BACKEND_URL="${BACKEND_API_URL:-http://127.0.0.1:7001}"
PASS=0
FAIL=0

test_endpoint() {
    local desc="$1"
    local method="$2"
    local endpoint="$3"
    local data="${4:-}"
    local expect_status="${5:-200}"

    echo -n "  Testing $desc ... "

    if [[ "$method" == "GET" ]]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL$endpoint" 2>/dev/null || echo "000")
    else
        status=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BACKEND_URL$endpoint" 2>/dev/null || echo "000")
    fi

    if [[ "$status" == "$expect_status" ]]; then
        echo -e "\e[32mPASS\e[0m (HTTP $status)"
        PASS=$((PASS + 1))
    else
        echo -e "\e[31mFAIL\e[0m (expected $expect_status, got $status)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "=========================================="
echo "  Rice-Reg Server — API 测试"
echo "  Backend URL: $BACKEND_URL"
echo "=========================================="
echo ""

# --- 1. Health check ---
echo "--- Health Check ---"
test_endpoint "GET /health" "GET" "/health"

echo ""

# --- 2. Predict (no ATAC source — should fail 400) ---
echo "--- Predict (missing ATAC source) ---"
test_endpoint "POST /predict/rice-reg (no ATAC)" "POST" "/predict/rice-reg" \
    '{"genome":"MH63RS3","chromosome":"chr1","start":0,"end":32678}' \
    400

echo ""

# --- 3. Predict (invalid genome — should fail 400) ---
echo "--- Predict (invalid genome) ---"
test_endpoint "POST /predict/rice-reg (bad genome)" "POST" "/predict/rice-reg" \
    '{"genome":"INVALID","chromosome":"chr1","start":0,"end":32678,"atac_source":"SAM2_MH63_1"}' \
    400

echo ""

# --- 4. Predict (valid request — should succeed if backend is properly configured) ---
echo "--- Predict (valid request) ---"
test_endpoint "POST /predict/rice-reg (valid)" "POST" "/predict/rice-reg" \
    '{"genome":"MH63RS3","chromosome":"chr1","start":0,"end":32678,"atac_source":"SAM2_MH63_1"}' \
    200

echo ""

# --- Summary ---
echo "=========================================="
if [[ $FAIL -eq 0 ]]; then
    echo -e "  \e[32mAll $PASS tests passed.\e[0m"
else
    echo -e "  \e[31m$FAIL/$((PASS + FAIL)) tests failed.\e[0m"
fi
echo "=========================================="

exit $FAIL
