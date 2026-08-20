#!/usr/bin/env bash
# ============================================================================
# Rice-Mutation Server — API 测试脚本
# 测试后端 API 端点的基本功能
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

BACKEND_URL="${BACKEND_API_URL:-http://127.0.0.1:8001}"
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
echo "  Rice-Mutation Server — API 测试"
echo "  Backend URL: $BACKEND_URL"
echo "=========================================="
echo ""

# --- 1. Health check ---
echo "--- Health Check ---"
test_endpoint "GET /health" "GET" "/health"

echo ""

# --- 2. Metadata endpoints ---
echo "--- Metadata ---"
test_endpoint "GET /genomes" "GET" "/genomes"
test_endpoint "GET /assays" "GET" "/assays"
test_endpoint "GET /biosamples" "GET" "/biosamples"

echo ""

# --- 3. Genome chromosomes endpoint (built-in genome) ---
echo "--- Genome chromosomes ---"
test_endpoint "GET /genomes/osa1_r7/chromosomes" "GET" "/genomes/osa1_r7/chromosomes"

echo ""

# --- 4. Predict (invalid genome) ---
echo "--- Predict (invalid genome) ---"
test_endpoint "POST /predict (bad genome)" "POST" "/predict" \
    '{"genome":"INVALID","chromosome":"Chr1","start":0,"end":32000}' \
    400

echo ""

# --- 5. Predict (valid request, alias chromosome) ---
echo "--- Predict (valid request) ---"
test_endpoint "POST /predict (valid)" "POST" "/predict" \
    '{"genome":"osa1_r7","chromosome":"chr01","start":0,"end":32000,"biosample_names":["NIP_CSQ"]}' \
    200

echo ""

# --- 6. Custom genome upload (auto .fai) ---
echo "--- Upload custom genome (valid .fa) ---"
UPLOAD_RESULT=$(mktemp -d)/tiny.fa
printf '>Chr1\n%s\n>Chr2\n%s\n' "$(printf 'A%.0s' $(seq 1 100))" "$(printf 'C%.0s' $(seq 1 100))" > "$UPLOAD_RESULT"
test_endpoint_upload() {
    local desc="$1" file="$2" expect_status="${3:-200}"
    echo -n "  Testing $desc ... "
    status=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -F "file=@$file" \
        "$BACKEND_URL/uploadFasta" 2>/dev/null || echo "000")
    if [[ "$status" == "$expect_status" ]]; then
        echo -e "\e[32mPASS\e[0m (HTTP $status)"
        PASS=$((PASS + 1))
    else
        echo -e "\e[31mFAIL\e[0m (expected $expect_status, got $status)"
        FAIL=$((FAIL + 1))
    fi
}
test_endpoint_upload "POST /uploadFasta (tiny genome)" "$UPLOAD_RESULT"

echo ""

# --- 7. SNV prediction (valid) ---
echo "--- SNV prediction ---"
echo -n "  Testing POST /predict/snv (valid) ... "
SNV_RESP=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"genome":"osa1_r7","chromosome":"chr01","start":0,"end":32000,"biosample_names":["NIP_CSQ"],"snv_index":16000,"snv_base":"A"}' \
    "$BACKEND_URL/predict/snv" 2>/dev/null || echo '{"error":"curl failed"}')
if echo "$SNV_RESP" | grep -q '"success": *true'; then
    SNV_ID=$(echo "$SNV_RESP" | sed -n 's/.*"snv_id":"\([^"]*\)".*/\1/p')
    if [[ -z "$SNV_ID" ]]; then
        echo -e "\e[31mFAIL\e[0m (no snv_id in response)"
        FAIL=$((FAIL + 1))
    else
        echo -e "\e[32mPASS\e[0m (snv_id=$SNV_ID)"
        PASS=$((PASS + 1))
    fi
else
    echo -e "\e[31mFAIL\e[0m (no success:true — $(echo "$SNV_RESP" | head -c 200))"
    FAIL=$((FAIL + 1))
fi

echo -n "  Testing POST /predict/snv (index out of window) ... "
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"genome":"osa1_r7","chromosome":"chr01","start":0,"end":32000,"snv_index":40000,"snv_base":"A"}' \
    "$BACKEND_URL/predict/snv" 2>/dev/null || echo "000")
if [[ "$status" == "400" ]]; then
    echo -e "\e[32mPASS\e[0m (HTTP $status)"
    PASS=$((PASS + 1))
else
    echo -e "\e[31mFAIL\e[0m (expected 400, got $status)"
    FAIL=$((FAIL + 1))
fi

echo -n "  Testing POST /predict/snv (invalid base) ... "
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{"genome":"osa1_r7","chromosome":"chr01","start":0,"end":32000,"snv_index":10,"snv_base":"X"}' \
    "$BACKEND_URL/predict/snv" 2>/dev/null || echo "000")
if [[ "$status" == "400" ]]; then
    echo -e "\e[32mPASS\e[0m (HTTP $status)"
    PASS=$((PASS + 1))
else
    echo -e "\e[31mFAIL\e[0m (expected 400, got $status)"
    FAIL=$((FAIL + 1))
fi

echo ""

# --- 8. SNV region stats ---
echo "--- SNV region stats ---"
if [[ -n "${SNV_ID:-}" ]]; then
    test_endpoint "POST /predict/snv/stat (valid)" "POST" "/predict/snv/stat" \
        "{\"snv_id\":\"$SNV_ID\",\"region_start\":5000,\"region_end\":20000}" \
        200
else
    echo "  Skipping stat valid test (no snv_id from previous step)"
    FAIL=$((FAIL + 1))
fi

test_endpoint "POST /predict/snv/stat (unknown id)" "POST" "/predict/snv/stat" \
    '{"snv_id":"snv_nonexistent","region_start":0,"region_end":100}' \
    404

test_endpoint "POST /predict/snv/stat (bad region)" "POST" "/predict/snv/stat" \
    "{\"snv_id\":\"$SNV_ID\",\"region_start\":20000,\"region_end\":5000}" \
    400

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
