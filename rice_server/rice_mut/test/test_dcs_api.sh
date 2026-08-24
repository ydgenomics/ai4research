#!/usr/bin/env bash
# ============================================================================
# DCS 适配层 API 测试 — 针对 backend/dcs_adapter.py 的 OpenAI 风格端点
#
# 端点:
#   GET  /api/aigress/openai/health
#   POST /api/aigress/openai/rice-mut         参考序列表达预测
#   POST /api/aigress/openai/rice-mut/snv     单碱基变异双轨预测
#
# 前置条件:dcs_adapter.py 服务已启动(默认 http://127.0.0.1:8001)
#   python backend/dcs_adapter.py
#
# 可选 --auth 模式(服务端配置了 DCS_API_KEY 时验证鉴权):
#   DCS_API_KEY=your-key bash test/test_dcs_api.sh --auth
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

# --auth 模式:校验 API Key。在 source .env 前捕获环境变量,
# 避免被 .env 中留空的 DCS_API_KEY= 覆盖。
CLI_DCS_API_KEY="${DCS_API_KEY:-}"
AUTH_KEY=""
if [[ "${1:-}" == "--auth" ]]; then
    AUTH_KEY="$CLI_DCS_API_KEY"
    if [[ -z "$AUTH_KEY" ]]; then
        echo -e "  \e[31m--auth 模式需要 DCS_API_KEY 环境变量或 .env 中配置:\e[0m"
        echo -e "      DCS_API_KEY=your-key bash test/test_dcs_api.sh --auth"
        exit 2
    fi
fi

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

BASE="${DCS_BASE_URL:-${BACKEND_API_URL:-http://127.0.0.1:8001}}"
PASS=0
FAIL=0

say()   { echo -e "$*"; }
ok()    { say "  \e[32mPASS\e[0m $1"; PASS=$((PASS + 1)); }
bad()   { say "  \e[31mFAIL\e[0m $1"; FAIL=$((FAIL + 1)); }

# json_field <json> <python-expr> — JSON 通过 stdin 传入(避免 argv 长度限制)
json_field() {
    printf '%s' "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$2" 2>/dev/null
}

# --auth 模式下 post_json 自动附带 Authorization: Bearer <DCS_API_KEY>
AUTH_ARGS=()
if [[ -n "$AUTH_KEY" ]]; then
    AUTH_ARGS=(-H "Authorization: Bearer $AUTH_KEY")
fi

# post_json <path> <data> — 只输出响应 JSON 本身(不含任何描述文本)
post_json() {
    curl -s -m 300 -X POST -H "Content-Type: application/json" "${AUTH_ARGS[@]}" -d "$2" "$BASE$1" 2>/dev/null
}

# 等待服务就绪
echo "==> 检查服务: $BASE/api/aigress/openai/health"
READY=""
for i in $(seq 1 60); do
    if curl -s -m 3 "$BASE/api/aigress/openai/health" 2>/dev/null | grep -q '"status":"ok"'; then
        READY=1; break
    fi
    sleep 1
done
if [[ -z "$READY" ]]; then
    say "  \e[31m服务未就绪。请先启动: python backend/dcs_adapter.py\e[0m"
    exit 1
fi
say "  服务已就绪。"

echo ""
echo "=========================================="
echo "  DCS Adapter API 测试"
echo "  Base URL: $BASE"
echo "=========================================="
echo ""

# ---------- 1. Health ----------
say "--- 1. Health ---"
RESP=$(curl -s -m 5 "$BASE/api/aigress/openai/health")
if echo "$RESP" | grep -q '"status":"ok"'; then
    GENOMES=$(json_field "$RESP" "d['genomes']")
    ok "health (genomes=$GENOMES)"
else
    bad "health → $RESP"
fi
echo ""

# ---------- 2. 参考预测:默认 full ----------
say "--- 2. 参考预测 full ---"
echo -n "  full 输出 ... "
RESP=$(post_json "/api/aigress/openai/rice-mut" \
    '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}')
if [[ -z "$RESP" ]]; then bad "无响应"; else
    STATUS=$(json_field "$RESP" "d['status']")
    PT=$(json_field "$RESP" "d['usage']['prompt_tokens']")
    CT=$(json_field "$RESP" "d['usage']['completion_tokens']")
    PS=$(json_field "$RESP" "d['result']['position_1based']['start']")
    PE=$(json_field "$RESP" "d['result']['position_1based']['end']")
    WL=$(json_field "$RESP" "d['result']['window_len']")
    CHR=$(json_field "$RESP" "d['result']['chromosome']")
    NTRACK=$(json_field "$RESP" "len(d['result']['values'])")
    [[ "$STATUS" == "200" ]] && ok "status=200" || bad "status=$STATUS"
    [[ "$PT" -ge 1 && "$CT" -ge 1 ]] && ok "usage prompt=$PT completion=$CT" || bad "usage 异常: $RESP"
    [[ "$CHR" == "Chr1" && "$PS" == "20716774" && "$PE" == "20749541" ]] \
        && ok "1-based 坐标保留: $CHR:$PS-$PE" || bad "坐标异常: $RESP"
    [[ "$WL" == "32768" ]] && ok "window_len=$WL" || bad "window_len=$WL"
    [[ "$NTRACK" -ge 1 ]] && ok "轨道数=$NTRACK" || bad "无轨道"
fi
echo ""

# ---------- 3. 参考预测:mean ----------
say "--- 3. 参考预测 mean ---"
echo -n "  mean 输出 ... "
RESP=$(post_json "/api/aigress/openai/rice-mut" \
    '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}')
if [[ -z "$RESP" ]]; then bad "无响应"; else
    VAL=$(json_field "$RESP" "list(d['result']['values'].values())[0].get('Leaf')")
    STATUS=$(json_field "$RESP" "d['status']")
    if [[ "$STATUS" == "200" && -n "$VAL" && "$VAL" != "None" ]]; then
        python3 -c "import sys; sys.exit(0 if isinstance($VAL, float) else 1)" 2>/dev/null \
            && ok "mean 输出标量: $VAL" || bad "mean 非标量: $RESP"
    else
        bad "mean 失败: $RESP"
    fi
fi
echo ""

# ---------- 4. 参考预测:downsample ----------
say "--- 4. 参考预测 downsample ---"
echo -n "  downsample max_points=1024 ... "
RESP=$(post_json "/api/aigress/openai/rice-mut" \
    '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}')
if [[ -z "$RESP" ]]; then bad "无响应"; else
    LEN=$(json_field "$RESP" "len(list(d['result']['values'].values())[0].get('Leaf',[]))")
    STATUS=$(json_field "$RESP" "d['status']")
    if [[ "$STATUS" == "200" && "$LEN" == "1024" ]]; then
        ok "downsample 长度=1024"
    else
        bad "downsample 失败(len=$LEN): $RESP"
    fi
fi
echo ""

# ---------- 5. SNV 预测 ----------
say "--- 5. SNV 预测 ---"
echo -n "  SNV T@20731844 ... "
RESP=$(post_json "/api/aigress/openai/rice-mut/snv" \
    '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T"}')
if [[ -z "$RESP" ]]; then bad "无响应"; else
    STATUS=$(json_field "$RESP" "d['status']")
    RB=$(json_field "$RESP" "d['result']['ref_base']")
    SB=$(json_field "$RESP" "d['result']['snv_base']")
    SI=$(json_field "$RESP" "d['result']['snv_index_1based']")
    HAS_REF=$(json_field "$RESP" "'ref_values' in d['result'] and 'mut_values' in d['result']")
    CT=$(json_field "$RESP" "d['usage']['completion_tokens']")
    [[ "$STATUS" == "200" ]] && ok "status=200" || bad "status=$STATUS"
    [[ "$HAS_REF" == "True" ]] && ok "含 ref_values + mut_values 双轨" || bad "缺双轨: $RESP"
    [[ "$SI" == "20731844" ]] && ok "snv_index_1based=$SI" || bad "snv_index=$SI"
    [[ -n "$RB" && -n "$SB" ]] && ok "ref_base=$RB → snv_base=$SB" || bad "碱基缺失: $RESP"
    [[ "$CT" -ge 1 ]] && ok "completion_tokens=$CT (双轨合计)" || bad "usage 异常"
fi
echo ""

# ---------- 6. 错误处理 ----------
say "--- 6. 错误处理 ---"

# 6a. 缺 start
echo -n "  缺 start ... "
RESP=$(post_json "/api/aigress/openai/rice-mut" '{"genome":"osa1_r7","chromosome":"chr01"}')
S=$(json_field "$RESP" "d['status']")
[[ "$S" == "400" ]] && ok "缺 start → 400" || bad "缺 start → status=$S"

# 6b. 非法 output_format
echo -n "  非法 output_format ... "
RESP=$(post_json "/api/aigress/openai/rice-mut" \
    '{"genome":"osa1_r7","start":1000,"output_format":"bogus"}')
S=$(json_field "$RESP" "d['status']")
[[ "$S" == "400" ]] && ok "非法 output_format → 400" || bad "非法 output_format → status=$S"

# 6c. 非法 snv_base
echo -n "  非法 snv_base ... "
RESP=$(post_json "/api/aigress/openai/rice-mut/snv" \
    '{"genome":"osa1_r7","start":1000,"snv_index":1000,"snv_base":"X"}')
S=$(json_field "$RESP" "d['status']")
[[ "$S" == "400" ]] && ok "非法 snv_base → 400" || bad "非法 snv_base → status=$S"

# 6d. 非法 JSON
echo -n "  非法 JSON ... "
S=$(curl -s -m 5 -X POST -H "Content-Type: application/json" "${AUTH_ARGS[@]}" -d 'not-json' \
    "$BASE/api/aigress/openai/rice-mut" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])" 2>/dev/null)
[[ "$S" == "400" ]] && ok "非法 JSON → 400" || bad "非法 JSON → status=$S"

# 6e. 未知基因组
echo -n "  未知基因组 ... "
RESP=$(post_json "/api/aigress/openai/rice-mut" \
    '{"genome":"INVALID","start":1000}')
S=$(json_field "$RESP" "d['status']")
[[ -n "$S" && "$S" != "200" ]] && ok "未知基因组 → $S" || bad "未知基因组 → status=$S"

echo ""

# ---------- 7. API Key 鉴权(--auth 模式;未启用则跳过) ----------
if [[ -n "$AUTH_KEY" ]]; then
    say "--- 7. API Key 鉴权 (AUTH_KEY=${AUTH_KEY:0:3}***) ---"

    # 7a. 无 header → 401
    echo -n "  无 header ... "
    RESP=$(curl -s -m 300 -X POST -H "Content-Type: application/json" \
        -d '{"genome":"osa1_r7","start":1000}' "$BASE/api/aigress/openai/rice-mut")
    S=$(json_field "$RESP" "d['status']")
    [[ "$S" == "401" ]] && ok "无 header → 401" || bad "无 header → status=$S"

    # 7b. 错误 Bearer → 401
    echo -n "  错误 Bearer ... "
    RESP=$(curl -s -m 300 -X POST -H "Content-Type: application/json" \
        -H "Authorization: Bearer wrong-key" \
        -d '{"genome":"osa1_r7","start":1000}' "$BASE/api/aigress/openai/rice-mut")
    S=$(json_field "$RESP" "d['status']")
    [[ "$S" == "401" ]] && ok "错误 Bearer → 401" || bad "错误 Bearer → status=$S"

    # 7c. 正确 Bearer → 200
    echo -n "  正确 Bearer ... "
    RESP=$(post_json "/api/aigress/openai/rice-mut" \
        '{"genome":"osa1_r7","start":1000}')
    S=$(json_field "$RESP" "d['status']")
    [[ "$S" == "200" ]] && ok "正确 Bearer → 200" || bad "正确 Bearer → status=$S"

    # 7d. X-API-Key → 200
    echo -n "  X-API-Key ... "
    RESP=$(curl -s -m 300 -X POST -H "Content-Type: application/json" \
        -H "X-API-Key: $AUTH_KEY" \
        -d '{"genome":"osa1_r7","start":1000}' "$BASE/api/aigress/openai/rice-mut")
    S=$(json_field "$RESP" "d['status']")
    [[ "$S" == "200" ]] && ok "X-API-Key → 200" || bad "X-API-Key → status=$S"

    # 7e. health 免鉴权
    echo -n "  health 免鉴权 ... "
    S=$(curl -s -m 5 "$BASE/api/aigress/openai/health" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])" 2>/dev/null)
    [[ "$S" == "ok" ]] && ok "health 免鉴权" || bad "health → $S"

    # 7f. SNV 无 header → 401
    echo -n "  SNV 无 header ... "
    RESP=$(curl -s -m 300 -X POST -H "Content-Type: application/json" \
        -d '{"genome":"osa1_r7","start":1000,"snv_index":1000,"snv_base":"A"}' "$BASE/api/aigress/openai/rice-mut/snv")
    S=$(json_field "$RESP" "d['status']")
    [[ "$S" == "401" ]] && ok "SNV 无 header → 401" || bad "SNV 无 header → status=$S"

    echo ""
else
    say "--- 7. 鉴权: 跳过(DCS_API_KEY 为空不校验;用 DCS_API_KEY=xxx bash test/test_dcs_api.sh --auth 验证) ---"
    echo ""
fi

# ---------- 汇总 ----------
echo "=========================================="
if [[ $FAIL -eq 0 ]]; then
    say "  \e[32mAll $PASS tests passed.\e[0m"
else
    say "  \e[31m$FAIL/$((PASS + FAIL)) tests failed.\e[0m"
fi
echo "=========================================="

exit $FAIL
