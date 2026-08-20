#!/usr/bin/env bash
# ============================================================================
# 网页端(Gradio)测试 — rice-mut 前端服务
#
# 前置条件:
#   1. 后端已启动:  bash backend/run_backend.sh      (默认 8001)
#   2. 前端已启动:  bash frontend/run_frontend.sh    (默认 8000)
#
# 测试内容:
#   - 前端 HTTP 可达性 & Gradio 配置加载
#   - 前端关键 UI 组件存在(基因组下拉框 / 染色体下拉框 / 启动按钮等)
#   - 后端健康检查(前端依赖后端,一损俱损)
#   - 前后端日志无致命错误
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:${FRONTEND_PORT:-8000}}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT:-8001}}"
PASS=0
FAIL=0

say() { echo -e "$*"; }
ok()  { say "  \e[32mPASS\e[0m $1"; PASS=$((PASS + 1)); }
bad() { say "  \e[31mFAIL\e[0m $1"; FAIL=$((FAIL + 1)); }

echo ""
echo "=========================================="
echo "  Rice-Mutation 网页端测试"
echo "  Frontend: $FRONTEND_URL"
echo "  Backend : $BACKEND_URL"
echo "=========================================="
echo ""

# ---------- 1. 前端可达性 ----------
say "--- 1. 前端可达性 ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 10 "$FRONTEND_URL/" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    ok "前端首页 HTTP 200"
else
    bad "前端首页 HTTP $HTTP_CODE (请先运行 frontend/run_frontend.sh)"
fi
echo ""

# ---------- 2. Gradio 配置 ----------
say "--- 2. Gradio 配置加载 ---"
CONFIG=$(curl -s -m 10 "$FRONTEND_URL/config" 2>/dev/null || echo "")
if echo "$CONFIG" | grep -q '"components"'; then
    ok "GET /config 返回组件配置"
    NCOMP=$(echo "$CONFIG" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('components',[])))" 2>/dev/null)
    [[ -n "$NCOMP" ]] && ok "组件数=$NCOMP" || say "  (组件数解析失败,跳过)"
else
    bad "GET /config 失败(前端可能未就绪或非 Gradio)"
fi
echo ""

# ---------- 3. 关键 UI 组件 ----------
say "--- 3. 关键 UI 组件存在性 ---"
if [[ -n "${CONFIG:-}" ]] && echo "$CONFIG" | grep -q '"components"'; then
    # 检查组件 label/type 关键字(区分大小写;旧版 Gradio 用 props,新版用 props 内字段)
    for KEY in "genome" "chromosome" "start" "end" "predict" "snv"; do
        if echo "$CONFIG" | grep -qi "$KEY"; then
            ok "组件含 '$KEY'"
        else
            bad "未找到组件 '$KEY'"
        fi
    done
else
    say "  (跳过:配置未加载)"
    FAIL=$((FAIL + 6))
fi
echo ""

# ---------- 4. 后端健康检查 ----------
say "--- 4. 后端健康检查 ---"
BACK_HEALTH=$(curl -s -m 10 "$BACKEND_URL/health" 2>/dev/null || echo "")
if echo "$BACK_HEALTH" | grep -q '"status": *"ok"\|"success": *true\|"ready"'; then
    ok "后端 /health 正常"
else
    # 部分后端用 /genomes 做探活
    if curl -s -m 10 "$BACKEND_URL/genomes" 2>/dev/null | grep -q 'osa1_r7'; then
        ok "后端 /genomes 正常"
    else
        bad "后端健康检查失败(请先运行 backend/run_backend.sh)"
    fi
fi
echo ""

# ---------- 5. 日志检查 ----------
say "--- 5. 日志致命错误检查 ---"
FRONT_LOG="$ROOT_DIR/frontend/logs/frontend.nohup.log"
BACK_LOG="$ROOT_DIR/backend/logs/backend.nohup.log"
for pair in "frontend:$FRONT_LOG" "backend:$BACK_LOG"; do
    name="${pair%%:*}"; log="${pair##*:}"
    if [[ -f "$log" ]]; then
        if grep -qiE "Traceback|Error|Exception|FATAL" "$log" | head -5 | grep -qiE "Traceback|Error|Exception|FATAL"; then
            say "  \e[33mWARN\e[0m $name 日志含错误关键字(可能为良性,详见 $log)"
        else
            ok "$name 日志无错误关键字"
        fi
    else
        say "  \e[33mSKIP\e[0m $name 日志不存在: $log"
    fi
done
echo ""

# ---------- 汇总 ----------
echo "=========================================="
if [[ $FAIL -eq 0 ]]; then
    say "  \e[32mAll $PASS tests passed.\e[0m"
else
    say "  \e[31m$FAIL/$((PASS + FAIL)) tests failed.\e[0m"
fi
echo "=========================================="

exit $FAIL
