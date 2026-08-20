#!/usr/bin/env bash
# ============================================================================
# Rice-Mutation 测试入口 — 一键运行全部测试
#
# 用法:
#   bash test/run_all_tests.sh                 # 全部(单测 + API + 前端)
#   bash test/run_all_tests.sh --unit          # 仅 Python 单元测试(离线)
#   bash test/run_all_tests.sh --api           # 仅 DCS API curl 测试(需服务)
#   bash test/run_all_tests.sh --frontend      # 仅网页端测试(需前后端)
#   bash test/run_all_tests.sh --no-models     # 跳过需要 GPU 模型的服务测试
#
# 环境变量(可选覆盖):
#   DCS_BASE_URL    DCS 适配层地址,默认 http://127.0.0.1:8001
#   FRONTEND_URL    前端地址,默认 http://127.0.0.1:8000
#   BACKEND_URL     后端地址,默认 http://127.0.0.1:8001
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

UNIT=1
API=1
FRONTEND=1
for arg in "$@"; do
    case "$arg" in
        --unit)     API=0; FRONTEND=0 ;;
        --api)      UNIT=0; FRONTEND=0 ;;
        --frontend) UNIT=0; API=0 ;;
        --no-models) API=0; FRONTEND=0 ;;
        *) echo "未知参数: $arg"; exit 2 ;;
    esac
done

PYTHON_BIN="${BACKEND_PYTHON_BIN:-/root/miniconda3/envs/vllm/bin/python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

TOTAL_FAIL=0

# ---------- 1. Python 单元测试(离线,不需要服务) ----------
if [[ "$UNIT" == "1" ]]; then
    echo ""
    echo "########## 1/3 Python 单元测试(离线) ##########"
    if "$PYTHON_BIN" -m pytest test/test_dcs_adapter.py -q 2>/dev/null; then
        echo "  单元测试通过"
    else
        # pytest 不可用时回退 unittest
        echo "  (pytest 不可用,回退 unittest)"
        if "$PYTHON_BIN" test/test_dcs_adapter.py; then
            echo "  单元测试通过"
        else
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    fi
fi

# ---------- 2. DCS API curl 测试 ----------
if [[ "$API" == "1" ]]; then
    echo ""
    echo "########## 2/3 DCS API 测试 ##########"
    if bash test/test_dcs_api.sh; then
        :
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
fi

# ---------- 3. 网页端测试 ----------
if [[ "$FRONTEND" == "1" ]]; then
    echo ""
    echo "########## 3/3 网页端测试 ##########"
    if bash test/test_frontend.sh; then
        :
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
fi

echo ""
echo "=========================================="
if [[ $TOTAL_FAIL -eq 0 ]]; then
    echo "  All test suites passed."
else
    echo "  $TOTAL_FAIL test suite(s) failed."
fi
echo "=========================================="
exit $TOTAL_FAIL
