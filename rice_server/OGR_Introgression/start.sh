#!/usr/bin/env bash
# =============================================================================
# OGR-Introgression 统一启动脚本（一个命令拉起整套网页应用）
#
# 设计说明：网页端与纯 API 调用走【同一后端进程、同一 DCS 子路径】——
#   - 网页端 Gradio (:5000) 直接调用后端 (:5001) 的
#     /api/aigress/openai/OGR-Introgression/Prediction（与 curl 同方言、同返回）
#   - 纯 API 调用方 / DCS 网关 也调用同一 DCS 风格子路径
#     /api/aigress/openai/OGR-Introgression/{Prediction,Genome,Chromosome,RegisterGenome}
#   - 两者共用同一进程、同一推理核心、同一统一返回结构
#     ({model_name, usage, token_cnt, status, message, result})；
#     Plotly 渲染数据完全来自该返回值（前端 = 填好参数的 curl）
#
# 因此只需一个命令：
#   bash start.sh            # 启动后端(API) + 前端(网页)
#   bash start.sh backend    # 只启动后端(纯 API 场景，不拉网页)
#   bash start.sh stop       # 停止后端 + 前端
#   bash start.sh status     # 查看状态
# =============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
ENV_FILE="$ROOT_DIR/.env"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# 加载 .env
if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
fi

# 默认 Python 解释器（.env 可覆盖）
BACKEND_PYTHON_BIN="${BACKEND_PYTHON_BIN:-/root/miniconda3/envs/vllm/bin/python}"
FRONTEND_PYTHON_BIN="${FRONTEND_PYTHON_BIN:-/root/miniconda3/envs/rice_reg/bin/python}"

BACKEND_PORT="${BACKEND_PORT:-5001}"
FRONTEND_PORT="${FRONTEND_PORT:-5000}"

backend_pid() { [ -f "$LOG_DIR/backend.pid" ] && cat "$LOG_DIR/backend.pid" 2>/dev/null || echo ""; }
frontend_pid() { [ -f "$LOG_DIR/frontend.pid" ] && cat "$LOG_DIR/frontend.pid" 2>/dev/null || echo ""; }

is_running() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_backend() {
    if is_running "$(backend_pid)"; then
        echo "Backend already running (PID=$(backend_pid)). Log: $LOG_DIR/backend.nohup.log"
        return 0
    fi
    echo "==> 启动后端 (API, :$BACKEND_PORT) ..."
    # startup 清理开关: 纯 API(backend)模式注入 OGR_ENABLE_STARTUP_CLEAN=0,
    # 后端跳过缓存目录删除/重建/清理线程(低权限部署兜底); all/restart(含前端)
    # 默认 =1 保持原有清理语义。环境变量赋值前缀不能经变量展开, 必须用 env。
    ( cd "$ROOT_DIR" && BACKEND_PORT="$BACKEND_PORT" env OGR_ENABLE_STARTUP_CLEAN="${STARTUP_CLEAN:-1}" \
        nohup "$BACKEND_PYTHON_BIN" backend/rice_introgression/main.py \
        >> "$LOG_DIR/backend.nohup.log" 2>&1 & echo $! > "$LOG_DIR/backend.pid" )
    echo "    PID=$(cat "$LOG_DIR/backend.pid")，日志 $LOG_DIR/backend.nohup.log"
}

start_frontend() {
    if is_running "$(frontend_pid)"; then
        echo "Frontend already running (PID=$(frontend_pid)). Log: $LOG_DIR/frontend.nohup.log"
        return 0
    fi
    echo "==> 启动前端 (网页, :$FRONTEND_PORT) ..."
    ( cd "$ROOT_DIR" && FRONTEND_PORT="$FRONTEND_PORT" nohup "$FRONTEND_PYTHON_BIN" frontend/app.py \
        >> "$LOG_DIR/frontend.nohup.log" 2>&1 & echo $! > "$LOG_DIR/frontend.pid" )
    echo "    PID=$(cat "$LOG_DIR/frontend.pid")，日志 $LOG_DIR/frontend.nohup.log"
}

stop_all() {
    for pid in "$(backend_pid)" "$(frontend_pid)"; do
        if is_running "$pid"; then
            echo "Stopping PID=$pid"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    rm -f "$LOG_DIR/backend.pid" "$LOG_DIR/frontend.pid"
    echo "Stopped."
}

status() {
    local bpid fpid
    bpid="$(backend_pid)"; fpid="$(frontend_pid)"
    if is_running "$bpid"; then echo "Backend  : running (PID=$bpid, :$BACKEND_PORT)"; else echo "Backend  : stopped"; fi
    if is_running "$fpid"; then echo "Frontend : running (PID=$fpid, :$FRONTEND_PORT)"; else echo "Frontend : stopped"; fi
}

case "${1:-all}" in
    all|"")
        start_backend
        # 等待后端 /health 就绪（模型加载最长 ~180s）
        echo "==> 等待后端就绪（最多 180s）..."
        for _ in $(seq 1 180); do
            if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
                echo "    ✓ 后端就绪"
                break
            fi
            sleep 1
        done
        start_frontend
        echo ""
        echo "=============================================="
        echo "  OGR-Introgression 已启动"
        echo "  网页端 : http://<host>:${FRONTEND_PORT}"
        echo "  纯 API : POST http://<host>:${BACKEND_PORT}/api/aigress/openai/OGR-Introgression/{Prediction,Genome,Chromosome,RegisterGenome}"
        echo "  (前端与纯 API 走同一 DCS 子路径，Plotly 渲染来自返回值)"
        echo "=============================================="
        ;;
    backend)
        # 纯 API 模式: 不拉前端, 后端跳过 startup 清理(不删缓存目录)
        STARTUP_CLEAN=0 start_backend
        echo "后端(API)已启动: http://127.0.0.1:${BACKEND_PORT}/api/aigress/openai/OGR-Introgression/{Prediction,Genome}"
        ;;
    stop)
        stop_all
        ;;
    status)
        status
        ;;
    restart)
        stop_all
        sleep 1
        start_backend
        for _ in $(seq 1 180); do
            if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then break; fi
            sleep 1
        done
        start_frontend
        echo "Restarted."
        ;;
    *)
        echo "Usage: bash start.sh [all|backend|stop|status|restart]"
        exit 1
        ;;
esac
