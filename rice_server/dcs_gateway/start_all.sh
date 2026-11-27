#!/usr/bin/env bash
# OGR 服务一键启动 / 停止 / 监控（四个后端 + 统一网关 dcs_gateway）
#
# 用法:
#   bash start_all.sh [start] [service...]    启动后端（默认全部；可指定子集）
#   bash start_all.sh web [service...]        启动后端 + 三个 Gradio 网页前端
#   bash start_all.sh monitor [service...]    启动后端并实时跟随日志（Ctrl-C 退出跟随,服务继续后台运行）
#   bash start_all.sh monitor-web [service...] web + 实时跟随日志（含前端与网关）
#   bash start_all.sh stop [service...]       停止后端（默认全部；可指定子集）
#   bash start_all.sh -h | --help             显示帮助
#   ENABLE_WEB=true bash start_all.sh         兼容旧写法：同时启动/停止网页前端
#   ENABLED_SERVICES=ogr_prediction,ogr_regulation bash start_all.sh   环境变量方式指定子集
#   RICE_OGR_PORT=8003 bash start_all.sh      shell 环境变量兜底（低于 .env）
#
# 监控模式说明: 服务仍以 nohup 后台启动(日志写 ${LOGDIR}/*.log)，
#   随后脚本 tail -f 实时跟随这些日志；Ctrl-C 只退出跟随、不影响后台服务。
#
# 前端说明: 默认(不带 web)只启动后端 API 与网关；带 web 时才额外拉起 Gradio 前端。
#   停止时 stop 全量会连带停前端与网关。
#
# 服务名与目录/端口映射:
#   ogr_prediction    → OGR_Prediction/backend    rice_mutation/main.py        (端口 8001)
#   ogr_regulation    → OGR_Regulation/backend    rice_reg/main.py            (端口 7001)
#   ogr_introgression → OGR_Introgression/backend rice_introgression/main.py  (端口 5001)
#   ogr_base          → OGR_Base                  dcs_adapter.py              (端口 6001)
#   dcs_gateway       → dcs_gateway/app.py                                    (端口 GATEWAY_PORT)
#
# 停止方式: 按各服务的精确命令行模式 pkill -f 匹配（不会误杀 Jupyter 等无关进程）;
#   网关按其工作目录(dcs_gateway)匹配,仅杀真正属于本部署的 app.py。
#
# 解释器优先级（从高到低）:
#   1) 本目录 .env 的 RICE_*_PYTHON（.env 里的值最高）
#   2) shell 环境变量 RICE_*_PYTHON
#   3) 各服务 .env 的 BACKEND_PYTHON_BIN（仅解释器回退）
#   4) /root/miniconda3/envs/vllm/bin/python → 系统 python
# 端口优先级: 本目录 .env（RICE_*_PORT / GATEWAY_PORT） > shell 环境变量 > 内置默认值
# 兼容 /bin/sh=dash（云平台 org_web:sanic 镜像 /bin/sh 为 dash，无 pipefail）:
#   - 去掉 -o pipefail，用 set -e 等价保护（本脚本无管道语义依赖）
#   - 关联数组 declare -A 改 POSIX 函数/小写变量写法
set -eu
cd "$(dirname "$0")/.."              # rice_server/
ROOT="$PWD"

usage() {
    cat <<'EOF'
dcs_gateway/start_all.sh —— OGR 服务一键启动 / 停止 / 监控

用法:
  bash start_all.sh [start] [service...]   启动后端（默认全部；可指定子集）
  bash start_all.sh web [service...]       启动后端 + Gradio 网页前端（全量）
  bash start_all.sh monitor [service...]   启动后端并实时跟随日志（Ctrl-C 退出跟随,服务继续后台运行）
  bash start_all.sh monitor-web [service...] web + 实时跟随日志（含前端与网关）
  bash start_all.sh stop [service...]      停止后端（默认全部；可指定子集）
  bash start_all.sh -h | --help            显示本帮助

监控模式:
  monitor / monitor-web 在服务启动后进入 tail -f 跟随；
  按 Ctrl-C 只退出日志跟随,服务仍在后台运行(可用 stop 停止)。

前端开关:
  默认(不带参数 / start) 只启动后端 API；带 web 才额外启动 Gradio 前端。
  旧写法 ENABLE_WEB=true 仍然兼容。

环境变量:
  ENABLE_WEB=true      同时启动/停止 Gradio 网页前端（prediction/regulation/introgression）
  ENABLED_SERVICES=... 逗号分隔的服务子集（缺省全部）
  RICE_*_PORT / GATEWAY_PORT / RICE_*_PYTHON  端口与解释器覆盖（本目录 .env 优先）

服务名与端口:
  ogr_prediction    8001   OGR_Prediction/backend    rice_mutation/main.py
  ogr_regulation    7001   OGR_Regulation/backend    rice_reg/main.py
  ogr_introgression 5001   OGR_Introgression/backend rice_introgression/main.py
  ogr_base          6001   OGR_Base                  dcs_adapter.py
  dcs_gateway       GATEWAY_PORT(.env 默认 4000)     dcs_gateway/app.py（统一网关）

示例:
  bash start_all.sh                          # 只启动全部后端 + 网关
  bash start_all.sh web                      # 启动全部后端 + 三个 Gradio 前端 + 网关
  bash start_all.sh monitor                  # 启动后端 + 实时看日志
  bash start_all.sh monitor-web              # 启动全部 + 前端 + 实时看日志(含网关)
  bash start_all.sh stop                     # 停止全部后端（含前端与网关）
  bash start_all.sh stop ogr_prediction ogr_regulation   # 只停这两个
EOF
}

# ---- 参数解析（-h/--help/stop/web/monitor/monitor-web/start 优先于服务子集位置参数） ----
ACTION="${1:-start}"
case "$ACTION" in
    -h|--help|help) usage; exit 0 ;;
    stop)  ACTION=stop;  shift ;;
    web)   ACTION=start; ENABLE_WEB=true; shift ;;
    monitor-web) ACTION=start; ENABLE_WEB=true; MONITOR=1; shift ;;
    monitor)     ACTION=start; MONITOR=1; shift ;;
    start) ACTION=start; shift ;;
esac

# 1) 加载 dcs_gateway/.env —— 本文件优先级最高
#    set -a 使 .env 变量成为导出变量，供 app.py / 后续进程读取
#    source 中的 KEY=VALUE 为普通赋值，会**覆盖** shell 中已存在的同名变量，
#    因此 .env 的值总是压过命令行/启动环境传入的同名变量（符合“.env 最高”约定）
if [ -f "$ROOT/dcs_gateway/.env" ]; then
    set -a
    . "$ROOT/dcs_gateway/.env"
    set +a
fi

# 2) 使用 .env 中的值（若 .env 未提供对应变量，则用 shell 环境变量 → 内置默认值）
RICE_MUT_PORT="${RICE_MUT_PORT:-8001}"
RICE_REG_PORT="${RICE_REG_PORT:-7001}"
RICE_INTRO_PORT="${RICE_INTRO_PORT:-5001}"
RICE_OGR_PORT="${RICE_OGR_PORT:-6001}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"

# 可选启动：ENABLED_SERVICES 逗号分隔（ogr_prediction/ogr_regulation/ogr_introgression/ogr_base），缺省全量
# 优先级：位置参数 > ENABLED_SERVICES（.env / shell 环境变量）> 全部
if [ "$#" -gt 0 ]; then
    ENABLED_SERVICES="$(IFS=','; echo "$*")"
else
    ENABLED_SERVICES="${ENABLED_SERVICES:-ogr_prediction,ogr_regulation,ogr_introgression,ogr_base}"
fi
echo "==> action: ${ACTION} | enabled services: $ENABLED_SERVICES"
RICE_MUT_PYTHON="${RICE_MUT_PYTHON:-}"
RICE_REG_PYTHON="${RICE_REG_PYTHON:-}"
RICE_INTRO_PYTHON="${RICE_INTRO_PYTHON:-}"
RICE_OGR_PYTHON="${RICE_OGR_PYTHON:-}"

LOGDIR="${LOGDIR:-/tmp/rice_dcs}"
mkdir -p "$LOGDIR"

# 解析某服务的 Python 解释器（env_python 内自动回退，见函数注释）
env_python() {
    local dir="$1" override="$2" py
    # 1) 已配置的 RICE_*_PYTHON（来自 .env 或 shell 环境变量）
    if [ -n "$override" ]; then echo "$override"; return; fi
    # 2) 各服务 .env 里的 BACKEND_PYTHON_BIN（部署约定均指向 vllm 环境）
    if [ -f "$ROOT/$dir/.env" ]; then
        py="$(sed -n 's/^BACKEND_PYTHON_BIN=//p' "$ROOT/$dir/.env" | tail -1)"
        [ -n "$py" ] && { echo "$py"; return; }
    fi
    # 3) 回退：统一 PATH 里的 python（部署机器上必须先激活 vllm 环境，
    #    不再写死绝对路径 vllm —— 各机器环境路径可能不同）
    if command -v python >/dev/null 2>&1; then echo python; return; fi
    echo python
}

# 注：RICE_*_PYTHON 已在上面被 .env 全覆盖（.env 优先级最高）
RICE_MUT_PYTHON="$(env_python "OGR_Prediction" "$RICE_MUT_PYTHON")"
RICE_REG_PYTHON="$(env_python "OGR_Regulation" "$RICE_REG_PYTHON")"
RICE_INTRO_PYTHON="$(env_python "OGR_Introgression" "$RICE_INTRO_PYTHON")"
RICE_OGR_PYTHON="$(env_python "OGR_Base" "$RICE_OGR_PYTHON")"

start_backend() {
    local name="$1" dir="$2" port="$3" py="$4" entry="${5:-dcs_adapter.py}"
    echo "==> [$name] 启动 ${entry} -> ${LOGDIR}/${name}.log  (port ${port}, python: ${py})"
    # OGR_Prediction / OGR_Regulation / OGR_Introgression 均已并入统一后端
    # (rice_mutation/main.py、rice_reg/main.py、rice_introgression/main.py =
    #  网页端与纯 API 同一进程)；仅 OGR_Base 仍为独立 dcs_adapter 进程。
    # setsid 使服务完全脱离终端会话/进程组: 终端 Ctrl-C / 关闭都影响不到服务。
    ( cd "$ROOT/$dir" && BACKEND_PORT="$port" setsid "$py" "$entry" \
        > "${LOGDIR}/${name}.log" 2>&1 < /dev/null & )
}

# ---- 停止逻辑 ----
# 各服务精确命令行模式（pkill -f 匹配，不误杀无关进程）
stop_pattern() {
    case "$1" in
        ogr_prediction)    echo "rice_mutation/main.py" ;;
        ogr_regulation)    echo "rice_reg/main.py" ;;
        ogr_introgression) echo "rice_introgression/main.py" ;;
        ogr_base)          echo "dcs_adapter.py" ;;
    esac
}

# 停止单个后端（pkill -f 精确模式；未运行时静默提示）
stop_backend() {
    local name="$1" pat
    pat="$(stop_pattern "$name")"
    if [ -z "$pat" ]; then return 1; fi
    if pgrep -f "$pat" >/dev/null 2>&1; then
        pkill -f "$pat" 2>/dev/null || true
        echo "    ✓ $name stopped"
    else
        echo "    - $name not running"
    fi
}

# 停止网关: 只杀工作目录为 dcs_gateway/ 的 app.py 进程（防止误杀 Jupyter 等）
stop_gateway() {
    local pid cwd found=0
    for pid in $(pgrep -f "app.py" 2>/dev/null || true); do
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
        if [ "$cwd" = "$ROOT/dcs_gateway" ]; then
            kill "$pid" 2>/dev/null || true
            echo "    ✓ dcs_gateway (pid $pid) stopped"
            found=1
        fi
    done
    [ "$found" = 1 ] || echo "    - dcs_gateway not running"
}

# 一键停止：全量停止时连带停前端与网关；指定子集时只停子集后端
do_stop() {
    echo "==> 停止服务..."
    local all=1 name
    for name in ogr_prediction ogr_regulation ogr_introgression ogr_base; do
        if ! enabled "$name"; then all=0; continue; fi
        stop_backend "$name"
    done
    if [ "$all" = 1 ]; then
        if pgrep -f "frontend/app.py" >/dev/null 2>&1; then
            pkill -f "frontend/app.py" 2>/dev/null || true
            echo "    ✓ 网页前端(Gradio) stopped"
        else
            echo "    - 网页前端 not running"
        fi
        stop_gateway
    fi
    echo "==> 停止完成（可用 ss -tlnp 检查端口是否释放）"
}

# 是否启用某服务（ENABLED_SERVICES 逗号分隔）
enabled() {
    local name="$1" e
    for e in $(echo "$ENABLED_SERVICES" | tr ',' ' '); do
        [ "$e" = "$name" ] && return 0
    done
    return 1
}

# ---- 停止模式：bash start_all.sh stop [service...] ----
if [ "$ACTION" = "stop" ]; then
    do_stop
    exit 0
fi

# ---- 启动模式 ----
if enabled ogr_prediction; then start_backend "ogr_prediction" "OGR_Prediction/backend" "$RICE_MUT_PORT" "$RICE_MUT_PYTHON" "rice_mutation/main.py"; else echo "    - skip ogr_prediction"; fi
if enabled ogr_regulation; then start_backend "ogr_regulation" "OGR_Regulation/backend"    "$RICE_REG_PORT" "$RICE_REG_PYTHON" "rice_reg/main.py"; else echo "    - skip ogr_regulation"; fi
if enabled ogr_introgression; then start_backend "ogr_introgression" "OGR_Introgression/backend" "$RICE_INTRO_PORT" "$RICE_INTRO_PYTHON" "rice_introgression/main.py"; else echo "    - skip ogr_introgression"; fi
if enabled ogr_base; then start_backend "ogr_base" "OGR_Base"         "$RICE_OGR_PORT" "$RICE_OGR_PYTHON"; else echo "    - skip ogr_base"; fi

# 可选：同时拉起网页前端（ENABLE_WEB=true 时）
# ogr_prediction / ogr_regulation / ogr_introgression 的网页端(Gradio)与后端(API)
# 是同一套服务，启动前端后网页与纯 API 都可用。
if [ "${ENABLE_WEB:-false}" = "true" ] && enabled ogr_prediction; then
    echo "==> 启动 OGR-Prediction 网页前端 ..."
    ( cd "$ROOT/OGR_Prediction" && setsid bash frontend/run_frontend.sh \
        > "${LOGDIR}/ogr_prediction_frontend.log" 2>&1 < /dev/null & )
fi
if [ "${ENABLE_WEB:-false}" = "true" ] && enabled ogr_regulation; then
    echo "==> 启动 OGR-Regulation 网页前端 ..."
    ( cd "$ROOT/OGR_Regulation" && setsid bash frontend/run_frontend.sh \
        > "${LOGDIR}/ogr_regulation_frontend.log" 2>&1 < /dev/null & )
fi
if [ "${ENABLE_WEB:-false}" = "true" ] && enabled ogr_introgression; then
    echo "==> 启动 OGR-Introgression 网页前端 ..."
    ( cd "$ROOT/OGR_Introgression" && setsid bash frontend/run_frontend.sh \
        > "${LOGDIR}/ogr_introgression_frontend.log" 2>&1 < /dev/null & )
fi

# 等待后端 /health 就绪（模型加载较慢，各最多 180s）
# POSIX 兼容：不用关联数组，改用 case 查端口
backend_port() {
    case "$1" in
        ogr_prediction) echo "$RICE_MUT_PORT" ;;
        ogr_regulation) echo "$RICE_REG_PORT" ;;
        ogr_introgression) echo "$RICE_INTRO_PORT" ;;
        ogr_base) echo "$RICE_OGR_PORT" ;;
    esac
}

echo "==> 等待后端就绪（最多 180s）..."
for name in ogr_prediction ogr_regulation ogr_introgression ogr_base; do
    if ! enabled "$name"; then continue; fi
    port="$(backend_port "$name")"
    url="http://127.0.0.1:${port}/health"
    ok=0
    for _ in $(seq 1 180); do
        if curl -sf "$url" >/dev/null 2>&1; then ok=1; break; fi
        sleep 1
    done
    if [ "$ok" = 1 ]; then
        echo "    ✓ $name ready"
    else
        echo "    ✗ $name NOT ready（看 ${LOGDIR}/${name}.log）" >&2
        tail -20 "${LOGDIR}/${name}.log" >&2 || true
    fi
done

echo "==> 启动 dcs_gateway -> GATEWAY_PORT=${GATEWAY_PORT}（后台运行,日志 ${LOGDIR}/gateway.log）"
cd "$ROOT/dcs_gateway"
GATEWAY_PORT="$GATEWAY_PORT" setsid bash run_gateway.sh > "${LOGDIR}/gateway.log" 2>&1 < /dev/null &

# ---- 监控模式: 实时跟随各服务日志（Ctrl-C 只退出跟随,不杀服务） ----
if [ "${MONITOR:-0}" = "1" ]; then
    echo
    echo "==> 监控模式: 正在跟随日志（Ctrl-C 退出跟随,服务继续后台运行）"
    # Ctrl-C 只杀 tail 子进程,保留服务进程
    trap 'echo; echo "==> 退出监控（服务仍在后台运行,可用 stop 停止）"; kill 0 2>/dev/null; exit 0' INT
    pids=""
    if enabled ogr_prediction; then tail -F "${LOGDIR}/ogr_prediction.log" & pids="$pids $!"; fi
    if enabled ogr_regulation; then tail -F "${LOGDIR}/ogr_regulation.log" & pids="$pids $!"; fi
    if enabled ogr_introgression; then tail -F "${LOGDIR}/ogr_introgression.log" & pids="$pids $!"; fi
    if enabled ogr_base; then tail -F "${LOGDIR}/ogr_base.log" & pids="$pids $!"; fi
    if [ "${ENABLE_WEB:-false}" = "true" ]; then
        tail -F "${LOGDIR}/ogr_prediction_frontend.log" 2>/dev/null & pids="$pids $!"
        tail -F "${LOGDIR}/ogr_regulation_frontend.log"  2>/dev/null & pids="$pids $!"
        tail -F "${LOGDIR}/ogr_introgression_frontend.log" 2>/dev/null & pids="$pids $!"
    fi
    # 网关日志最后跟随（保持在前台可见的末尾）
    tail -F "${LOGDIR}/gateway.log" 2>/dev/null
    # 任一 tail 退出(文件被删等)时清理其余
    for p in $pids; do kill "$p" 2>/dev/null || true; done
fi