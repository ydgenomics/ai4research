#!/usr/bin/env bash
# 一键启动：三个后端 dcs_adapter + dcs_gateway（统一网关，单端口收口）
#
# 用法:
#   bash start_all.sh                                      # 启动全部服务（读本目录 .env + 各服务 .env）
#   bash start_all.sh rice_mut rice_reg                    # 位置参数：只启动指定服务（等价于 ENABLED_SERVICES）
#   ENABLED_SERVICES=rice_mut,rice_reg bash start_all.sh   # 环境变量方式（逗号分隔；.env 的 ENABLED_SERVICES 缺省=全部）
#   RICE_OGR_PORT=8003 bash start_all.sh                   # shell 环境变量兜底（低于 .env）
#   RICE_MUT_PYTHON=/path/to/python bash start_all.sh      # 同上，覆盖解释器
#
# 解释器优先级（从高到低）:
#   0) conda activate vllm（脚本启动时先激活 vllm 环境，python=该环境 python）
#   1) 本目录 .env 的 RICE_*_PYTHON（.env 里的值最高）
#   2) shell 环境变量 RICE_*_PYTHON
#   3) 各服务 .env 的 BACKEND_PYTHON_BIN（仅解释器回退）
#   4) 激活后 python（即 vllm 环境）→ /root/miniconda3/envs/vllm/bin/python
# 端口优先级: 本目录 .env（RICE_*_PORT / GATEWAY_PORT） > shell 环境变量 > 内置默认值
# 兼容 /bin/sh=dash（云平台 org_web:sanic 镜像 /bin/sh 为 dash，无 pipefail）:
#   - 去掉 -o pipefail，用 set -e 等价保护（本脚本无管道语义依赖）
#   - 关联数组 declare -A 改 POSIX 函数/小写变量写法（见下方 wait_health）
set -eu
cd "$(dirname "$0")/.."              # rice_server/
ROOT="$PWD"

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
RICE_OGR_PORT="${RICE_OGR_PORT:-6001}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"

# 可选启动：ENABLED_SERVICES 逗号分隔（rice_mut/rice_reg/rice_ogr），缺省全量
# 优先级：位置参数 > ENABLED_SERVICES（.env / shell 环境变量）> 全部
if [ "$#" -gt 0 ]; then
    ENABLED_SERVICES="$(IFS=','; echo "$*")"
else
    ENABLED_SERVICES="${ENABLED_SERVICES:-rice_mut,rice_reg,rice_ogr}"
fi
echo "==> enabled services: $ENABLED_SERVICES"
RICE_MUT_PYTHON="${RICE_MUT_PYTHON:-}"
RICE_REG_PYTHON="${RICE_REG_PYTHON:-}"
RICE_OGR_PYTHON="${RICE_OGR_PYTHON:-}"

LOGDIR="${LOGDIR:-/tmp/rice_dcs}"
mkdir -p "$LOGDIR"

# 解析某服务的 Python 解释器（env_python 内自动回退，见函数注释）
env_python() {
    local dir="$1" override="$2" py
    # 1) 已配置的 RICE_*_PYTHON（来自 .env 或 shell 环境变量）
    if [ -n "$override" ]; then echo "$override"; return; fi
    # 2) 各服务 .env 里的 BACKEND_PYTHON_BIN（三服务部署约定均指向 vllm 环境）
    if [ -f "$ROOT/$dir/.env" ]; then
        py="$(sed -n 's/^BACKEND_PYTHON_BIN=//p' "$ROOT/$dir/.env" | tail -1)"
        [ -n "$py" ] && { echo "$py"; return; }
    fi
    # 3) 回退：vllm 环境（含 sanic，rice_OGR 依赖）→ 系统 python
    for cand in /root/miniconda3/envs/vllm/bin/python python; do
        if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then echo "$cand"; return; fi
    done
    echo python
}

# 注：RICE_*_PYTHON 已在上面被 .env 全覆盖（.env 优先级最高）
RICE_MUT_PYTHON="$(env_python "rice_mut" "$RICE_MUT_PYTHON")"
RICE_REG_PYTHON="$(env_python "rice_reg" "$RICE_REG_PYTHON")"
RICE_OGR_PYTHON="$(env_python "rice_OGR" "$RICE_OGR_PYTHON")"

start_backend() {
    local name="$1" dir="$2" port="$3" py="$4"
    echo "==> [$name] 启动 dcs_adapter -> ${LOGDIR}/${name}.log  (port ${port}, python: ${py})"
    ( cd "$ROOT/$dir" && BACKEND_PORT="$port" nohup "$py" dcs_adapter.py \
        > "${LOGDIR}/${name}.log" 2>&1 & )
}

# 是否启用某服务（ENABLED_SERVICES 逗号分隔）
enabled() {
    local name="$1" e
    for e in $(echo "$ENABLED_SERVICES" | tr ',' ' '); do
        [ "$e" = "$name" ] && return 0
    done
    return 1
}

if enabled rice_mut; then start_backend "rice_mut" "rice_mut/backend" "$RICE_MUT_PORT" "$RICE_MUT_PYTHON"; else echo "    - skip rice_mut"; fi
if enabled rice_reg; then start_backend "rice_reg" "rice_reg/backend" "$RICE_REG_PORT" "$RICE_REG_PYTHON"; else echo "    - skip rice_reg"; fi
if enabled rice_ogr; then start_backend "rice_ogr" "rice_OGR"         "$RICE_OGR_PORT" "$RICE_OGR_PYTHON"; else echo "    - skip rice_ogr"; fi

# 等待三个后端 /health 就绪（模型加载较慢，各最多 180s）
# POSIX 兼容：不用关联数组，改用 case 查端口
backend_port() {
    case "$1" in
        rice_mut) echo "$RICE_MUT_PORT" ;;
        rice_reg) echo "$RICE_REG_PORT" ;;
        rice_ogr) echo "$RICE_OGR_PORT" ;;
    esac
}

echo "==> 等待后端就绪（最多 180s）..."
for name in rice_mut rice_reg rice_ogr; do
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

echo "==> 启动 dcs_gateway -> GATEWAY_PORT=${GATEWAY_PORT}（前台运行，Ctrl-C 停止）"
cd "$ROOT/dcs_gateway"
GATEWAY_PORT="$GATEWAY_PORT" bash run_gateway.sh