# dcs_gateway — 统一网关（单端口收口三服务）

DCS 平台侧只需配置 **一个外部地址**，网关按请求体 `model_sub` 字段把请求
路由到三个后端 `dcs_adapter` 进程（**不加载任何模型、不持有 GPU**，纯轻量反向代理）：

| `model_sub` | 后端服务 | 后端监听 | 路径重写目标 |
|---|---|---|---|
| `rice_mut` | rice_mut（变异对比表达预测） | `127.0.0.1:8001` | `/api/aigress/openai/rice_mut` |
| `rice_reg` | rice_reg（ATAC 条件表达预测） | `127.0.0.1:7001` | `/api/aigress/openai/rice_reg` |
| `rice_ogr`（**缺省**） | rice_OGR（embedding / 碱基预测） | `127.0.0.1:6001`（`.env` 可改） | `/api/aigress/openai/rice_ogr` |

外部唯一入口：

```
POST https://www.dcs.cloud/api/aigress/openai/OGR
```

> 三种服务文档：`../README.md`（总览）、`../dcs.md`（DCS 规范）、`../quick_start.md`（快手上手）。

---

## 1. 目录结构

```
dcs_gateway/
├── app.py              # 网关主程序（FastAPI，标准库 http.client 转发）
├── run_gateway.sh      # 启动脚本：加载 .env 后 exec python app.py
├── start_all.sh        # 一键拉起：三个后端 + 网关（前端脚本，见 §2.3）
├── .env.example        # 配置模板（复制为 .env 使用）
└── README.md
```

依赖：仅 **Python 标准库（http.client）+ fastapi / uvicorn**（与三个后端同栈，无需额外安装）。

---

## 2. 启动

### 2.0 推荐：一键拉起（三个后端 + 网关）

```bash
cd dcs_gateway && bash start_all.sh
```

脚本按下述顺序**自动解析每个后端的 Python 解释器**（见 §2.3 源码）：
**本目录 `.env` 的 `RICE_*_PYTHON`**（最高）> shell 环境变量 > 各服务 `.env` 的
`BACKEND_PYTHON_BIN` > `/root/miniconda3/envs/vllm/bin/python` > `python`。
端口同理：**`.env`（`RICE_*_PORT` / `GATEWAY_PORT`）优先级最高**，shell 环境变量仅兜底。
按默认端口启动三个后端（rice_mut=8001、rice_reg=7001、rice_OGR=6001，
后端日志写在 `/tmp/rice_dcs/*.log`），等待各自 `/health` 就绪（最长 180s，模型加载较慢）
后前台拉起网关（网关监听见 `GATEWAY_PORT`，默认 9000）。

### 2.1 手动：先启动三个后端（各占一个 GPU 进程）

网关路由的目标地址由环境变量控制（默认见 §3），后端端口不匹配时改网关 `.env` 或后端 `PORT`。
三个后端的 `.env` 均配置了 `BACKEND_PYTHON_BIN`（部署约定统一指向 vllm 环境，
含 sanic 等 rice_OGR 依赖），手动启动时建议用该解释器：

```bash
# ── 后端 1：rice_mut（端口 8001）────────────────────────────
cd ../rice_mut/backend && /root/miniconda3/envs/vllm/bin/python dcs_adapter.py

# ── 后端 2：rice_reg（端口 7001）────────────────────────────
cd ../rice_reg/backend && /root/miniconda3/envs/vllm/bin/python dcs_adapter.py

# ── 后端 3：rice_OGR（gateway .env 的 RICE_OGR_PORT 决定，示例 6001）──
cd ../rice_OGR && BACKEND_PORT=6001 /root/miniconda3/envs/vllm/bin/python dcs_adapter.py
# 注意：rice_OGR/.env 默认 BACKEND_PORT=8001，本地同机联调必须用环境变量覆盖为 gateway 期望的端口
```

### 2.2 手动：启动网关

```bash
cd dcs_gateway

# 方式一：直接运行（默认 0.0.0.0:9000，可用环境变量覆盖）
python app.py

# 方式二：脚本运行（自动加载 .env；DCS 部署通常用平台注入的 PORT）
bash run_gateway.sh

# 方式三：显式指定端口
GATEWAY_PORT=9111 python app.py
```

监听端口优先级：**`PORT`（DCS 平台注入）> `GATEWAY_PORT` > 9000**。

### 2.3 启动脚本源码

`run_gateway.sh`（加载 `.env` 后启动网关）：

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; source .env; set +a; fi
exec python app.py
```

`start_all.sh`（一键拉起：三后端 + 网关；**端口与解释器以本目录 `.env` 为最高优先级**，
shell 环境变量（如 `RICE_OGR_PORT=8003`、`GATEWAY_PORT=9111`）仅兜底）：

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."              # rice_server/
ROOT="$PWD"

# 1) 加载 dcs_gateway/.env —— 本文件优先级最高（set -a 导出，供 app.py 读取）
if [ -f "$ROOT/dcs_gateway/.env" ]; then
    set -a
    source "$ROOT/dcs_gateway/.env"
    set +a
fi

RICE_MUT_PORT="${RICE_MUT_PORT:-8001}"
RICE_REG_PORT="${RICE_REG_PORT:-7001}"
RICE_OGR_PORT="${RICE_OGR_PORT:-6001}"
GATEWAY_PORT="${GATEWAY_PORT:-9000}"
RICE_MUT_PYTHON="${RICE_MUT_PYTHON:-}"
RICE_REG_PYTHON="${RICE_REG_PYTHON:-}"
RICE_OGR_PYTHON="${RICE_OGR_PYTHON:-}"

LOGDIR="${LOGDIR:-/tmp/rice_dcs}"
mkdir -p "$LOGDIR"

# 解析某服务的 Python 解释器（.env 的 RICE_*_PYTHON 优先，其次各服务 .env 的
# BACKEND_PYTHON_BIN，最后 vllm 环境 / python）
env_python() {
    local dir="$1" override="$2" py
    if [ -n "$override" ]; then echo "$override"; return; fi
    if [ -f "$ROOT/$dir/.env" ]; then
        py="$(sed -n 's/^BACKEND_PYTHON_BIN=//p' "$ROOT/$dir/.env" | tail -1)"
        [ -n "$py" ] && { echo "$py"; return; }
    fi
    for cand in /root/miniconda3/envs/vllm/bin/python python; do
        if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then echo "$cand"; return; fi
    done
    echo python
}

RICE_MUT_PYTHON="$(env_python "rice_mut" "$RICE_MUT_PYTHON")"
RICE_REG_PYTHON="$(env_python "rice_reg" "$RICE_REG_PYTHON")"
RICE_OGR_PYTHON="$(env_python "rice_OGR" "$RICE_OGR_PYTHON")"

start_backend() {
    local name="$1" dir="$2" port="$3" py="$4"
    echo "==> [$name] 启动 dcs_adapter -> ${LOGDIR}/${name}.log  (port ${port}, python: ${py})"
    ( cd "$ROOT/$dir" && BACKEND_PORT="$port" nohup "$py" dcs_adapter.py \
        > "${LOGDIR}/${name}.log" 2>&1 & )
}

start_backend "rice_mut" "rice_mut/backend" "$RICE_MUT_PORT" "$RICE_MUT_PYTHON"
start_backend "rice_reg" "rice_reg/backend" "$RICE_REG_PORT" "$RICE_REG_PYTHON"
start_backend "rice_ogr" "rice_OGR"         "$RICE_OGR_PORT" "$RICE_OGR_PYTHON"

# 等待三个后端 /health 就绪（模型加载较慢，各最多 180s）
echo "==> 等待后端就绪（最多 180s）..."
declare -A URLS=(
    [rice_mut]="http://127.0.0.1:${RICE_MUT_PORT}/health"
    [rice_reg]="http://127.0.0.1:${RICE_REG_PORT}/health"
    [rice_ogr]="http://127.0.0.1:${RICE_OGR_PORT}/health"
)
for name in rice_mut rice_reg rice_ogr; do
    ok=0
    for _ in $(seq 1 180); do
        if curl -sf "${URLS[$name]}" >/dev/null 2>&1; then ok=1; break; fi
        sleep 1
    done
    [ "$ok" = 1 ] && echo "    ✓ $name ready" \
        || { echo "    ✗ $name NOT ready（看 ${LOGDIR}/${name}.log）" >&2;
             tail -20 "${LOGDIR}/${name}.log" >&2 || true; }
done

echo "==> 启动 dcs_gateway -> GATEWAY_PORT=${GATEWAY_PORT}（前台运行，Ctrl-C 停止）"
cd "$ROOT/dcs_gateway"
GATEWAY_PORT="$GATEWAY_PORT" bash run_gateway.sh
```

> 提示：`start_all.sh` 前台运行网关（便于 Ctrl-C 停整条链路）；后端以 nohup 后台运行，
> 停止用 `pkill -f dcs_adapter.py`（网关本身 Ctrl-C 即可）。

---

## 3. 配置（.env）

**`dcs_gateway/.env` 优先级最高**——端口与解释器优先读它，其次才是 shell 环境变量、
服务 `.env`、内置默认值。修改配置请直接编辑该文件：

| 变量 | 当前值（.env） | 说明 |
|---|---|---|
| `RICE_MUT_HOST` / `RICE_MUT_PORT` | `127.0.0.1` / `8001` | rice_mut 后端地址 |
| `RICE_REG_HOST` / `RICE_REG_PORT` | `127.0.0.1` / `7001` | rice_reg 后端地址 |
| `RICE_OGR_HOST` / `RICE_OGR_PORT` | `127.0.0.1` / `6001` | rice_OGR 后端地址（与 rice_mut 错开） |
| `RICE_MUT_PYTHON` / `RICE_REG_PYTHON` / `RICE_OGR_PYTHON` | `/root/miniconda3/envs/vllm/bin/python` | 各后端解释器（可留空自动回退） |
| `GATEWAY_HOST` | `0.0.0.0` | 网关监听地址 |
| `GATEWAY_PORT` | `9000` | 网关监听端口（DCS 平台注入的 `PORT` 仍优先，逻辑在 `app.py`） |

> 跨机部署时把三个后端地址改成对应 IP / 服务名即可，其余不用动。
> `start_all.sh` 通过 `set -a; source .env` 使这些变量同时成为**导出变量**，
> 因此 `app.py`（读 `os.getenv`）与各后端进程（`BACKEND_PORT`）都会读到同一份配置。

---

## 4. 验证

### 4.1 聚合健康检查（免鉴权）

```bash
curl -s http://127.0.0.1:9000/health
# → {"status":"ok|degraded","services":{"rice_mut":{...},"rice_reg":{...},"rice_ogr":{...}}}
```

### 4.2 通过网关调用三服务

```bash
api_key="${api_key}"
GW="http://127.0.0.1:9000/api/aigress/openai/OGR"

api_key=""
GW=""

# rice_mut（predict）
curl -X POST ${GW} -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_mut","mode":"predict","genome":"osa1_r7","chromosome":"chr09",
       "start":20716774,"end":20749541,"output_format":"mean"}'

# rice_reg（predict）
curl -X POST ${GW} -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_reg","mode":"predict","genome":"MH63RS3","chromosome":"chr01",
       "start":1,"end":32678,"atac_source":"SAM2_MH63_1","output_format":"mean"}'

# rice_OGR（embedding；model_sub 缺省即 rice_ogr，可省略）
curl -X POST ${GW} -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_ogr", "mode":"dna_embedding","model_name":"1B_8k","sequence":"ACGTTGCATGCAACGTACGTTGCATGCAACGT",
       "pooling_method":"mean"}'


# 10KP api
# rice_mut（predict）
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_mut","mode":"predict","genome":"osa1_r7","chromosome":"chr09",
       "start":20716774,"end":20749541,"output_format":"mean"}'

curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_mut","mode":"health"}'

# rice_reg（predict）
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_reg","mode":"predict","genome":"MH63RS3","chromosome":"chr01",
       "start":1,"end":32678,"atac_source":"SAM2_MH63_1","output_format":"mean"}'

# rice_OGR（embedding；model_sub 缺省即 rice_ogr，可省略）
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{"model":"OGR", "model_sub":"rice_ogr", "mode":"dna_embedding","model_name":"1B_8k","sequence":"ACGTTGCATGCAACGTACGTTGCATGCAACGT",
       "pooling_method":"mean"}'

```

### 4.3 错误语义

| 场景 | 返回 |
|---|---|
| 后端不可达 / 超时 | `502`，`message: gateway: backend 127.0.0.1:8003 unreachable (...)` |
| 未知 `model_sub` | `400`，`message: 未知 model_sub 'xxx',可选: rice_mut / rice_reg / rice_ogr` |
| 请求体非 JSON | `400`，`message: 请求体不是合法 JSON 对象` |
| 参数错误（透传） | 后端原样 400/401/500 语义，`result` 不变 |

---

## 5. 行为约定

- **路由**：仅读请求体 JSON 的 `model_sub`（小写匹配，缺省 `rice_ogr`）；不按 URL 前缀区分服务。
- **透传**：转发前剥离 `model_sub`，其余字段（`mode`/`sequence`/`start`/…）与请求头
  （`Authorization` / `X-API-Key` 等）原样透传，后端计费 / 错误语义不变。
- **路径重写**：网关收到 `.../OGR` 后转发到各后端 `/api/aigress/openai/<service>`。
- **健康检查**：`/health`、`/api/aigress/openai/health` 为聚合探活（本地合成，不转发）；
  `_probe` 对各后端 `/health` 免鉴权探测，超时 5s。
- **转发超时**：推理请求较长，转发超时 180s。
- **token 计费**：仍由后端按 1 bp = 1 token 计算，网关不改动 `usage`。