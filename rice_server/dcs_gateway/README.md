# dcs_gateway — 统一网关（单端口收口多服务）

DCS 平台侧只需配置 **一个外部地址**，网关按请求体 `model_sub` 字段把请求
路由到各后端 `dcs_adapter` 进程（**不加载任何模型、不持有 GPU**，纯轻量反向代理）：

| `model_sub` | 后端服务 | 后端监听（默认） |
|---|---|---|
| `rice_mut` | rice_mut（变异对比表达预测） | `127.0.0.1:8001` |
| `rice_reg` | rice_reg（ATAC 条件表达预测） | `127.0.0.1:7001` |
| `rice_intro` | rice_intro（粳/籼血缘渗入分析） | `127.0.0.1:5001` |
| `rice_ogr`（**缺省**） | rice_OGR（embedding / 碱基预测） | `127.0.0.1:6001`（`.env` 可改） |

外部唯一入口：`POST https://www.dcs.cloud/api/aigress/openai/OGR`（URL 路径或 body 路由两种方式）。

> **API 调用（本机 + DCS、三合一完整说明 + 测试代码）见 [API.md](API.md)**。
> 其它文档：`../README.md`（总览）、`../dcs.md`（DCS 部署要求）、`../quick_start.md`（快手上手）。

---

## 1. 目录结构

```
dcs_gateway/
├── app.py              # 网关主程序（FastAPI，标准库 http.client 转发）
├── run_gateway.sh      # 启动脚本：加载 .env 后 exec python app.py
├── start_all.sh        # 一键拉起：各后端 + 网关（支持服务开关，见 §2.0）
├── .env.example        # 配置模板（复制为 .env 使用）
├── API.md              # ★ 多合一 API 调用文档（本机 + DCS 测试代码）
└── README.md
```

依赖：仅 **Python 标准库（http.client）+ fastapi / uvicorn**（与各后端同栈，无需额外安装）。

---

## 2. 启动

### 2.0 推荐：一键拉起（各后端 + 网关）

```bash
cd dcs_gateway && bash start_all.sh        # 全部服务
bash start_all.sh rice_mut rice_reg        # 位置参数：只启动指定后端 + 网关（服务开关）
ENABLED_SERVICES=rice_mut,rice_reg bash start_all.sh   # 或环境变量方式（.env 的 ENABLED_SERVICES 亦可）
```

脚本启动时**先 `conda activate vllm`**（`python`=vllm 环境，含 sanic/transformers/torch+cuda），
再自动解析每个后端的 Python 解释器（**本目录 `.env` 的 `RICE_*_PYTHON`** > shell 环境变量 >
各服务 `.env` 的 `BACKEND_PYTHON_BIN` > 激活后的 `python`）；
端口同理（`.env` 的 `RICE_*_PORT` / `GATEWAY_PORT` 优先级最高）。
后端日志写在 `/tmp/rice_dcs/*.log`，等待各自 `/health` 就绪（最长 180s）后前台拉起网关（默认 9000）。

### 2.1 手动：先启动三个后端（各占一个 GPU 进程）

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

python app.py                    # 直接运行（默认 0.0.0.0:9000）
bash run_gateway.sh              # 自动加载 .env（DCS 部署通常用平台注入的 PORT）
GATEWAY_PORT=9111 python app.py  # 显式指定端口
```

监听端口优先级：**`PORT`（DCS 平台注入）> `GATEWAY_PORT` > 9000**。

> 停止：后端 `pkill -f dcs_adapter.py`；网关 Ctrl-C 即可。

---

## 3. 配置（.env）

**`dcs_gateway/.env` 优先级最高**——端口与解释器优先读它，其次才是 shell 环境变量、
服务 `.env`、内置默认值。修改配置请直接编辑该文件：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RICE_MUT_HOST` / `RICE_MUT_PORT` | `127.0.0.1` / `8001` | rice_mut 后端地址 |
| `RICE_REG_HOST` / `RICE_REG_PORT` | `127.0.0.1` / `7001` | rice_reg 后端地址 |
| `RICE_INTRO_HOST` / `RICE_INTRO_PORT` | `127.0.0.1` / `5001` | rice_intro 后端地址 |
| `RICE_OGR_HOST` / `RICE_OGR_PORT` | `127.0.0.1` / `6001` | rice_OGR 后端地址（与 rice_mut 错开） |
| `RICE_MUT_PYTHON` / `RICE_REG_PYTHON` / `RICE_INTRO_PYTHON` / `RICE_OGR_PYTHON` | （空，自动回退） | 各后端解释器（留空 = 用激活后的 vllm python） |
| `ENABLED_SERVICES` | `rice_mut,rice_reg,rice_intro,rice_ogr` | 一键拉起时的服务开关（逗号分隔；`bash start_all.sh <svc>...` 位置参数优先） |
| `GATEWAY_HOST` | `0.0.0.0` | 网关监听地址 |
| `GATEWAY_PORT` | `9000` | 网关监听端口（DCS 平台注入的 `PORT` 仍优先） |

> 跨机部署时把三个后端地址改成对应 IP / 服务名即可，其余不用动。
> `start_all.sh` 通过 `set -a; source .env` 使这些变量同时成为导出变量，
> 因此 `app.py`（读 `os.getenv`）与各后端进程（`BACKEND_PORT`）都会读到同一份配置。

---

## 4. 验证

```bash
# 聚合健康检查（免鉴权）
curl -s http://127.0.0.1:9000/health
# → {"status":"ok|degraded","services":{"rice_mut":{...},"rice_reg":{...},"rice_ogr":{...}}}
```

通过网关调用三服务的完整示例见 **[API.md](API.md)**。

---

## 5. 行为约定

- **路由**：URL 路径段优先级 > body `model_sub`（小写匹配，缺省 `rice_ogr`）。
- **透传**：转发前剥离 `model_sub`/`mode`，其余字段与请求头 `Authorization`/`X-API-Key` 原样透传，后端计费 / 错误语义不变。
- **路径重写**：网关收到 `.../OGR/{sub}[/{mode}]` 后转发到各后端 `/api/aigress/openai/<service>[/{mode}]`。
- **健康检查**：`/health`、`/api/aigress/openai/health` 为聚合探活（本地合成，不转发）；`_probe` 对各后端 `/health` 免鉴权探测，超时 5s。
- **转发超时**：推理请求较长，转发超时 180s。
- **token 计费**：仍由后端按 1 bp = 1 token 计算，网关不改动 `usage`。
