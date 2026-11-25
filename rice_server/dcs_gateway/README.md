# dcs_gateway — 统一网关（单端口收口多服务）

DCS 平台侧只需配置 **一个外部地址**，网关按请求体 `model_sub` 字段把请求
路由到各后端 `dcs_adapter` 进程（**不加载任何模型、不持有 GPU**，纯轻量反向代理）：

| `model_sub` | 后端服务 | 后端监听（默认） |
|---|---|---|
| `OGR-Prediction` | OGR-Prediction（变异对比表达预测） | `127.0.0.1:8001` |
| `OGR-Regulation` | OGR-Regulation（ATAC 条件表达预测） | `127.0.0.1:7001` |
| `OGR-Introgression` | OGR-Introgression（粳/籼血缘渗入分析） | `127.0.0.1:5001` |
| `OGR-Base`（**缺省**） | OGR-Base（embedding / 碱基预测） | `127.0.0.1:6001`（`.env` 可改） |

外部唯一入口：`POST https://www.dcs.cloud/api/aigress/openai/OGR`（URL 路径或 body 路由两种方式）。

> **自定义文件（FASTA / GFF）**：三个服务（OGR-Prediction / OGR-Regulation / OGR-Introgression）
> 支持 `POST .../OGR/{sub}/RegisterGenome` **传本地文件路径注册**（文件需已放在后端同机，
> 不做文件传输），预测时用返回的 `genome` 引用；详见 [API.md §4.5](API.md)。

> **API 调用（本机 + DCS、四合一完整说明 + 测试代码）见 [API.md](API.md)**。
> 其它文档：`../README.md`（总览）、`../dcs.md`（DCS 部署要求）、`../quick_start.md`（快手上手）。

---

## 1. 目录结构

```
dcs_gateway/
├── app.py              # 网关主程序（FastAPI，httpx 连接池转发）
├── run_gateway.sh      # 启动脚本：加载 .env 后 exec python app.py
├── start_all.sh        # 一键拉起：各后端 + 网关（支持服务开关，见 §2.0）
├── .env                # 网关配置（端口 / 后端地址 / 解释器 / 服务开关）
├── API.md              # ★ 四合一 API 调用文档（本机 + DCS 测试代码）
└── README.md
```

依赖：仅 **httpx（0.28+）+ fastapi / uvicorn**（与各后端同栈，无需额外安装）。

---

## 2. 启动

### 2.0 推荐：一键拉起（各后端 + 网关）

```bash
cd dcs_gateway && bash start_all.sh        # 全部服务
bash start_all.sh ogr_prediction ogr_regulation        # 位置参数：只启动指定后端 + 网关（服务开关）
ENABLED_SERVICES=ogr_prediction,ogr_regulation bash start_all.sh   # 或环境变量方式（.env 的 ENABLED_SERVICES 亦可）
```

脚本启动时**先 `conda activate vllm`**（`python`=vllm 环境，含 sanic/transformers/torch+cuda），
再自动解析每个后端的 Python 解释器（**本目录 `.env` 的 `RICE_*_PYTHON`** > shell 环境变量 >
各服务 `.env` 的 `BACKEND_PYTHON_BIN` > 激活后的 `python`）；
端口同理（`.env` 的 `RICE_*_PORT` / `GATEWAY_PORT` 优先级最高）。
后端日志写在 `/tmp/rice_dcs/*.log`，等待各自 `/health` 就绪（最长 180s）后前台拉起网关（默认 4000）。

### 2.1 手动：先启动四个后端（各占一个 GPU 进程）

```bash
# ── 后端 1：OGR-Prediction（端口 8001）────────────────────────────
cd ../OGR_Prediction/backend && /root/miniconda3/envs/vllm/bin/python dcs_adapter.py

# ── 后端 2：OGR-Regulation（端口 7001）────────────────────────────
cd ../OGR_Regulation/backend && /root/miniconda3/envs/vllm/bin/python dcs_adapter.py

# ── 后端 3：OGR-Introgression（端口 5001）─────────────────────────
cd ../OGR_Introgression/backend && /root/miniconda3/envs/vllm/bin/python dcs_adapter.py

# ── 后端 4：OGR-Base（gateway .env 的 RICE_OGR_PORT 决定，示例 6001）──
cd ../OGR_Base && BACKEND_PORT=6001 /root/miniconda3/envs/vllm/bin/python dcs_adapter.py
# 注意：OGR_Base/.env 默认 BACKEND_PORT=8001，本地同机联调必须用环境变量覆盖为 gateway 期望的端口
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
| `RICE_MUT_HOST` / `RICE_MUT_PORT` | `127.0.0.1` / `8001` | OGR-Prediction 后端地址 |
| `RICE_REG_HOST` / `RICE_REG_PORT` | `127.0.0.1` / `7001` | OGR-Regulation 后端地址 |
| `RICE_INTRO_HOST` / `RICE_INTRO_PORT` | `127.0.0.1` / `5001` | OGR-Introgression 后端地址 |
| `RICE_OGR_HOST` / `RICE_OGR_PORT` | `127.0.0.1` / `6001` | OGR-Base 后端地址（与 OGR-Prediction 错开） |
| `RICE_MUT_PYTHON` / `RICE_REG_PYTHON` / `RICE_INTRO_PYTHON` / `RICE_OGR_PYTHON` | （空，自动回退） | 各后端解释器（留空 = 用激活后的 vllm python） |
| `ENABLED_SERVICES` | `ogr_prediction,ogr_regulation,ogr_introgression,ogr_base` | 一键拉起时的服务开关（逗号分隔；`bash start_all.sh <svc>...` 位置参数优先） |
| `GATEWAY_HOST` | `0.0.0.0` | 网关监听地址 |
| `GATEWAY_PORT` | `4000` | 网关监听端口（DCS 平台注入的 `PORT` 仍优先） |

> 跨机部署时把四个后端地址改成对应 IP / 服务名即可，其余不用动。
> `start_all.sh` 通过 `set -a; source .env` 使这些变量同时成为导出变量，
> 因此 `app.py`（读 `os.getenv`）与各后端进程（`BACKEND_PORT`）都会读到同一份配置。

---

## 4. 验证

```bash
# 聚合健康检查（免鉴权）
curl -s http://127.0.0.1:4000/health
# → {"status":"ok|degraded","services":{"ogr_prediction":{...},"ogr_regulation":{...},"ogr_base":{...}}}
```

通过网关调用四服务的完整示例见 **[API.md](API.md)**。

---

## 5. 行为约定

- **路由**：URL 路径段优先级 > body `model_sub`（大小写/连字符不敏感，缺省 `ogr_base`）。
- **透传**：转发前剥离 `model_sub`（及仅用于路由的路径段），其余字段与请求头 `Authorization`/`X-API-Key` 原样透传，后端计费 / 错误语义不变。
- **路径重写**：网关收到 `.../OGR/{sub}[/{subpath}]` 后转发到各后端 `/api/aigress/openai/<service>/{subpath}`（四个后端均为纯子路径路由，无 body mode 注入）。
- **健康检查**：`/health`、`/api/aigress/openai/health` 为聚合探活（本地合成，不转发）；`_probe` 对各后端 `/health` 免鉴权探测，超时 5s。
- **转发超时**：推理请求较长，connect 5s / read 600s；连接失败→502，读超时→504。
- **token 计费**：仍由后端按 1 bp = 1 token 计算，网关不改动 `usage`。
- **文件系统探查**（调试用，默认关闭）：`EXPLORE_ENABLED=true` 时开放 `POST .../OGR/fs/{list,stat,find,cat}`
  （网关本地只读实现，不转发后端），用于远程排查部署机文件位置与目录结构（详见 [API.md §4.6](API.md)）；
  仅 list/stat/find/cat 四项只读操作，配置 `DCS_API_KEY` 后强制鉴权，cat 限 64KB；**排查完请关闭**。
