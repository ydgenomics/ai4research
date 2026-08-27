# Rice-Server — 水稻基因组学模型服务仓库

本仓库包含 **三个水稻基因组学服务**（基于 Genos DNA 大模型衍生的 1B 水稻基模），
共享同一份模型/基因组数据（`source/`），并通过 **DCS 平台网关** 以 OpenAI 风格对外提供 API。

> 部署与维护细节见 [AGENTS.md](AGENTS.md)（部署指南）、[dcs.md](dcs.md)（DCS 部署要求）。
> **API 调用**（本机 + DCS 两套测试代码）：三合一完整版见 [dcs_gateway/API.md](dcs_gateway/API.md)，各服务见 [rice_mut/API.md](rice_mut/API.md)、[rice_reg/API.md](rice_reg/API.md)、[rice_OGR/API.md](rice_OGR/API.md)。
> **外部用户**（只关心 API 怎么调）直接看 [quick_start.md](quick_start.md)（一页速查）。

---

## 1. 三个服务一览

| 服务 | 任务 | 基础模型 | 输入 | 输出 | 本地端口(网页/API) | DCS 入口 |
|---|---|---|---|---|---|---|
| **rice_mut** | DNA → 多组学表达预测（**参考 vs SNV 变异对比**） | `rice_1B_stage2_8k_hf` | DNA 序列 | assay×biosample 多维表达 + bigWig | 8000 / 8001 | 统一入口 `…/OGR` + `model_sub=rice_mut` |
| **rice_reg** | DNA + **ATAC** → RNA-seq 表达预测 | `rice_1B_32k_hf` | DNA + ATAC bigWig | RNA-seq ± 两通道 + bigWig | 7000 / 7001 | 统一入口 `…/OGR` + `model_sub=rice_reg` |
| **rice_OGR** | **embedding 提取 / 下游碱基预测**（基模直接开放） | 两者均可（注册表） | DNA 序列 | 1024 维向量 / 预测碱基 | 8000(本地 Sanic) / **6001**(DCS 适配) | 统一入口 `…/OGR`（`model_sub` 缺省即此） |

> ⚠️ **端口规划**：rice_OGR 的 DCS 适配层默认 **6001**（与 rice_mut 8001 错开，见 `dcs_gateway/.env`）。
> **统一网关**：`dcs_gateway/` 单端口（默认 9000）收口三个服务，对外**唯一入口**
> `POST /api/aigress/openai/OGR`（URL 路径或 body `model_sub` 路由，详见 [dcs_gateway/API.md](dcs_gateway/API.md)）。

| 服务 | 快速入口 |
|---|---|
| rice_mut | [README](rice_mut/README.md) · [API 调用](rice_mut/API.md) |
| rice_reg | [README](rice_reg/README.md) · [API 调用](rice_reg/API.md) |
| rice_OGR | [README](rice_OGR/README.md) · [API 调用](rice_OGR/API.md) |

---

## 2. 目录结构

```
rice_server/
├── README.md                # 本文档(索引导航)
├── AGENTS.md                # ★ 部署/维护指南(端口、.env、缓存、并发、FAQ)
├── dcs.md                   # ★ DCS 平台部署要求
├── quick_start.md           # ★ 外部用户一页速查(三服务功能 + curl 示例)
│
├── rice_mut/                # 服务一：变异对比表达预测 (网页:8000 / API:8001)
│   ├── backend/             #   FastAPI 后端 + dcs_adapter.py
│   ├── frontend/            #   Gradio 前端 + IGV.js
│   └── API.md               #   API 调用文档（本机 + DCS）
├── rice_reg/                # 服务二：ATAC 条件表达预测 (网页:7000 / API:7001)
│   ├── backend/             #   FastAPI 后端 + dcs_adapter.py
│   ├── frontend/            #   Gradio 前端
│   └── API.md               #   API 调用文档（本机 + DCS）
├── rice_OGR/                # 服务三：embedding 提取 / 碱基预测 (Sanic:8000 / DCS:6001)
│   ├── dna_embedding.py     #   Sanic 原服务
│   ├── dcs_adapter.py       #   FastAPI DCS 适配层(单入口 + mode 分发)
│   ├── README.md            #   服务介绍 / 配置 / 启动
│   └── API.md               #   API 调用文档（本机 + DCS）
│
├── dcs_gateway/            # ★ 统一网关：单端口收口三个服务的 DCS API
│   ├── app.py               #   FastAPI 轻量反代（model_sub 路由，不加载模型）
│   ├── .env.example         #   后端地址 / 端口配置
│   └── run_gateway.sh       #   启动脚本
│
├── docker/                  # 交付镜像
│   ├── org_web/             #   DCS 平台镜像(包含依赖)
│   ├── org_web-jupyter/     #   Jupyter 变体
│   └── push_org_web.sh      #   推送脚本(ydgenomics/org_web)
│
└── source/                  # ★ 模型 + 基因组数据(约 17G, 交付必带)
    ├── rice_mut/            #   rice_1B_stage2_8k_hf + csq-5 checkpoint + osa1_r7 基因组
    └── rice_reg/            #   rice_1B_32k_hf + three_sam2 checkpoint + MH63/NIP + ATAC
```

---

## 3. 快速开始（本地）

每个服务独立的启动方式见各自 README。共性流程：

```bash
# 1. 配置 .env(必改：模型/基因组/缓存路径, 参考各项目 .env.example)
cd rice_mut && cp .env.example .env && vim .env

# 2. 启动后端 + 前端(以 rice_mut 为例)
bash backend/run_backend.sh
bash frontend/run_frontend.sh
# 网页 → http://<host>:8000 ; 后端健康检查 → http://<host>:8001/health

# 3. DCS 适配层(供 DCS 网关转发, 与网页版独立端口)
python backend/dcs_adapter.py        # rice_mut: 8001(与网页后端同端口, 二选一)
```

> rice_OGR 无网页前端，直接：`python dcs_adapter.py`（:6001）即可提供 DCS API；
> 或用 `python dna_embedding.py`（:8000）提供 Sanic 原生接口。

---

## 4. DCS 平台部署（统一网关模式）

三个服务的 DCS 适配层遵循**同一套约定**，对外由 **`dcs_gateway/` 统一收口**（三合一完整说明 + 测试代码见 [dcs_gateway/API.md](dcs_gateway/API.md)）：

- **唯一入口**：`POST /api/aigress/openai/OGR`，请求体 **`model_sub`** 路由（`rice_mut`/`rice_reg`/`rice_ogr` 缺省）
- **统一返回**：`{"usage": {prompt_tokens, completion_tokens}, "status", "message", "result"}`
- **鉴权**：`.env` 配 `DCS_API_KEY` 后 POST 需 `Authorization: Bearer <key>`（`health` 免鉴权）
- **计费**：`prompt_tokens = 输入碱基数 × 系数`，`completion_tokens = 输出元素数 × 系数`（系数可调）

```bash
# 示例：经统一入口调用 rice_OGR 提取 embedding
curl --location 'https://<dcs-host>/api/aigress/openai/OGR' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{"model_sub": "rice_ogr", "mode": "dna_embedding", "model_name": "1B_8k", "sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}'
```

---

## 5. 部署注意事项（摘要，详见 AGENTS.md）

1. **安全**：rice_mut / rice_reg 后端将根文件系统挂载为 `/static-files` 静态服务（IGV 加载 bigWig 用），**暴露公网前必须改白名单目录或加鉴权**。
2. **数据必带**：`source/`（约 17G 模型+基因组）无法从网络重取，打包交付必须包含；`cache/`、`logs/`、`*.pid` 为运行时产物，打包前删除。
3. **路径耦合**：`.env` 均为开发机绝对路径，部署机必须逐项修改。
4. **版本锁定**：`requirements.txt` 用 `>=`，交付前建议 `pip freeze > requirements-lock.txt`。