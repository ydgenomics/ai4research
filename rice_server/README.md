# Rice-Server — 水稻基因组学模型服务仓库

本仓库包含 **三个水稻基因组学服务**（基于 Genos DNA 大模型衍生的 1B 水稻基模），
共享同一份模型/基因组数据（`source/`），并通过 **DCS 平台网关** 以 OpenAI 风格对外提供 API。

> 部署与维护细节见 [AGENTS.md](AGENTS.md)（部署指南）、[dcs.md](dcs.md)（DCS 调用规范）。
> **外部用户**（只关心 API 怎么调）直接看 [quick_start.md](quick_start.md)（一页速查）。

---

## 1. 三个服务一览

| 服务 | 任务 | 基础模型 | 输入 | 输出 | 本地端口(网页/API) | DCS 单入口 |
|---|---|---|---|---|---|---|
| **rice_mut** | DNA → 多组学表达预测（**参考 vs SNV 变异对比**） | `rice_1B_stage2_8k_hf` | DNA 序列 | assay×biosample 多维表达 + bigWig | 8000 / 8001 | `/api/aigress/openai/rice_mut` |
| **rice_reg** | DNA + **ATAC** → RNA-seq 表达预测 | `rice_1B_32k_hf` | DNA + ATAC bigWig | RNA-seq ± 两通道 + bigWig | 7000 / 7001 | `/api/aigress/openai/rice_reg` |
| **rice_OGR** | **embedding 提取 / 下游碱基预测**（基模直接开放） | 两者均可（注册表） | DNA 序列 | 1024 维向量 / 预测碱基 | 8000(本地 Sanic) / 8001(DCS 适配) | `/api/aigress/openai/rice_ogr` |

> ⚠️ **端口规划**：rice_mut 与 rice_OGR 本地都以 8000/8001 为默认值；**同机部署时二者必须错开**
> （例如 rice_OGR 用 `PORT=8002` / `BACKEND_PORT=8003`，见各项目 `.env`）。

| 服务 | 快速入口 |
|---|---|
| rice_mut | [README](rice_mut/README.md) · [DCS 调用](rice_mut/DCS_API.md) · [快速上手](rice_mut/DCS_API_quick_start.md) |
| rice_reg | [README](rice_reg/README.md) · [DCS 调用](rice_reg/DCS_API.md) |
| rice_OGR | [README](rice_OGR/README.md) · 参考：[Genos 适配](rice_OGR/ref_genos.md) |

---

## 2. 目录结构

```
rice_server/
├── README.md                # 本文档(索引导航)
├── AGENTS.md                # ★ 部署/维护指南(端口、.env、缓存、并发、FAQ)
├── dcs.md                   # ★ DCS 平台 API 规范(返回结构、鉴权、计费、三服务调用速查)
├── quick_start.md           # ★ 外部用户一页速查(三服务功能 + curl 示例)
│
├── rice_mut/                # 服务一：变异对比表达预测 (网页:8000 / API:8001)
│   ├── backend/             #   FastAPI 后端 + dcs_adapter.py
│   ├── frontend/            #   Gradio 前端 + IGV.js
│   └── DCS_API.md           #   DCS 调用文档
├── rice_reg/                # 服务二：ATAC 条件表达预测 (网页:7000 / API:7001)
│   ├── backend/             #   FastAPI 后端 + dcs_adapter.py
│   ├── frontend/            #   Gradio 前端
│   └── DCS_API.md           #   DCS 调用文档
├── rice_OGR/                # 服务三：embedding 提取 / 碱基预测 (Sanic:8000 / DCS:8001)
│   ├── dna_embedding.py     #   Sanic 原服务
│   ├── dcs_adapter.py       #   FastAPI DCS 适配层(单入口 + mode 分发)
│   └── README.md            #   DCS 调用章节
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

> rice_OGR 无网页前端，直接：`python dcs_adapter.py`（:8001）即可提供 DCS API；
> 或用 `python dna_embedding.py`（:8000）提供 Sanic 原生接口。

---

## 4. DCS 平台部署（三服务一致的模式）

三个服务的 DCS 适配层遵循**同一套约定**（详见 [dcs.md](dcs.md)）：

- **单入口**：`POST /api/aigress/openai/<service>`，用请求体 `mode` 区分具体操作
- **统一返回**：`{"usage": {prompt_tokens, completion_tokens}, "status", "message", "result"}`
- **鉴权**：`.env` 配 `DCS_API_KEY` 后 POST 需 `Authorization: Bearer <key>`（`health` 免鉴权）
- **计费**：`prompt_tokens = 输入碱基数 × 系数`，`completion_tokens = 输出元素数 × 系数`（系数可调）

```bash
# 示例：调用 rice_OGR 提取 embedding（model=服务名，model_name=实际模型名）
curl --location 'https://<dcs-host>/api/aigress/openai/rice_ogr' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{"model": "rice_ogr", "model_name": "1B_8k", "mode": "dna_embedding", "sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}'
```

---

## 5. 部署注意事项（摘要，详见 AGENTS.md）

1. **安全**：rice_mut / rice_reg 后端将根文件系统挂载为 `/static-files` 静态服务（IGV 加载 bigWig 用），**暴露公网前必须改白名单目录或加鉴权**。
2. **数据必带**：`source/`（约 17G 模型+基因组）无法从网络重取，打包交付必须包含；`cache/`、`logs/`、`*.pid` 为运行时产物，打包前删除。
3. **路径耦合**：`.env` 均为开发机绝对路径，部署机必须逐项修改。
4. **版本锁定**：`requirements.txt` 用 `>=`，交付前建议 `pip freeze > requirements-lock.txt`。