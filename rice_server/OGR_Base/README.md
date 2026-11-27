# OGR-Base — DNA Embedding 提取 / 下游碱基预测 API 服务

基于 **Genos DNA 大模型**与**水稻基因组基模**（`rice_1B_stage2_8k_hf`）的序列 Embedding 提取服务。
提供 RESTful API，支持 GPU / 昇腾 NPU / CPU，多设备并行，四种池化方式，以及下游碱基预测（NTP）。

> 上游参考：[BGI-HangzhouAI/Genos](https://github.com/BGI-HangzhouAI/Genos/tree/main)
> **API 调用（本机 + DCS 测试代码）见 [API.md](API.md)**。

## 1. 项目介绍

### 1.1 功能特性

- **DNA 序列 Embedding 提取**：支持 `mean` / `max` / `last` / `none` 四种池化方式
- **DNA 核苷酸预测（NTP）**：自回归预测下游碱基
- **多设备支持**：自动检测，优先级 **NPU > GPU > CPU**；支持多卡 `device_map` 并行
- **模型固定**：调用不再传 `model_name` / `mode`，模型固定为注册表中的 `1B_8k`；**功能由 URL 子路径决定**（`Embedding` | `NTP`）
- **两套入口**：Sanic 原生服务（`dna_embedding.py`，本地直连）+ DCS 适配层（`dcs_adapter.py`，供网关转发）

### 1.2 目录结构

```
OGR_Base/
├── dna_embedding.py        # Sanic 原生服务（/OGR-Base/Embedding、/OGR-Base/NTP，默认 8000）
├── dcs_adapter.py          # DCS 适配层（FastAPI，/api/aigress/openai/OGR-Base/*，默认 6001）
├── .env                    # ★ 模型注册表 + 服务/设备配置（部署时必改）
├── .env.example            # 配置模板
├── README.md               # 本文档
├── API.md                  # ★ API 调用文档（本机 + DCS 测试代码）
└── ref_genos.md            # Genos 在 NPU/MetaX 上的 Docker 部署参考
```

### 1.3 模型注册表(.env)

每个模型通过一组 `MODEL_<NAME>_*` 环境变量注册，`<NAME>` 即注册表中模型名（适配层固定取 `1B_8k`）：

| 配置项 | 必填 | 说明 |
|---|---|---|
| `MODEL_<NAME>_PATH` | 是 | 模型目录路径(HF 格式, 含 `config.json`/`tokenizer.json`/`model.safetensors`) |
| `MODEL_<NAME>_TYPE` | 否 | `flash`(默认) 或 `no_flash` |
| `MODEL_<NAME>_MAX_LEN` | 否 | 最大序列长度, 超长截断(防 OOM) |
| `MODEL_<NAME>_SPECIAL` | 否 | 是否添加特殊 token: `1`/`0`（水稻基模 tokenizer 默认不加特殊 token，统一建议 `0`） |

示例（本仓库默认 `.env` 已内置）:

```bash
# ---- 水稻 1B 基模 (OGR_Prediction: stage2 8k) ----
MODEL_1B_8k_PATH=/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/model_data/rice_1B_stage2_8k_hf
MODEL_1B_8k_TYPE=flash
MODEL_1B_8k_MAX_LEN=131072
MODEL_1B_8k_SPECIAL=0

# ---- 水稻 1B 基模 (OGR_Regulation: 32k) —— 本部署默认注释，仅 OGR-Base 用 1B_8k ----
# MODEL_1B_32k_PATH=/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/model_data/rice_1B_32k_hf
# MODEL_1B_32k_TYPE=flash
# MODEL_1B_32k_MAX_LEN=32678
# MODEL_1B_32k_SPECIAL=0
```

> 未配置 `MODEL_<NAME>_PATH` 的模型将回退到旧逻辑：`{model_path_prefix}/Genos-<NAME>`（如 `1.2B`、`10B`）。
> 适配层与 Sanic 服务均**固定使用 `1B_8k`**；注册表无 `1B_8k` 时回退到第一个已注册模型。

### 1.4 服务/设备配置(.env)

| 配置项 | 默认 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | Sanic 原生服务监听地址 |
| `PORT` | `8000` | Sanic 原生服务监听端口 |
| `DEVICE` | 空(自动) | `cuda:0` / `npu:0` / `cpu`；多卡逗号分隔 `cuda:0,cuda:1` |
| `DEVICE_MAP` | 空 | 多设备映射: `auto` / `balanced` / `sequential` |
| `MEMORY_RATIO` | `0.9` | 每卡内存分配比例 |
| `FORCE_CPU` | `false` | 强制 CPU |
| `BACKEND_HOST` / `BACKEND_PORT` | `0.0.0.0` / `8001` | **DCS 适配层**监听（端口优先级 `PORT` > `BACKEND_PORT` > 8001） |
| `DCS_API_KEY` | 空 | 配置后 POST 需鉴权；留空免鉴权 |
| `DCS_PROMPT_TOKEN_MULTIPLIER` / `DCS_COMPLETION_TOKEN_MULTIPLIER` | `1` / `1` | 计费系数 |

> ⚠️ **端口陷阱**：`.env` 默认 `BACKEND_PORT=8001`，与 OGR-Prediction 冲突。同机联调必须用
> `BACKEND_PORT=6001` 覆盖（`start_all.sh` 通过 `.env` 的 `RICE_OGR_PORT=6001` 自动覆盖）。

## 2. 启动服务

### 2.1 安装依赖

```bash
pip install -r requirements.txt      # 或：pip install torch transformers sanic safetensors fastapi uvicorn
# 可选加速: flash-attn (GPU/NPU)
```

### 2.2 配置 .env

```bash
cp .env.example .env
# 编辑 .env, 设置 MODEL_*_PATH / DEVICE / BACKEND_PORT 等
```

### 2.3 启动

```bash
cd OGR_Base

# ── 方式 A：DCS 适配层（供统一网关转发，推荐给 DCS 部署）──
BACKEND_PORT=6001 python dcs_adapter.py          # 默认 6001（与 OGR-Prediction 8001 错开）

# ── 方式 B：Sanic 原生服务（本地直连，子路径与适配层一致）──
python dna_embedding.py                          # 按 .env 注册表加载，默认 8000
python dna_embedding.py --model 1B_8k            # 只加载注册表中的 1B_8k
python dna_embedding.py --device cuda:0,cuda:1 --device_map auto   # 多卡
python dna_embedding.py --force_cpu              # 强制 CPU
```

## 3. API 接口

完整 API 调用说明（**本机 + DCS 测试代码**）见 **[API.md](API.md)**。

简要接口一览（**子路径决定功能，无 `mode` / `model_name` 参数**）：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/OGR-Base/Embedding` | POST | 提取 DNA 序列 Embedding（Sanic 原生服务 :8000） |
| `/OGR-Base/NTP` | POST | 预测下游碱基 NTP（Sanic 原生服务 :8000） |
| `/health` | GET | 健康检查 + 设备/模型信息（Sanic :8000） |
| `/models` | GET | 注册/已加载模型列表（Sanic :8000） |
| `/api/aigress/openai/OGR-Base/Embedding`（或 `/OGR-Base/Embedding`） | POST | DCS 适配层 Embedding（FastAPI :6001） |
| `/api/aigress/openai/OGR-Base/NTP`（或 `/OGR-Base/NTP`） | POST | DCS 适配层 NTP（FastAPI :6001） |
| `/api/aigress/openai/health`、`/health` | GET/POST | 健康检查 + 诊断（适配层，免鉴权） |
| `/api/aigress/openai/models`、`/models` | GET | 模型列表（适配层，免鉴权） |

> 两个服务进程：**Sanic 原生服务**（`dna_embedding.py`，端口 8000）用于本地直连；
> **DCS 适配层**（`dcs_adapter.py`，默认 **6001**，与 OGR-Prediction 8001 错开）供 DCS 网关转发。
> 两者的子路径一致（`Embedding` / `NTP`），返回结构在适配层外层包统一信封。

## 4. DCS 适配层部署说明

DCS 适配层 `dcs_adapter.py`（FastAPI）对外暴露 **OpenAI 风格子路径**（`OGR-Base/Embedding`、`OGR-Base/NTP`），
复用 `dna_embedding.EmbeddingExtractor` 的推理逻辑，并自动注入 `usage` 计费字段、支持鉴权与 JSON 末尾换行。

### 4.1 架构总览

```
DCS 网关(https://<dcs-host>/api/aigress/openai/OGR/OGR-Base/{Embedding|NTP})
        │  OpenAI 风格请求体 {sequence, pooling_method|predict_length}
        ▼
OGR_Base/dcs_adapter.py  (FastAPI, 默认 6001)
        │  子路径分发: Embedding | NTP（无 mode 参数）
        ▼
dna_embedding.EmbeddingExtractor  (模型加载/设备/推理, 复用原逻辑)
        │
        ▼
{"model_name":"OGR","usage":{...},"token_cnt":...,"status":200,"message":"...","result":{...}}  (+换行)
```

### 4.2 启动适配层

```bash
cd OGR_Base
BACKEND_PORT=6001 python dcs_adapter.py    # 预加载注册表模型, 监听 6001(BACKEND_PORT)
PORT=6001 python dcs_adapter.py            # 显式指定端口(平台注入 PORT 时自动生效)
```

> 适配层**容错预加载**：模型加载失败不会退出进程，错误写入 `_INIT_ERROR`，由 `/health` 与推理接口返回 503；
> 先 `curl /health` 确认 `diagnostics.init_error` 为 `null` 且 `models.loaded` 非空再调用。
> 鉴权：`.env` 配了 `DCS_API_KEY` 时本地 POST 也要带 `Authorization: Bearer <key>`（否则 401）。

### 4.3 容器内路径约定

DCS 容器内模型路径与本地 .env 不同，通过环境变量覆盖：

```bash
# 容器内启动适配层(模型挂载在 /AI_models/ 下)
MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf \
PORT=6001 \
python /code/dcs_adapter.py
```

## 5. 常见问题

- **FlashAttention 报错**：`MODEL_<NAME>_TYPE` 改为 `no_flash`，或移除 `.env` 中的 `_TYPE=flash`
- **OOM**：减小 `MODEL_<NAME>_MAX_LEN`；或 `--device cuda:0,cuda:1 --device_map auto`
- **NPU 多卡**：`DEVICE=npu:0,npu:1`（accelerate 不支持 NPU 设备的 max_memory，会自动手动切层分配）
- **embedding 为 bf16**：服务自动转 fp16 输出，与下游 numpy 兼容
- **端口冲突**：`BACKEND_PORT=8001` 与 OGR-Prediction 冲突 → 用 `BACKEND_PORT=6001` 覆盖
- **接口 404**：确认用的是 `/OGR-Base/Embedding` / `/OGR-Base/NTP`（旧 `/extract`、`/predict` 已移除）
