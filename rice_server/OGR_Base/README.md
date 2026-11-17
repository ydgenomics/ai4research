# Rice-OGR — DNA Embedding 提取 API 服务

基于 **Genos DNA 大模型**与**水稻基因组基模**（`rice_1B_stage2_8k_hf` / `rice_1B_32k_hf`）的序列 Embedding 提取服务。
提供 RESTful API，支持 GPU / 昇腾 NPU / CPU，多设备并行，多种池化方式，以及下游碱基预测。

> 上游参考：[BGI-HangzhouAI/Genos](https://github.com/BGI-HangzhouAI/Genos/tree/main)
> **API 调用（本机 + DCS、Sanic 原生 + DCS 适配两套测试代码）见 [API.md](API.md)**。

## 1. 项目介绍

### 1.1 功能特性

- **DNA 序列 Embedding 提取**：支持 `mean` / `max` / `last` / `none` 四种池化方式
- **DNA 核苷酸预测**：自回归预测下游碱基（`/predict`）
- **多设备支持**：自动检测，优先级 **NPU > GPU > CPU**；支持多卡 `device_map` 并行
- **多模型注册**：支持任意数量模型同时加载，通过 `--model` 选择或 `.env` 注册表，`/health` 查看
- **兼容水稻基模**：原生支持两个水稻 1B 基模（rice_mut 的 8k / rice_reg 的 32k 变体）

### 1.2 目录结构

```
rice_OGR/
├── dna_embedding.py        # 主服务程序(EmbeddingExtractor + Sanic API, 端口 8000)
├── dcs_adapter.py          # DCS 适配层(FastAPI 单入口, 默认 6001)
├── .env                    # ★ 模型注册表 + 服务/设备配置(部署时必改)
├── .env.example            # 配置模板
├── README.md               # 本文档
├── API.md                  # ★ API 调用文档（本机 + DCS 测试代码）
└── ref_genos.md            # Genos 在 NPU/MetaX 上的 Docker 部署参考
```

### 1.3 模型注册表(.env)

每个模型通过一组 `MODEL_<NAME>_*` 环境变量注册，`<NAME>` 即 API 中 `model_name`：

| 配置项 | 必填 | 说明 |
|---|---|---|
| `MODEL_<NAME>_PATH` | 是 | 模型目录路径(HF 格式, 含 `config.json`/`tokenizer.json`/`model.safetensors`) |
| `MODEL_<NAME>_TYPE` | 否 | `flash`(默认) 或 `no_flash` |
| `MODEL_<NAME>_MAX_LEN` | 否 | 最大序列长度, 超长截断(防 OOM) |
| `MODEL_<NAME>_SPECIAL` | 否 | 是否添加特殊 token: `1`/`0`（水稻基模 tokenizer 默认不加特殊 token，统一建议 `0`） |

示例（本仓库默认 `.env` 已内置）:

```bash
# ---- 水稻 1B 基模 (rice_mut: stage2 8k) ----
MODEL_1B_8k_PATH=/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/rice_1B_stage2_8k_hf
MODEL_1B_8k_TYPE=flash
MODEL_1B_8k_MAX_LEN=32768
MODEL_1B_8k_SPECIAL=0

# ---- 水稻 1B 基模 (rice_reg: 32k) ----
# 注: 两基模 tokenizer 均为单碱基编码且默认不加特殊token, SPECIAL 统一 0
MODEL_1B_32k_PATH=/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_reg/rice_1B_32k_hf
MODEL_1B_32k_TYPE=flash
MODEL_1B_32k_MAX_LEN=32678
MODEL_1B_32k_SPECIAL=0
```

> 未配置 `MODEL_<NAME>_PATH` 的模型将回退到旧逻辑：`{model_path_prefix}/Genos-<NAME>`（如 `1.2B`、`10B`）。

### 1.4 服务/设备配置(.env)

| 配置项 | 默认 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `DEVICE` | 空(自动) | `cuda:0` / `npu:0` / `cpu`；多卡逗号分隔 `cuda:0,cuda:1` |
| `DEVICE_MAP` | 空 | 多设备映射: `auto` / `balanced` / `sequential` |
| `MEMORY_RATIO` | `0.9` | 每卡内存分配比例 |
| `FORCE_CPU` | `false` | 强制 CPU |

## 2. 启动服务

### 2.1 安装依赖

```bash
pip install torch transformers sanic safetensors
# 可选加速: flash-attn (GPU/NPU)
```

### 2.2 配置 .env

```bash
cp .env.example .env
# 编辑 .env, 设置 MODEL_*_PATH 等
```

### 2.3 启动

```bash
cd rice_OGR
python dna_embedding.py                       # 按 .env 注册表加载全部模型
python dna_embedding.py --model 1B_8k         # 只加载水稻 8k 基模
python dna_embedding.py --device cuda:0,cuda:1 --device_map auto   # 多卡
python dna_embedding.py --force_cpu           # 强制 CPU
```
## 3. API 接口

完整 API 调用说明（**本机 + DCS 两套测试代码**）见 **[API.md](API.md)**。

简要接口一览：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 + 设备/模型信息 |
| `/models` | GET | 已注册/已加载模型列表 |
| `/extract` | POST | 提取 DNA 序列 Embedding（Sanic 原生服务） |
| `/predict` | POST | 预测下游碱基（Sanic 原生服务） |
| `/api/aigress/openai/rice_ogr`（或 `/rice_ogr`） | POST | DCS 适配层单入口，按 `mode` 分发 `dna_embedding` / `predict` |
| `/api/aigress/openai/rice_ogr/dna_embedding`、`…/predict` | POST | DCS 适配层子路径（等价 mode 写法） |
| `/api/aigress/openai/health`、`/health` | GET/POST | 健康检查 + 诊断（免鉴权） |
| `/api/aigress/openai/models`、`/models` | GET | 模型列表（免鉴权） |

> 两个服务进程：**Sanic 原生服务**（`dna_embedding.py`，端口 8000）用于本地直连；
> **DCS 适配层**（`dcs_adapter.py`，默认 **6001**，与 rice_mut 8001 错开）供 DCS 网关转发。

## 4. DCS 适配层部署说明

DCS 适配方式与 rice_mut 一致：独立适配层 `dcs_adapter.py`（FastAPI）对外暴露 **OpenAI 风格单入口**，
复用 `dna_embedding.EmbeddingExtractor` 的推理逻辑，并自动注入 `usage` 计费字段、支持鉴权与 JSON 末尾换行。

### 4.1 架构总览

```
DCS 网关(https://<dcs-host>/api/aigress/openai/OGR/rice_ogr)
        │  OpenAI 风格请求体 {model_name, sequence, ...}
        ▼
rice_OGR/dcs_adapter.py  (FastAPI, 默认 6001)
        │  mode 分发: dna_embedding | predict
        ▼
dna_embedding.EmbeddingExtractor  (模型加载/设备/推理, 复用原逻辑)
        │
        ▼
{"usage": {...}, "status": 200, "message": "...", "result": {...}}  (+换行)
```

### 4.2 启动适配层

```bash
cd rice_OGR
python dcs_adapter.py                    # 按 .env 注册表预加载全部模型, 监听 6001(BACKEND_PORT)
PORT=6001 python dcs_adapter.py          # 显式指定端口(平台注入 PORT 时自动生效)
```

> 本地联调时若模型未预加载完成就发请求, 适配层会返回 503 — 先 `curl /health` 确认
> `init_error` 为 `null` 且 `models.loaded` 非空再调用。
> 鉴权：`.env` 配了 `DCS_API_KEY` 时本地 POST 也要带 `Authorization: Bearer <key>`(否则 401)。

### 4.3 容器内路径约定

DCS 容器内模型路径与本地 .env 不同，通过环境变量覆盖：

```bash
# 容器内启动适配层(模型挂载在 /AI_models/ 下)
MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf \
MODEL_1B_32k_PATH=/AI_models/rice_reg/rice_1B_32k_hf \
PORT=6001 \
python /code/dcs_adapter.py
```

## 5. 常见问题

- **FlashAttention 报错**：`model_type` 改为 `no_flash`，或去掉 `.env` 中的 `_TYPE=flash`
- **OOM**：减小 `MODEL_<NAME>_MAX_LEN`；或 `--device cuda:0,cuda:1 --device_map auto`
- **NPU 多卡**：`DEVICE=npu:0,npu:1`（accelerate 不支持 NPU 设备的 max_memory，会自动手动切层分配）
- **embedding 为 bf16**：服务自动转 fp16 输出，与下游 numpy 兼容
