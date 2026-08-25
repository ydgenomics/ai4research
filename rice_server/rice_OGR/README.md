# Rice-OGR — DNA Embedding 提取 API 服务

基于 **Genos DNA 大模型**与**水稻基因组基模**（`rice_1B_stage2_8k_hf` / `rice_1B_32k_hf`）的序列 Embedding 提取服务。
提供 RESTful API，支持 GPU / 昇腾 NPU / CPU，多设备并行，多种池化方式，以及下游碱基预测。

> 上游参考：[BGI-HangzhouAI/Genos](https://github.com/BGI-HangzhouAI/Genos/tree/main)

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
├── dna_embedding.py        # 主服务程序(EmbeddingExtractor + Sanic API)
├── .env                    # ★ 模型注册表 + 服务/设备配置(部署时必改)
├── .env.example            # 配置模板
├── README.md               # 本文档
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

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 + 设备/模型信息 |
| `/models` | GET | 已注册/已加载模型列表 |
| `/extract` | POST | 提取 DNA 序列 Embedding |
| `/predict` | POST | 预测下游碱基 |

### 3.1 嵌入提取 `/extract`

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "model_name": "1B_8k",
    "pooling_method": "mean"
  }'
```

参数：
- `sequence`（必填）：DNA 序列
- `model_name`（必填）：注册表模型名，如 `1B_8k` / `1B_32k`
- `pooling_method`（可选）：`mean`（默认）/ `max` / `last` / `none`

响应示例：

```json
{
  "success": true,
  "message": "客户端序列embedding提取成功",
  "result": {
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "sequence_length": 24,
    "token_count": 24,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "mean",
    "model_type": "flash",
    "device": "cuda:0",
    "embedding": [0.123, 0.456, ...]
  }
}
```

> 水稻基模（1B_8k / 1B_32k）为单碱基编码：1 bp = 1 token，`token_count` 即输入碱基数（受 `MAX_LEN` 截断）。

### 3.2 碱基预测 `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "1B_32k", "predict_length": 10}'
```

## 4. Host 本地测试

### 4.1 健康检查 / 模型列表

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/models | python3 -m json.tool
```

### 4.2 Embedding 提取冒烟测试

```bash
curl -s -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGT", "model_name": "1B_8k", "pooling_method": "mean"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('shape:', r['embedding_shape'], 'dim:', r['embedding_dim'], 'device:', r['device'])"
# 期望输出: shape: [1, 1024] dim: 1024 device: cuda:0
```

四种池化对比：

```bash
for m in mean max last none; do
  echo "== $m =="
  curl -s -X POST http://localhost:8000/extract -H "Content-Type: application/json" \
    -d "{\"sequence\":\"ACGTTGCATGCAACGT\",\"model_name\":\"1B_8k\",\"pooling_method\":\"$m\"}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('shape:', r['embedding_shape'])"
done
# mean/max/last → [1, 1024]; none → [1, L, 1024]
```

### 4.3 超长序列截断测试（验证 MAX_LEN）

> 注意：水稻基模 tokenizer 为**单碱基编码**（1 bp = 1 token，A→8 / C→5 / G→6 / T→7 / N→9）。
> 因此 `MODEL_<NAME>_MAX_LEN` 截断的是 **token 数 = 碱基数**。

```bash
python3 -c "
import urllib.request, json
seq = 'ACGT' * 20000   # 8 万 bp = 8 万 tokens
req = urllib.request.Request('http://localhost:8000/extract',
      data=json.dumps({'sequence': seq, 'model_name': '1B_8k', 'pooling_method': 'mean'}).encode(),
      headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req).read())['result']['token_count'])
"   # 期望 = 32768 (MAX_LEN 截断生效)
```

### 4.4 两个水稻基模对比

```bash
curl -s -X POST http://localhost:8000/extract -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT", "model_name": "1B_32k", "pooling_method": "mean"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('32k dim:', r['embedding_dim'], 'tokens:', r['token_count'])"
```

> 注意：1B_8k 与 1B_32k 的 `rope_theta` 不同（5e7 vs 1e6），同一序列的 embedding 不具可比性，分别用于各自下游任务（rice_mut / rice_reg）。

### 4.5 碱基预测 `/predict` 本地测试

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "1B_32k", "predict_length": 10}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('预测碱基:', r.get('predicted_sequence', r))"
```

> `predict_length` 上限 1000；返回字段见下方 DCS 版响应示例（`predicted_sequence` / `token_count` 等）。

### 4.6 tokenizer 编码对照

水稻基模 tokenizer 为 BPE(ByteLevel) + 单碱基词表（vocab=18），映射关系：

| 碱基 | token id |
|---|---|
| A | 8 |
| C | 5 |
| G | 6 |
| T | 7 |
| N | 9 |

特殊 token：`<CLS>`(10) `<SEP>`(11) `<EOD>`(12) `<MASK>`(13) `<PAD>`(14) `<s>`(15) `</s>`(16) `<UNK>`(17)。

## 5. DCS 平台部署与测试

DCS 适配方式与 rice_mut 一致：新增独立适配层 `dcs_adapter.py`（FastAPI），
对外暴露 **OpenAI 风格单入口**，复用 `dna_embedding.EmbeddingExtractor` 的推理逻辑，
并自动注入 `usage` 计费字段、支持鉴权与 JSON 末尾换行。

### 5.1 架构总览

```
DCS 网关(https://<dcs-host>/api/aigress/openai/rice_ogr)
        │  OpenAI 风格请求体 {model, mode, sequence, ...}
        ▼
rice_OGR/dcs_adapter.py  (FastAPI, 默认端口 8001)
        │  mode 分发: dna_embedding | predict
        ▼
dna_embedding.EmbeddingExtractor  (模型加载/设备/推理, 复用原逻辑)
        │
        ▼
{"usage": {...}, "status": 200, "message": "...", "result": {...}}  (+换行)
```

> Sanic 原服务 (`dna_embedding.py`, 端口 8000) 保留用于本地直连 / 网页版；
> DCS 部署走适配层 (`dcs_adapter.py`, 默认 8001)，两个端口互不冲突。

### 5.2 启动适配层

```bash
cd rice_OGR
python dcs_adapter.py                    # 按 .env 注册表预加载全部模型, 监听 8001
PORT=8001 python dcs_adapter.py          # 显式指定端口(平台注入 PORT 时自动生效)
```

> 本地联调时若模型未预加载完成就发请求, 适配层会返回 503 — 先 `curl /health` 确认
> `init_error` 为 `null` 且 `models.loaded` 非空再调用(见 §5.5 第 1 步)。
> 鉴权：`.env` 配了 `DCS_API_KEY` 时本地 POST 也要带 `Authorization: Bearer <key>`(否则 401)。

### 5.3 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/aigress/openai/rice_ogr`、`/rice_ogr` | POST | 单入口，按 `mode` 分发 `dna_embedding` / `predict` |
| `/api/aigress/openai/rice_ogr/dna_embedding` | POST | 子路径方式(等价 `mode=dna_embedding`) |
| `/api/aigress/openai/rice_ogr/predict` | POST | 子路径方式(等价 `mode=predict`) |
| `/api/aigress/openai/health`、`/health` | GET/POST | 健康检查 + 诊断(免鉴权) |
| `/api/aigress/openai/models`、`/models` | GET | 模型列表(免鉴权) |

### 5.4 单入口请求示例(DCS 网关风格)

> `model` 为**服务名**(`rice_ogr`,与入口 path 末段一致);实际**模型名**放在 `model_name`
> (如 `1B_8k` / `1B_32k`)。向后兼容:`model` 不等于 `rice_ogr` 时仍视为模型名。

```bash
# mode=dna_embedding(默认,--data 中可省略 mode)
curl --location 'https://<dcs-host>/api/aigress/openai/rice_ogr' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "mode": "dna_embedding",
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# mode=predict(下游碱基预测)
curl --location 'https://<dcs-host>/api/aigress/openai/rice_ogr' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "mode": "predict",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'
```

`mode` 未指定时自动推断：带 `predict_length` → `predict`，否则 → `dna_embedding`(向后兼容)。

### 5.5 本地冒烟测试

**1) 健康检查 / 模型列表(末尾自动换行, 不会与 shell 提示符粘连)**

```bash
curl -s http://localhost:8001/api/aigress/openai/health | python3 -m json.tool
curl -s http://localhost:8001/api/aigress/openai/models | python3 -m json.tool
```

关键字段：`status`（`ok`）、`diagnostics.init_error`（**必须为 `null`**）、`models.loaded`（已加载模型名）。

**2) 等待模型就绪(可选, 防止 503)**

```bash
until curl -s http://localhost:8001/api/aigress/openai/health \
  | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok' and d['diagnostics']['init_error'] is None" 2>/dev/null; do
  echo "⏳ 模型未就绪, 2s 后重试..."; sleep 2
done; echo "✅ 模型就绪"
```

**3) embedding 提取(默认 mode)**

```bash
curl -s -X POST http://localhost:8001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" \
  -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}' \
  | python3 -m json.tool
# 期望: usage.prompt_tokens=16, result.embedding_shape=[1, 1024], pooling_method=mean, model_name=1B_8k
```

四种池化对比:

```bash
for m in mean max last none; do
  echo "== $m =="
  curl -s -X POST http://localhost:8001/api/aigress/openai/rice_ogr -H "Content-Type: application/json" \
    -d "{\"model\":\"rice_ogr\",\"model_name\":\"1B_8k\",\"sequence\":\"ACGTTGCATGCAACGT\",\"pooling_method\":\"$m\"}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('shape:', r['embedding_shape'])"
done
# mean/max/last → [1, 1024]; none → [1, L, 1024]
```

**4) 碱基预测(mode=predict)**

```bash
curl -s -X POST http://localhost:8001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" \
  -d '{"model": "rice_ogr", "model_name": "1B_8k", "mode": "predict", "sequence": "ACGTTGCATGCAACGT", "predict_length": 5}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('预测碱基:', r['predicted_bases'], '| 全长:', r['predicted_sequence'])"
```

**5) 鉴权测试(配了 `DCS_API_KEY` 时)**

```bash
# 不带 key → 401
curl -s -o /dev/null -w "无key: HTTP %{http_code}\n" -X POST http://localhost:8001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT"}'

# 带 key → 200
curl -s -o /dev/null -w "带key: HTTP %{http_code}\n" -X POST http://localhost:8001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT"}'
# 期望: 无key: HTTP 401 / 带key: HTTP 200
```

**6) 子路径(等价 mode 写法)**

```bash
curl -s -X POST http://localhost:8001/api/aigress/openai/rice_ogr/dna_embedding \
  -H "Content-Type: application/json" -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT"}'
curl -s -X POST http://localhost:8001/api/aigress/openai/rice_ogr/predict \
  -H "Content-Type: application/json" -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT", "predict_length": 5}'
```

> 以上地址将 `localhost:8001` 换成 DCS 网关地址 + 带 `Authorization: Bearer <key>` 即为线上调用方式。

### 5.6 返回结构与计费

遵循 `rice_server/dcs.md` 规范，返回 `{"usage", "status", "message", "result"}`。

**mode=dna_embedding 响应示例：**

```json
{
  "usage": {"prompt_tokens": 16, "completion_tokens": 0},
  "status": 200,
  "message": "DNA sequence embedding 提取成功",
  "result": {
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "mode": "dna_embedding",
    "sequence": "ACGTTGCATGCAACGT",
    "token_count": 16,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "mean",
    "embedding": [...]
  }
}
```

**mode=predict 响应示例（`result` 与 Sanic `/predict` 一致）：**

```json
{
  "usage": {"prompt_tokens": 16, "completion_tokens": 5},
  "status": 200,
  "message": "下游碱基预测成功",
  "result": {
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "mode": "predict",
    "original_sequence": "ACGTTGCATGCAACGT",
    "predicted_sequence": "ACGTTGCATGCAACGTTCGAT",
    "predicted_bases": "TCGAT",
    "predict_length": 5,
    "total_length": 21,
    "elapsed_seconds": 0.1234
  }
}
```

- `prompt_tokens` = 输入 token 数(水稻基模单碱基编码: 1 bp = 1 token = 原序列长度 = `total_length - predict_length`)
- `completion_tokens` = predict 模式为**预测碱基数**(= `predict_length`); dna_embedding 模式恒为 0
- 计费系数可通过 `DCS_PROMPT_TOKEN_MULTIPLIER` / `DCS_COMPLETION_TOKEN_MULTIPLIER` 调整(默认 1)
- 鉴权: `.env` 配置 `DCS_API_KEY` 后 POST 需带 `Authorization: Bearer <key>` / `X-API-Key: <key>`; 留空则免鉴权

### 5.7 容器内路径约定

DCS 容器内模型路径与本地 .env 不同，通过环境变量覆盖：

```bash
# 容器内启动适配层(模型挂载在 /AI_models/ 下)
MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf \
MODEL_1B_32k_PATH=/AI_models/rice_reg/rice_1B_32k_hf \
PORT=8001 \
python /code/dcs_adapter.py
```

## 6. 常见问题

- **FlashAttention 报错**：`model_type` 改为 `no_flash`，或去掉 `.env` 中的 `_TYPE=flash`
- **OOM**：减小 `MODEL_<NAME>_MAX_LEN`；或 `--device cuda:0,cuda:1 --device_map auto`
- **NPU 多卡**：`DEVICE=npu:0,npu:1`（accelerate 不支持 NPU 设备的 max_memory，会自动手动切层分配）
- **embedding 为 bf16**：服务自动转 fp16 输出，与下游 numpy 兼容