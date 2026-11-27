# OGR-Base API 调用文档

> 适用服务：`OGR_Base/dna_embedding.py`（Sanic 原生服务）+ `OGR_Base/dcs_adapter.py`（DCS 适配层）
> 功能：**DNA 序列 Embedding 提取**（四种池化） 与 **下游碱基预测 NTP**（自回归续写）。
> **子路径决定功能（`Embedding` | `NTP`），无 `mode` / `model_name` 参数；模型固定为注册表中的 `1B_8k`。**
> 本文档包含 **本机（host）测试** 与 **DCS 平台测试** 两套可直接复制的 curl 代码。

---

## 1. 入口总览

| 场景 | 入口 | 说明 |
|---|---|---|
| **本机 Sanic 原生服务** | `POST http://127.0.0.1:8000/OGR-Base/{Embedding,NTP}` | 本地直连（`dna_embedding.py`，端口 8000）；响应为 `{success,status,message,result}` |
| **本机 DCS 适配层** | `POST http://127.0.0.1:6001/api/aigress/openai/OGR-Base/{Embedding,NTP}`（简写 `/OGR-Base/...`） | DCS 适配（`dcs_adapter.py`，默认 6001）；外层统一信封 |
| **DCS 统一网关（URL 路径，推荐）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Base/{Embedding,NTP}` | 服务与功能放路径段，body 无 `model_sub`/`mode` |
| **兼容：缺省服务** | `POST .../api/aigress/openai/OGR/{Embedding,NTP}` | `OGR-Base` 为缺省服务，`/OGR/Embedding` 等价 `/OGR/OGR-Base/Embedding` |
| **DCS 统一网关（body 字段，兼容）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR` + body `model_sub=OGR-Base` | 旧式调用 |

> 统一网关由 `dcs_gateway/` 提供（唯一入口 `…/OGR`），完整说明见 [dcs_gateway/API.md](../dcs_gateway/API.md)。
> 端口约定：网关 `.env` 的 `RICE_OGR_PORT`（默认 **6001**，与 OGR-Prediction 8001 错开）；本机运行 Sanic 原生服务用 8000。

### 1.1 子路径与固定模型

| 子路径 | 功能 | 必填键 | 可选键 |
|---|---|---|---|
| `Embedding` | 提取序列 embedding | `sequence` | `pooling_method`（`mean`/`max`/`last`/`none`，默认 `mean`） |
| `NTP` | 下游碱基预测（自回归续写） | `sequence` | `predict_length`（1–1000，默认 10） |

> **不再有 `mode` 分发、不再接受 `model_name` / `model` 键**：模型固定为注册表 `1B_8k`
> （注册表无 `1B_8k` 时回退第一个已注册模型）。请求体中若带 `mode` / `model_name`，会被忽略。

---

## 2. 请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；部署端 `.env` 配置了 `DCS_API_KEY` 时 POST 必填，`health`/`models` 免鉴权 |
| `X-API-Key` | ⚠️ | `Authorization` 的等价替代 |

> 服务端 `.env` 未配置 `DCS_API_KEY` 则所有 POST 免鉴权。

### 2.1 模型注册（`.env` 的 `MODEL_<NAME>_*`）

| 配置项 | 必填 | 说明 |
|---|---|---|
| `MODEL_<NAME>_PATH` | 是 | 模型目录路径（HF 格式，含 `config.json`/`tokenizer.json`/`model.safetensors`） |
| `MODEL_<NAME>_TYPE` | 否 | `flash`（默认）或 `no_flash` |
| `MODEL_<NAME>_MAX_LEN` | 否 | 最大序列长度，超长截断（防 OOM）；本部署 8k 基模为 `131072`（128k） |
| `MODEL_<NAME>_SPECIAL` | 否 | 是否添加特殊 token：`1`/`0`（水稻基模统一建议 `0`） |

> 服务运行时固定取注册表中的 **`1B_8k`**，无需在请求里指定模型名。

---

## 3. 本机 Sanic 原生服务测试（`dna_embedding.py`，端口 8000）

### 3.1 健康检查 / 模型列表

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/models | python3 -m json.tool
```

### 3.2 Embedding 提取 `/OGR-Base/Embedding`

```bash
curl -X POST http://localhost:8000/OGR-Base/Embedding \
  -H "Content-Type: application/json" \
  -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "pooling_method": "mean"}'
```

参数：
- `sequence`（必填）：DNA 序列
- `pooling_method`（可选）：`mean`（默认）/ `max` / `last` / `none`

响应示例（Sanic 原生层为 `{success,status,message,result}`，无 `usage` 信封）：

```json
{
  "success": true,
  "status": 200,
  "message": "DNA sequence embedding extraction succeeded",
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

> 水稻基模（`1B_8k`）为单碱基编码：1 bp = 1 token，`token_count` 即输入碱基数（受 `MAX_LEN` 截断）。

### 3.3 碱基预测 `/OGR-Base/NTP`

```bash
curl -X POST http://localhost:8000/OGR-Base/NTP \
  -H "Content-Type: application/json" \
  -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "predict_length": 10}'
```

### 3.4 冒烟测试

```bash
curl -s -X POST http://localhost:8000/OGR-Base/Embedding \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('shape:', r['embedding_shape'], 'dim:', r['embedding_dim'], 'device:', r['device'])"
# 期望输出: shape: [1, 1024] dim: 1024 device: cuda:0
```

四种池化对比：

```bash
for m in mean max last none; do
  echo "== $m =="
  curl -s -X POST http://localhost:8000/OGR-Base/Embedding -H "Content-Type: application/json" \
    -d "{\"sequence\":\"ACGTTGCATGCAACGT\",\"pooling_method\":\"$m\"}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('shape:', r['embedding_shape'])"
done
# mean/max/last → [1, 1024]; none → [1, L, 1024]
```

### 3.5 超长序列截断测试（验证 MAX_LEN）

> 水稻基模 tokenizer 为**单碱基编码**（1 bp = 1 token，A→8 / C→5 / G→6 / T→7 / N→9）。
> 因此 `MODEL_<NAME>_MAX_LEN` 截断的是 **token 数 = 碱基数**。

```bash
python3 -c "
import urllib.request, json
seq = 'ACGT' * 40000   # 16 万 bp
req = urllib.request.Request('http://localhost:8000/OGR-Base/Embedding',
      data=json.dumps({'sequence': seq, 'pooling_method': 'mean'}).encode(),
      headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req).read())['result']['token_count'])
"   # 期望 = 131072（本部署 MAX_LEN 截断生效）
```

### 3.6 两个水稻基模对比（如注册表同时启用）

```bash
# 仅当 .env 同时注册了 1B_32k 时才可用；OGR-Base 服务固定使用 1B_8k
curl -s -X POST http://localhost:8000/OGR-Base/Embedding -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT", "pooling_method": "mean"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('dim:', r['embedding_dim'], 'tokens:', r['token_count'])"
```

> 注意：`1B_8k` 与 `1B_32k` 的 `rope_theta` 不同（5e7 vs 1e6），同一序列的 embedding 不具可比性，分别用于各自下游任务（rice_mut / rice_reg）。

---

## 4. DCS 适配层测试（`dcs_adapter.py`，默认 6001）

### 4.1 健康检查 / 模型列表

```bash
# 本机直连适配层
curl -s http://localhost:6001/api/aigress/openai/health | python3 -m json.tool
curl -s http://localhost:6001/api/aigress/openai/models | python3 -m json.tool
# 经网关（默认服务 OGR-Base）
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/models | python3 -m json.tool
```

关键字段：`status`（`ok`）、`diagnostics.init_error`（**必须为 `null`**）、`models.loaded`（已加载模型名）。

### 4.2 子路径请求示例（无 `mode` / `model_name`）

```bash
# Embedding
curl -s -X POST http://127.0.0.1:6001/api/aigress/openai/OGR-Base/Embedding \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}' \
  | python3 -m json.tool
# 期望: usage.prompt_tokens=16, result.embedding_shape=[1,1024], model_name=1B_8k

# NTP（下游碱基预测）
curl -s -X POST http://127.0.0.1:6001/api/aigress/openai/OGR-Base/NTP \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGT", "predict_length": 5}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('预测碱基:', r['predicted_bases'], '| 全长:', r['predicted_sequence'])"
```

> 简写路径（无 `/api/aigress/openai` 前缀）同样可用：`POST http://127.0.0.1:6001/OGR-Base/Embedding`、`/OGR-Base/NTP`。

### 4.3 鉴权测试（配了 `DCS_API_KEY` 时）

```bash
# 不带 key → 401
curl -s -o /dev/null -w "无key: HTTP %{http_code}\n" -X POST http://127.0.0.1:6001/api/aigress/openai/OGR-Base/Embedding \
  -H "Content-Type: application/json" -d '{"sequence": "ACGT"}'

# 带 key → 200
curl -s -o /dev/null -w "带key: HTTP %{http_code}\n" -X POST http://127.0.0.1:6001/api/aigress/openai/OGR-Base/Embedding \
  -H "Content-Type: application/json" -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{"sequence": "ACGT"}'
# 期望: 无key: HTTP 401 / 带key: HTTP 200
```

---

## 5. DCS 平台测试（统一网关）

```bash
# ── URL 路径方式（推荐，body 无 model_sub / mode）──
# embedding
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Base/Embedding' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# 碱基预测
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Base/NTP' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'

# ── 省略 OGR-Base（缺省服务）──
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR/Embedding' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{"sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}'

# ── body 字段方式（兼容；model_sub 缺省即 OGR-Base，可省略）──
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model_sub": "OGR-Base",
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'
```

> 将 `localhost:6001` 换成 DCS 网关地址 + 带 `Authorization: Bearer <key>` 即为线上调用方式。
> `OGR-Base` 为网关缺省服务：`/OGR/Embedding`、`/OGR/NTP` 直接用子路径亦可路由到 OGR-Base。

---

## 6. 返回结构与计费

适配层遵循 `rice_server/dcs_gateway/API.md` 统一规范，返回 `{"model_name", "usage", "token_cnt", "status", "message", "result"}`。

**Embedding 响应示例：**

```json
{
  "model_name": "OGR",
  "usage": {"prompt_tokens": 16, "completion_tokens": 0},
  "token_cnt": 16,
  "status": 200,
  "message": "DNA sequence embedding extraction succeeded",
  "result": {
    "model": "OGR-Base",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "sequence_length": 16,
    "token_count": 16,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "mean",
    "embedding": [ ... ]
  }
}
```

**NTP 响应示例（`result` 与 Sanic 层一致）：**

```json
{
  "model_name": "OGR",
  "usage": {"prompt_tokens": 16, "completion_tokens": 5},
  "token_cnt": 21,
  "status": 200,
  "message": "Downstream base (NTP) prediction succeeded",
  "result": {
    "model": "OGR-Base",
    "model_name": "1B_8k",
    "original_sequence": "ACGTTGCATGCAACGT",
    "predicted_sequence": "ACGTTGCATGCAACGTTCGAT",
    "predicted_bases": "TCGAT",
    "predict_length": 5,
    "total_length": 21,
    "elapsed_seconds": 0.1234
  }
}
```

- `prompt_tokens` = 输入 token 数（水稻基模单碱基编码：1 bp = 1 token）
  - Embedding：= `token_count`（= 输入序列长度，超长截断后）
  - NTP：= `total_length - predict_length`（上下文碱基数）
- `completion_tokens` = Embedding 恒为 0；NTP = `predict_length`（预测碱基数）
- `token_cnt`（顶层）= `prompt_tokens + completion_tokens`
- 计费系数可通过 `DCS_PROMPT_TOKEN_MULTIPLIER` / `DCS_COMPLETION_TOKEN_MULTIPLIER` 调整（默认 1）
- 鉴权：`.env` 配置 `DCS_API_KEY` 后 POST 需带 `Authorization: Bearer <key>` / `X-API-Key: <key>`；留空则免鉴权

---

## 7. 容器内路径约定

DCS 容器内模型路径与本地 .env 不同，通过环境变量覆盖：

```bash
# 容器内启动适配层（模型挂载在 /AI_models/ 下）
MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf \
PORT=6001 \
python /code/dcs_adapter.py
```

---

## 8. 常见问题

- **FlashAttention 报错**：`MODEL_<NAME>_TYPE` 改为 `no_flash`，或移除 `.env` 中的 `_TYPE=flash`
- **OOM**：减小 `MODEL_<NAME>_MAX_LEN`；或 `--device cuda:0,cuda:1 --device_map auto`
- **NPU 多卡**：`DEVICE=npu:0,npu:1`（accelerate 不支持 NPU 设备的 max_memory，会自动手动切层分配）
- **接口 404**：旧 `/extract`、`/predict`、`/rice_ogr` 已移除，改用 `/OGR-Base/Embedding`、`/OGR-Base/NTP`
- **不再接受 `model_name` / `mode`**：模型固定 `1B_8k`，功能由子路径决定；带这两个键会被忽略
- **端口冲突**：适配层默认 `BACKEND_PORT=8001` 与 OGR-Prediction 同端口，同机必须用 `BACKEND_PORT=6001` 覆盖
- **embedding 为 bf16**：服务自动转 fp16 输出，与下游 numpy 兼容