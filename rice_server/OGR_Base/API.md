# rice_OGR API 调用文档

> 适用服务：`rice_OGR/dna_embedding.py`（Sanic 原生服务）+ `rice_OGR/dcs_adapter.py`（DCS 适配层）
> 功能：**DNA 序列 Embedding 提取**（四种池化） 与 **下游碱基预测**（自回归续写）。
> 本文档包含 **本机（host）测试** 与 **DCS 平台测试** 两套可直接复制的 curl 代码。

---

## 1. 入口总览

| 场景 | 入口 | 说明 |
|---|---|---|
| **本机 Sanic 原生服务** | `POST http://127.0.0.1:8000/extract`、`…/predict` | 本地调试（`dna_embedding.py`，端口 8000） |
| **本机 DCS 适配层** | `POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr` | DCS 适配（`dcs_adapter.py`，默认 6001） |
| **DCS 统一网关（URL 路径，推荐）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/{predict,dna_embedding}` | 服务与功能放路径段，body 无 `model_sub`/`mode` |
| **DCS 统一网关（body 字段，兼容）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR` + body `model_sub=rice_ogr` | 旧式调用（`model_sub` 缺省即 `rice_ogr`） |

> 统一网关由 `dcs_gateway/` 提供（唯一入口 `…/OGR`），完整说明见 [dcs_gateway/API.md](../dcs_gateway/API.md)。
> 端口约定：网关 `.env` 的 `RICE_OGR_PORT`（默认 **6001**，与 rice_mut 8001 错开）；本机运行 Sanic 原生服务用 8000。

### 1.1 模式分发（单入口 `POST /rice_ogr` + body `mode`）

| 请求体 | 推断模式 | 说明 |
|---|---|---|
| `{"mode":"dna_embedding", "sequence":..., ...}` | **dna_embedding** | 提取序列向量（默认） |
| `{"mode":"predict", "sequence":..., "predict_length":N}` | **predict** | 下游碱基预测 |
| 带 `predict_length`（无 mode） | **predict** | 自动推断 |
| 其他（有 `sequence`，无 mode） | **dna_embedding** | 自动推断（向后兼容） |

> **显式 `mode` 优先级最高**；未指定时按「带 `predict_length` → predict、否则 → dna_embedding」自动推断。

---

## 2. 请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；部署端 `.env` 配置了 `DCS_API_KEY` 时 POST 必填，`health`/`models` 免鉴权 |

> 鉴权同时兼容 `X-API-Key: <DCS_API_KEY>` 头。服务端 `.env` 未配置 `DCS_API_KEY` 则所有 POST 免鉴权。

### 2.1 模型注册（`.env` 的 `MODEL_<NAME>_*`）

| 配置项 | 必填 | 说明 |
|---|---|---|
| `MODEL_<NAME>_PATH` | 是 | 模型目录路径（HF 格式，含 `config.json`/`tokenizer.json`/`model.safetensors`） |
| `MODEL_<NAME>_TYPE` | 否 | `flash`（默认）或 `no_flash` |
| `MODEL_<NAME>_MAX_LEN` | 否 | 最大序列长度，超长截断（防 OOM） |
| `MODEL_<NAME>_SPECIAL` | 否 | 是否添加特殊 token：`1`/`0`（水稻基模统一建议 `0`） |

> 实际模型名用 `model_name` 字段指定（如 `1B_8k` / `1B_32k`）；`model_name` 缺省时向后兼容取 `model`；两者都缺省取注册表第一个。

---

## 3. 本机 Sanic 原生服务测试（`dna_embedding.py`，端口 8000）

### 3.1 健康检查 / 模型列表

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/models | python3 -m json.tool
```

### 3.2 Embedding 提取 `/extract`

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "1B_8k", "pooling_method": "mean"}'
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

### 3.3 碱基预测 `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "1B_32k", "predict_length": 10}'
```

### 3.4 冒烟测试

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

### 3.5 超长序列截断测试（验证 MAX_LEN）

> 水稻基模 tokenizer 为**单碱基编码**（1 bp = 1 token，A→8 / C→5 / G→6 / T→7 / N→9）。
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

### 3.6 两个水稻基模对比

```bash
curl -s -X POST http://localhost:8000/extract -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT", "model_name": "1B_32k", "pooling_method": "mean"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('32k dim:', r['embedding_dim'], 'tokens:', r['token_count'])"
```

> 注意：1B_8k 与 1B_32k 的 `rope_theta` 不同（5e7 vs 1e6），同一序列的 embedding 不具可比性，分别用于各自下游任务（rice_mut / rice_reg）。

---

## 4. DCS 适配层测试（`dcs_adapter.py`，默认 6001）

### 4.1 健康检查 / 模型列表

```bash
# 本机直连适配层
curl -s http://localhost:6001/api/aigress/openai/health | python3 -m json.tool
curl -s http://localhost:6001/api/aigress/openai/models | python3 -m json.tool
# 经网关（默认路由 rice_ogr）
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/models | python3 -m json.tool
```

关键字段：`status`（`ok`）、`diagnostics.init_error`（**必须为 `null`**）、`models.loaded`（已加载模型名）。

### 4.2 单入口请求示例（DCS 网关风格）

> `model` 为**服务名**（`rice_ogr`，与入口 path 末段一致）；实际**模型名**放在 `model_name`（如 `1B_8k` / `1B_32k`）。向后兼容：`model` 不等于 `rice_ogr` 时仍视为模型名。

```bash
# mode=dna_embedding（默认，--data 中可省略 mode）
curl -s -X POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" \
  -d '{"model": "rice_ogr", "model_name": "1B_8k", "mode": "dna_embedding", "sequence": "ACGTTGCATGCAACGT", "pooling_method": "mean"}' \
  | python3 -m json.tool
# 期望: usage.prompt_tokens=16, result.embedding_shape=[1, 1024], pooling_method=mean, model_name=1B_8k

# mode=predict（下游碱基预测）
curl -s -X POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" \
  -d '{"model": "rice_ogr", "model_name": "1B_8k", "mode": "predict", "sequence": "ACGTTGCATGCAACGT", "predict_length": 5}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print('预测碱基:', r['predicted_bases'], '| 全长:', r['predicted_sequence'])"
```

### 4.3 子路径（等价 mode 写法）

```bash
curl -s -X POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr/dna_embedding \
  -H "Content-Type: application/json" -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT"}'
curl -s -X POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr/predict \
  -H "Content-Type: application/json" -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT", "predict_length": 5}'
```

### 4.4 鉴权测试（配了 `DCS_API_KEY` 时）

```bash
# 不带 key → 401
curl -s -o /dev/null -w "无key: HTTP %{http_code}\n" -X POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT"}'

# 带 key → 200
curl -s -o /dev/null -w "带key: HTTP %{http_code}\n" -X POST http://127.0.0.1:6001/api/aigress/openai/rice_ogr \
  -H "Content-Type: application/json" -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{"model": "rice_ogr", "model_name": "1B_8k", "sequence": "ACGT"}'
# 期望: 无key: HTTP 401 / 带key: HTTP 200
```

---

## 5. DCS 平台测试（统一网关）

```bash
# ── URL 路径方式（推荐，body 无 model_sub / mode）──
# embedding
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/dna_embedding' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# 碱基预测
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/predict' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'

# ── body 字段方式（兼容；model_sub 缺省即 rice_ogr，可省略）──
curl --location 'https://www.dcs.cloud/api/aigress/openai/OGR' \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model_sub": "rice_ogr",
    "mode": "dna_embedding",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'
```

> 将 `localhost:6001` 换成 DCS 网关地址 + 带 `Authorization: Bearer <key>` 即为线上调用方式。

---

## 6. 返回结构与计费

遵循 `rice_server/dcs.md` 规范，返回 `{"usage", "status", "message", "result"}`。

**mode=dna_embedding 响应示例：**

```json
{
  "usage": {"prompt_tokens": 16, "completion_tokens": 0},
  "status": 200,
  "message": "DNA sequence embedding extraction succeeded",
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
  "message": "Downstream base prediction succeeded",
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

- `prompt_tokens` = 输入 token 数（水稻基模单碱基编码：1 bp = 1 token = 原序列长度 = `total_length - predict_length`）
- `completion_tokens` = predict 模式为**预测碱基数**（= `predict_length`）；dna_embedding 模式恒为 0
- 计费系数可通过 `DCS_PROMPT_TOKEN_MULTIPLIER` / `DCS_COMPLETION_TOKEN_MULTIPLIER` 调整（默认 1）
- 鉴权：`.env` 配置 `DCS_API_KEY` 后 POST 需带 `Authorization: Bearer <key>` / `X-API-Key: <key>`；留空则免鉴权

---

## 7. 容器内路径约定

DCS 容器内模型路径与本地 .env 不同，通过环境变量覆盖：

```bash
# 容器内启动适配层（模型挂载在 /AI_models/ 下）
MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf \
MODEL_1B_32k_PATH=/AI_models/rice_reg/rice_1B_32k_hf \
PORT=6001 \
python /code/dcs_adapter.py
```

---

## 8. 常见问题

- **FlashAttention 报错**：`model_type` 改为 `no_flash`，或去掉 `.env` 中的 `_TYPE=flash`
- **OOM**：减小 `MODEL_<NAME>_MAX_LEN`；或 `--device cuda:0,cuda:1 --device_map auto`
- **NPU 多卡**：`DEVICE=npu:0,npu:1`（accelerate 不支持 NPU 设备的 max_memory，会自动手动切层分配）
- **embedding 为 bf16**：服务自动转 fp16 输出，与下游 numpy 兼容