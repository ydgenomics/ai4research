# 快速上手（外部用户版）

本文档面向 **外部 API 调用方**：用最少的篇幅说明三个服务各自能做什么、DCS API 怎么调。
不涉及部署/维护细节。

---

## 0. 三个服务能做什么

| 服务 | 能力 | 一句话 |
|---|---|---|
| **rice_mut** | DNA → RNA表达预测 | 输入一段 DNA 序列，输出参考基因组与**单碱基变异（SNV）**对比下的多维表达轨迹 |
| **rice_reg** | DNA + ATAC → RNA-seq 表达预测 | 输入 DNA 区域 + 可选 ATAC 信号源，输出该条件下的 RNA-seq 表达轨迹 |
| **rice_OGR** | 基模直接开放 | 输入 DNA 序列，输出 **1024 维 embedding 向量**，或**续写预测下游碱基** |

---

## 1. 调用方式

三个服务各有**独立入口**（本地联调 / 单独部署时直连），也可经**统一网关入口**（DCS 平台只配一个地址）调用：

| 入口 | 地址 | 适用场景 |
|---|---|---|
| **独立入口**（每服务一个） | `POST ${host}/api/aigress/openai/rice_mut` / `.../rice_reg` / `.../rice_ogr` | 本地联调 / 单个服务独立部署 |
| **统一网关入口** | `POST ${dcs_host}/api/aigress/openai/OGR` + body `model_sub` 选服务 | DCS 平台（唯一地址） |

公共约定（两种入口一致）：

- **服务路由**（仅统一入口）：请求体带 **`model_sub`**：`rice_mut` / `rice_reg` / `rice_ogr`（**缺省 = `rice_ogr`**）
- **功能选择**：请求体 `mode` 字段选择服务内功能（见下）；其余字段随服务不同
- **鉴权**：请求头带 `Authorization: Bearer ${api_key}`，或 `X-API-Key: ${api_key}`
- **返回**：统一为 `{"usage": {...}, "status": 200, "message": "...", "result": {...}}`
- **计费口径**：水稻基模单碱基编码，**1 bp = 1 token**；`usage.prompt_tokens` 即输入序列的 token 数

> DCS 平台入口 `${dcs_host}` 由平台提供；本地联调替换为 `localhost:<端口>`（各服务端口见对应章节）。

```bash
# ── 变量 ──────────────────────────────────────────────
# for local（本地直连：各服务独立入口，端口见各服务章节）
api_key="${api_key}"
host="http://0.0.0.0"

# for dcs（统一网关：DCS 平台唯一地址 + 统一入口路径）
dcs_api_key="sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"
dcs_host="https://www.dcs.cloud"
dcs_entry="/api/aigress/openai/OGR"
```

## 2. rice_mut — 变异对比表达预测

**独立入口**：`POST ${host}:8001/api/aigress/openai/rice_mut`（本地后端端口 8001）
**统一入口**：`POST ${dcs_host}${dcs_entry}` + `"model_sub": "rice_mut"`

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | 参考序列表达预测 | `chromosome` + `start` |
| `snv` | 单突变位点的窗口轨迹对比 | `chromosome` + `start` + `snv_index` + `snv_base` |

> 染色体统一 `chr01`–`chr12` 命名；位置 **1-based inclusive**。
> `genome`/`end`/`output_format` 可选（缺省：唯一基因组 / 32768 窗口 / `full`）。
> `mode=snv` 可省略——body 带 `snv_index` 时自动进入 SNV 模式。

```bash
# ── 方式一：独立入口（本地直连后端 :8001 / 服务单独部署）──
# mode=predict：参考序列表达预测（mean 格式，每轨道一个标量均值）
curl -X POST ${host}:8001/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'

# mode=snv：chr09:20731844 位点 A→T 的 ref/mut 轨迹对比
curl -X POST ${host}:8001/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "snv",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }'

# ── 方式二：统一网关入口（DCS 平台，body 加 model_sub）──
curl -X POST ${dcs_host}${dcs_entry} \
  -H "Authorization: Bearer ${dcs_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_sub": "rice_mut",
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'
```

---

## 3. rice_reg — ATAC 条件表达预测

**独立入口**：`POST ${host}:7001/api/aigress/openai/rice_reg`（本地后端端口 7001）
**统一入口**：`POST ${dcs_host}${dcs_entry}` + `"model_sub": "rice_reg"`

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | ATAC 条件下的 RNA-seq 表达预测 | `chromosome` + `start` + `atac_source`/`uploaded_atac` |

> `ATAC 输入`：`atac_source`（内置源，如 `SAM2_MH63_1`）与 `uploaded_atac`（文件路径）**必须二选一**。
> 位置 **1-based inclusive**；窗口长度固定 32678。

```bash
# ── 方式一：独立入口（本地直连后端 :7001）──
curl -X POST ${host}:7001/api/aigress/openai/rice_reg \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 1,
    "end": 32678,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'

curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_reg",
    "mode": "health"
  }'

curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_reg",
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 1,
    "end": 32678,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'

# ── 方式二：统一网关入口（DCS 平台，body 加 model_sub）──
curl -X POST ${dcs_host}${dcs_entry} \
  -H "Authorization: Bearer ${dcs_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_sub": "rice_reg",
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 1,
    "end": 32678,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'
```

---

## 4. rice_OGR — embedding 提取 / 碱基预测

**独立入口**：`POST ${host}:6001/api/aigress/openai/rice_ogr`（本地后端端口 6001）
**统一入口**：`POST ${dcs_host}${dcs_entry}`（`model_sub` 缺省即 `rice_ogr`，可省略）

| mode | 功能 | 必填参数 | 说明 |
|---|---|---|---|
| `dna_embedding` | 提取序列向量（默认） | `sequence`（+ `model_name`） | `pooling_method`: `mean`(默认)/`max`/`last` → 输出 `[1, 1024]`；`none` → 保留每位 `[1, L, 1024]` |
| `predict` | 续写预测下游碱基 | `sequence` + `predict_length` | 输出预测的碱基序列 |

> `mode` 可省略：请求体带 `predict_length` 自动走 `predict`，否则走 `dna_embedding`。
> **实际模型名**用 `model_name` 指定，取平台提供的模型注册名（如 `1B_8k` / `1B_32k`）。

```bash
# ── 方式一：独立入口（本地直连后端 :6001）──
# mode=dna_embedding：提取整条序列的 1024 维向量
curl -X POST ${host}:6001/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "dna_embedding",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_ogr",
    "mode": "dna_embedding",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# mode=predict：基于前 16 bp 预测后续 8 个碱基
curl -X POST ${host}:6001/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }'

curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_ogr",
    "mode": "predict",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }'

# ── 方式二：统一网关入口（DCS 平台；model_sub 缺省=rice_ogr 可省略）──
curl -X POST ${dcs_host}${dcs_entry} \
  -H "Authorization: Bearer ${dcs_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_sub": "rice_ogr",
    "mode": "dna_embedding",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'
```

---


## 5. 常见错误速查

| HTTP | 含义 | 处理 |
|---|---|---|
| 400 | 参数缺漏/格式错误 | 检查 `mode`、必填字段、`chr01-12` 命名、1-based 坐标；`model_sub` 只能是 `rice_mut`/`rice_reg`/`rice_ogr` |
| 401 | 鉴权失败 | 确认请求头 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` |
| 500 | 服务内部推理异常 | 记录返回的 `message`/`result` 并反馈管理员 |
| 503 | 模型尚未就绪 | 稍后重试；长时间如此看 `/health` 的 `init_error` |

---

## 7. 命名速查表（一页版）

```
# ── 独立入口（每服务一个，本地联调 / 单独部署）────────────────
POST ${host}:8001/api/aigress/openai/rice_mut   # rice_mut（端口 8001）
POST ${host}:7001/api/aigress/openai/rice_reg   # rice_reg（端口 7001）
POST ${host}:6001/api/aigress/openai/rice_ogr   # rice_OGR（端口 6001）
  HEADER: Authorization: Bearer ${api_key}

# ── 统一网关入口（DCS 平台唯一地址）──────────────────────────
POST ${dcs_host}/api/aigress/openai/OGR
  HEADER: Authorization: Bearer ${api_key}
  BODY:   model_sub 选服务 + mode 选功能

model_sub=rice_mut  mode=predict | mode=snv   # 坐标: chromosome+start(+snv_index+snv_base)
model_sub=rice_reg  mode=predict              # genome+chromosome+start + atac_source|uploaded_atac
model_sub=rice_ogr(缺省) mode=dna_embedding(默认) | mode=predict   # model_name=实际模型名

GET  ${host}/api/aigress/openai/<service>/health   # 独立入口免鉴权探活
GET  ${dcs_host}/api/aigress/openai/OGR/health     # 统一网关聚合探活
```

> 三合一完整调用规范（本机 + DCS 测试代码）见 [dcs_gateway/API.md](dcs_gateway/API.md)；
> 服务级细节见 [rice_mut/API.md](rice_mut/API.md)、[rice_reg/API.md](rice_reg/API.md)、[rice_OGR/API.md](rice_OGR/API.md)；部署与维护见 [AGENTS.md](AGENTS.md)。