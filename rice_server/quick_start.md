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

## 1. 调用方式（统一约定）

三个服务共用 **一套调用方式**：

- **单入口**：`POST ${host}/api/aigress/openai/<service>`
- **服务命名**：`rice_mut` / `rice_reg` / `rice_ogr`（注意大小写）
- **请求体**：JSON，用 `mode` 字段选择功能（见下）；其余字段随服务不同
- **鉴权**：请求头带 `Authorization: Bearer ${api_key}`，或 `X-API-Key: ${api_key}`
- **返回**：统一为 `{"usage": {...}, "status": 200, "message": "...", "result": {...}}`
- **计费口径**：水稻基模单碱基编码，**1 bp = 1 token**；`usage.prompt_tokens` 即输入序列的 token 数

> `<dcs-host>` 由平台提供；本地联调时替换为 `localhost:<port>` 即可（见各服务入口表）。

---

```bash
# for host
api_key="${api_key}"
host="http://0.0.0.0:8001"

# for dcs
api_key="sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"
host="https://www.dcs.cloud"
```

## 2. rice_mut — 变异对比表达预测

**入口命名**：`POST /api/aigress/openai/rice_mut`

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | 参考序列表达预测 | `chromosome` + `start` |
| `snv` | 单突变位点的窗口轨迹对比 | `chromosome` + `start` + `snv_index` + `snv_base` |

> 染色体统一 `chr01`–`chr12` 命名；位置 **1-based inclusive**。
> `genome`/`end`/`output_format` 可选（缺省：唯一基因组 / 32768 窗口 / `full`）。
> `mode=snv` 可省略——body 带 `snv_index` 时自动进入 SNV 模式。

```bash
curl -X POST ${host}/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR-Mutation",
    "mode": "health"}'

# mode=predict：参考序列表达预测（mean 格式，每轨道一个标量均值）
curl -X POST ${host}/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR-Mutation",
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'

# mode=snv：chr01:20731844 位点 A→T 的 ref/mut 轨迹对比
curl -X POST ${host}/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR-Mutation",
    "mode": "snv",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }'
```

---

## 3. rice_reg — ATAC 条件表达预测

**入口命名**：`POST /api/aigress/openai/rice_reg`

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | ATAC 条件下的 RNA-seq 表达预测 | `chromosome` + `start` + `atac_source`/`uploaded_atac` |

> `ATAC 输入`：`atac_source`（内置源，如 `SAM2_MH63_1`）与 `uploaded_atac`（文件路径）**必须二选一**。

```bash
curl -X POST ${host}/api/aigress/openai/rice_reg \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR-Reg",
    "mode": "health"
  }'

curl -X POST ${host}/api/aigress/openai/rice_reg \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR-Reg",
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

**入口命名**：`POST /api/aigress/openai/rice_ogr`

| mode | 功能 | 必填参数 | 说明 |
|---|---|---|---|
| `dna_embedding` | 提取序列向量（默认） | `sequence`（+ `model_name`） | `pooling_method`: `mean`(默认)/`max`/`last` → 输出 `[1, 1024]`；`none` → 保留每位 `[1, L, 1024]` |
| `predict` | 续写预测下游碱基 | `sequence` + `predict_length` | 输出预测的碱基序列 |

> `mode` 可省略：请求体带 `predict_length` 自动走 `predict`，否则走 `dna_embedding`。
> `model` 为**服务名**（`rice_ogr`）；**实际模型名**用 `model_name` 指定，取平台提供的模型注册名
> （如 `1B_8k` / `1B_32k`）。

```bash
# mode=dna_embedding：提取整条序列的 1024 维向量
curl -X POST ${host}/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "mode": "dna_embedding",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# mode=predict：基于前 16 bp 预测后续 8 个碱基
curl -X POST ${host}/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "mode": "predict",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }'
```

---


## 5. 常见错误速查

| HTTP | 含义 | 处理 |
|---|---|---|
| 400 | 参数缺漏/格式错误 | 检查 `mode`、必填字段、`chr01-12` 命名、1-based 坐标 |
| 401 | 鉴权失败 | 确认请求头 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` |
| 500 | 服务内部推理异常 | 记录返回的 `message`/`result` 并反馈管理员 |
| 503 | 模型尚未就绪 | 稍后重试；长时间如此看 `/health` 的 `init_error` |

---

## 7. 命名速查表（一页版）

```
POST /api/aigress/openai/<service>      # 通用入口
  HEADER: Authorization: Bearer ${api_key}

rice_mut  mode=predict | mode=snv   # 坐标: chromosome+start(+snv_index+snv_base)
rice_reg  mode=predict              # genome+chromosome+start + atac_source|uploaded_atac
rice_ogr  mode=dna_embedding(默认) | mode=predict   # model=服务名 + model_name=模型名
GET  /api/aigress/openai/<service>/health   # 免鉴权
```

> 完整调用规范（计费/字段说明/返回结构）见 [dcs.md](dcs.md)；部署与维护见 [AGENTS.md](AGENTS.md)。