# DCS 平台调用规范（三服务速查）

> 本文档面向 **DCS 平台上的调用方**，统一描述 `rice_server` 三个服务的 DCS API。
> 三个服务均为 **OpenAI 风格单入口**：`POST /api/aigress/openai/<service>` + body `mode` 分发。
> 本地联调地址与启动命令见 §1；DCS 网关地址由平台提供（`https://<dcs-host>/api/aigress/openai/<service>`）。
> 只想快速调用：见 [quick_start.md](quick_start.md)（外部用户一页速查）。

| 服务 | DCS 入口 path | 本地 API 地址 | 本地启动命令 |
|---|---|---|---|
| rice_mut（变异预测/SNV） | `/api/aigress/openai/rice_mut` | `http://localhost:8001` | `cd rice_mut/backend && python dcs_adapter.py` |
| rice_reg（ATAC 条件预测） | `/api/aigress/openai/rice_reg` | `http://localhost:7001` | `cd rice_reg/backend && python dcs_adapter.py` |
| rice_OGR（embedding/碱基预测） | `/api/aigress/openai/rice_ogr` | `http://localhost:8001` | `cd rice_OGR && python dcs_adapter.py` |

---

## 1. 统一返回结构与计费

三个服务返回结构一致：

```json
{
  "usage": { "prompt_tokens": 32768, "completion_tokens": 128 },
  "status": 200,
  "message": "成功提示",
  "result": { ... }
}
```

- **计费口径**：`prompt_tokens = 输入 token 数 × DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1）；`completion_tokens = 输出元素数 × DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）。水稻基模为单碱基编码，**1 bp = 1 token**。
- **鉴权**：`.env` 配置 `DCS_API_KEY` 后，除 `/health` 外均需 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；key 未配置（空）则不鉴权。
- **返回末尾自带换行**：curl 查看结果不粘连提示符。

---

## 2. 通用健康检查（免鉴权）

```bash
curl -s http://localhost:8001/health | python3 -m json.tool    # 本地
curl -s https://<dcs-host>/api/aigress/openai/rice_mut/health   # DCS 网关（三服务同构，替换 path）
```

返回（以 rice_OGR 为例）：`status` 200 / `device` 设备 / `loaded_models` 已加载模型 / **`init_error` 必须为 `null`**（非 null 表示模型加载失败，见 §7 503）。

---

## 3. rice_mut：变异预测（DNA → 多组学表达对比 + SNV）

mode 均为必填。坐标窗口统一 `chr01`–`chr12` 命名，位置 **1-based inclusive**。

> **模式自动推断**（显式 `mode` 优先级最高）：空 body → `health`；带 `snv_index` → `snv`；其余（有 `start`）→ `predict`。
> `model` 字段固定为 `rice_mut`（服务名），非必填。

### 3.1 mode=predict（参考序列表达预测）

```bash
curl -s http://localhost:8001/api/aigress/openai/rice_mut \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"           # full(默认) / mean / downsample
  }'
```

返回 `result` 含 `values`（assay × biosample 轨迹，按 `output_format` 为逐碱基数组 / 标量均值）、`genome/chromosome/position_1based/window_len` 等元数据。

### 3.2 mode=snv（单突变居中的窗口轨迹）

```bash
curl -s http://localhost:8001/api/aigress/openai/rice_mut \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "snv",
    "genome": "osa1_r7",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,            # 1-based 变异位点(须在窗口内)
    "snv_base": "T",                  # A/C/G/T/N
    "output_format": "mean"
  }'
```

返回双轨 `ref_values` / `mut_values` 对比。`mode=snv` 可省略：body 带 `snv_index` 时自动进入 SNV 模式。

---

## 4. rice_reg：ATAC 条件表达预测（DNA + ATAC → RNA-seq）

```bash
curl -s http://localhost:7001/api/aigress/openai/rice_reg \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 12000000,                # 必填(1-based inclusive)
    "end": 12050000,
    "atac_source": "SAM2_MH63_1",     # 内置 ATAC 源(与 uploaded_atac 二选一,必填其一)
    "output_format": "mean"           # full(默认) / mean / downsample
  }'
```

### 4.1 各 mode 速查

| mode | 说明 | 关键参数 |
|---|---|---|
| `health` | 健康检查（免鉴权） | 无 |
| `genomes` | 列出可用参考基因组 | 无 |
| `chromosomes` | 列出基因组染色体 | `genome`（必填） |
| `predict` | ATAC 条件表达预测 | `chromosome` + `start`（必填）+ `atac_source`/`uploaded_atac`（必填其一） |

`result` 含 `values`（`RNA-seq_+` / `RNA-seq_-` 两通道，按 `output_format` 输出）、`genome/atac_source/position_1based/window_len` 等。

---

## 5. rice_OGR：embedding 提取 / 下游碱基预测

mode 可省略（自动推断）：带 `predict_length` → `predict`，否则 → `dna_embedding`。

### 5.1 mode=dna_embedding（默认，提取序列向量）

```bash
curl -s http://localhost:8001/api/aigress/openai/rice_ogr \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "dna_embedding",
    "model": "rice_ogr",            # 服务名（与入口 path 末段一致）
    "model_name": "1B_8k",          # 实际模型注册名（.env MODEL_<NAME>_PATH，<NAME> 即此值）
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"        # mean(默认) / max / last / none
  }'
```

- `pooling_method=mean|max|last` → 输出 `embedding_shape: [1, 1024]`（1B 模型 hidden_size=1024）；
  `none` → `[1, L, 1024]`（保留每个位置的向量）。
- `model` 为服务名（`rice_ogr`）；实际模型名放 `model_name`（`model_name` 缺省时向后兼容取 `model` 作为模型名；两者都缺省取注册表第一个）。
- `result` 含 `embedding`、`embedding_shape`、`token_count`（= 序列长度）、`device`、`model_name`。

### 5.2 mode=predict（下游碱基预测）

```bash
curl -s http://localhost:8001/api/aigress/openai/rice_ogr \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "predict",
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }'
```

### 5.3 免鉴权辅助接口

```bash
curl -s http://localhost:8001/models    # 列出已注册/已加载模型
```

---

## 6. 错误返回排查表

| HTTP | 场景 | 排查 |
|---|---|---|
| 400 | 参数缺漏/格式错 | 检查 `mode`、必填参数（如 rice_mut 的 `start`）、序列命名 `chr01-12` 与长度上限 |
| 400 | mode 不在白名单 | 见各服务 MODES 常量：rice_mut=`predict,snv`；rice_reg=`health,genomes,chromosomes,predict`；rice_OGR=`dna_embedding,predict` |
| 401 | 鉴权失败 | 带 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；检查 `.env` 的 `DCS_API_KEY` |
| 500 | 推理异常 | 看 `result`/日志 traceback；模型路径、dtype/device 是否可用 |
| 503 | 模型未加载完成 | `/health` 查 `init_error`；初始化失败时适配层缓存错误并整体返回 503，重启容器 |

---

## 7. 部署注意

- **端口**：rice_mut 8000/8001、rice_reg 7000/7001、rice_OGR 默认 8000/8001 —— **rice_OGR 与 rice_mut 冲突，同机部署用 `PORT=8002`/`BACKEND_PORT=8003` 错开**。
- **鉴权开关**：`.env` 中 `DCS_API_KEY` 留空 = 不鉴权（仅限内网联调），上 DCS 必须配置。
- **容器内模型路径**：DCS 容器中通过环境变量注入，如 `MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf`（rice_OGR）；rice_mut/rice_reg 同理覆盖 checkpoint 路径。
- **多卡**：rice_OGR 支持 `--device cuda:0,cuda:1 --device_map auto`；rice_mut/rice_reg 用 `CUDA_VISIBLE_DEVICES` 控制。
- 各服务完整开发文档：`rice_mut/README.md`、`rice_reg/README.md`、`rice_OGR/README.md`（第 5 章为 DCS 适配层说明）。
