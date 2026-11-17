# OGR 四应用调用指南（网页端 + DCS API 端）

> 适用目录：rice_server（OGR-Prediction / OGR-Regulation / OGR-Introgression / OGR-Base）
> 端口：前端 8000 / 7000 / 5000 / NA；DCS 适配 8001 / 7001 / 5001 / 6001；统一网关 4000（DCS 平台注入 `PORT` 优先）

## 0. 总览

| 应用 | 能力 | 网页端 | DCS model_sub | DCS 子路径 |
|---|---|---|---|---|
| OGR-Prediction | DNA → 多组学表达预测（参考 vs SNV） | Gradio :8000 | `OGR-Prediction` | `Prediction` / `SNV` |
| OGR-Regulation | DNA + ATAC → RNA-seq 表达预测 | Gradio :7000 | `OGR-Regulation` | `Genome` / `Chromosome` / `Prediction` |
| OGR-Introgression | 粳/籼血缘渗入分析 | Gradio :5000 | `OGR-Introgression` | `predict` |
| OGR-Base | DNA embedding / 下游碱基预测 | 无网页端（纯 API） | `OGR-Base`（**缺省**） | `Embedding` / `NTP` |

**统一入口（DCS）**：`POST https://<dcs-host>/api/aigress/openai/OGR/{model_sub}/{subpath}`
**公共返回**：`{"usage": {prompt_tokens, completion_tokens}, "status", "message", "result"}`
**鉴权**：请求头 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`（`health` 免鉴权）
**计费**：水稻基模单碱基编码，**1 bp = 1 token**；`prompt_tokens = 输入碱基数 × 系数`（默认 1）

---

## 1. OGR-Prediction（变异对比表达预测）

### 1.1 网页端（Gradio :8000）

| 控件（键） | 类型 | 取值 |
|---|---|---|
| Genome（基因组） | 下拉 | `osa1_r7` / `📤 Custom Genome`（上传自定义 FASTA） |
| Chromosome（染色体） | 下拉 | `chr01`–`chr12` |
| Start（起始位置） | 数字 | 1-based 起始坐标，预测窗口 32768 bp 中心对齐 |
| SNV_position（SNV 位置） | 数字（可选） | 1-based 变异位点，须在窗口内 |
| SNV_base（SNV 碱基） | 下拉（可选） | `A` / `C` / `G` / `T` |
| 上传 Genome FASTA | 文件（自定义时） | `.fa` / `.fasta` / `.fna` |
| 上传 Annotation GFF | 文件（自定义时） | `.gff` / `.gff3` / `.gtf`（可 .gz） |

> 参考预测：SNV 两字段留空；SNV 对比：同时填 SNV_position + SNV_base。结果在 IGV 中显示（灰=参考，彩色=突变），右侧柱状图为窗口内平均表达量。

### 1.2 DCS API 端

**子路径**：`POST .../OGR/OGR-Prediction/Prediction`（参考预测）· `.../OGR/OGR-Prediction/SNV`（变异对比）

| 键 | 必填 | 类型 | 取值 |
|---|---|---|---|
| Genome | ❌ | string | 默认唯一基因组 `osa1_r7` |
| Chromosome | ❌ | string | 默认 `chr01`；`chr01`/`Chr1`/`1` 自动通配 |
| Start | ✅ | int | **1-based inclusive**（必填，缺失 400） |
| Format | ❌ | string | `Full`（默认，逐碱基数组）/ `Mean`（每轨道标量均值） |

SNV 额外两个必填键：

| 键 | 必填 | 类型 | 取值 |
|---|---|---|---|
| SNV_position | ✅ | int | 1-based 变异位点，须落在窗口内 |
| SNV_base | ✅ | string | `A` / `C` / `G` / `T` |

**调用示例**：

```bash
dcs_host="https://www.dcs.cloud"; dcs_api_key="<key>"
GW="${dcs_host}/api/aigress/openai/OGR"

# 参考预测（Mean）
curl -X POST ${GW}/OGR-Prediction/Prediction \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR","Genome":"osa1_r7","Chromosome":"chr09","Start":20716774,"Format":"Mean"}'

# SNV 变异对比（C@20731844 → T）
curl -X POST ${GW}/OGR-Prediction/SNV \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR","Genome":"osa1_r7","Chromosome":"chr09","Start":20716774,"SNV_position":20731844,"SNV_base":"T","Format":"Mean"}'
```

**结果解读（参考预测）**：

```json
{
  "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
  "status": 200, "message": "Reference expression prediction succeeded",
  "result": {
    "model": "OGR-Prediction", "Genome": "osa1_r7", "Chromosome": "Chr9",
    "Start": 20716774, "window_len": 32768, "Format": "mean",
    "values": {"RNA-seq": {"Leaf": 0.430746}}
  }
}
```

- `usage.prompt_tokens = window_len = 32768`（1 bp = 1 token）；`completion_tokens = 输出数组元素数`。
- `values` = `{assay: {biosample: 表达值}}`；`Format=Mean` 时值为标量均值，`Full` 时为 32768 长度数组。
- assay/biosample 显示名由 .env 的 `DISPLAY_HEADS` / `DISPLAY_BIOSAMPLES` 覆盖（当前 = `RNA-seq` / `Leaf`）。

**结果解读（SNV）**：`result` 含 `ref_values`（参考）/ `mut_values`（突变）双轨 + `SNV_position`、`ref_base`、`SNV_base`；对比两轨同一 biosample 的数值即该变异对表达的影响。`completion_tokens = 两轨元素数之和`。

---

## 2. OGR-Regulation（ATAC 条件表达预测）

### 2.1 网页端（Gradio :7000）

| 控件（键） | 类型 | 取值 |
|---|---|---|
| Genome（基因组） | 下拉 | `MH63` / `NIP` / 自定义 |
| Chromosome（染色体） | 下拉 | `chr01`–`chr12` |
| Start（起始位置） | 数字 | 1-based，窗口 32768 bp |
| ATAC-seq_Source（ATAC 数据源） | 下拉 | `SAM2_MH63_1` / `SAM2_NIP_1` / `Upload bigWig…` |
| Upload bigWig | 文件 | `.bw` / `.bigwig`（选择上传时） |
| 上传 Genome FASTA / GFF | 文件（自定义时） | 同 OGR-Prediction |

> ATAC 源与基因组映射：`ATAC_GENOME_MAP_<GENOME>`（MH63→SAM2_MH63_1，NIP→SAM2_NIP_1）。

### 2.2 DCS API 端

**子路径**：`POST .../OGR/OGR-Regulation/Prediction`（预测）

| 键 | 必填 | 类型 | 取值 |
|---|---|---|---|
| Genome | ✅ | string | `MH63` / `NIP`（缺省取第一个配置的基因组） |
| Chromosome | ❌ | string | 默认 `chr01` |
| Start | ✅ | int | 1-based inclusive |
| ATAC-seq_Source | △ | string | 内置源：`SAM2_MH63_1` / `SAM2_NIP_1` |
| Format | ❌ | string | `Full`（默认）/ `Mean` |


**调用示例**：

```bash
curl -X POST ${GW}/OGR-Regulation/Prediction \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR","Genome":"MH63","Chromosome":"chr01","Start":1,"ATAC-seq_Source":"SAM2_MH63_1","Format":"Mean"}'
```

**结果解读**：

```json
{
  "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
  "status": 200, "message": "ATAC→RNA-seq expression prediction succeeded",
  "result": {
    "model": "OGR-Regulation", "Genome": "MH63", "Chromosome": "Chr1",
    "Start": 1, "window_len": 32768, "ATAC-seq_Source": "SAM2_MH63_1",
    "atac_path": ".../ATAC_SAM2_MH63_1...bw", "Format": "mean",
    "values": {"RNA-seq_+": 0.599, "RNA-seq_-": 0.588}
  }
}
```

- `values` 分 **正链 `RNA-seq_+` / 负链 `RNA-seq_-`** 两通道；`Format=Mean` 为标量，`Full` 为数组。


---

## 3. OGR-Introgression（粳/籼血缘渗入分析）

### 3.1 网页端（Gradio :5000）

| 控件（键） | 类型 | 取值 |
|---|---|---|
| Genome（基因组） | 下拉 | `YF47` 等（.env `GENOME_*_FASTA`）/ 自定义 |
| Chromosome（染色体） | 下拉 | FASTA **实际命名**（如 `Chr01`，不归一化） |
| Start（可选） | 文本 | 1-based；留空 = 最大窗口 |
| End（可选） | 文本 | 1-based；留空 = `start + 256k` |
| 上传 Genome FASTA | 文件（自定义时） | `.fa` / `.fasta` / `.fna` |

> 窗口模式：Start/End 都填 → 匹配网格窗口；只填 Start → 1 个覆盖 Start 的最大 256k 窗口；都空 → 整条染色体（受 `MAX_NUMBER_256W` 限制）。

### 3.2 DCS API 端

**子路径**：`POST .../OGR/OGR-Introgression/predict`

| 键 | 必填 | 类型 | 取值 |
|---|---|---|---|
| Genome | ❌ | string | 默认第一个 `GENOME_*_FASTA`（如 `YF47`） |
| Chromosome | ❌ | string | 默认 `Chr01`（FASTA 实际命名） |
| Start | ❌ | int | 1-based inclusive（可选） |
| End | ❌ | int | 1-based inclusive（可选） |

**调用示例**：

```bash
curl -X POST ${GW}/OGR-Introgression/predict \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR","Genome":"YF47","Chromosome":"Chr01","Start":100001,"End":356001}'
```

**结果解读**：

```json
{
  "usage": {"prompt_tokens": 1024000, "completion_tokens": 4},
  "status": 200, "message": "Introgression analysis succeeded",
  "result": {
    "model": "OGR-Introgression", "genome": "YF47", "chromosome": "Chr01",
    "chrom_len": 43896792, "mode": "window",
    "position_1based": {"start": 128001, "end": 256000},
    "window_len": 256000, "n_windows": 4,
    "windows": [
      {"win_start": 128001, "win_end": 384000, "center": 256001, "n_segments": 32,
       "topk_mean_jap": 0.544, "topk_mean_ind": 0.547, "group": "Ind"}, ...
    ],
    "params": {"segment_size": 8000, "window_size": 256000, "window_step": 64000,
               "top_k": 10, "threshold_jap": 0.55519, "threshold_ind": 0.53473},
    "threshold_rule": "Ind: prob_jap < 0.5552 & prob_ind >= 0.5347; ..."
  }
}
```

**关键解读（重要）**：
- **`position_1based.end` ≠ 输入的 `End` 是正常的**：服务使用**固定 256k 网格**（窗口宽 256k、步长 64k、起点在网格上），输入区间只用于**匹配重叠最大的网格窗口**，返回的是网格窗口覆盖区间，不是你的原始输入。
- `windows[]` 是匹配到的 **N 个网格窗口**（`n_windows` 受 `.env MAX_NUMBER_256W` 控制，当前=4），每个窗口 32 个 8k 片段。
- **分组规则**（`group`）：`Jap` = prob_jap ≥ 0.55519 且 prob_ind < 0.53473；`Ind` = prob_jap < 0.55519 且 prob_ind ≥ 0.53473；否则 `uncertain`。
- `usage.prompt_tokens = 窗口数 × 256000`（如 4×256000=1024000，即推理总碱基数）；`completion_tokens = 返回窗口数`。

---

## 4. OGR-Base（DNA embedding / 下游碱基预测）

### 4.1 网页端

**无网页前端**（纯 API 服务）。

### 4.2 DCS API 端

**子路径**：`POST .../OGR/OGR-Base/Embedding`（embedding 提取）· `.../OGR/OGR-Base/NTP`（碱基预测）
> `model_sub` **缺省即 `OGR-Base`**，`/OGR/OGR-Base/Embedding` 可简写为 `/OGR/Embedding`。

| 键 | 必填 | 类型 | 取值 |
|---|---|---|---|
| sequence | ✅ | string | DNA 序列（大写），最长32768（超过会只保留前32768） |
| pooling_method | ❌ | string | `mean`（默认）/ `max` / `last` / `none`（仅 Embedding） |
| predict_length | ❌ | int | 1–1000，默认 10（仅 NTP） |

> 模型固定为注册表 `1B_8k`（不接受 `model_name`/`mode` 键）。长度上限 32768 bp，超出截断。

**调用示例**：

```bash
# Embedding
curl -X POST ${GW}/OGR-Base/Embedding \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR","sequence":"ACGTTGCATGCAACGTACGTTGCATGCAACGT","pooling_method":"mean"}'
# NTP（基于前 16 bp 续写 8 bp）
curl -X POST ${GW}/OGR-Base/NTP \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{"model":"OGR","sequence":"ACGTTGCATGCAACGT","predict_length":8}'
```

**结果解读（Embedding）**：

```json
{
  "usage": {"prompt_tokens": 32, "completion_tokens": 0},
  "status": 200, "message": "DNA sequence embedding extraction succeeded",
  "result": {
    "model": "OGR-Base", "model_name": "1B_8k", "sequence": "...",
    "sequence_length": 32, "token_count": 32,
    "embedding_shape": [1, 1024], "embedding_dim": 1024,
    "pooling_method": "mean", "model_type": "flash", "device": "cuda:0",
    "embedding": [...1024 个浮点数...]
  }
}
```

- **`usage`**：`prompt_tokens = token_count = 输入序列长度`（1 bp = 1 token，如 32 bp → 32）；`completion_tokens = 0`（提取无"生成"输出）。
- **`embedding_shape`** 决定返回向量的维度与含义：
  - `pooling_method=mean|max|last` → `[1, 1024]`：整条序列被池化为 **1 个 1024 维向量**，可直接用于下游分类/回归/相似度计算。
  - `pooling_method=none` → `[1, L, 1024]`：**逐碱基**保留每位点的 1024 维向量（L = token_count），适合需要位置级特征的任务（如变异效应定位），但数据量约为前者的 L 倍。
- **`token_count`** 即实际送入模型的碱基数（超长截断后 = 32768，此时 `sequence_length` 仍显示原始输入长度，二者可能不一致）。
- **`embedding`** 为 JSON 数组（bf16 已自动转 fp16/numpy 序列化，可直接 `np.array()` 使用）。

**结果解读（NTP，下游碱基预测）**：

```json
{
  "usage": {"prompt_tokens": 16, "completion_tokens": 8},
  "status": 200, "message": "Downstream base (NTP) prediction succeeded",
  "result": {
    "model": "OGR-Base", "model_name": "1B_8k",
    "original_sequence": "ACGTTGCATGCAACGT",
    "predicted_sequence": "ACGTTGCATGCAACGTACAACAAAAA",
    "predicted_bases": "ACAACAAAAA",
    "predict_length": 8, "total_length": 24,
    "elapsed_seconds": 0.317
  }
}
```

- **`predicted_bases`**：自回归续写的 **8 个下游碱基**（本示例 `ACAACAAAAA`）。
- **`predicted_sequence`** = `original_sequence` + `predicted_bases`（原序列 16 bp + 续写 8 bp = `total_length` 24 bp）。
- **`usage`**：`prompt_tokens = total_length - predict_length = 16`（输入上下文碱基数）；`completion_tokens = predict_length = 8`（生成碱基数）。这是四个服务中**唯一 completion_tokens > 0 的生成类调用**。
- **自回归续写**：将 `predicted_bases`（或整条 `predicted_sequence`）作为新的 `sequence` 再次调用，可继续向后续写任意长度（每次最多 1000 bp）。


## 使用说明

```shell
# ============================================================================
# OGR 四应用 DCS API 调用速查（统一入口）
#   入口:  POST https://<dcs-host>/api/aigress/openai/OGR/{model_sub}/{subpath}
#   鉴权:  Authorization: Bearer <key>（health 免鉴权）
#   计费:  1 bp = 1 token
# ============================================================================

# ---- 公共变量 ----
dcs_host="https://www.dcs.cloud"
dcs_api_key="<YOUR_API_KEY>"
GW="${dcs_host}/api/aigress/openai/OGR"
AUTH=(-H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json")


# ============================================================================
# 1. OGR-Prediction —— DNA → 多组学表达预测（参考 vs SNV 双轨）
#    窗口固定 32768 bp（中心对齐）；坐标 1-based inclusive
# ============================================================================

# 1a. 参考预测（Format: Full=逐碱基数组 | Mean=每轨道均值）
curl -s -X POST ${GW}/OGR-Prediction/Prediction "${AUTH[@]}" \
  -d '{"model":"OGR","Genome":"osa1_r7","Chromosome":"chr09","Start":20716774,"Format":"Mean"}'
  
# 返回: values = {assay: {biosample: 值}}，如 {"RNA-seq": {"Leaf": 0.430746}}

# 1b. SNV 变异对比（ref vs mut 双轨；SNV_base 仅 A/C/G/T）
curl -s -X POST ${GW}/OGR-Prediction/SNV "${AUTH[@]}" \
  -d '{"model":"OGR","Genome":"osa1_r7","Chromosome":"chr09","Start":20716774,"SNV_position":20731844,"SNV_base":"T","Format":"Mean"}'
# 返回: ref_values(参考) + mut_values(突变)，对比两轨数值即变异效应

# ============================================================================
# 2. OGR-Regulation —— DNA + ATAC → RNA-seq 表达预测
#    输出分正链 RNA-seq_+ / 负链 RNA-seq_-
# ============================================================================

curl -s -X POST ${GW}/OGR-Regulation/Prediction "${AUTH[@]}" \
  -d '{"model":"OGR","Genome":"MH63","Chromosome":"chr01","Start":1,"ATAC-seq_Source":"SAM2_MH63_1","Format":"Mean"}'

# Genome: MH63|NIP；ATAC 源: SAM2_MH63_1↔MH63, SAM2_NIP_1↔NIP
# 返回: values = {"RNA-seq_+": 0.599, "RNA-seq_-": 0.588}


# ============================================================================
# 3. OGR-Introgression —— 粳/籼血缘渗入分析（256k 网格窗口）
#    输入区间只匹配网格窗口；返回 position/windows 均为网格坐标
# ============================================================================

curl -s -X POST ${GW}/OGR-Introgression/predict "${AUTH[@]}" \
  -d '{"model":"OGR","Genome":"YF47","Chromosome":"Chr01","Start":100001,"End":356001}'
# 窗口模式: Start+End → 重叠最大的 N 窗（.env MAX_NUMBER_256W 控制，当前=4）
#          只填 Start → 1 个覆盖 Start 的最大 256k 窗；都空 → 整条染色体
# 返回: windows[] = {win_start, win_end, topk_mean_jap, topk_mean_ind, group}
#       group: Jap | Ind | uncertain（阈值: Jap≥0.55519, Ind≥0.53473）

# ============================================================================
# 4. OGR-Base —— 基模直接开放（model_sub 缺省即 OGR-Base）
#    模型固定 1B_8k；sequence 最长 32768 bp，超出截断（只保留前 32768）
# ============================================================================

# 4a. Embedding 提取（pooling: mean|max|last → [1,1024]；none → [1,L,1024]）
curl -s -X POST ${GW}/OGR-Base/Embedding "${AUTH[@]}" \
  -d '{"model":"OGR","sequence":"ACGTTGCATGCAACGTACGTTGCATGCAACGT","pooling_method":"mean"}'
# 返回: embedding(1024 维数组), embedding_shape, token_count(=序列长度)

# 4b. NTP 下游碱基预测（自回归续写，predict_length 1-1000）
curl -s -X POST ${GW}/OGR-Base/NTP "${AUTH[@]}" \
  -d '{"model":"OGR","sequence":"ACGTTGCATGCAACGT","predict_length":8}'
# 返回: predicted_bases(续写碱基), predicted_sequence(原+续), total_length
#       prompt_tokens=上下文碱基数, completion_tokens=生成碱基数

```