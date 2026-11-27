# OGR-Prediction API 调用文档

> 适用服务：`rice_server/OGR_Prediction/backend/rice_mutation/api.py`（DNA → 多组学表达预测）
> 功能：**参考序列表达预测** 与 **单碱基变异（SNV）ref/mut 双轨对比**（assays × biosamples 多维输出）。
> 本文档提供可直接复制的 curl / Python 代码。**网页端（Gradio）与纯 API（curl）走同一套端点、同一套参数键、同一套坐标语义** —— 所有 1-based → 0-based 转换都在后端完成。

---

## 1. 入口总览

后端统一进程（默认 `:8001`）暴露 5 个 **纯子路径** 端点（camelCase 键、1-based 坐标）。每个端点都有「带前缀」和「简写」两个等价路径：

| 功能 | 端点（两个路径等价） | 说明 |
|---|---|---|
| 参考序列预测 | `POST /api/aigress/openai/OGR-Prediction/Prediction`<br>`POST /OGR-Prediction/Prediction` | 参考序列表达预测 |
| SNV 变异预测 | `POST /api/aigress/openai/OGR-Prediction/SNV`<br>`POST /OGR-Prediction/SNV` | ref/mut 双轨对比 |
| 基因组列表 | `POST /api/aigress/openai/OGR-Prediction/Genome`<br>`POST /OGR-Prediction/Genome` | 列出已注册基因组 |
| 染色体列表 | `POST /api/aigress/openai/OGR-Prediction/Chromosome`<br>`POST /OGR-Prediction/Chromosome` | 列出某基因组的染色体 |
| 注册基因组 | `POST /api/aigress/openai/OGR-Prediction/RegisterGenome`<br>`POST /OGR-Prediction/RegisterGenome` | 按本机路径注册 FASTA / 挂载 GFF |
| 健康检查 | `GET /health` | 免鉴权，模型加载状态 |

> **没有 `mode` 分发、没有 `model_sub`、没有 `end`、没有 `downsample`/`max_points`**。功能完全由 URL 子路径决定，body 里不再出现这些字段。

### 1.1 与旧版（`rice_mut` + `dcs_adapter.py`）的区别

| 项 | 旧 | 新 |
|---|---|---|
| 入口 | `…/rice_mut` + body `mode` 分发 | `…/OGR-Prediction/{Prediction,SNV,…}` 纯子路径 |
| 参数键 | `genome/chromosome/start/end/…`（snake_case） | `Genome/Chromosome/Start/…`（camelCase） |
| 格式字段 | `output_format: full/mean/downsample` | `Format: Full/Mean` |
| SNV 键 | `snv_index` | `SNV_position` / `SNV_base` |
| 坐标转换 | 前端/调用方自行换算 | 后端统一 `start_0 = start_1 - 1` |

---

## 2. 请求头与鉴权

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；仅当后端 `.env` 配置了 `DCS_API_KEY` 时必填 |
| `X-API-Key` | ⚠️ | `Authorization` 的等价替代 |

> - 后端 `_check_api_key` 会校验 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`。
> - 后端 `.env` 的 `DCS_API_KEY` **为空时所有请求免鉴权**（本地直连 / 开发测试）。
> - **网页端**：Gradio 页面在 Predict 按钮上方有一个 **API Key 输入框（必填）**，运行时由用户填写；前端只把输入框的值作为 `Authorization: Bearer <key>` 发送，**不读 `.env`**。后端启用鉴权时留空会得到 401。
> - 本机联调若 `localhost` 提示 `address already in use`，说明 8001 已有服务：`fuser -k 8001/tcp` 后重启。

---

## 3. 健康检查（GET /health，免鉴权）

```bash
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
```

示例返回：

```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["osa1_r7"],
    "assays": ["RNA-seq"],
    "biosamples": ["Leaf"]
}
```

| 字段 | 含义 |
|---|---|
| `predictor_initialized` | 模型是否加载完成（`false` 时请等待模型加载约 30–60s） |
| `genomes` | 已注册基因组 ID 列表 |
| `assays` / `biosamples` | 模型输出头（展示名，见 §4.1 `biosample_names` 注意） |

---

## 4. 统一响应信封

所有 5 个 DCS 子路径（含错误）都返回同一外层结构：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
    "token_cnt": 65536,
    "status": 200,
    "message": "Prediction completed in 2.7s",
    "result": { ... }
}
```

| 字段 | 说明 |
|---|---|
| `model_name` | 固定 `"OGR"` |
| `usage.prompt_tokens` | 输入窗口碱基数 × `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `usage.completion_tokens` | 输出数组元素总数 × `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）；SNV 为 ref + mut 两轨之和 |
| `token_cnt` | `prompt_tokens + completion_tokens` |
| `status` | HTTP 语义状态（200/400/401/500） |
| `result` | 成功时携带业务数据；失败时为 `null` |

---

## 5. 参考序列表达预测（Prediction）

### 5.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `Genome` | ❌ | string | 基因组 ID；缺省取已注册的第一个（默认 `osa1_r7`） |
| `Chromosome` | ❌ | string | 染色体；缺省 `chr01`；`chr01`/`Chr1`/`1` 均可（后端自动通配为 FASTA 实际命名） |
| `Start` | ✅ | int | **1-based** 起始坐标；后端居中到 32768 bp 窗口 |
| `Format` | ❌ | string | `Full`（默认，逐碱基数组）/ `Mean`（每轨道标量均值） |
| `biosample_names` | ❌ | string/array | 逗号分隔字符串或数组；缺省全部 |
| `fasta_path` | ❌ | string | 可选：按本机路径懒注册自定义基因组 FASTA（见 §8） |
| `gff_path` | ❌ | string | 可选：随 `fasta_path` 一起挂载 GFF 注释 |

> ⚠️ **`biosample_names` 必须用内部名**（来自 `index_stat.json` 的 `biosample_order`，如 `NIP_leaf`）；展示名（如 `Leaf`）只用于 IGV 标签和返回键。传 `Leaf` 会得到 400「Biosample 'Leaf' not found」。

> **自定义基因组：直接传路径即可（懒注册）**。携带 `fasta_path`（+ 可选 `gff_path`）时，后端就地注册该 FASTA 并用它预测，**无需先调用 RegisterGenome**；若同时给了 `Genome`，`fasta_path` 会覆盖之。成功响应里会附 `fasta_path` / `gff_path` 回显。

### 5.2 Mean 格式（每轨道标量均值，最常用）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Prediction \
  -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr01",
    "Start": 20716774,
    "Format": "Mean"
  }' | python3 -m json.tool

curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Prediction \
  -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr01",
    "Start": 20716774,
    "fasta_path": "/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/OGR_Regulation/genome/MH63RS2.fa",
    "Format": "Mean"
  }' | python3 -m json.tool
```

示例返回（节选 `result`）：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 32768, "completion_tokens": 1},
    "token_cnt": 32769,
    "status": 200,
    "message": "Prediction completed in 2.7s",
    "result": {
        "model": "OGR-Prediction",
        "Genome": "osa1_r7",
        "Chromosome": "Chr1",
        "Start": 20716774,
        "window_start": 20716773,
        "window_len": 32768,
        "Format": "Mean",
        "values": {"RNA-seq": {"Leaf": 0.599956}},
        "igv_payload": { "...": "..." },
        "ref_id": "ref_1730000000000"
    }
}
```

> Mean 模式下 `values` 已是标量，因此**不再返回** `mean_values`（它是 Full 模式专属的重复字段）。

### 5.3 Full 格式（逐碱基数组，适合落盘分析）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Prediction \
  -H "Content-Type: application/json" \
  -d '{"Genome":"osa1_r7","Chromosome":"chr01","Start":20716774}' \
  -o predict_full.json && ls -lh predict_full.json

python3 -c "import json; d=json.load(open('predict_full.json'))['result']; print('窗口长度:', d['window_len'], '窗口起点(0-based):', d['window_start'])"
```

Full 模式下 `result` 额外包含：

| 字段 | 说明 |
|---|---|
| `values` | `{assay: {biosample: [float, ...]}}`，逐碱基数组（6 位小数） |
| `mean_values` | `{assay: {biosample: scalar}}`，与 `values` 同窗口的每轨道均值（**仅 Full 模式**） |
| `window_start` | **0-based** 窗口起点（`Start - 1`） |
| `Start` | **1-based** 窗口起点（回显） |
| `window_len` | 窗口长度（默认 32768） |
| `igv_payload` | `{reference:{fastaURL,indexURL}, locus, tracks:[{name,url}]}`，IGV 可视化数据 |
| `ref_id` | 本次预测 ID（可继续用于 `/predict/bar` 视口均值，见 §10） |

---

## 6. SNV 变异预测（SNV，双轨对比）

### 6.1 请求体参数

在参考预测的基础上额外：

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `SNV_position` | ✅ | int | **1-based** 变异位点 |
| `SNV_base` | ✅ | string | 变异碱基 `A`/`C`/`G`/`T` |

> **自定义基因组：直接传路径即可（懒注册）**。SNV 与参考预测一样，请求里可直接携带 `fasta_path`（可选 `gff_path`），后端会就地注册该 FASTA 并用它做预测，**不需要先单独调用 RegisterGenome**。已注册的自定义基因组则照常传 `Genome` 即可。网页端选中自定义基因组后，预测时也会自动携带注册时记录的 `fasta_path`/`gff_path`。

```bash
# 直接传 fasta_path（无需单独注册）
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/SNV \
  -H "Content-Type: application/json" \
  -d '{
    "fasta_path": "/abs/path/to/my.fa",
    "gff_path": "/abs/path/to/my.gff3",
    "Chromosome": "Chr1",
    "Start": 20716774,
    "SNV_position": 20731844,
    "SNV_base": "T",
    "Format": "Mean"
  }' | python3 -m json.tool
```

### 6.2 Mean 格式

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/SNV \
  -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr01",
    "Start": 20716774,
    "SNV_position": 20731844,
    "SNV_base": "T",
    "Format": "Mean"
  }' | python3 -m json.tool
```

示例返回（节选 `result`）：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 32768, "completion_tokens": 2},
    "token_cnt": 32770,
    "status": 200,
    "message": "SNV prediction completed in 5.1s",
    "result": {
        "model": "OGR-Prediction",
        "Genome": "osa1_r7",
        "Chromosome": "Chr1",
        "Start": 20716774,
        "window_start": 20716773,
        "window_len": 32768,
        "SNV_position": 20731844,
        "ref_base": "T",
        "SNV_base": "T",
        "Format": "Mean",
        "ref_values": {"RNA-seq": {"Leaf": 0.599956}},
        "mut_values": {"RNA-seq": {"Leaf": 0.599956}},
        "igv_payload": { "...": "..." },
        "snv_id": "snv_1730000000000"
    }
}
```

| 字段 | 说明 |
|---|---|
| `SNV_position` | **1-based** 变异位点（回显） |
| `ref_base` | 参考碱基（FASTA 实际碱基） |
| `SNV_base` | 请求的变异碱基 |
| `ref_values` / `mut_values` | 参考 / 突变两轨数值 |
| `snv_id` | 本次预测 ID |

> Full 模式下 `result` 额外包含 `ref_mean_values` / `mut_mean_values`（两轨各自的每轨道均值）；Mean 模式移除此二字段（`ref_values`/`mut_values` 已是标量）。

### 6.3 同时保存参考/突变两轨道（Full）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/SNV \
  -H "Content-Type: application/json" \
  -d '{"Genome":"osa1_r7","Chromosome":"chr01","Start":20716774,"SNV_position":20731844,"SNV_base":"T"}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
json.dump(d['ref_values'], open('snv.ref.json','w'), ensure_ascii=False, indent=2)
json.dump(d['mut_values'], open('snv.mut.json','w'), ensure_ascii=False, indent=2)
json.dump({'SNV_position': d['SNV_position'], 'ref_base': d['ref_base'], 'SNV_base': d['SNV_base']}, open('snv.meta.json','w'), ensure_ascii=False, indent=2)
print('已保存: snv.ref.json / snv.mut.json / snv.meta.json')
"
```

---

## 7. 列表端点

### 7.1 基因组列表（Genome）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Genome \
  -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
```

返回 `result`：`{"model":"OGR-Prediction", "Genome": ["osa1_r7", "fe_1730000000000_my.fa"]}`

### 7.2 染色体列表（Chromosome）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Chromosome \
  -H "Content-Type: application/json" \
  -d '{"Genome":"osa1_r7"}' | python3 -m json.tool
```

返回 `result`：`{"model":"OGR-Prediction", "Genome":"osa1_r7", "Chromosome": ["Chr1", "Chr2", ...]}`

---

## 8. 注册自定义基因组（RegisterGenome）

按**本机路径**注册（前后端同机，零拷贝、自动建 `.fai` 索引）。文件必须先存在于后端机器上（不传输文件内容）。

| 字段 | 说明 |
|---|---|
| `fasta_path` | 注册新基因组：FASTA 绝对路径（必填，用于注册模式） |
| `genome_id` | 可选；缺省自动生成 `fe_<时间戳>` |
| `gff_path` | 可选；注册时一并挂载 GFF 注释 |

两种模式：

- **注册**：`fasta_path` 非空 → 注册新基因组（可带 `genome_id` / `gff_path`）。
- **挂载注释**：`fasta_path` 为空 + `genome_id` + `gff_path` → 给已注册的自定义基因组挂载 GFF。

```bash
# 注册
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/RegisterGenome \
  -H "Content-Type: application/json" \
  -d '{"fasta_path":"/abs/path/to/my.fa","gff_path":"/abs/path/to/my.gff3"}' | python3 -m json.tool

# 挂载注释（给已注册基因组）
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/RegisterGenome \
  -H "Content-Type: application/json" \
  -d '{"genome_id":"fe_1730000000000_my.fa","gff_path":"/abs/path/to/my.gff3"}' | python3 -m json.tool
```

返回 `result`：`{"model":"OGR-Prediction", "Genome":"<final_id>", "fasta_path":..., "gff_path":..., "chromosomes":[...]}`

> **不是必需步骤**：`Prediction` / `SNV` 请求里直接携带 `fasta_path`（+ 可选 `gff_path`）即可完成懒注册并预测（见 §5.1 / §6.1）。`RegisterGenome` 主要用于预先注册以固定 `genome_id`、或给已注册基因组单独挂载 GFF。

> **网页端（CACHE_MODE=false）**：选择「📤 自定义基因组」后，FASTA / GFF 两栏是**文本框**（不是上传按钮），直接填写后端机器上的**文件绝对路径**（如 `/mnt/.../my.fa`）。前端注册后会记住这两个路径，预测（参考 / SNV）时自动携带，无需再次填写。

---

## 9. Python 调用示例

```python
import json
import urllib.request

BASE = "http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction"
API_KEY = ""  # 后端 .env 启用鉴权时填入

def call(subpath: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"{BASE}/{subpath}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# 参考预测（Mean）
r = call("Prediction", {
    "Genome": "osa1_r7", "Chromosome": "chr01",
    "Start": 20716774, "Format": "Mean",
})
print(r["result"]["values"])          # {"RNA-seq": {"Leaf": 0.599956}}

# SNV 预测（Full）
r = call("SNV", {
    "Genome": "osa1_r7", "Chromosome": "chr01",
    "Start": 20716774, "SNV_position": 20731844, "SNV_base": "T",
})
print(r["result"]["window_start"])    # 0-based 窗口起点

# 自定义基因组：SNV 直接传 fasta_path/gff_path（懒注册，无需 RegisterGenome）
r = call("SNV", {
    "fasta_path": "/abs/path/to/my.fa",
    "gff_path": "/abs/path/to/my.gff3",
    "Chromosome": "Chr1",
    "Start": 20716774, "SNV_position": 20731844, "SNV_base": "T",
})
print(r["result"]["Genome"])         # 懒注册后自动分配的 genome id
```

---

## 10. 前端内部端点（仅供网页端，非 curl 方言）

网页端 IGV 视口柱状图使用以下端点，它们**不属于** curl 统一方言（参数与 DCS 子路径不同）：

| 端点 | 用途 | 请求体 |
|---|---|---|
| `POST /predict/bar` | IGV 视口内每 (assay, biosample) 均值 | `{prediction_id, region_start, region_end}`（0-based 视口） |
| `POST /predict/snv/stat` | ref/mut 区间差异统计 | `{snv_id, region_start, region_end}` |

> 这两个端点依赖内存中的 `ref_id` / `snv_id`，由 `Prediction` / `SNV` 响应提供，且受缓存上限约束，超时会失效。

---

## 11. 错误语义与计费

### 11.1 HTTP 状态码

| 状态 | 含义 |
|---|---|
| 200 | 成功（信封内 `status` 也为 200） |
| 400 | 参数校验错误（缺 `Start`/`SNV_position`、非法 `Format`/`SNV_base`/`biosample_names` 等） |
| 401 | API Key 无效或缺失（仅当后端 `.env` 配置了 `DCS_API_KEY`） |
| 404 | `RegisterGenome` 文件不存在 |
| 500 | 预测执行错误 |

### 11.2 错误响应结构

错误也走统一信封，`result` 为 `null`，`detail` 回显请求摘要：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "token_cnt": 0,
    "status": 400,
    "message": "Reference prediction failed: Missing required parameter 'Start' (1-based int). Usage: ...",
    "result": null,
    "detail": {"request": {"Genome": "osa1_r7", "Format": "Mean"}}
}
```

| 场景 | `detail` 内容 |
|---|---|
| 400 | `detail.request` —— 回显触发错误的请求摘要字段（`model/Genome/Chromosome/Start/SNV_position/SNV_base/Format/biosample_names/fasta_path/genome_id/gff_path`） |
| 401 | 无 `detail`（`message` 为 `Invalid or missing API Key`） |
| 404 / 500 | 400 同款 `detail.request`；500 的完整 traceback 打印在后端日志 |

### 11.3 计费口径

| 字段 | 计算方式 |
|---|---|
| `prompt_tokens` | 输入窗口碱基数（`window_len`）× `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `completion_tokens` | 输出数组元素总数 × `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）；SNV 为 ref + mut 两轨之和 |

---

## 12. 常见问题排查

| 现象 | 排查方法 |
|---|---|
| `Start` 坐标偏移 | 确认传入的是 **1-based** 坐标；`result.Start`（1-based）与 `result.window_start`（0-based）相差 1 |
| `biosample_names` 报 400 | 用内部名（`index_stat.json` 的 `biosample_order`，如 `NIP_leaf`），不要用展示名 `Leaf` |
| 401 未授权 | 后端 `.env` 配了 `DCS_API_KEY` 时需带 `Authorization: Bearer <key>`；本地开发把 `DCS_API_KEY` 留空即可免鉴权 |
| 模型未加载 | `GET /health` 看 `predictor_initialized`，`false` 时等待约 30–60s |
| `Format` 报 400 | 仅接受 `Full` / `Mean`（大小写不敏感）；不再有 `downsample` |
| 请求体不是 JSON | 返回 400「Request body is not valid JSON」 |
| 端口被占用 | `fuser -k 8001/tcp` 后重启后端 |