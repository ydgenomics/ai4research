# OGR-Regulation API 调用文档

> 适用服务：`OGR_Regulation/backend/rice_reg/api.py`（ATAC → RNA-seq 表达预测）
> 功能：输入 DNA 区域 + ATAC 信号，输出 **RNA-seq +/− 两通道** 表达值（`Format=Full` 逐碱基数组 / `Format=Mean` 标量均值）。
> **网页端（Gradio :7000）与纯 API（curl）走同一套端点、同一套参数键、同一套坐标语义、同一返回信封** —— 一个后端进程（:7001）同时服务网页(/backend/* 反代)与纯 API。所有 1-based → 0-based 转换都在后端完成。

---

## 1. 入口总览

后端统一进程（默认 `:7001`）暴露 4 个 **纯子路径** 端点（camelCase 键、1-based 坐标）。每个端点都有「带前缀」和「简写」两个等价路径：

| 功能 | 端点（两个路径等价） | 说明 |
|---|---|---|
| ATAC→RNA-seq 预测 | `POST /api/aigress/openai/OGR-Regulation/Prediction`<br>`POST /OGR-Regulation/Prediction` | 表达预测（+ IGV payload） |
| 基因组列表 | `POST /api/aigress/openai/OGR-Regulation/Genome`<br>`POST /OGR-Regulation/Genome` | 已配置/已注册基因组 |
| 染色体列表 | `POST /api/aigress/openai/OGR-Regulation/Chromosome`<br>`POST /OGR-Regulation/Chromosome` | 某基因组的染色体 |
| 注册基因组 | `POST /api/aigress/openai/OGR-Regulation/RegisterGenome`<br>`POST /OGR-Regulation/RegisterGenome` | 按本机路径注册 FASTA / 挂载 GFF |
| 健康检查 | `GET /health` | 免鉴权，模型加载状态 |

> **没有 `mode` 分发、没有 `model_sub`、没有 `end`、没有 `downsample`/`max_points`/`output_format`**（`Format` 为 `Full`/`Mean`）。功能完全由 URL 子路径决定。

### 1.1 与旧版（`rice_reg` 单入口 + body `mode` + 独立 `dcs_adapter.py`）的区别

| 项 | 旧 | 新 |
|---|---|---|
| 进程/端口 | 网页版 `api.py` 与 `dcs_adapter.py` 两个进程抢 7001 | **同一进程、同一端口 7001** |
| 入口 | `…/rice_reg` + body `mode` 分发 | `…/OGR-Regulation/{Prediction,Genome,Chromosome,RegisterGenome}` |
| 参数键 | `genome/chromosome/start/end/atac_source`（snake_case） | `Genome/Chromosome/Start/ATAC-seq_Source/uploaded_atac`（camelCase） |
| 格式 | `output_format: full/mean/downsample` | `Format: Full|Mean` |
| 坐标 | 前端 `-1` 转 0-based | 1-based `Start`，后端 `start_0 = Start - 1` |
| 上传 | multipart 传文件 | 传本机路径（懒注册/懒加载） |

---

## 2. 请求头与鉴权

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；仅当后端 `.env` 配置了 `DCS_API_KEY` 时必填 |
| `X-API-Key` | ⚠️ | `Authorization` 的等价替代 |

> - 后端 `_check_api_key` 校验 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`。
> - 后端 `.env` 的 `DCS_API_KEY` **为空时所有请求免鉴权**（本地直连 / 开发测试）。
> - **网页端**：Gradio 页面在 Predict 按钮上方有 **API Key 输入框（必填）**，运行时由用户填写；前端只把输入框的值作为 `Authorization: Bearer <key>` 发送，**不读 `.env`**。
> - 若提示 `address already in use`，说明 7001 已有服务：`fuser -k 7001/tcp` 后重启。

---

## 3. 健康检查（GET /health，免鉴权）

```bash
curl -s http://127.0.0.1:7001/health | python3 -m json.tool
```

```json
{"status": "ok", "predictor_initialized": true}
```

| 字段 | 含义 |
|---|---|
| `predictor_initialized` | 模型是否加载完成（`false` 时等待约 30–40s） |

---

## 4. 统一响应信封

所有 4 个 DCS 子路径（含错误）都返回同一外层结构：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 32768, "completion_tokens": 65536},
    "token_cnt": 98304,
    "status": 200,
    "message": "...",
    "result": { ... }
}
```

| 字段 | 说明 |
|---|---|
| `model_name` | 固定 `"OGR"` |
| `usage.prompt_tokens` | 输入窗口碱基数 × `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `usage.completion_tokens` | 输出数组元素总数 × `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1） |
| `token_cnt` | `prompt_tokens + completion_tokens` |
| `status` | 200/400/401/404/500 |
| `result` | 成功时业务数据；失败时 `null` |

---

## 5. ATAC → RNA-seq 表达预测（Prediction）

### 5.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `Genome` | ❌ | string | 基因组 ID（如 `MH63`、`NIP`）；缺省取第一个已配置基因组 |
| `Chromosome` | ❌ | string | 染色体（如 `chr01`），支持别名；缺省 `chr01` |
| `Start` | ✅ | int | **1-based** 起始坐标；后端居中到 `TARGET_LEN`(32768) 窗口 |
| `ATAC-seq_Source` | ⚠️ | string | 内置 ATAC 源 ID（如 `SAM2_MH63_1`） |
| `uploaded_atac` | ⚠️ | string | ATAC bigWig 的**本机路径**（优先于 `ATAC-seq_Source`） |
| `Format` | ❌ | string | `Full`（默认，数组）/ `Mean`（标量） |
| `fasta_path` | ❌ | string | 可选：懒注册自定义基因组 FASTA |
| `gff_path` | ❌ | string | 可选：随 `fasta_path` 一并挂载 GFF |

> ⚠️ `ATAC-seq_Source` 与 `uploaded_atac` **至少提供一个**；两者都给时 `uploaded_atac` 优先。
> **自定义基因组直接传路径即可（懒注册）**：带 `fasta_path`（+ 可选 `gff_path`）时后端就地注册并使用，**无需先 RegisterGenome**。

### 5.2 Mean 格式（标量均值，最常用）

```bash
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation/Prediction \
  -H "Content-Type: application/json" \
  -d '{
    "Genome": "MH63",
    "Chromosome": "chr01",
    "Start": 20716774,
    "ATAC-seq_Source": "SAM2_MH63_1",
    "Format": "Mean"
  }' | python3 -m json.tool
```

示例返回（节选 `result`）：

```json
{
    "result": {
        "model": "OGR-Regulation",
        "Genome": "MH63",
        "fasta_path": "",
        "gff_path": "",
        "Chromosome": "chr1",
        "Start": 20716774,
        "window_start": 20716773,
        "window_len": 32768,
        "ATAC-seq_Source": "SAM2_MH63_1",
        "atac_path": "/mnt/.../ATAC_SAM2_MH63_1.MH63RS2.q30.bin5.CPM.bw",
        "Format": "Mean",
        "values": {"RNA-seq_+": 6.6e-05, "RNA-seq_-": 6.3e-05},
        "igv_payload": { "reference": {...}, "locus": "chr1:20,716,774-20,749,541", "tracks": [...] }
    }
}
```

### 5.3 Full 格式（逐碱基数组）

```bash
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation/Prediction \
  -H "Content-Type: application/json" \
  -d '{"Genome":"MH63","Chromosome":"chr01","Start":20716774,"ATAC-seq_Source":"SAM2_MH63_1","Format":"Full"}' \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)['result']
print('window_start(0-based):', r['window_start'], '| len:', r['window_len'])
print('plus[:5] :', r['values']['RNA-seq_+'][:5])
print('minus[:5]:', r['values']['RNA-seq_-'][:5])
"
```

`result` 关键字段：

| 字段 | 说明 |
|---|---|
| `Genome` / `Chromosome` | 回显（`Chromosome` 为归一化后的 FASTA 实际名，如 `chr1`） |
| `Start` | **1-based** 窗口起点 |
| `window_start` | **0-based** 窗口起点 |
| `window_len` | 窗口长度（默认 32768） |
| `ATAC-seq_Source` / `atac_path` | 请求的内置源 / 解析到的 bigWig 路径 |
| `Format` | `Full` / `Mean` |
| `values` | `{"RNA-seq_+": ..., "RNA-seq_-": ...}`（Full 数组 / Mean 标量） |
| `igv_payload` | IGV 可视化（reference/locus/tracks） |

> **自定义 ATAC（按路径）**：`uploaded_atac` 传本机 bigWig 绝对路径，后端懒加载，无文件传输。

---

## 6. 基因组 / 染色体 / 注册

### 6.1 基因组列表

```bash
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation/Genome \
  -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
```

`result`：`{"model":"OGR-Regulation","Genome":["MH63","NIP"]}`

### 6.2 染色体列表

```bash
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation/Chromosome \
  -H "Content-Type: application/json" -d '{"Genome":"MH63"}' | python3 -m json.tool
```

`result`：`{"model":"OGR-Regulation","Genome":"MH63","Chromosome":["chr01",...,"chr12"]}`

### 6.3 注册自定义基因组（RegisterGenome）

按**本机路径**注册（零拷贝、自动建 `.fai`）。两种模式：

- **注册**：`fasta_path` 非空 → 注册新基因组（可带 `genome_id` / `gff_path`）。
- **挂载注释**：`fasta_path` 为空 + `genome_id` + `gff_path` → 给已注册基因组挂 GFF。

```bash
# 注册
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation/RegisterGenome \
  -H "Content-Type: application/json" \
  -d '{"fasta_path":"/abs/path/to/my.fa","gff_path":"/abs/path/to/my.gff3"}' | python3 -m json.tool

# 挂载注释
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation/RegisterGenome \
  -H "Content-Type: application/json" \
  -d '{"genome_id":"test_mh63","gff_path":"/abs/path/to/my.gff3"}' | python3 -m json.tool
```

`result`：`{"model":"OGR-Regulation","Genome":"<final_id>","fasta_path":...,"gff_path":...,"chromosomes":[...]}`

> **不是必需步骤**：`Prediction` 请求直接携带 `fasta_path`（+ `gff_path`）即可懒注册并预测。
> **网页端**：选择「📤 自定义基因组」后，FASTA / GFF 两栏是**文本框**（填本机绝对路径）；注册后前端记住路径，预测时自动携带。

---

## 7. Python 调用示例

```python
import json, urllib.request

BASE = "http://127.0.0.1:7001/api/aigress/openai/OGR-Regulation"
API_KEY = ""  # 后端 .env 启用鉴权时填入

def call(subpath: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{BASE}/{subpath}",
        data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# 参考（内置 ATAC）
r = call("Prediction", {"Genome": "MH63", "Chromosome": "chr01",
                        "Start": 20716774, "ATAC-seq_Source": "SAM2_MH63_1",
                        "Format": "Mean"})
print(r["result"]["values"])              # {"RNA-seq_+":..., "RNA-seq_-":...}

# 自定义 ATAC（按路径）+ 自定义基因组（懒注册）
r = call("Prediction", {"fasta_path": "/abs/to/genome.fa", "Chromosome": "chr01",
                        "Start": 20716774, "uploaded_atac": "/abs/to/atac.bw",
                        "Format": "Mean"})
print(r["result"]["Genome"])
```

---

## 8. 错误语义与计费

### 8.1 HTTP 状态码

| 状态 | 含义 |
|---|---|
| 200 | 成功（信封内 `status` 也为 200） |
| 400 | 参数校验错误（缺 `Start`、非法 `Format`、缺 ATAC 输入等） |
| 401 | API Key 无效或缺失（仅当后端配置了 `DCS_API_KEY`） |
| 404 | ATAC / FASTA 文件不存在 |
| 500 | 预测执行错误 |

错误也走统一信封，`result` 为 `null`，`detail.request` 回显请求摘要：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "token_cnt": 0,
    "status": 400,
    "message": "Prediction failed: Missing ATAC input: provide 'ATAC-seq_Source' (built-in) or 'uploaded_atac' (file path). Usage: ...",
    "result": null,
    "detail": {"request": {"Genome": "MH63", "Start": 1000}}
}
```

### 8.2 计费口径

| 字段 | 计算方式 |
|---|---|
| `prompt_tokens` | 窗口碱基数（`window_len`）× `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `completion_tokens` | 输出数组元素总数（plus + minus）× `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1） |

---

## 9. 部署与排查

### 9.1 启动（单进程，网页 + API 同端口）

```bash
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/OGR_Regulation
bash backend/run_backend.sh      # 后端(:7001)，含 DCS 子路径
bash frontend/run_frontend.sh    # 网页(:7000)
```

- 监听端口：`BACKEND_PORT`（默认 7001）
- `dcs_adapter.py` 独立进程已废弃，其 DCS 子路径已合并进 `rice_reg/api.py`。

### 9.2 常见排查

| 症状 | 检查 |
|---|---|
| `Start` 坐标偏移 | 传 **1-based** `Start`；`result.Start`(1-based) 与 `result.window_start`(0-based) 差 1 |
| 400 缺 ATAC | 需 `ATAC-seq_Source` 或 `uploaded_atac` 至少一个 |
| 401 | 后端配置了 `DCS_API_KEY` 时需带 `Authorization: Bearer <key>`；本地开发留空即免鉴权 |
| `predictor_initialized: false` | `GET /health` 查看，等待模型加载约 30–40s |
| 端口冲突 | 改 `PORT` 或 `BACKEND_PORT` 后重启 |