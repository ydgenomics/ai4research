# rice_reg API 调用文档

> 适用服务：`rice_reg/backend/dcs_adapter.py`（ATAC → RNA-seq 表达预测）
> 功能：输入 DNA 区域 + ATAC 信号，输出 **RNA-seq +/− 两通道** 表达轨迹（按 `output_format` 为逐碱基数组 / 标量均值 / 降采样）。
> 本文档包含 **本机（host）测试** 与 **DCS 平台测试** 两套可直接复制的 curl 代码。

---

## 1. 入口总览

| 场景 | 入口 | 说明 |
|---|---|---|
| **本机后端直连** | `POST http://127.0.0.1:7001/api/aigress/openai/rice_reg` | 本地调试（模型加载约 30–40s） |
| **本机后端简写** | `POST http://127.0.0.1:7001/rice_reg` | 同上，简写路径 |
| **DCS 统一网关（URL 路径，推荐）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/{predict,genomes,chromosomes}` | 服务与功能放路径段，body 无 `model_sub`/`mode` |
| **DCS 统一网关（body 字段，兼容）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR` + body `model_sub=rice_reg` | 旧式调用 |

> 统一网关由 `dcs_gateway/` 提供（唯一入口 `…/OGR`），完整说明见 [dcs_gateway/API.md](../dcs_gateway/API.md)。
> 后端自身也保留子路径：`/api/aigress/openai/rice_reg/predict`、`…/rice_reg/genomes`、`…/rice_reg/chromosomes`、`…/health`（免鉴权）。

### 1.1 模式分发（单入口 `POST /rice_reg` + body `mode`）

| 请求体 | 推断模式 | 说明 |
|---|---|---|
| `{"mode":"health"}` | **health** | 健康检查（免鉴权） |
| `{"mode":"genomes"}` | **genomes** | 已配置基因组列表 |
| `{"mode":"chromosomes","genome":"MH63RS3"}` | **chromosomes** | 指定基因组的染色体列表 |
| `{"mode":"predict", ...}` | **predict** | ATAC→RNA-seq 表达预测 |
| `{}` 空 body | **health** | 自动推断 |
| 其他（含 `start` 等，无 mode） | **predict** | 默认预测 |
| `{"mode":"xxx"}` | 400 | 未知 mode 报错 |

> **显式 `mode` 优先级最高**；未指定时按「空 body → health、其余 → predict」自动推断。

---

## 2. 请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；部署端 `.env` 配置了 `DCS_API_KEY` 时 POST 推理必填，`health` 始终免鉴权 |

> 鉴权同时兼容 `X-API-Key: <DCS_API_KEY>` 头。服务端 `.env` 未配置 `DCS_API_KEY` 则所有 POST 免鉴权。
> 本机 `.env` 未配置 `DCS_API_KEY` 时**无需** `Authorization` 头；若提示 `address already in use`，说明 7001 已有服务：`fuser -k 7001/tcp` 清理后重启。

---

## 3. 健康检查（mode=health）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"health"}' | python3 -m json.tool

# ── DCS（URL 路径 /health）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/health \
  -H "Content-Type: application/json" | python3 -m json.tool
```

示例返回（节选）：

```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["MH63RS3", "NIP"],
    "atac_sources": ["SAM2_MH63_1", "SAM2_NIP_1"],
    "diagnostics": {
        "python": {"version": "3.12.12", ...},
        "deps": {"torch": ..., "pyBigWig": ...},
        "gpu": {"cuda_available": true, "device_count": 1, "device_name": "NVIDIA A40"},
        "files": {"base_model_exists": true, "checkpoint_exists": true},
        "genomes": {"MH63RS3": true, "NIP": true},
        "atac_sources": {"SAM2_MH63_1": true, "SAM2_NIP_1": true},
        "listen": {"BACKEND_HOST": "0.0.0.0", "BACKEND_PORT": "7001", "PORT": "", "actual_host": "0.0.0.0", "actual_port": 7001},
        "init_error": null
    },
    "gateway": {"received_path": "/api/aigress/openai/rice_reg", "served_port": 7001, "actual_port": 7001, "path_matches": true, ...}
}
```

### 关键字段解读

| 字段 | 含义 | 排查用途 |
|---|---|---|
| `predictor_initialized` | 模型是否加载完成 | `false` 时看 `diagnostics.init_error` |
| `diagnostics.genomes` | 各基因组 FASTA 文件是否存在 | — |
| `diagnostics.atac_sources` | 各内置 ATAC 源 bigWig 是否存在 | — |
| `diagnostics.listen.actual_port` | **容器内实际监听端口** | 与 `gateway.served_port` 比对 |
| `gateway.served_port` | **平台实际转发到达的容器端口** | 转发端口配错时二者不一致 |
| `gateway.received_path` | 平台实际转发进来的路径 | 排查 404 / 路径剥离问题 |

---

## 4. ATAC → RNA-seq 表达预测（predict，默认模式）

### 4.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `genome` | ✅* | string | 基因组 ID（如 `MH63RS3`、`NIP`）；缺省时用第一个已配置基因组 |
| `chromosome` | ❌ | string | 染色体（如 `chr01`），支持别名；缺省 `chr01` |
| `start` | ✅ | int | **1-based inclusive** 起始坐标 |
| `end` | ❌ | int | 1-based inclusive 结束坐标；缺省则自动取 `TARGET_LEN` 窗口 |
| `atac_source` | ⚠️ | string | 内置 ATAC 源 ID（如 `SAM2_MH63_1` → `.env` 的 `ATAC_PATH_SAM2_MH63_1`） |
| `uploaded_atac` | ⚠️ | string | 服务器上已上传的 ATAC bigWig 文件路径（**优先于** `atac_source`） |
| `output_format` | ❌ | string | `full`（默认）/ `mean` / `downsample` |
| `max_points` | ❌ | int | `downsample` 时的目标点数（默认 1024） |

> ⚠️ `atac_source` 与 `uploaded_atac` **至少提供一个**；两者都给时 `uploaded_atac` 优先。
> 坐标约定：网页版输入框是 1-based inclusive，`start` 与 `end` 都按此理解；内部自动归一化到模型窗口（center-align + truncate/pad 到 `TARGET_LEN`）。

### 4.2 mean 格式（每轨道标量均值，最常用）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }' | python3 -m json.tool

# ── DCS（推荐：URL 路径带服务与功能，body 无 mode）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/predict \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }' | python3 -m json.tool
```

示例返回：

```json
{
    "usage": {"prompt_tokens": 32768, "completion_tokens": 2},
    "status": 200,
    "message": "ATAC→RNA-seq expression prediction succeeded",
    "result": {
        "model": "rice_reg",
        "genome": "MH63RS3",
        "chromosome": "chr01",
        "position_1based": {"start": 20703125, "end": 20735892},
        "window_len": 32768,
        "atac_source": "SAM2_MH63_1",
        "atac_path": "/mnt/.../ATAC_SAM2_MH63_1.bw",
        "output_format": "mean",
        "values": {
            "RNA-seq_+": 2.830175,
            "RNA-seq_-": 2.851024
        }
    }
}
```

### 4.3 full 格式（完整逐碱基数组，每个值都是 `(window_len,)` 的浮点数组）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "full"
  }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
r = d['result']['values']
print('shape: +', len(r['RNA-seq_+']), '-', len(r['RNA-seq_-']))
print('plus[:5] :', r['RNA-seq_+'][:5])
print('minus[:5]:', r['RNA-seq_-'][:5])
"
```

### 4.4 downsample 格式（降采样到 max_points 点，适合画图）

```bash
# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/predict \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "downsample",
    "max_points": 1024
  }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
r = d['result']['values']
print('status:', d['status'], '| points:', len(r['RNA-seq_+']))
"
```

> `downsample` 用 `np.linspace` 均匀取点，保留整体趋势。

---

## 5. 基因组 / 染色体查询

### 5.1 基因组列表（mode=genomes）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"genomes"}' | python3 -m json.tool

# ── DCS（URL 路径方式）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/genomes \
  -H "Content-Type: application/json" | python3 -m json.tool
```

```json
{
    "status": 200,
    "result": {"model": "rice_reg", "genomes": ["MH63RS3", "NIP"]}
}
```

### 5.2 染色体列表（mode=chromosomes）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"chromosomes","genome":"MH63RS3"}' | python3 -m json.tool

# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/chromosomes \
  -H "Content-Type: application/json" \
  -d '{"genome":"MH63RS3"}' | python3 -m json.tool
```

```json
{
    "status": 200,
    "result": {
        "model": "rice_reg",
        "genome": "MH63RS3",
        "chromosomes": ["chr01", "chr02", "chr03", "...", "chr12"]
    }
}
```

---

## 6. 错误语义

| HTTP code（`status` 字段）| 场景 | `message` 示例 |
|---|---|---|
| 400 | 请求参数错误 | `Request body is not a valid JSON object` / `Prediction failed: missing ATAC input:...` / `Unknown mode 'xxx',...` |
| 401 | API Key 无效 | `Invalid or missing API Key` |
| 404 | 文件不存在（ATAC / FASTA） | `Prediction failed: ATAC bigWig not found: ...` |
| 500 | 预测/服务内部错误 | `Prediction failed: <error>`（带 `detail.error_type` / `traceback`） |
| 503 | 预测器未初始化 | `Predictor initialization failed, cannot infer: ...`（带 `detail.init_error`） |

错误响应结构：

```json
{
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "status": 400,
    "message": "Prediction failed: missing ATAC input: need 'atac_source' (built-in) or 'uploaded_atac' (file path)",
    "result": null,
    "detail": {"request": {"genome": "MH63RS3", "start": 1000}}
}
```

> 400 的 `detail` 带 `request`（请求摘要）；500 额外带 `error_type` + `traceback`；503 带 `init_error`（含完整 traceback）。所有错误信息已随响应返回，无需 SSH 进容器即可定位。

---

## 7. 计费口径

| 字段 | 口径 | 可调环境变量 |
|---|---|---|
| `usage.prompt_tokens` | 输入窗口碱基数（`position.end - position.start`） | `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `usage.completion_tokens` | 输出元素总数（plus + minus 数组元素数） | `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1） |

---

## 8. 部署与排查

### 8.1 启动

```bash
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/rice_reg
nohup /path/to/python backend/dcs_adapter.py > backend/logs/dcs_adapter.nohup.log 2>&1 &
```

- 监听端口优先级：`PORT`（平台注入）> `BACKEND_PORT`（默认 7001）
- 模型初始化失败**不会退出进程**：服务照常监听（平台探活不挂），原因写入 `diagnostics.init_error`，health 即可查看

### 8.2 常见排查

| 症状 | 检查 |
|---|---|
| 外网调用 404 | `gateway.received_path` 看平台实际转发路径；确认容器跑的是最新 `dcs_adapter.py`（不是旧的 `api.py` 入口） |
| 外网调用无响应/连接失败 | `gateway.served_port` 与 `diagnostics.listen.actual_port` 是否一致（转发端口配置） |
| `predictor_initialized: false` | `diagnostics.init_error` 看模型加载失败原因（路径 / HF 模型名等） |
| 端口冲突 | 改 `PORT` 或 `BACKEND_PORT` 后重启 |