# DCS API 调用说明（rice_reg 单入口）

> 适用服务：`backend/dcs_adapter.py`（ATAC → RNA-seq 表达预测）
> 核心背景：**DCS 平台只提供一个转发地址，无法使用 `/health`、`/genomes`、`/predict/rice-reg` 等子路径**，
> 因此全部通过 **单入口 `POST /rice_reg` + 请求体 `mode` 字段** 区分调用模式。

---

## 1. 入口与模式总览

### 1.1 平台外网地址（DCS 部署后）

```
POST https://www.dcs.cloud/api/aigress/openai/rice_reg
```

### 1.2 模式分发规则

| 请求体 | 推断模式 | 说明 |
|---|---|---|
| `{"mode":"health"}` | **health** | 健康检查（免鉴权） |
| `{"mode":"genomes"}` | **genomes** | 已配置基因组列表 |
| `{"mode":"chromosomes","genome":"MH63RS3"}` | **chromosomes** | 指定基因组的染色体列表 |
| `{"mode":"predict", ...}` | **predict** | ATAC→RNA-seq 表达预测 |
| `{}` 空 body | **health** | 自动推断 |
| 其他（含 `start` 等，无 mode） | **predict** | 默认预测 |
| `{"mode":"xxx"}` | 400 | 未知 mode 报错 |

> **显式 `mode` 优先级最高**，未指定时按"空 body → health、其余 → predict"自动推断。

### 1.3 本机（host）测试说明

部署在 DCS 之前先在**宿主机/本机**验证，任何模式的 host 命令都指向 `http://127.0.0.1:7001`：

```bash
# 启动服务（模型加载约 30-40s，health 返回 ok 即可调用）
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/rice_reg
/root/miniconda3/envs/vllm/bin/python backend/dcs_adapter.py

# 等价路由（本机两种路径均可用）
#   POST http://127.0.0.1:7001/api/aigress/openai/rice_reg  ← 与 DCS 外网同路径，推荐
#   POST http://127.0.0.1:7001/rice_reg                    ← 简写入口
```

> 本机 `.env` 未配置 `DCS_API_KEY` 时**无需** `Authorization` 头；配置后所有 POST 需带 `Bearer <DCS_API_KEY>`。
> 若提示 `address already in use`，说明 7001 已有服务：`fuser -k 7001/tcp` 清理后重启。

### 1.4 保留的传统路由（本地调试用，DCS 单地址场景下不可用）

| 路径 | 方法 | 作用 |
|---|---|---|
| `/api/aigress/openai/health`、`/health` | GET/POST | 健康检查（免鉴权） |
| `/api/aigress/openai/rice_reg`、`/rice_reg` | POST | 单入口（本文档主推） |
| `/predict/rice-reg` 等 | — | 原网页版路由（本地调试用） |

---

## 2. 请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；部署端 `.env` 配置了 `DCS_API_KEY` 时 POST 推理必填，`health` 模式始终免鉴权 |

> 鉴权同时还兼容 `X-API-Key: <DCS_API_KEY>` 头。
> 若服务端 `.env` 未配置 `DCS_API_KEY`，则所有 POST 均免鉴权。

---

## 3. 健康检查（mode=health）

```bash
# host 本机测试
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"health"}' | python3 -m json.tool

# DCS 外网（部署后）
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"health"}' | python3 -m json.tool
```

示例返回（节选）：

```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["MH63RS3", "NIP"],
    "atac_sources": ["ATAC_PATH_SAM2_MH63_1", "ATAC_PATH_SAM2_NIP_1"],
    "diagnostics": {
        "python": {...},
        "deps": {"torch": ..., "pyBigWig": ...},
        "gpu": {"cuda_available": true, "device_count": 1, "device_name": "NVIDIA A40"},
        "files": {"base_model_exists": true, "checkpoint_exists": true},
        "genomes": {"MH63RS3": true, "NIP": true},
        "atac_sources": {"SAM2_MH63_1": true, "SAM2_NIP_1": true},
        "listen": {
            "BACKEND_HOST": "0.0.0.0",
            "BACKEND_PORT": "7001",
            "PORT": "",
            "actual_host": "0.0.0.0",
            "actual_port": 7001
        },
        "init_error": null
    },
    "gateway": {
        "received_path": "/api/aigress/openai/rice_reg",
        "served_host": "127.0.0.1",
        "served_port": 7001,
        "host_header": "127.0.0.1:7001",
        "remote_addr": "127.0.0.1:43218",
        "port_matches": true,
        "path_matches": true
    }
}
```

### 关键字段解读

| 字段 | 含义 | 排查用途 |
|---|---|---|
| `status` | 恒为 `ok`（HTTP 服务存活，不破坏平台探活） | — |
| `predictor_initialized` | 模型是否加载完成 | `false` 时看 `diagnostics.init_error` |
| `diagnostics.genomes` | 各基因组 FASTA 文件是否存在 | — |
| `diagnostics.atac_sources` | 各内置 ATAC 源 bigWig 是否存在 | — |
| `diagnostics.listen.actual_port` | **容器内实际监听端口** | 与 `gateway.served_port` 比对 |
| `gateway.served_port` | **平台实际转发到达的容器端口** | 转发端口配错时二者不一致 |
| `gateway.received_path` | 平台实际转发进来的路径 | 排查 404 / 路径剥离问题 |
| `gateway.port_matches` | `served_port == actual_port` 便捷布尔 | — |
| `gateway.path_matches` | 转发路径是否为已知路由 | — |

---

## 4. ATAC → RNA-seq 表达预测（predict，默认模式）

### 4.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `genome` | ✅* | string | 基因组 ID（如 `MH63RS3`、`NIP`）；缺省时用第一个已配置基因组 |
| `chromosome` | ❌ | string | 染色体（如 `chr01`），支持别名；缺省 `chr01` |
| `start` | ✅ | int | 1-based inclusive 起始坐标 |
| `end` | ❌ | int | 1-based inclusive 结束坐标；缺省则自动取 `TARGET_LEN` 窗口 |
| `atac_source` | ⚠️ | string | 内置 ATAC 源 ID（如 `SAM2_MH63_1` → `.env` 的 `ATAC_PATH_SAM2_MH63_1`） |
| `uploaded_atac` | ⚠️ | string | 服务器上已上传的 ATAC bigWig 文件路径（**优先于** `atac_source`） |
| `output_format` | ❌ | string | `full`（默认）/ `mean` / `downsample` |
| `max_points` | ❌ | int | `downsample` 时的目标点数（默认 1024） |

> ⚠️ `atac_source` 与 `uploaded_atac` **至少提供一个**；两者都给时 `uploaded_atac` 优先。
> 坐标约定：网页版输入框是 1-based inclusive，`start` 与 `end` 都按此理解；内部自动归一化到模型窗口（center-align + truncate/pad 到 `TARGET_LEN`）。

### 4.2 mean 格式（每轨道标量均值，最常用）

```bash
# host 本机测试
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

# DCS 外网（部署后）
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'
```

示例返回：

```json
{
    "usage": {"prompt_tokens": 32768, "completion_tokens": 2},
    "status": 200,
    "message": "ATAC→RNA-seq 表达预测成功",
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

### 4.3 full 格式（完整逐碱基数组）

每个值都是 `(window_len,)` 的浮点数组（默认窗口 32678，受 `TARGET_LEN` 控制）：

```bash
# host 本机测试
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

```bash
# DCS 外网（部署后）
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "full"
  }'
```

### 4.4 downsample 格式（降采样到 max_points 点）

```bash
# host 本机测试
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
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

```bash
# DCS 外网（部署后）
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "atac_source": "SAM2_MH63_1",
    "output_format": "downsample",
    "max_points": 1024
  }'
```

> `downsample` 用 `np.linspace` 均匀取点，保留整体趋势，适合画图。

---

## 5. 基因组 / 染色体查询

### 5.1 基因组列表（mode=genomes）

```bash
# host 本机测试
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"genomes"}'

# DCS 外网（部署后）
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"genomes"}'
```

```json
{
    "status": 200,
    "result": {"model": "rice_reg", "genomes": ["MH63RS3", "NIP"]}
}
```

### 5.2 染色体列表（mode=chromosomes）

```bash
# host 本机测试
curl -s -X POST http://127.0.0.1:7001/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"chromosomes","genome":"MH63RS3"}'

# DCS 外网（部署后）
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -d '{"mode":"chromosomes","genome":"MH63RS3"}'
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
| 400 | 请求参数错误 | `请求体不是合法 JSON` / `预测失败: 缺少 ATAC 输入:...` / `未知 mode 'xxx',...` |
| 401 | API Key 无效 | `无效或缺失的 API Key` |
| 404 | 文件不存在（ATAC / FASTA） | `预测失败: ATAC bigWig not found: ...` |
| 500 | 预测/服务内部错误 | `预测失败: <error>`（带 `detail.error_type` / `traceback`） |
| 503 | 预测器未初始化 | `预测器初始化失败,无法推理: ...`（带 `detail.init_error`） |

错误响应结构：

```json
{
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "status": 400,
    "message": "预测失败: 缺少 ATAC 输入:需提供 'atac_source'(内置)或 'uploaded_atac'(文件路径)",
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

---

## 9. 与网页版（原 `/predict/rice-reg`）的差异

| 维度 | 网页版 `/predict/rice-reg` | DCS 单入口（本适配层） |
|---|---|---|
| 输入 | web 表单（含文件上传） | JSON body（`atac_source` / `uploaded_atac` 路径） |
| 输出 | IGV payload（tracks 指向静态文件 URL） | 数值数组（`values.RNA-seq_+/-`），无 IGV 依赖 |
| 缓存 | 内容寻址缓存 + 预测 bigWig 落盘 | 复用同一 predict 链路（窗口归一化、染色体别名、ATAC 校验一致），不写 bigWig |
| 鉴权 | 无 | 可选 `DCS_API_KEY` |