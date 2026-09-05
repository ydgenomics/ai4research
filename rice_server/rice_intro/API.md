# rice_intro API 调用文档

> 适用服务：`rice_intro/backend/dcs_adapter.py`（DNA → 粳/籼血缘渗入分析）
> 功能：输入基因组 + 染色体 + 可选区间，输出 **256k 滑动窗口级渗入聚合**（`topk_mean_jap` / `topk_mean_ind` + 分组 `Ind`/`Jap`/`uncertain`）。
> 本文档包含 **本机（host）测试** 与 **DCS 平台测试** 两套可直接复制的 curl / Python 代码。

---

## 1. 入口总览

| 场景 | 入口 | 说明 |
|---|---|---|
| **本机后端直连** | `POST http://127.0.0.1:5001/api/aigress/openai/rice_intro` | 本地调试（模型加载约 30–60s） |
| **本机后端简写** | `POST http://127.0.0.1:5001/rice_intro` | 同上，简写路径 |
| **DCS 统一网关（URL 路径，推荐）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_intro/predict` | 服务与功能放路径段，body 无 `model_sub`/`mode` |
| **DCS 统一网关（body 字段，兼容）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR` + body `model_sub=rice_intro` | 旧式调用 |

> 统一网关由 `dcs_gateway/` 提供（唯一入口 `…/OGR`），完整说明见 [dcs_gateway/API.md](../dcs_gateway/API.md)。
> 后端自身也保留子路径：`/api/aigress/openai/rice_intro`、`…/health`（免鉴权）。
> DCS 部署监听约定 **5001**（平台注入 `PORT` 优先，回退 `BACKEND_PORT`）；与网页版 5001 是**不同机器**。

### 1.1 模式分发（单入口 `POST /rice_intro` + body `mode`）

| 请求体 | 推断模式 | 说明 |
|---|---|---|
| `{"mode":"health"}` | **health** | 健康检查（免鉴权） |
| `{"mode":"predict", ...}` | **predict** | 渗入分析（默认） |
| `{}` 空 body | **health** | 自动推断（向后兼容） |
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
> 本机联调若 `localhost` 提示 `address already in use`，说明 5001 已有进程：`fuser -k 5001/tcp` 后重启。

---

## 3. 健康检查（mode=health）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/rice_intro \
  -H "Content-Type: application/json" \
  -d '{"mode":"health"}' | python3 -m json.tool

# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_intro/health \
  -H "Content-Type: application/json" | python3 -m json.tool
```

示例返回（节选）：

```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["YF47"],
    "diagnostics": {
        "python": {"version": "3.12.12", ...},
        "deps": {"torch": {...}, "transformers": {...}, "fastapi": {...}},
        "gpu": {"cuda_available": true, "device_count": 2, "device_name": "NVIDIA A40"},
        "files": {"base_model_exists": true, "checkpoint_exists": true, "genome_yf47_exists": true},
        "listen": {"BACKEND_HOST": "0.0.0.0", "BACKEND_PORT": "5001", "PORT": "", "actual_host": "0.0.0.0", "actual_port": 5001},
        "init_error": null
    },
    "gateway": {"received_path": "/api/aigress/openai/rice_intro", "served_port": 5001, "actual_port": 5001, "path_matches": true, ...}
}
```

### 关键字段解读

| 字段 | 含义 | 排查用途 |
|---|---|---|
| `predictor_initialized` | 模型是否加载完成 | `false` 时看 `diagnostics.init_error` |
| `diagnostics.listen.actual_port` | **容器内实际监听端口** | 与 `gateway.served_port` 比对 |
| `gateway.served_port` | **平台实际转发到达的容器端口** | 转发端口配错时二者不一致 |
| `gateway.received_path` | 平台实际转发进来的路径 | 排查 404 / 路径剥离问题 |
| `gateway.port_matches` | `served_port == actual_port` 便捷布尔 | — |

---

## 4. 渗入分析（predict，默认模式）

### 4.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `genome` | ❌ | string | 基因组 ID（默认第一个 `GENOME_*_FASTA`，如 `YF47`） |
| `chromosome` | ❌ | string | 染色体（默认 `Chr01`；FASTA 实际命名，不归一化） |
| `start` | ❌ | int | **1-based inclusive** 起始坐标 |
| `end` | ❌ | int | 1-based inclusive 结束坐标 |

**窗口选择语义**（与网页版 `/analyze` 逐位一致，受 `.env` `MAX_NUMBER_256W` 限制）：

| start | end | 行为 |
|---|---|---|
| 空 | 空 | 整条染色体：`MAX_NUMBER_256W` 空=全部窗口；N=前 N 个 256k 窗口 |
| 填 | 空 | 1 个覆盖 start 的最大 256k 窗口（忽略 `MAX_NUMBER_256W`） |
| 填 | 填 | 与该区间覆盖度最大的 ≤`MAX_NUMBER_256W` 个 256k 窗口 |

### 4.2 示例请求

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/rice_intro \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "genome": "YF47",
    "chromosome": "Chr01",
    "start": 100001,
    "end": 356001
  }' | python3 -m json.tool

# ── DCS（推荐：URL 路径带服务与功能，body 无 mode）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_intro/predict \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{
    "genome": "YF47",
    "chromosome": "Chr01",
    "start": 100001,
    "end": 356001
  }' | python3 -m json.tool
```

### 4.3 返回结构

```json
{
    "usage": {"prompt_tokens": 512000, "completion_tokens": 2},
    "status": 200,
    "message": "Introgression analysis succeeded",
    "result": {
        "model": "rice_intro",
        "genome": "YF47",
        "chromosome": "Chr01",
        "chrom_len": 43896792,
        "mode": "window",
        "position_1based": {"start": 64001, "end": 384001},
        "window_len": 256000,
        "n_windows": 2,
        "windows": [
            {
                "win_start": 64001,
                "win_end": 320001,
                "center": 192001,
                "n_segments": 32,
                "topk_mean_jap": 0.5312,
                "topk_mean_ind": 0.5481,
                "group": "Ind"
            },
            {
                "win_start": 128001,
                "win_end": 384001,
                "center": 256001,
                "n_segments": 32,
                "topk_mean_jap": 0.5401,
                "topk_mean_ind": 0.5422,
                "group": "Ind"
            }
        ],
        "params": {
            "segment_size": 8000,
            "window_size": 256000,
            "window_step": 64000,
            "top_k": 10,
            "threshold_jap": 0.55519,
            "threshold_ind": 0.53473
        },
        "threshold_rule": "Ind: prob_jap < 0.5552 & prob_ind >= 0.5347; Jap: prob_jap >= 0.5552 & prob_ind < 0.5347; uncertain: otherwise"
    }
}
```

### 4.4 返回字段说明

| 字段 | 含义 |
|---|---|
| `usage` | 计费：`prompt_tokens`=推理总碱基数（片段数×8000），`completion_tokens`=返回窗口数；可被 `DCS_*_MULTIPLIER` 缩放 |
| `result.mode` | `window`（窗口模式）或 `chromosome`（整条模式） |
| `result.position_1based` | 实际推理窗口覆盖区间（1-based inclusive；整条模式=整条染色体） |
| `result.windows[]` | 每个 256k 窗口的聚合：1-based 坐标、top-k 均值、分组 |
| `result.params` | 本次生效的分析参数（阈值等） |
| `result.threshold_rule` | 分组规则文本（Jap / Ind / uncertain） |

**分组规则**：`Jap` = `topk_mean_jap ≥ thr_jap` 且 `topk_mean_ind < thr_ind`；`Ind` = `topk_mean_jap < thr_jap` 且 `topk_mean_ind ≥ thr_ind`；否则 `uncertain`。

---

## 5. 易错点 / 排障

| 现象 | 处理 |
|---|---|
| 外网调用 404 | 看 `gateway.received_path` / `gateway.served_port`；确认容器跑的是 `dcs_adapter.py`（不是网页版 `main.py`/`api.py`） |
| 模型未加载 | `predictor_initialized=false` → 看 `diagnostics.init_error` |
| 推理请求慢/超时 | 模型是 1B LoRA，单窗 32 段 8k 序列；整条模式 + MAX 空时窗口数可达 683（约 5488 片段），注意网关 180s 超时 |
| 鉴权 401 | 确认 `.env` 配置了 `DCS_API_KEY` 且请求带 `Authorization: Bearer <key>` |
| 端口冲突 | `fuser -k 5001/tcp`（本机联调时清理残留进程） |