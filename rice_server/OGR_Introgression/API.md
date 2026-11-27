# OGR-Introgression API 调用文档

> 适用服务：`OGR_Introgression`（DNA → 粳/籼血缘渗入分析）
> 功能：输入基因组 + 染色体 + **Start（必填）** + 可选 End，输出 **256k 滑动窗口级渗入聚合**（`topk_mean_jap` / `topk_mean_ind` + 分组 `Ind`/`Jap`/`uncertain`）。
> 本文档包含 **本机（host）测试** 与 **DCS 平台测试** 两套可直接复制的 curl / Python 代码。

---

## 1. 入口总览

| 场景 | 入口 | 说明 |
|---|---|---|
| **网页端（同一进程、同一子路径）** | Gradio :5000 → 后端 :5001 `…/OGR-Introgression/Prediction` | 网页与纯 API 走**同一 DCS 子路径**，渲染数据来自该返回值 |
| **本机后端直连（DCS 子路径）** | `POST http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/Prediction` | 推荐（模型加载约 30–60s） |
| **本机后端简写** | `POST http://127.0.0.1:5001/OGR-Introgression/Prediction` | 同上，简写路径 |
| **DCS 统一网关（URL 路径）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Introgression/Prediction` | 服务与功能放路径段，body 无 `model_sub` |
| **DCS 统一网关（body 字段，兼容）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR` + body `model_sub=OGR-Introgression` | 旧式调用 |

> 统一网关由 `dcs_gateway/` 提供（唯一入口 `…/OGR`），完整说明见 [dcs_gateway/API.md](../dcs_gateway/API.md)。
> **网页端与纯 API 调用是同一进程、同一子路径、同一返回结构**（`backend/rice_introgression/main.py`）：
> 前端就是“填好参数的 curl”，Plotly 渲染完全来自 `Prediction` 子路径返回的 `result`。
> 独立 dcs_adapter 进程已删除；网页端辅助查询（基因组/染色体列表、进度）仍走内部端点。
> DCS 部署监听约定 **5001**（平台注入 `PORT` 优先，回退 `BACKEND_PORT`）。

### 1.1 子路径一览（无 body mode 分发）

| 子路径 | 功能 | 鉴权 |
|---|---|---|
| `POST …/OGR-Introgression/Prediction`（兼容 `…/predict`） | 渗入分析（Start 必填） | 需鉴权（配置后） |
| `POST …/OGR-Introgression/Genome` | 基因组列表 | 需鉴权（配置后） |
| `POST …/OGR-Introgression/Chromosome` | 染色体列表 | 需鉴权（配置后） |
| `POST …/OGR-Introgression/RegisterGenome` | 按本机路径注册自定义基因组（传路径，不传文件） | 需鉴权（配置后） |
| `POST …/OGR-Introgression/health` | 健康检查 | 免鉴权 |

---

## 2. 请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；部署端 `.env` 配置了 `DCS_API_KEY` 时 POST 必填，`health` 始终免鉴权 |

> 鉴权同时兼容 `X-API-Key: <DCS_API_KEY>` 头。服务端 `.env` 未配置 `DCS_API_KEY` 则所有 POST 免鉴权。
> 前端网页端的 API Key 由页面输入框提供（必填 UI 提示），不读 .env。
> 本机联调若 `localhost` 提示 `address already in use`：`fuser -k 5001/tcp` 后重启。

---

## 3. 健康检查（免鉴权）

```bash
# ── 本机 ──
curl -s http://127.0.0.1:5001/health | python3 -m json.tool

# ── DCS 子路径（免鉴权）──
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/health \
  -H "Content-Type: application/json" | python3 -m json.tool
```

示例返回（节选）：

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "token_cnt": 0,
    "status": 200,
    "message": "health",
    "result": {
        "model": "OGR-Introgression",
        "status": "ok",
        "predictor_initialized": true,
        "genomes": ["YF47"]
    }
}
```

---

## 4. 渗入分析（Prediction）

### 4.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `Genome` | ❌ | string | 基因组 ID（默认第一个 `GENOME_*_FASTA`，如 `YF47`） |
| `Chromosome` | ❌ | string | 染色体（默认 `Chr01`；FASTA 实际命名，不归一化） |
| `Start` | ✅ | int | **1-based inclusive** 起始坐标（必填） |
| `End` | ❌ | int | 1-based inclusive 结束坐标（可选） |
| `save_intermediate` | ❌ | bool | 是否使用/写入磁盘内容寻址缓存。**缺省 `false`**：纯 API/curl 调用每次重跑推理、不写磁盘缓存（不污染网页端命中）；网页端显式传 `true` 以启用缓存（相同窗口复用 GPU 结果）。数值结果（`segments`/`windows`/`regions`）两种模式都完整返回 |

**窗口选择语义**（start 锚定，取消网格吸附；受 `.env` `MAX_NUMBER_256W` 限制）：

| start | end | 行为 |
|---|---|---|
| 填 | 空 | **恒 1 个窗口** `[Start, Start+256k)`（忽略 `MAX_NUMBER_256W`） |
| 填 | 填 | **重叠滑窗**：起点 = start, start+64k, ...（step=64k）；起点 > end 停止；最后一个起点 ≤ end 的窗口即使尾部 > end 也保留输出；窗口数 > N 取前 N 个 |
| 空 | — | **400 报错**（Start 必填；不再支持整条染色体模式） |

**超限语义**（start 必填时）：
- 第 1 个窗口尾部 > 染色体末端 → **返回空**（HTTP 200 + `message` 说明，`windows` 为空）
- 后续窗口尾部 > 染色体末端 → 丢弃该窗口及之后（break），保留前面的

**返回坐标**：统一 **1-based inclusive**（`win_start` = 请求 Start 本身；`win_end` = 窗口闭区间末尾）。

### 4.2 示例请求

```bash
# ── 本机（DCS 子路径）──
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/Prediction \
  -H "Content-Type: application/json" \
  -d '{
    "Genome": "YF47",
    "Chromosome": "Chr01",
    "Start": 1000,
    "End": 356000
  }' | python3 -m json.tool

# ── DCS 统一网关（URL 路径）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Introgression/Prediction \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{
    "Genome": "YF47",
    "Chromosome": "Chr01",
    "Start": 1000,
    "End": 356000
  }' | python3 -m json.tool
```

**只填 Start**（恒 1 个窗口）：

```bash
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/Prediction \
  -H "Content-Type: application/json" \
  -d '{"Genome": "YF47", "Chromosome": "Chr01", "Start": 1000}' | python3 -m json.tool
```

> ⚠️ 每个 256k 窗口需推理 32 段 8k 序列（约 20–30s/窗）；建议先用小区域（Start + 短 End）验证。

### 4.3 返回结构

> 以下返回对应**只填 `Start=1000`**（恒 1 个窗口 `[1000, 256999]`，1-based）。
> 若传 `End`（如 `Start=1000, End=356000`），`n_windows` 按滑窗计算（step 64k，起点 > End 停止），
> 最后一个窗口尾部可越过 `End`（"若窗口 > end，则输出一个"），`position_1based.end` = 最后一个窗口的 1-based 闭区间末尾。

```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 256000, "completion_tokens": 1},
    "token_cnt": 256001,
    "status": 200,
    "message": "Introgression analysis succeeded",
    "result": {
        "model": "OGR-Introgression",
        "genome": "YF47",
        "fasta_path": "",
        "chromosome": "Chr01",
        "chrom_len": 43896792,
        "mode": "genome",
        "position_1based": {"start": 1000, "end": 256999},
        "window_len": 256000,
        "n_windows": 1,
        "windows": [
            {
                "win_start": 1000,
                "win_end": 256999,
                "center": 129000,
                "n_segments": 32,
                "topk_mean_jap": 0.5375,
                "topk_mean_ind": 0.5536,
                "group": "Ind"
            }
        ],
        "segments": [
            {"start": 1000, "end": 8999, "prob_jap": 0.5123, "prob_ind": 0.4981}, ...
        ],
        "regions": {
            "Ind":    [{"start": 1000, "end": 256999, "chromosome": "Chr01"}],
            "Jap":    [],
            "uncertain": []
        },
        "chromosomes": ["Chr12", "Chr11", ..., "Chr01"],
        "chromosome_lengths": {"Chr01": 43896792, "Chr02": ..., "Chr12": ...},
        "message": "",
        "cached": false,
        "elapsed_seconds": 3.2,
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

> 该返回**同时是网页端 Plotly 渲染的数据源**（前端与 curl 拿同一份 `result`）：
> `chromosomes` / `chromosome_lengths` 画 12 条染色体车道骨架，`regions[].chromosome` 定位色带，
> `windows` / `segments` 供统计与 hover。

### 4.4 返回字段说明

| 字段 | 含义 |
|---|---|
| `usage` | 计费：`prompt_tokens`=推理总碱基数（片段数×8000），`completion_tokens`=返回窗口数；可被 `DCS_*_MULTIPLIER` 缩放 |
| `result.mode` | `genome`（注入全基因组展示上下文后的展示模式） |
| `result.position_1based` | 实际推理窗口覆盖区间（**1-based inclusive**） |
| `result.windows[]` | 每个 256k 窗口的聚合：**1-based 坐标**、top-k 均值、分组 |
| `result.segments[]` | 8k 片段级结果（1-based `start`/`end` + `prob_jap`/`prob_ind`） |
| `result.regions{}` | `Ind`/`Jap`/`uncertain` 三组连续区域（带 `chromosome` 字段） |
| `result.chromosomes` / `chromosome_lengths` | 全基因组 12 条染色体车道骨架（前端绘图） |
| `result.message` | 空结果（首窗超末端）时的说明文本；正常为 `""` |
| `result.cached` | 是否命中缓存（仅 `CACHE_MODE=true` 时有意义） |
| `result.elapsed_seconds` | 本次推理耗时（s） |
| `result.params` | 本次生效的分析参数（阈值等） |
| `result.threshold_rule` | 分组规则文本（Jap / Ind / uncertain） |

**分组规则**：`Jap` = `topk_mean_jap ≥ thr_jap` 且 `topk_mean_ind < thr_ind`；`Ind` = `topk_mean_jap < thr_jap` 且 `topk_mean_ind ≥ thr_ind`；否则 `uncertain`。

### 4.5 错误响应结构

非 200 时统一返回（HTTP 状态码与 `status` 字段一致）：

```json
{
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "status": 400,
    "message": "Introgression prediction failed: Missing required parameter 'Start' (1-based int). ...",
    "result": null,
    "detail": {"request": {"Genome": "YF47", "Chromosome": "Chr01"}}
}
```

| HTTP | 触发场景 | 排查 |
|---|---|---|
| 400 | 参数缺漏/格式错（**Start 缺失**、`Start<1`、未知基因组） | 检查请求体字段与 Start 必填 |
| 401 | 鉴权失败 | 检查 `Authorization: Bearer <key>` / `X-API-Key: <key>` 与 `.env` 的 `DCS_API_KEY` |
| 404 | 基因组 FASTA / 染色体不存在 | 检查 `Chromosome` 拼写（FASTA 实际命名，不归一化） |
| 500 | 推理异常 | `detail.traceback` / 服务端日志 |
| 503 | 模型初始化失败，无法推理 | `/health` 查 `predictor_initialized`，重启容器 |

---

## 5. RegisterGenome（自定义基因组路径注册）

按**本机路径**注册自定义基因组（**不传输文件**；CACHE_MODE=false 轻量模式零拷贝）。

```bash
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/RegisterGenome \
  -H "Content-Type: application/json" \
  -d '{
    "fasta_path": "/abs/path/to/xxx.fa",
    "genome_id": "optional_id"
  }' | python3 -m json.tool
```

返回（节选）：

```json
{
    "status": 200,
    "message": "Custom genome registered: 1795685024_xxx.fa",
    "result": {
        "model": "OGR-Introgression",
        "genome": "1795685024_xxx.fa",
        "fasta_path": "/abs/path/to/xxx.fa",
        "chromosomes": ["Chr01", "Chr02", "..."],
        "chromosome_count": 12,
        "note": "Files are referenced by path only (no upload). Use the returned 'genome' as the Genome field in predict requests."
    }
}
```

> 注册后预测请求直接用返回的 `genome` 作为 `Genome` 字段；或预测请求 body 直接带 `fasta_path` 懒注册（自动生成 genome id）。

---

## 6. 易错点 / 排障

| 现象 | 处理 |
|---|---|
| 外网调用 404 | 确认服务跑的是统一后端 `main.py`（网页与纯 API 同一进程），看 `gateway.received_path` |
| 模型未加载 | `predictor_initialized=false` → 看服务端日志 |
| 推理请求慢/超时 | 模型是 1B LoRA，单窗 32 段 8k 序列；注意网关 180s 超时，先用小区域验证 |
| 鉴权 401 | 确认 `.env` 配置了 `DCS_API_KEY` 且请求带 `Authorization: Bearer <key>` |
| Start 必填 400 | 请求体必须含 `Start`（1-based ≥ 1）；不再支持整条模式 |
| 端口冲突 | `fuser -k 5001/tcp`（本机联调时清理残留进程） |

---

## 7. Python 调用示例（httpx）

```python
import httpx

# ── 本机后端直连（DCS 子路径；模型加载约 30–60s，首次推理慢属正常）──
resp = httpx.post(
    "http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/Prediction",
    json={
        "Genome": "YF47",
        "Chromosome": "Chr01",
        "Start": 1000,
        "End": 356000,
    },
    timeout=300,
)
data = resp.json()
print(data["status"], data["message"])          # 200 Introgression analysis succeeded
for w in data["result"]["windows"]:
    print(w["win_start"], w["win_end"],          # 1-based
          round(w["topk_mean_jap"], 4), round(w["topk_mean_ind"], 4), w["group"])

# ── DCS 平台（URL 路径路由 + 鉴权）──
resp = httpx.post(
    "https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Introgression/Prediction",
    headers={"Authorization": "Bearer <DCS_API_KEY>"},
    json={"Genome": "YF47", "Chromosome": "Chr02", "Start": 1000, "End": 356000},
    timeout=300,
)
print(resp.json()["usage"])                      # {"prompt_tokens": ..., "completion_tokens": ...}
```
