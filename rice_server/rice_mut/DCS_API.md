# DCS API 调用说明（rice_mut 单入口）

> 适用服务：`backend/dcs_adapter.py`（DNA → 多组学表达预测）
> 核心背景：**DCS 平台只提供一个转发地址，无法使用 `/health`、`/rice_mut/snv` 等子路径**，
> 因此全部通过 **单入口 `POST /rice_mut` + 请求体 `mode` 字段** 区分调用模式。

---

## 1. 入口与模式总览

### 1.1 平台外网地址（DCS 部署后）

```
POST https://www.dcs.cloud/api/aigress/openai/rice_mut
```

### 1.2 模式分发规则

| 请求体 | 推断模式 | 说明 |
|---|---|---|
| `{"mode":"health"}` | **health** | 健康检查（免鉴权） |
| `{"mode":"snv", ...}` | **snv** | 单碱基变异双轨预测 |
| `{"mode":"predict", ...}` | **predict** | 参考序列表达预测 |
| `{}` 空 body | **health** | 自动推断（向后兼容） |
| 含 `snv_index`（无 mode） | **snv** | 自动推断（现有 SNV 调用零改动） |
| 其他（含 `start` 等，无 mode） | **predict** | 默认参考预测 |
| `{"mode":"xxx"}` | 400 | 未知 mode 报错 |

> **显式 `mode` 优先级最高**，未指定时按"空 body → health、含 snv_index → snv、其余 → predict"自动推断。

### 1.3 保留的传统路由（本地调试用，DCS 单地址场景下不可用）

| 路径 | 方法 | 作用 |
|---|---|---|
| `/api/aigress/openai/health`、`/health` | GET/POST | 健康检查（免鉴权） |
| `/api/aigress/openai/rice_mut`、`/rice_mut` | POST | 单入口（本文档主推） |
| `/api/aigress/openai/rice_mut/snv`、`/rice_mut/snv` | POST | SNV 子路径（与 body mode=snv 等价） |

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
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"mode":"health"}' | python3 -m json.tool
```

示例返回（节选）：

```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["osa1_r7"],
    "diagnostics": {
        "python": {...},
        "deps": {...},
        "gpu": {"cuda_available": true, "device_count": 2, "device_name": "NVIDIA A40"},
        "files": {"base_model_exists": true, "checkpoint_exists": true, ...},
        "listen": {
            "BACKEND_HOST": "0.0.0.0",
            "BACKEND_PORT": "8001",
            "PORT": "",
            "actual_host": "0.0.0.0",
            "actual_port": 8001
        },
        "init_error": null
    },
    "gateway": {
        "received_path": "/api/aigress/openai/rice_mut",
        "served_host": "127.0.0.1",
        "served_port": 8001,
        "host_header": "127.0.0.1:8001",
        "remote_addr": "127.0.0.1:42626",
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
| `diagnostics.listen.actual_port` | **容器内实际监听端口** | 与 `gateway.served_port` 比对 |
| `gateway.served_port` | **平台实际转发到达的容器端口** | 转发端口配错时二者不一致 |
| `gateway.received_path` | 平台实际转发进来的路径 | 排查 404 / 路径剥离问题 |
| `gateway.port_matches` | `served_port == actual_port` 便捷布尔 | — |
| `gateway.path_matches` | 转发路径是否为已知路由 | — |

---

## 4. 参考序列表达预测（predict）

### 4.1 mean 格式（每轨道标量均值，最常用）

```bash
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }' | python3 -m json.tool
```

示例返回：

```json
{
    "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
    "status": 200,
    "message": "参考序列表达预测成功",
    "result": {
        "model": "rice_mut",
        "genome": "osa1_r7",
        "chromosome": "Chr1",
        "position_1based": {"start": 20716774, "end": 20749541},
        "window_len": 32768,
        "output_format": "mean",
        "values": {"RNA-seq": {"Leaf": 0.599956}}
    }
}
```

> **不写 `mode` 也能 predict**（自动推断），见第 1.2 节。显式 `mode:"predict"` 语义更清晰，建议业务调用都带上。

### 4.2 full 格式（逐碱基数组，约 286KB 响应）

```bash
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"mode":"predict","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}' | python3 -m json.tool | head -30
```

### 4.3 downsample 格式（均匀降采样到指定点数）

```bash
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"mode":"predict","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}' | python3 -m json.tool | head -30
```

---

## 5. SNV 变异预测（mode=snv，双轨对比）

### 5.1 显式 mode=snv

```bash
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{
    "mode": "snv",
    "genome": "osa1_r7",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }' | python3 -m json.tool
```

### 5.2 自动识别（不写 mode）

body 含 `snv_index` 时自动走 SNV，现有 SNV 调用无需改动：

```bash
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T","output_format":"mean"}' | python3 -m json.tool
```

示例返回（双轨道）：

```json
{
    "usage": {"prompt_tokens": 32768, "completion_tokens": 65536},
    "status": 200,
    "message": "SNV 预测成功 (ref T → T)",
    "result": {
        "model": "rice_mut",
        "genome": "osa1_r7",
        "chromosome": "Chr1",
        "position_1based": {"start": 20716774, "end": 20749541},
        "window_len": 32768,
        "snv_index_1based": 20731844,
        "ref_base": "T",
        "snv_base": "T",
        "output_format": "mean",
        "ref_values": {"RNA-seq": {"Leaf": 0.599956}},
        "mut_values": {"RNA-seq": {"Leaf": 0.599956}}
    }
}
```

---

## 6. 参数速查

| 参数 | 适用模式 | 必填 | 说明 |
|---|---|---|---|
| `mode` | 全部 | ❌ | `health` / `snv` / `predict`；缺省自动推断 |
| `genome` | predict/snv | ❌ | 默认唯一基因组 `osa1_r7` |
| `chromosome` | predict/snv | ❌ | 默认 `chr01`；`chr01`/`Chr1`/`1` 均可 |
| `start` | predict/snv | ✅ | 1-based inclusive（缺失返回 400） |
| `end` | predict/snv | ❌ | 缺省自动取 32768 窗口 |
| `biosample_names` | predict/snv | ❌ | 逗号分隔字符串或数组，缺省全部 |
| `output_format` | predict/snv | ❌ | `full`（默认）/ `mean` / `downsample` |
| `max_points` | predict/snv | ❌ | downsample 目标点数，默认 1024 |
| `snv_index` | snv | ✅ | 1-based 变异位点（须在窗口内） |
| `snv_base` | snv | ✅ | 目标碱基 `A`/`C`/`G`/`T`/`N` |

---

## 7. 错误语义与计费

### 7.1 HTTP 状态码

| 状态 | 含义 |
|---|---|
| 200 | 成功（体内部 `status` 也为 200） |
| 400 | 参数校验错误（缺 `start`、非法 `output_format`/`mode`/`snv_base` 等） |
| 401 | API Key 无效或缺失（仅当服务端启用了鉴权） |
| 503 | 预测器初始化失败，无法推理 |
| 500 | 预测执行错误 |

### 7.2 错误响应结构

```json
{
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "status": 400,
    "message": "参考预测失败: output_format 必须是 full/mean/downsample,收到 'bad_fmt'",
    "result": null,
    "detail": {
        "request": {"model": "rice_mut", "genome": "osa1_r7", "start": 20716774, "output_format": "bad_fmt"}
    }
}
```

> `detail` 字段（增强排障用）：
> - 400：`detail.request` —— 回显触发错误的请求摘要字段；
> - 500：`detail.error_type` + `detail.traceback`（后 2000 字符）+ `detail.request`；
> - 503：`detail.init_error`（初始化失败原因与堆栈）+ `detail.request`。

### 7.3 计费口径

| 字段 | 计算方式 |
|---|---|
| `prompt_tokens` | 输入窗口碱基数 × `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `completion_tokens` | 输出数组元素总数 × `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）；SNV 为 ref + mut 两轨之和 |

---

## 8. 常见问题排查

| 现象 | 排查方法 |
|---|---|
| 平台调用 404 | `health` 的 `gateway.received_path` 看平台实际转发路径；确认转发到了容器内单入口 `/rice_mut` |
| 转发了但连不上服务 | `gateway.served_port`（转发目标端口）与 `diagnostics.listen.actual_port`（容器监听端口）比对 |
| health 返回 `predictor_initialized: false` | 看 `diagnostics.init_error` 获取模型/文件加载失败原因 |
| 缺 model 参数是否报错 | 不报错——`model` 字段非必填，服务只认 `mode`/`snv_index` 做分发，其他参数见第 6 节 |
| 想验证网关确实转发到了 | 请求任意未匹配路径（例如 `POST /anything`）会命中调试回显路由，返回 `debug_received_path` |

---

## 9. Python 调用示例

```python
import json
import urllib.request

BASE = "https://www.dcs.cloud/api/aigress/openai/rice_mut"
API_KEY = "<DCS_API_KEY>"  # 服务端未启用鉴权时可留空

def call(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"} if API_KEY else {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# 健康检查
print(call({"mode": "health"})["predictor_initialized"])

# 参考预测（mean）
print(call({
    "mode": "predict",
    "genome": "osa1_r7", "chromosome": "chr01",
    "start": 20716774, "end": 20749541,
    "output_format": "mean",
})["result"]["values"])

# SNV 预测（自动识别）
print(call({
    "genome": "osa1_r7", "chromosome": "chr01",
    "start": 20716774, "end": 20749541,
    "snv_index": 20731844, "snv_base": "T",
    "output_format": "mean",
})["result"]["ref_values"])
```