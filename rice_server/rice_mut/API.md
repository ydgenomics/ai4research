# rice_mut API 调用文档

> 适用服务：`rice_mut/backend/dcs_adapter.py`（DNA → 多组学表达预测）
> 功能：**参考序列表达预测** 与 **单碱基变异（SNV）ref/mut 双轨对比**（assays × biosamples 多维输出）。
> 本文档包含 **本机（host）测试** 与 **DCS 平台测试** 两套可直接复制的 curl / Python 代码。

---

## 1. 入口总览

| 场景 | 入口 | 说明 |
|---|---|---|
| **本机后端直连** | `POST http://127.0.0.1:8001/api/aigress/openai/rice_mut` | 本地调试（模型加载约 30–60s） |
| **本机后端简写** | `POST http://127.0.0.1:8001/rice_mut` | 同上，简写路径 |
| **DCS 统一网关（URL 路径，推荐）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/{predict,snv}` | 服务与功能放路径段，body 无 `model_sub`/`mode` |
| **DCS 统一网关（body 字段，兼容）** | `POST https://www.dcs.cloud/api/aigress/openai/OGR` + body `model_sub=rice_mut` | 旧式调用 |

> 统一网关由 `dcs_gateway/` 提供（唯一入口 `…/OGR`），完整说明见 [dcs_gateway/API.md](../dcs_gateway/API.md)。
> 后端自身也保留子路径：`/api/aigress/openai/rice_mut/predict`、`…/rice_mut/snv`、`…/health`（免鉴权）。

### 1.1 模式分发（单入口 `POST /rice_mut` + body `mode`）

| 请求体 | 推断模式 | 说明 |
|---|---|---|
| `{"mode":"health"}` | **health** | 健康检查（免鉴权） |
| `{"mode":"snv", ...}` | **snv** | 单碱基变异双轨预测 |
| `{"mode":"predict", ...}` | **predict** | 参考序列表达预测 |
| `{}` 空 body | **health** | 自动推断（向后兼容） |
| 含 `snv_index`（无 mode） | **snv** | 自动推断（现有 SNV 调用零改动） |
| 其他（含 `start` 等，无 mode） | **predict** | 默认参考预测 |
| `{"mode":"xxx"}` | 400 | 未知 mode 报错 |

> **显式 `mode` 优先级最高**；未指定时按「空 body → health、含 snv_index → snv、其余 → predict」自动推断。

---

## 2. 请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | ✅ | `application/json` |
| `Authorization` | ⚠️ | `Bearer <DCS_API_KEY>`；部署端 `.env` 配置了 `DCS_API_KEY` 时 POST 推理必填，`health` 始终免鉴权 |

> 鉴权同时兼容 `X-API-Key: <DCS_API_KEY>` 头。服务端 `.env` 未配置 `DCS_API_KEY` 则所有 POST 免鉴权。
> 本机联调若 `localhost` 提示 `address already in use`，说明 8001 已有服务：`fuser -k 8001/tcp` 后重启。

---

## 3. 健康检查（mode=health）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"mode":"health"}' | python3 -m json.tool

# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/health \
  -H "Content-Type: application/json" | python3 -m json.tool
```

示例返回（节选）：

```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["osa1_r7"],
    "diagnostics": {
        "python": {"version": "3.12.12", ...},
        "deps": {"torch": {...}, "pyBigWig": {...}},
        "gpu": {"cuda_available": true, "device_count": 2, "device_name": "NVIDIA A40"},
        "files": {"base_model_exists": true, "checkpoint_exists": true, ...},
        "listen": {"BACKEND_HOST": "0.0.0.0", "BACKEND_PORT": "8001", "PORT": "", "actual_host": "0.0.0.0", "actual_port": 8001},
        "init_error": null
    },
    "gateway": {"received_path": "/api/aigress/openai/rice_mut", "served_port": 8001, "actual_port": 8001, "path_matches": true, ...}
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
| `gateway.path_matches` | 转发路径是否为已知路由 | — |

---

## 4. 参考序列表达预测（predict）

### 4.1 请求体参数

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `genome` | ❌ | string | 基因组 ID（默认唯一基因组 `osa1_r7`） |
| `chromosome` | ❌ | string | 染色体（默认 `chr01`）；`chr01`/`Chr1`/`1` 均可（后端自动通配） |
| `start` | ✅ | int | **1-based inclusive** 起始坐标 |
| `end` | ❌ | int | 1-based inclusive 结束坐标；缺省自动取 32768 窗口 |
| `biosample_names` | ❌ | string/array | 逗号分隔字符串或数组，缺省全部 |
| `output_format` | ❌ | string | `full`（默认）/ `mean` / `downsample` |
| `max_points` | ❌ | int | downsample 目标点数，默认 1024 |

### 4.2 mean 格式（每轨道标量均值，最常用）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr01",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }' | python3 -m json.tool

# ── DCS（推荐：URL 路径带服务与功能，body 无 mode）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/predict \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{
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
    "message": "Reference expression prediction succeeded",
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

### 4.3 full 格式（逐碱基数组，约 286KB 响应，适合落盘分析）

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"mode":"predict","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}' \
  -o predict_full.json && ls -lh predict_full.json
python3 -c "import json; d=json.load(open('predict_full.json')); print('窗口长度:', d['result']['window_len'])"
```

### 4.4 downsample 格式（均匀降采样到指定点数，兼顾大小与形状）

```bash
# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/predict \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{"genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}' \
  | python3 -m json.tool | head -30
```

---

## 5. SNV 变异预测（mode=snv，双轨对比）

### 5.1 显式 mode=snv

```bash
# ── 本机 ──
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
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

# ── DCS（URL 路径方式）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/snv \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{
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

body 含 `snv_index` 时自动走 SNV，现有 SNV 调用零改动：

```bash
# ── DCS（兼容：body 字段路由）──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR \
  -H "Authorization: Bearer <DCS_API_KEY>" -H "Content-Type: application/json" \
  -d '{"model_sub":"rice_mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T","output_format":"mean"}' \
  | python3 -m json.tool
```

示例返回（双轨道）：

```json
{
    "usage": {"prompt_tokens": 32768, "completion_tokens": 65536},
    "status": 200,
    "message": "SNV prediction succeeded (ref T → T)",
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

### 5.3 同时保存参考/突变两轨道（解析后拆分）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"mode":"snv","genome":"osa1_r7","chromosome":"chr09","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T","output_format":"downsample","max_points":256}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
json.dump(d['ref_values'], open('snv.ref.json','w'), ensure_ascii=False, indent=2)
json.dump(d['mut_values'], open('snv.mut.json','w'), ensure_ascii=False, indent=2)
json.dump({'snv_index': d['snv_index_1based'], 'ref_base': d['ref_base'], 'snv_base': d['snv_base']}, open('snv.meta.json','w'), ensure_ascii=False, indent=2)
print('已保存: snv.ref.json / snv.mut.json / snv.meta.json')
"
```

---

## 6. 批量落盘案例（Python 脚本，循环调用保存）

```bash
cat > batch_predict.py <<'EOF'
import json, time, urllib.request

BASE = "http://127.0.0.1:8001/api/aigress/openai/rice_mut"
API_KEY = ""  # 服务端启用鉴权时填入

def call(payload):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# 场景1:多个区间的参考预测,逐个保存
regions = [
    {"chromosome": "chr01", "start": 20716774, "end": 20749541},
    {"chromosome": "chr01", "start": 20800000, "end": 20832767},
    {"chromosome": "chr02", "start": 1000000,  "end": 1032767},
]
for i, r in enumerate(regions, 1):
    out = call({**r, "mode": "predict", "output_format": "mean"})
    fname = f"region_{i}.json"
    with open(fname, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[{i}] 保存 {fname} -> {out['result']['chromosome']}:{out['result']['position_1based']}")

# 场景2:同一区间多个碱基的 SNV,结果合并到一个文件
snv_results = []
for base in ["A", "C", "G", "T"]:
    out = call({"mode": "snv", "chromosome": "chr01", "start": 20716774,
                "end": 20749541, "snv_index": 20731844, "snv_base": base,
                "output_format": "mean"})
    r = out["result"]
    snv_results.append({"snv_base": base, "ref_base": r["ref_base"],
                        "ref_value": r["ref_values"]["RNA-seq"]["Leaf"],
                        "mut_value": r["mut_values"]["RNA-seq"]["Leaf"]})
    time.sleep(0.1)
with open("snv_all_bases.json", "w") as f:
    json.dump(snv_results, f, ensure_ascii=False, indent=2)
print("已保存 snv_all_bases.json:", snv_results)
EOF
python3 batch_predict.py
```

---

## 7. Python 调用示例

```python
import json
import urllib.request

BASE = "https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut"  # URL 路径方式
API_KEY = "<DCS_API_KEY>"  # 服务端未启用鉴权时可留空

def call(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/predict",                       # 参考预测;SNV 用 BASE + "/snv"
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"} if API_KEY else {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# 参考预测（mean）
print(call({
    "genome": "osa1_r7", "chromosome": "chr01",
    "start": 20716774, "end": 20749541,
    "output_format": "mean",
})["result"]["values"])
```

---

## 8. 错误语义与计费

### 8.1 HTTP 状态码

| 状态 | 含义 |
|---|---|
| 200 | 成功（体内部 `status` 也为 200） |
| 400 | 参数校验错误（缺 `start`、非法 `output_format`/`mode`/`snv_base` 等） |
| 401 | API Key 无效或缺失（仅当服务端启用了鉴权） |
| 503 | 预测器初始化失败，无法推理（看 `detail.init_error`） |
| 500 | 预测执行错误 |

### 8.2 错误响应结构

```json
{
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    "status": 400,
    "message": "Reference prediction failed: output_format must be full/mean/downsample, got 'bad_fmt'",
    "result": null,
    "detail": {"request": {"genome": "osa1_r7", "start": 20716774, "output_format": "bad_fmt"}}
}
```

> `detail` 字段（增强排障用）：
> - 400：`detail.request` —— 回显触发错误的请求摘要字段；
> - 500：`detail.error_type` + `detail.traceback`（后 2000 字符）+ `detail.request`；
> - 503：`detail.init_error`（初始化失败原因与堆栈）+ `detail.request`。

### 8.3 计费口径

| 字段 | 计算方式 |
|---|---|
| `prompt_tokens` | 输入窗口碱基数 × `DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1） |
| `completion_tokens` | 输出数组元素总数 × `DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）；SNV 为 ref + mut 两轨之和 |

---

## 9. 常见问题排查

| 现象 | 排查方法 |
|---|---|
| 平台调用 404 | `health` 的 `gateway.received_path` 看平台实际转发路径；确认转发到了容器内单入口 `/rice_mut` |
| 转发了但连不上服务 | `gateway.served_port`（转发目标端口）与 `diagnostics.listen.actual_port`（容器监听端口）比对 |
| health 返回 `predictor_initialized: false` | 看 `diagnostics.init_error` 获取模型/文件加载失败原因 |
| 缺 model 参数是否报错 | 不报错——`model` 字段非必填，服务只认 `mode`/`snv_index` 做分发，其他参数见 §4.1 |
| 想验证网关确实转发到了 | 请求任意未匹配路径（例如 `POST /anything`）会命中调试回显路由，返回 `debug_received_path` |