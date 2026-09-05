# dcs_gateway API 调用文档（四合一统一入口）

> 本文件是 rice_server 四个服务（rice_mut / rice_reg / rice_intro / rice_OGR）**经统一网关调用的唯一完整 API 文档**，
> 含 **本机（localhost）测试** 与 **DCS 平台测试** 两套可直接复制的 curl 代码。
> 四合一内容（URL 路径路由 + body 字段路由）只出现在本文件；各服务 README 不介绍 API，
> 服务级细节（参数表/返回结构/计费）见各服务 `API.md`。

---

## 1. 统一入口与路由

**外部（DCS 平台）只要一个地址**，两种路由方式：

| 方式 | 写法 | 说明 |
|---|---|---|
| **URL 路径路由（推荐）** | `POST .../OGR/{model_sub}[/{mode}]` | 服务与功能放路径段，body 只留业务参数（可省 `model_sub`/`mode`） |
| body 字段路由（兼容） | `POST .../OGR` + body `model_sub`/`mode` | 旧式调用，完全兼容 |

```text
POST {host}/api/aigress/openai/OGR/{model_sub}[/{mode}]
```

- **`model_sub` 路径段**选服务：`rice_mut` / `rice_reg` / `rice_intro` / `rice_ogr`
  （缺省默认 `rice_ogr`，即主服务 embedding/碱基预测）。
- **`mode` 路径段**选服务内功能（可省略，缺省由后端自动推断）：
  `/OGR/rice_ogr/{dna_embedding,predict}`、`/OGR/rice_mut/{predict,snv}`、
  `/OGR/rice_reg/{predict,genomes,chromosomes}`、`/OGR/rice_intro/{predict}`，`/OGR/{sub}/health` 返回网关聚合健康状态。
- 路径段只用于路由，**不传给后端**；其余字段（`sequence`/`start`/`genome`/…）与
  请求头（`Authorization`/`X-API-Key`）原样透传，后端计费/错误语义不变。
- 网关为轻量反代（`dcs_gateway/app.py`），**不加载模型**：四个模型仍由各自后端进程加载。
- 网关探活：`GET/POST /OGR/health`（或 `/health`、`/api/aigress/openai/health`）返回四个后端的聚合状态。

> **`model` 字段说明**：rice_OGR 后端仍以 `model_name` 指定实际模型名（如 `1B_8k`）；
> `model_sub` 是**网关专用**的二级路由字段，不要与后端 `model`/`model_name` 混淆。

---

## 2. 路由表（model_sub → 后端）

| `model_sub` | 后端服务 | 后端监听 | 可用 mode |
|---|---|---|---|
| `rice_mut` | rice_mut（变异对比表达预测） | `127.0.0.1:8001` | `predict` / `snv` |
| `rice_reg` | rice_reg（ATAC 条件表达预测） | `127.0.0.1:7001` | `predict` / `genomes` / `chromosomes` |
| `rice_intro` | rice_intro（粳/籼血缘渗入分析） | `127.0.0.1:5001` | `predict` |
| `rice_ogr`（**缺省**） | rice_OGR（embedding / 碱基预测） | `127.0.0.1:6001`（`.env` 可改） | `predict` / `dna_embedding` |

> 端口以 `dcs_gateway/.env` 的 `RICE_*_PORT` 为准；默认 rice_mut=8001、rice_reg=7001、rice_intro=5001、rice_OGR=6001。
> 网关自身监听：`GATEWAY_PORT`（默认 9000）；DCS 平台注入 `PORT` 时自动覆盖。

---

## 3. 公共约定

### 3.1 返回结构

四个服务返回结构一致：

```json
{
  "usage": { "prompt_tokens": 32768, "completion_tokens": 128 },
  "status": 200,
  "message": "成功提示",
  "result": { ... }
}
```

### 3.2 计费口径

- `prompt_tokens = 输入 token 数 × DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1）；
  `completion_tokens = 输出元素数 × DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）。
  水稻基模为单碱基编码，**1 bp = 1 token**。

### 3.3 鉴权

- `.env` 配置 `DCS_API_KEY` 后，除 `/health` 外均需 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；
- key 未配置（空）则不鉴权。
- 返回末尾自带换行：curl 查看结果不粘连提示符。

### 3.4 错误返回排查表

| HTTP | 场景 | 排查 |
|---|---|---|
| 400 | 参数缺漏/格式错 | 检查 `mode`、必填参数（如 rice_mut 的 `start`）、序列命名 `chr01-12` 与长度上限 |
| 400 | mode 不在白名单 | 见各服务 MODES 常量：rice_mut=`predict,snv`；rice_reg=`health,genomes,chromosomes,predict`；rice_intro=`health,predict`；rice_OGR=`dna_embedding,predict` |
| 400 | 未知 `model_sub` | `model_sub` 只能是 `rice_mut` / `rice_reg` / `rice_intro` / `rice_ogr` |
| 401 | 鉴权失败 | 带 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；检查 `.env` 的 `DCS_API_KEY` |
| 500 | 推理异常 | 看 `result`/日志 traceback；模型路径、dtype/device 是否可用 |
| 502 | 后端不可达 / 超时 | `message: gateway: backend … unreachable`；检查后端进程/端口 |
| 503 | 模型未加载完成 | `/health` 查 `init_error`；初始化失败时适配层缓存错误并整体返回 503，重启容器 |

---

## 4. 通用健康检查（免鉴权）

```bash
# ── 本机 ──
curl -s http://127.0.0.1:9000/health | python3 -m json.tool
# → {"status":"ok|degraded","services":{"rice_mut":{...},"rice_reg":{...},"rice_intro":{...},"rice_ogr":{...}}}

# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/health | python3 -m json.tool
```

返回 `status`（`ok`/`degraded`）+ 四个后端的 `reachable`/`predictor_initialized`/`init_error`。
各后端 `/health` 的 `init_error` **必须为 `null`**（非 null 表示模型加载失败）。

---

## 5. 本机（localhost）测试

前提：`bash dcs_gateway/start_all.sh`（或手动启动四个后端 + 网关）已就绪，
网关在 `127.0.0.1:9000`，四个后端均在 `127.0.0.1`（rice_mut 8001 / rice_reg 7001 / rice_intro 5001 / rice_OGR 6001）。
本机 `.env` 未配置 `DCS_API_KEY` 时无需 `Authorization` 头。

```bash
# ── 变量 ──
api_key=""                                        # 服务端启用鉴权时填入
GW="http://127.0.0.1:9000/api/aigress/openai/OGR" # 本机网关
```

### 5.1 rice_mut：参考表达预测（URL 路径，body 无 model_sub/mode）

```bash
curl -X POST ${GW}/rice_mut/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }' | python3 -m json.tool
```

### 5.2 rice_mut：SNV ref/mut 对比

```bash
curl -X POST ${GW}/rice_mut/snv \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }' | python3 -m json.tool
```

### 5.3 rice_reg：ATAC 条件表达预测

```bash
curl -X POST ${GW}/rice_reg/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 1,
    "end": 32678,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }' | python3 -m json.tool

# 辅助查询
curl -X POST ${GW}/rice_reg/genomes      -H "Content-Type: application/json" | python3 -m json.tool
curl -X POST ${GW}/rice_reg/chromosomes  -H "Content-Type: application/json" \
  -d '{"genome":"MH63RS3"}' | python3 -m json.tool
```

### 5.4 rice_intro：粳/籼血缘渗入分析

```bash
curl -X POST ${GW}/rice_intro/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "genome": "YF47",
    "chromosome": "Chr01",
    "start": 100001,
    "end": 356001
  }' | python3 -m json.tool
# 整条模式（窗口数受后端 MAX_NUMBER_256W 限制）
curl -X POST ${GW}/rice_intro/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"genome": "YF47", "chromosome": "Chr01"}' | python3 -m json.tool
```

### 5.5 rice_OGR：embedding（model_sub 缺省即 rice_ogr）

```bash
curl -X POST ${GW}/rice_ogr/dna_embedding \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }' | python3 -m json.tool

# 碱基预测
curl -X POST ${GW}/rice_ogr/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }' | python3 -m json.tool
```

### 5.6 兼容：body 字段路由（旧式，完全等价）

```bash
curl -X POST ${GW} \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "model_sub": "rice_mut",
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }' | python3 -m json.tool
```

---

## 6. DCS 平台测试（统一网关）

> `${dcs_host}` 由 DCS 平台提供；`${dcs_api_key}` 为平台分配密钥（本文示例保留真实 key 用于直接复制测试，
> 生产环境请走密钥管理）。

```bash
dcs_host="https://www.dcs.cloud"
dcs_entry="/api/aigress/openai/OGR"
dcs_api_key="sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"
```

### 6.1 健康检查（免鉴权）

```bash
curl -s ${dcs_host}${dcs_entry}/health | python3 -m json.tool
```

### 6.2 rice_mut：参考表达预测（推荐：URL 路径带服务与功能）

```bash
curl -X POST ${dcs_host}${dcs_entry}/rice_mut/predict \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }' | python3 -m json.tool
```

### 6.3 rice_mut：SNV 变异预测（双轨对比）

```bash
curl -X POST ${dcs_host}${dcs_entry}/rice_mut/snv \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }' | python3 -m json.tool
```

### 6.4 rice_reg：ATAC 条件表达预测

```bash
curl -X POST ${dcs_host}${dcs_entry}/rice_reg/predict \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 1,
    "end": 32678,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }' | python3 -m json.tool
```

### 6.5 rice_intro：粳/籼血缘渗入分析

```bash
curl -X POST ${dcs_host}${dcs_entry}/rice_intro/predict \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "genome": "YF47",
    "chromosome": "Chr01",
    "start": 100001,
    "end": 356001
  }' | python3 -m json.tool
```

### 6.6 rice_OGR：embedding 提取（model_sub 缺省即 rice_ogr）

```bash
curl -X POST ${dcs_host}${dcs_entry}/rice_ogr/dna_embedding \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }' | python3 -m json.tool
```

### 6.7 rice_OGR：下游碱基预测

```bash
curl -X POST ${dcs_host}${dcs_entry}/rice_ogr/predict \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }' | python3 -m json.tool
```

### 6.8 兼容：body 字段路由（旧式）

```bash
curl -X POST ${dcs_host}${dcs_entry} \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "model_sub": "rice_mut",
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }' | python3 -m json.tool
```

---

## 7. 各服务详情

| 服务 | 内容 | 文件 |
|---|---|---|
| rice_mut | 参数表 / 返回结构 / 计费 / 排查 | [rice_mut/API.md](../rice_mut/API.md) |
| rice_reg | 参数表 / 返回结构 / 计费 / 排查 | [rice_reg/API.md](../rice_reg/API.md) |
| rice_intro | 参数表 / 返回结构 / 计费 / 排查 | [rice_intro/API.md](../rice_intro/API.md) |
| rice_OGR | 参数表 / 返回结构 / 计费 / 排查 | [rice_OGR/API.md](../rice_OGR/API.md) |

> 命名速查（一页版）见 [quick_start.md](../quick_start.md)；部署与维护见 [AGENTS.md](../AGENTS.md)、[dcs.md](../dcs.md)。