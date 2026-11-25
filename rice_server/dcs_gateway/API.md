# dcs_gateway API 调用文档（四合一统一入口）

> 本文件是 rice_server 四个服务（OGR-Prediction / OGR-Regulation / OGR-Introgression / OGR-Base）**经统一网关调用的唯一完整 API 文档**，
> 含 **本机（localhost）测试** 与 **DCS 平台测试** 两套可直接复制的 curl 代码。
> 四合一内容（URL 路径路由 + body 字段路由）只出现在本文件；各服务 README 不介绍 API，
> 服务级细节（参数表/返回结构/计费）见各服务 `API.md`。

---

## 1. 统一入口与路由

**外部（DCS 平台）只要一个地址**，两种路由方式：

| 方式 | 写法 | 说明 |
|---|---|---|
| **URL 路径路由（推荐）** | `POST .../OGR/{model_sub}[/{subpath}]` | 服务与功能放路径段，body 只留业务参数（可省 `model_sub`） |
| body 字段路由（兼容） | `POST .../OGR` + body `model_sub` | 旧式调用，完全兼容 |

```text
POST {host}/api/aigress/openai/OGR/{model_sub}[/{subpath}]
```

- **`model_sub` 路径段**选服务：`OGR-Prediction` / `OGR-Regulation` / `OGR-Introgression` / `OGR-Base`
  （大小写与连字符不敏感；缺省默认 `OGR-Base`，即主服务 embedding/碱基预测）。
- **`subpath` 路径段**选服务内功能（四个后端均为**纯子路径路由，无 body `mode`**）：
  - `/OGR/OGR-Prediction/{Prediction,SNV,RegisterGenome}`
  - `/OGR/OGR-Regulation/{Genome,Chromosome,Prediction,RegisterGenome}`
  - `/OGR/OGR-Introgression/{predict,RegisterGenome}`
  - `/OGR/OGR-Base/{Embedding,NTP}`
  - `/OGR/{sub}/health` 返回网关聚合健康状态。
- 路径段只用于路由，**不传给后端**；其余字段（`Genome`/`Chromosome`/`Start`/`sequence`/…）与
  请求头（`Authorization`/`X-API-Key`）原样透传，后端计费/错误语义不变。
- 网关为轻量反代（`dcs_gateway/app.py`），**不加载模型**：四个模型仍由各自后端进程加载。
- 网关探活：`GET/POST /OGR/health`（或 `/health`、`/api/aigress/openai/health`）返回四个后端的聚合状态。

> **参数命名说明**：四个后端请求体均使用大写驼峰参数（`Genome`/`Chromosome`/`Start`/`Format`/`SNV_position`/`SNV_base` 等），
> 不使用 `mode`/`end`/`max_points`/`output_format` 等旧键；`model` 字段可省略（网关与后端都不依赖它）。

---

## 2. 路由表（model_sub → 后端）

| `model_sub` | 后端服务 | 后端监听 | 可用 subpath |
|---|---|---|---|
| `OGR-Prediction` | OGR-Prediction（变异对比表达预测） | `127.0.0.1:8001` | `Prediction` / `SNV` / `RegisterGenome` |
| `OGR-Regulation` | OGR-Regulation（ATAC 条件表达预测） | `127.0.0.1:7001` | `Genome` / `Chromosome` / `Prediction` / `RegisterGenome` |
| `OGR-Introgression` | OGR-Introgression（粳/籼血缘渗入分析） | `127.0.0.1:5001` | `predict` / `RegisterGenome` |
| `OGR-Base`（**缺省**） | OGR-Base（embedding / 碱基预测） | `127.0.0.1:6001`（`.env` 可改） | `Embedding` / `NTP` |

> 端口以 `dcs_gateway/.env` 的 `RICE_*_PORT` 为准；默认 OGR-Prediction=8001、OGR-Regulation=7001、OGR-Introgression=5001、OGR-Base=6001。
> 网关自身监听：`GATEWAY_PORT`（默认 4000）；DCS 平台注入 `PORT` 时自动覆盖。

---

## 3. 公共约定

### 3.1 返回结构

四个服务返回结构一致：

```json
{
  "usage": { "prompt_tokens": 32768, "completion_tokens": 128 },
  "token_cnt": 32896,
  "status": 200,
  "message": "成功提示",
  "result": { ... }
}
```

### 3.2 计费口径

- `prompt_tokens = 输入 token 数 × DCS_PROMPT_TOKEN_MULTIPLIER`（默认 1）；
  `completion_tokens = 输出元素数 × DCS_COMPLETION_TOKEN_MULTIPLIER`（默认 1）。
  水稻基模为单碱基编码，**1 bp = 1 token**。
- **`token_cnt`**（顶层键，四个服务统一）= `prompt_tokens + completion_tokens`（按折算后两值求和；错误/未鉴权时 = 0）。

### 3.3 鉴权

- `.env` 配置 `DCS_API_KEY` 后，除 `/health` 外均需 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；
- key 未配置（空）则不鉴权。
- ⚠️ **未启用鉴权时不要带空 `Authorization` 头**：`-H "Authorization: Bearer ${api_key}"`（`api_key` 为空）
  会展开成非法的 `Bearer ` 头，网关透传时会被 httpx 拒绝（`LocalProtocolError: Illegal header value`，网关误报 502）。
  网关已对空值 header 做透传过滤，但建议调用方直接省略该头。
- 返回末尾自带换行：curl 查看结果不粘连提示符。

### 3.4 错误返回排查表

| HTTP | 场景 | 排查 |
|---|---|---|
| 400 | 参数缺漏/格式错 | 检查 `Start`、`SNV_position`/`SNV_base`、`Genome`、`ATAC-seq_Source`/`uploaded_atac` 等必填参数；参数错误时后端返回**英文使用说明**（`message` 字段） |
| 400 | 未知 `model_sub` | `model_sub` 只能是 `OGR-Prediction` / `OGR-Regulation` / `OGR-Introgression` / `OGR-Base` |
| 400 | 未知 subpath | 后端子路径只接受各服务白名单（见 §2 路由表），其余返回英文使用说明 |
| 401 | 鉴权失败 | 带 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；检查 `.env` 的 `DCS_API_KEY` |
| 500 | 推理异常 | 看 `result`/日志 traceback；模型路径、dtype/device 是否可用 |
| 502 | 后端不可达 / 超时 | `message: gateway: backend … unreachable`；检查后端进程/端口 |
| 503 | 模型未加载完成 | `/health` 查 `init_error`；初始化失败时适配层缓存错误并整体返回 503，重启容器 |

---

## 4. 通用健康检查（免鉴权）

```bash
# ── 本机 ──
curl -s http://127.0.0.1:4000/health | python3 -m json.tool
# → {"status":"ok|degraded","services":{"ogr_prediction":{...},"ogr_regulation":{...},"ogr_introgression":{...},"ogr_base":{...}}}

# ── DCS ──
curl -s https://www.dcs.cloud/api/aigress/openai/OGR/health | python3 -m json.tool
```

返回 `status`（`ok`/`degraded`）+ 四个后端的 `reachable`/`predictor_initialized`/`extractor_initialized`/`init_error`。
各后端 `/health` 的 `init_error` **必须为 `null`**（非 null 表示模型加载失败）。

---

## 4.5 自定义文件注册（传本地路径，不做文件传输）

> 调用方**预先**把文件放到与后端同一台机器上，API 只需传**文件路径**，
> 后端校验（存在性 + 建 `.fai` 索引 + 读染色体名）后注册为可用基因组，
> 预测请求用返回的 `genome` 字段引用。**无需 multipart / base64，无任何传输动作。**

支持 `OGR-Prediction` / `OGR-Regulation` / `OGR-Introgression` 三个服务
（`OGR-Base` 无自定义文件场景）。

```bash
# 本机
curl -X POST ${GW}/OGR-Prediction/RegisterGenome \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"fasta_path": "/data/custom/genome.fa"}' | python3 -m json.tool

# 可指定 genome_id（自定义 id，避免自动时间戳 id）与 GFF 注释
curl -X POST ${GW}/OGR-Regulation/RegisterGenome \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "fasta_path": "/data/custom/genome.fa",
    "genome_id": "my_genome",
    "gff_path": "/data/custom/genes.gff3"
  }' | python3 -m json.tool
```

- **请求体**（JSON，大小写敏感）：
  - `fasta_path`：**必填**，本地 `.fa` / `.fasta` / `.fna` 绝对路径；
  - `genome_id`：可选，自定义基因组 id（缺省自动生成 `custom_<timestamp>`）；
  - `gff_path`：可选（仅 OGR-Prediction / OGR-Regulation），附加注释 GFF。
- **返回**：`result` 含 `genome`（id）、`fasta_path`、`gff_path`、`chromosomes`、
  `chromosome_count`；`usage` 为 0（注册不计 token）。
- **后续预测**：把返回的 `genome` 作为 `Genome` 字段调用预测接口。
- **OGR-Regulation 的 ATAC bigWig 无需注册**：预测请求直接传 `uploaded_atac` 文件路径即可（已有能力）。
- 错误（文件不存在 / 格式错 / 缺少参数）：body `status: 400`，`message` 附原因。
- ⚠️ 注册表在**后端进程内存**中（进程重启后失效），且受各应用 `UPLOADED_GENOMES_TTL_HOURS`
  （默认 0.5h）空闲清理影响——**注册后尽快预测**；需要长期使用请用 `genome_id` 并在
  每次会话前重新注册。

---

## 4.6 文件系统探查接口（调试排查用，默认关闭）

> 部署机不在本地、无法 SSH 时，用这四个**只读**接口远程查看文件位置与目录结构
> （网关本地实现，**不转发后端、不碰模型进程**，token 计费为 0）。
> 生产环境请保持 `EXPLORE_ENABLED=false`；排查结束后立即关闭。

**开启方式**：`dcs_gateway/.env` 设 `EXPLORE_ENABLED=true` 并重启网关。

| 接口 | 等价命令 | 请求体（JSON） | 说明 |
|---|---|---|---|
| `POST .../OGR/fs/list` | `ls -lh /path` | `{path}` | 列目录，默认 200 条，超出截断 |
| `POST .../OGR/fs/stat` | `ll /path` | `{path}` | 单文件/目录详情（type/size/mode/mtime/软链 target） |
| `POST .../OGR/fs/find` | `find /path -name PATTERN` | `{path, pattern, max_depth, limit}` | 递归按文件名模式搜索，默认深度 4 / 200 条 |
| `POST .../OGR/fs/cat` | `head -c 64K /path` | `{path, tail, limit}` | 读文件内容，默认/上限 64KB，UTF-8 容错解码 |

- **路径**：绝对路径，`realpath` 解析；不限目录（排查阶段可访问部署机任意路径）。
- **鉴权**：与其它接口一致——配置了 `DCS_API_KEY` 后需 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`；
  未配置（空）则不鉴权。
- **返回**：统一结构 `{usage, token_cnt, status, message, result}`；`usage` 为 0（探查不计 token）。
- **错误**：404（路径不存在）/ 403（无权限或未启用）/ 400（参数错，如相对路径）/ 500（IO 异常）。

```bash
# 开启后的调用示例（本机网关；DCS 则换成 https://www.dcs.cloud/api/aigress/openai/OGR）
GW="http://127.0.0.1:4000/api/aigress/openai/OGR"
api_key=""   # 服务端启用鉴权时填入；为空时省略 -H "Authorization: ..." 头

# 1. 列目录（等价 ls -lh）
curl -X POST ${GW}/fs/list \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"path": "/mnt/rice/default/Workspace/yangdong/ai4research/rice_server"}' | python3 -m json.tool

# 2. 单文件/目录详情（等价 ll /path）
curl -X POST ${GW}/fs/stat \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"path": "/tmp/rice_dcs/ogr_base.log"}' | python3 -m json.tool

# 3. 递归找某类文件（等价 find /path -name "*.bw"，限深度 4）
curl -X POST ${GW}/fs/find \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"path": "/data", "pattern": "*.bw", "max_depth": 4}' | python3 -m json.tool

# 4. 看日志尾部（等价 tail -c 64K）
curl -X POST ${GW}/fs/cat \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"path": "/tmp/rice_dcs/ogr_prediction.log", "tail": true}' | python3 -m json.tool
```

---

## 5. 本机（localhost）测试

前提：`bash dcs_gateway/start_all.sh`（或手动启动四个后端 + 网关）已就绪，
网关在 `127.0.0.1:4000`，四个后端均在 `127.0.0.1`（OGR-Prediction 8001 / OGR-Regulation 7001 / OGR-Introgression 5001 / OGR-Base 6001）。
本机 `.env` 未配置 `DCS_API_KEY` 时无需 `Authorization` 头。

```bash
# ── 变量 ──
api_key=""                                        # 服务端启用鉴权时填入；为空（未启用鉴权）时省略 -H "Authorization: ..." 头
GW="http://127.0.0.1:4000/api/aigress/openai/OGR" # 本机网关
```

### 5.1 OGR-Prediction：参考表达预测（URL 路径，body 无 model_sub/mode）

```bash
curl -X POST ${GW}/OGR-Prediction/Prediction \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr09",
    "Start": 20716774,
    "Format": "Mean"
  }' | python3 -m json.tool
```

### 5.2 OGR-Prediction：SNV ref/mut 对比

```bash
curl -X POST ${GW}/OGR-Prediction/SNV \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr09",
    "Start": 20716774,
    "SNV_position": 20731844,
    "SNV_base": "T",
    "Format": "Mean"
  }' | python3 -m json.tool
```

### 5.3 OGR-Regulation：ATAC 条件表达预测

```bash
curl -X POST ${GW}/OGR-Regulation/Prediction \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "MH63",
    "Chromosome": "chr01",
    "Start": 1,
    "ATAC-seq_Source": "SAM2_MH63_1",
    "Format": "Mean"
  }' | python3 -m json.tool

# 辅助查询
curl -X POST ${GW}/OGR-Regulation/Genome      -H "Content-Type: application/json" | python3 -m json.tool
curl -X POST ${GW}/OGR-Regulation/Chromosome  -H "Content-Type: application/json" \
  -d '{"Genome":"MH63"}' | python3 -m json.tool
```

### 5.4 OGR-Introgression：粳/籼血缘渗入分析

```bash
curl -X POST ${GW}/OGR-Introgression/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "YF47",
    "Chromosome": "Chr01",
    "Start": 100001,
    "End": 356001
  }' | python3 -m json.tool
# 整条模式（窗口数受后端 MAX_NUMBER_256W 限制）
curl -X POST ${GW}/OGR-Introgression/predict \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{"Genome": "YF47", "Chromosome": "Chr01"}' | python3 -m json.tool
```

### 5.5 OGR-Base：embedding（model_sub 缺省即 OGR-Base）

```bash
curl -X POST ${GW}/OGR-Base/Embedding \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }' | python3 -m json.tool

# 碱基预测
curl -X POST ${GW}/OGR-Base/NTP \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }' | python3 -m json.tool
```

### 5.6 兼容：body 字段路由（旧式，完全等价）

```bash
curl -X POST ${GW} \
  -H "Authorization: Bearer ${api_key}" -H "Content-Type: application/json" \
  -d '{
    "model_sub": "OGR-Prediction",
    "Genome": "osa1_r7",
    "Chromosome": "chr09",
    "Start": 20716774,
    "Format": "Mean"
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

### 6.2 OGR-Prediction：参考表达预测（推荐：URL 路径带服务与功能）

```bash
curl -X POST ${dcs_host}${dcs_entry}/OGR-Prediction/Prediction \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr09",
    "Start": 20716774,
    "Format": "Mean"
  }' | python3 -m json.tool
```

### 6.3 OGR-Prediction：SNV 变异预测（双轨对比）

```bash
curl -X POST ${dcs_host}${dcs_entry}/OGR-Prediction/SNV \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "osa1_r7",
    "Chromosome": "chr09",
    "Start": 20716774,
    "SNV_position": 20731844,
    "SNV_base": "T",
    "Format": "Mean"
  }' | python3 -m json.tool
```

### 6.4 OGR-Regulation：ATAC 条件表达预测

```bash
curl -X POST ${dcs_host}${dcs_entry}/OGR-Regulation/Prediction \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "MH63",
    "Chromosome": "chr01",
    "Start": 1,
    "ATAC-seq_Source": "SAM2_MH63_1",
    "Format": "Mean"
  }' | python3 -m json.tool
```

### 6.5 OGR-Introgression：粳/籼血缘渗入分析

```bash
curl -X POST ${dcs_host}${dcs_entry}/OGR-Introgression/predict \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "Genome": "YF47",
    "Chromosome": "Chr01",
    "Start": 100001,
    "End": 356001
  }' | python3 -m json.tool
```

### 6.6 OGR-Base：embedding 提取（model_sub 缺省即 OGR-Base）

```bash
curl -X POST ${dcs_host}${dcs_entry}/OGR-Base/Embedding \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "sequence": "ACGTTGCATGCAACGTACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }' | python3 -m json.tool
```

### 6.7 OGR-Base：下游碱基预测

```bash
curl -X POST ${dcs_host}${dcs_entry}/OGR-Base/NTP \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 8
  }' | python3 -m json.tool
```

### 6.8 兼容：body 字段路由（旧式）

```bash
curl -X POST ${dcs_host}${dcs_entry} \
  -H "Authorization: Bearer ${dcs_api_key}" -H "Content-Type: application/json" \
  -d '{
    "model_sub": "OGR-Prediction",
    "Genome": "osa1_r7",
    "Chromosome": "chr09",
    "Start": 20716774,
    "Format": "Mean"
  }' | python3 -m json.tool
```

---

## 7. 各服务详情

| 服务 | 内容 | 文件 |
|---|---|---|
| OGR-Prediction | 参数表 / 返回结构 / 计费 / 排查 | [OGR_Prediction/API.md](../OGR_Prediction/API.md) |
| OGR-Regulation | 参数表 / 返回结构 / 计费 / 排查 | [OGR_Regulation/API.md](../OGR_Regulation/API.md) |
| OGR-Introgression | 参数表 / 返回结构 / 计费 / 排查 | [OGR_Introgression/API.md](../OGR_Introgression/API.md) |
| OGR-Base | 参数表 / 返回结构 / 计费 / 排查 | [OGR_Base/API.md](../OGR_Base/API.md) |

> 命名速查（一页版）见 [quick_start.md](../quick_start.md)；部署与维护见 [AGENTS.md](../AGENTS.md)、[dcs.md](../dcs.md)。