# 水稻模型 OGR 及其应用模型的 API 部署

> ⚠️ **本文档为作者工作笔记(yangdong)**,与 `README.md`(总览)、`AGENTS.md`(部署指南)、
> `dcs.md`(DCS 规范)、`quick_start.md`(外部调用速查)、`dcs_gateway/API.md`(三合一完整 API)并存;
> 本文聚焦**模型能力与实测 demo**,外部调用细节以 `quick_start.md` / `dcs_gateway/API.md` 为准。

本文档面向**模型能力演示与后端/API 部署**。网页部署需要分前端（frontend）展示和后端（backend）计算，
以及前后端的联动 API。网页部署完全参考 Genos 的部署，前端使用 Python 包 Gradio，后端即模型推理，
打包为一个 docker（`ydgenomics/org_web:sanic`）。

## 0. 部署架构一览

```
浏览器 (Gradio 前端 :8000/:7000) ── HTTP ──▶ FastAPI 后端 (:8001/:7001 网页版)
                                              ├── rice_mut:   DNA → 多组学表达 (参考 vs SNV 对比)
                                              ├── rice_reg:   DNA + ATAC → RNA-seq 表达
                                              └── rice_OGR:   embedding 提取 / 碱基预测 (Sanic :8000)

DCS 网关 (dcs_gateway/, :9000) ── model_sub 路由 ──▶ 三个 dcs_adapter 进程
    外部唯一入口: POST /api/aigress/openai/OGR/{model_sub}[/{mode}]
```

- **端口规划**: rice_mut 8000/8001、rice_reg 7000/7001、rice_OGR Sanic 8000 / DCS 适配 **6001**、网关 9000。
- **统一网关模式(推荐)**: 三个服务各跑一个 `dcs_adapter.py` 进程，由 `dcs_gateway/` 单端口收口，
  外部只配置 `/api/aigress/openai/OGR` 一个 DCS 转发地址（详见 `dcs_gateway/API.md`）。
- **一键启动**: `cd dcs_gateway && bash start_all.sh`（先 `conda activate vllm`，再按 `.env` 拉起三后端 + 网关）。

## 1. 模型使用

水稻泛基因组（水稻+野生稻）DNA 模型 **OneGenomeRice** 基于 MoE 架构（Decoder-only，next token）
学到了生命的"语法"，one-hot 编码即一个碱基一个 token，具有长文本能力（1 M）。基于基模衍生其下游应用，
例如接入下游应用架构（U-net）的 RNA 表达预测（单碱基分辨率，32k 窗口）：
- **OGR-Mutation**：直接从序列到表达 sequence-to-expression（DNA → 多组学表达，支持单碱基突变对比）；
- **OGR-Reg**：多模态的表达预测 DNA+ATAC → RNA（ATAC 条件下的 RNA-seq 表达）。

### 1.1 OGR

水稻基模提供 **embedding 提取** 和 **下游碱基预测** 的服务。API 使用默认最大可提取 **32678 bp** 长的序列（`MODEL_1B_32k_MAX_LEN`）；`1B_8k` 最长 32768 bp（`MODEL_1B_8k_MAX_LEN`），超出截断。

**入口**（推荐走 DCS 网关统一入口 `/OGR/rice_ogr/...`；本地直连适配层 `:6001`）：

| 入口 | 方法 | 说明 |
|---|---|---|
| `POST /api/aigress/openai/OGR/rice_ogr/{dna_embedding,predict}` | POST | **DCS 网关 URL 路径路由（推荐）** |
| `POST /api/aigress/openai/OGR` + body `model_sub=rice_ogr` | POST | DCS 网关 body 路由（`model_sub` 缺省即 `rice_ogr`） |
| `POST /api/aigress/openai/rice_ogr`、`/rice_ogr` | POST | 适配层单入口，按 `mode` 分发 `dna_embedding` / `predict` |
| `POST /api/aigress/openai/rice_ogr/dna_embedding`、`…/predict` | POST | 适配层子路径（等价 `mode` 写法） |
| `GET /health`、`GET /models` | GET | 健康检查 / 已注册模型列表（免鉴权） |

**参数**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `sequence` | ✅ | DNA 序列（大/小写均可） |
| `model_name` | ✅ | 注册表模型名：`1B_8k` / `1B_32k`（`.env` 的 `MODEL_<NAME>_*`） |
| `mode` | 可选 | `dna_embedding`（默认）/ `predict`；未指定时按字段自动推断（带 `predict_length` → `predict`） |
| `pooling_method` | 可选 | `mean`（默认）/ `max` / `last` / `none`；`none` 输出逐位 `[1, L, 1024]` |
| `predict_length` | predict 用 | 预测下游碱基数（1–1000，默认 10） |

> `model` 为**服务名**（`rice_ogr`，与入口 path 末段一致）；实际**模型名**放在 `model_name`
> （如 `1B_8k` / `1B_32k`）。向后兼容：`model` 不等于 `rice_ogr` 时仍视为模型名。

**返回结构**（三个服务统一）：`{"usage": {"prompt_tokens", "completion_tokens"}, "status", "message", "result"}`。
计费口径：单碱基编码 **1 bp = 1 token**；`prompt_tokens = 输入碱基数 × 系数`，`completion_tokens = 输出元素数 × 系数`（系数见 `.env`，默认 1）。

```bash
# mode=dna_embedding(默认):提取整条序列的 1024 维向量
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/dna_embedding \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "OGR",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# mode=predict(下游碱基预测):基于前 16 bp 续写后续 10 个碱基
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "model_name": "1B_8k",
    "mode": "predict",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'
```

<details> <summary> demo </summary>

```shell
[01:46:31] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/predict   -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"   -H "Content-Type: application/json"   -d '{
    "model": "OGR",
    "model_name": "1B_8k",
    "mode": "predict",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'
{"usage":{"prompt_tokens":16,"completion_tokens":10},"status":200,"message":"Base prediction succeeded","result":{"model":"OGR","model_name":"1B_8k","mode":"predict","original_sequence":"ACGTTGCATGCAACGT","predicted_sequence":"ACGTTGCATGCAACGTACAACAAAAA","predicted_bases":"ACAACAAAAA","predict_length":10,"total_length":26,"elapsed_seconds":0.317}}
```

> 说明：`prompt_tokens=16`（输入 16 bp = 16 token）、`completion_tokens=10`（预测 10 bp）；
> `predicted_bases` 即续写的下游碱基（可继续作为下一轮输入做自回归续写）。

</details>


### 1.2 OGR-Mutation

水稻 RNA 表达预测模型（在 **4 个品种**的配对基因组 + 叶转录组数据上训练），具有**单碱基分辨率**的
RNA 表达预测能力。每次预测输入 32k 窗口（不足则 padding 到 32k，超过则被修剪到 32k，
窗口与输入区间中心对齐），输出对应 32k 窗口的表达峰图；支持**单碱基突变（SNV）**对比。

**入口**（推荐 `/OGR/rice_mut/{predict,snv}`；本地直连适配层 `:8001`）：统一入口
`POST /api/aigress/openai/OGR/rice_mut/...`（或 body `model_sub=rice_mut` + `mode`）。

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | 参考序列表达预测 | `chromosome` + `start` |
| `snv` | 单突变位点的窗口轨迹对比 | `chromosome` + `start` + `snv_index` + `snv_base` |

> `mode` 可省略：body 带 `snv_index` 自动进入 `snv`；否则有 `start` 即 `predict`。

**参数**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `genome` | 可选 | 参考基因组，默认唯一基因组 `osa1_r7` |
| `chromosome` | ✅ | 统一 `chr01`–`chr12` 命名（后端自动通配 `Chr1`/`1` 等） |
| `start` | ✅ | 窗口起点，**1-based inclusive**（默认窗口长 32768，中心对齐） |
| `end` | 可选 | 窗口终点（缺省按窗口规则补齐） |
| `snv_index` | snv 用 | 变异位点**绝对基因组坐标**（1-based，须落在窗口内） |
| `snv_base` | snv 用 | 突变碱基：`A` / `C` / `G` / `T` / `N` |
| `output_format` | 可选 | `full`（默认，逐碱基数组）/ `mean`（每轨道标量均值）/ `downsample`（降采样） |

```shell
# mode=predict 参考序列表达预测(mean 格式:每轨道一个标量均值)
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'

# mode=snv：chr09:20731844 位点 C→T 的 ref/mut 轨迹对比
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/snv \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OGR",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }'
```

<details> <summary> demo </summary>

```shell
[01:48:07] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/predict   -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"   -H "Content-Type: application/json"   -d '{
    "model": "OGR",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'
{"usage":{"prompt_tokens":32768,"completion_tokens":32768},"status":200,"message":"Reference expression prediction succeeded","result":{"model":"rice_mut","genome":"osa1_r7","chromosome":"Chr9","position_1based":{"start":20716774,"end":20749541},"window_len":32768,"output_format":"mean","values":{"RNA-seq":{"Leaf":0.430133}}}}

[01:49:17] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/snv   -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"   -H "Content-Type: application/json"   -d '{
    "model": "OGR",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }'
{"usage":{"prompt_tokens":32768,"completion_tokens":65536},"status":200,"message":"SNV prediction succeeded (ref C → T)","result":{"model":"rice_mut","genome":"osa1_r7","chromosome":"Chr9","position_1based":{"start":20716774,"end":20749541},"window_len":32768,"snv_index_1based":20731844,"ref_base":"C","snv_base":"T","output_format":"mean","ref_values":{"RNA-seq":{"Leaf":0.430133}},"mut_values":{"RNA-seq":{"Leaf":0.43156}}}}
```

> 说明：`ref_values` = 参考序列表达（"RNA-seq"×"Leaf"，即 index_stat 的 assay×biosample 组合，
> 网页端/IGV 以 `DISPLAY_HEADS`/`DISPLAY_BIOSAMPLES` 覆盖显示名）；
> `mut_values` = 单碱基突变后表达；`output_format=mean` 时各轨为标量均值，`full` 时为 32768 长度逐碱基数组。
> 本 demo 中 ref 0.430133 → mut 0.43156（C→T 在 chr09:20731844 的效应）。

</details>


### 1.3 OGR-Reg

水稻多模态 RNA 表达预测模型（在 **2 个品种**的配对基因组、茎尖分生组织的 ATAC 和 RNA 数据上训练），
具有**单碱基分辨率**的 RNA 表达预测能力。每次预测输入 32k 窗口的 **DNA 序列 + ATAC 信号**
（不足则 padding 到 32k，超过则被修剪到 32k），输出对应 32k 窗口的 RNA 表达峰图（**区分正负链**）。

**入口**（推荐 `/OGR/rice_reg/predict`；本地直连适配层 `:7001`）：统一入口
`POST /api/aigress/openai/OGR/rice_reg/...`（或 body `model_sub=rice_reg` + `mode`）。

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | ATAC 条件下的 RNA-seq 表达预测 | `genome` + `chromosome` + `start` + `atac_source`/`uploaded_atac`（二选一） |
| `genomes` / `chromosomes` | 查询可用基因组 / 染色体列表 | `chromosomes` 需 `genome` |

**参数**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `genome` | ✅ | 参考基因组：`MH63RS3` / `NIP` |
| `chromosome` | ✅ | 统一 `chr01`–`chr12` 命名 |
| `start` | ✅ | 窗口起点，**1-based inclusive**（窗口固定 `TARGET_LEN=32678`） |
| `end` | 可选 | 窗口终点（缺省按窗口规则补齐） |
| `atac_source` | 二选一 | 内置 ATAC 源 ID：`SAM2_MH63_1`（对应 MH63）/ `SAM2_NIP_1`（对应 NIP） |
| `uploaded_atac` | 二选一 | 服务器上已上传的 ATAC bigWig 文件路径（与 `atac_source` 二选一，优先） |
| `output_format` | 可选 | `full`（默认）/ `mean` / `downsample` |

> **ATAC 源与基因组的映射**：`atac_source` 需与 `genome` 匹配 —— `MH63RS3` ↔ `SAM2_MH63_1`、
> `NIP` ↔ `SAM2_NIP_1`（`.env` 的 `ATAC_GENOME_MAP_<GENOME>` 控制）。

```shell
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{
    "model": "OGR",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'
```

<details> <summary> demo </summary>

```shell
[01:50:19] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -s -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/predict   -H "Content-Type: application/json"   -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx"   -d '{
    "model": "OGR",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'
{"usage":{"prompt_tokens":32678,"completion_tokens":65356},"status":200,"message":"ATAC→RNA-seq expression prediction succeeded","result":{"model":"rice_reg","genome":"MH63RS3","chromosome":"chr1","position_1based":{"start":20716775,"end":20749452},"window_len":32678,"atac_source":"SAM2_MH63_1","atac_path":"/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_reg/ATAC/ATAC_SAM2_MH63_1.MH63RS2.q30.bin5.CPM.bw","output_format":"mean","values":{"RNA-seq_+":0.000255,"RNA-seq_-":0.11602}}}
```

> 说明：`values` 分两通道：`RNA-seq_+`（正链）/ `RNA-seq_-`（负链）；
> `completion_tokens=65356` = 2 通道 × 32678 窗口长度（输出元素数）；
> `atac_path` 回显实际使用的 ATAC bigWig 绝对路径（便于核对内置源是否命中）。

</details>


## 2. 模型测试

> 三服务的**完整测试代码**（本机 + DCS 两套 curl）见 [dcs_gateway/API.md](dcs_gateway/API.md) §5/§6；
> 各服务 API 细节见 `rice_mut/API.md`、`rice_reg/API.md`、`rice_OGR/API.md`。

### 2.1 一键启动（本地联调）

```bash
cd dcs_gateway && bash start_all.sh    # 先 conda activate vllm → 按 .env 拉起三后端 + 网关
curl -s http://127.0.0.1:9000/health | python3 -m json.tool   # 聚合健康检查(三服务全绿)
```

### 2.2 本机网关冒烟测试（URL 路径路由）

```bash
GW="http://127.0.0.1:9000/api/aigress/openai/OGR"

# rice_mut —— 参考表达预测
curl -X POST ${GW}/rice_mut/predict -H "Content-Type: application/json" \
  -d '{"genome":"osa1_r7","chromosome":"chr09","start":20716774,"output_format":"mean"}'

# rice_mut —— SNV 对比
curl -X POST ${GW}/rice_mut/snv -H "Content-Type: application/json" \
  -d '{"genome":"osa1_r7","chromosome":"chr09","start":20716774,"snv_index":20731844,"snv_base":"T","output_format":"mean"}'

# rice_reg —— ATAC 条件预测
curl -X POST ${GW}/rice_reg/predict -H "Content-Type: application/json" \
  -d '{"genome":"MH63RS3","chromosome":"chr01","start":20716774,"atac_source":"SAM2_MH63_1","output_format":"mean"}'

# rice_ogr —— embedding
curl -X POST ${GW}/rice_ogr/dna_embedding -H "Content-Type: application/json" \
  -d '{"model_name":"1B_8k","sequence":"ACGTTGCATGCAACGTACGTTGCATGCAACGT","pooling_method":"mean"}'

# rice_ogr —— 碱基预测
curl -X POST ${GW}/rice_ogr/predict -H "Content-Type: application/json" \
  -d '{"model_name":"1B_8k","sequence":"ACGTTGCATGCAACGT","predict_length":8}'
```

### 2.3 DCS 平台联调

三个服务同一入口 `https://www.dcs.cloud/api/aigress/openai/OGR`，仅路径段不同
（`/rice_mut/{predict,snv}`、`/rice_reg/predict`、`/rice_ogr/{dna_embedding,predict}`），
请求头带 `Authorization: Bearer ${dcs_api_key}`。示例见 §1.1–§1.3 各节 curl。

### 2.4 常见错误速查

| HTTP | 含义 | 处理 |
|---|---|---|
| 400 | 参数缺漏/格式错 | 检查 `mode`、必填字段、`chr01-12` 命名、1-based 坐标；`model_sub` 只能是 `rice_mut`/`rice_reg`/`rice_ogr` |
| 401 | 鉴权失败 | 确认请求头 `Authorization: Bearer <key>` / `X-API-Key: <key>` |
| 5xx | 模型未就绪/推理异常 | `/health` 查 `init_error`；看后端日志 `/tmp/rice_dcs/*.log` |
| 502 | 后端不可达 | 检查对应后端进程/端口是否存活 |

## 3. 模型部署心得

- **地址不要包含 `-`**（DCS 平台入口/服务命名规范）。
- 使用 **tosutil** 将模型/基因组文件上传到云平台对象存储。
- **基因组文件 `fa` 和 `fa.fai`：一定要先传 `fa`，再传 `fai`**（索引依赖主文件）。
- **一键部署（推荐）**：`dcs_gateway/start_all.sh` 会自动 `conda activate vllm` 并用 `.env`
  指定的解释器拉起三后端 + 网关（日志 `/tmp/rice_dcs/*.log`）；跨机部署时改 `dcs_gateway/.env`
  的 `RICE_*_HOST`/`RICE_*_PORT` 与 `RICE_*_PYTHON`（详见 `dcs_gateway/README.md` §3）。
- **容器内模型路径**：DCS 容器中通过环境变量注入模型路径覆盖，如
  `MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf`（rice_OGR 注册表）；
  rice_mut/rice_reg 同理覆盖 checkpoint 路径。
- **端口提醒**：rice_OGR 的 DCS 适配层用 **6001**（与 rice_mut 的 8001 错开），
  同机部署**不要**用 rice_OGR/.env 里默认的 `BACKEND_PORT=8001`（会与 rice_mut 冲突）。
- **多卡**：rice_OGR 支持 `--device cuda:0,cuda:1 --device_map auto`；rice_mut/rice_reg 用 `CUDA_VISIBLE_DEVICES` 控制。
- **安全**：rice_mut/rice_reg 网页后端将根目录挂载为 `/static-files` 静态服务（IGV 加载 bigWig 用），
  暴露公网前必须改白名单目录或加鉴权（详见 `AGENTS.md` §4）。
- **缓存**：预测结果写为 bigWig（`cache/predictions`），由后台线程按 TTL 自动清理；
  Docker 镜像不需要包含 `cache/`、`logs/`、`*.pid` 等运行时产物。