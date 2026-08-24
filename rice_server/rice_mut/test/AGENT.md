# AGENT.md — rice-mut 测试套件使用说明

本目录包含 rice-mut 项目的完整测试套件:**单元测试(离线)** + **DCS API curl 测试** + **网页端(Gradio)测试**,以及一键运行入口。

> 本文件供**开发者 / AI 编码代理 / 测试人员**快速了解:测什么、怎么跑、需要什么前置条件、常见问题怎么排查。

---

## 1. 文件清单

| 文件 | 类型 | 测试对象 | 是否需 GPU 服务 |
|---|---|---|---|
| `test_dcs_adapter.py` | Python 单元测试(20 项) | `backend/dcs_adapter.py` 的纯逻辑:请求解析、坐标换算、output_format、计费、响应形状 | ❌ 离线 |
| `test_dcs_api.sh` | Shell + curl(18 项;`--auth` 模式另加 6 项鉴权测试) | DCS 适配层 API 端点(OpenAI 风格) | ✅ 需 dcs_adapter 服务 |
| `test_frontend.sh` | Shell + curl(12 项) | 前端 Gradio 页面 + 后端健康 + 日志检查 | ✅ 需前后端服务 |
| `test_api.sh` | Shell + curl | 原后端 8001 端口端点(`/predict`、`/snv`、`/uploadFasta`、`/stat` 等) | ✅ 需后端服务 |
| `run_all_tests.sh` | 一键入口 | 按参数组合上面 3 套件 | 取决于参数 |

---

## 2. 前置条件

- **模型/权重路径**:由 `rice-mut/.env` 配置(`GENOME_osa1_r7_FASTA`、模型 safetensors、`index_stat.json` 等)
- **GPU**:推理需要 CUDA 设备(实测 2×A40,加载 ~30s,首轮推理 ~7s,缓存命中 ~0.73s)
- **Python 环境**:`.env` 中 `BACKEND_PYTHON_BIN=/root/miniconda3/envs/vllm/bin/python`(torch 2.8.0 + CUDA)

---

## 3. 服务启动方式

按测试目标启动对应服务(都在 `rice-mut/` 根目录下执行):

```bash
cd rice-mut

# 方式 A:仅 DCS 适配服务(跑 test_dcs_api.sh 用)
BACKEND_PORT=8001 /root/miniconda3/envs/vllm/bin/python backend/dcs_adapter.py

# 方式 B:原前后端(跑 test_frontend.sh / test_api.sh 用)
bash backend/run_backend.sh      # 后端 → http://127.0.0.1:8001,PID 文件 backend/logs/backend.pid
bash frontend/run_frontend.sh    # 前端 → http://127.0.0.1:8000,PID 文件 frontend/logs/frontend.pid
```

停止服务:

```bash
kill $(cat backend/logs/backend.pid) $(cat frontend/logs/frontend.pid) 2>/dev/null
pgrep -af "rice_mutation/main.py|frontend/app.py|dcs_adapter"   # 确认无残留
```

---

## 4. 运行测试

### 4.1 单元测试(离线,最快,无需服务)

```bash
python test/test_dcs_adapter.py      # 直接运行
python -m pytest test/test_dcs_adapter.py -q   # 或 pytest
```

### 4.2 DCS API 测试(需方式 A 服务)

```bash
bash test/test_dcs_api.sh                      # 默认模式(未配置 DCS_API_KEY)
DCS_API_KEY=your-key bash test/test_dcs_api.sh --auth   # 鉴权模式(额外 6 项)
```

覆盖:health / 参考预测 full / mean / downsample / SNV 双轨 / 5 种错误处理(缺 start、非法 output_format、非法 snv_base、非法 JSON、未知基因组);`--auth` 模式另覆盖:无 header → 401、错误 Bearer → 401、正确 Bearer → 200、`X-API-Key` → 200、health 免鉴权、SNV 无 header → 401。

### 4.3 网页端测试(需方式 B 服务)

```bash
bash test/test_frontend.sh
```

覆盖:首页 HTTP 可达性 / `/config` 组件配置 / 关键 UI 组件(genome、chromosome、start、end、predict、snv)/ 后端健康 / 前后端日志无致命错误。

### 4.4 一键入口

```bash
bash test/run_all_tests.sh               # 全部(需先自行启动对应服务)
bash test/run_all_tests.sh --unit        # 仅单测(离线)
bash test/run_all_tests.sh --api         # 仅 API
bash test/run_all_tests.sh --frontend    # 仅前端
bash test/run_all_tests.sh --no-models   # 跳过需 GPU 的服务测试
```

> 注意:`run_all_tests.sh` 只负责**执行**,不负责**启动服务**。全量模式前需按第 3 节先启动服务。

---

## 5. 环境变量覆盖

测试脚本会先加载 `rice-mut/.env`,再允许环境变量覆盖:

| 变量 | 默认值 | 用途 |
|---|---|---|
| `DCS_BASE_URL` | `$BACKEND_API_URL` → `http://127.0.0.1:8001` | DCS 适配层地址(API 测试) |
| `BACKEND_API_URL` | `http://127.0.0.1:8001` | 后端地址 |
| `FRONTEND_URL` | `http://127.0.0.1:8000` | 前端地址 |
| `BACKEND_URL` | `http://127.0.0.1:8001` | 后端地址(前端测试用) |
| `BACKEND_PYTHON_BIN` | `/root/miniconda3/envs/vllm/bin/python` | Python 解释器 |
| `DCS_API_KEY` | 空(不鉴权) | API Key;在 `.env` 配置后 POST 需 `Authorization: Bearer` / `X-API-Key` |

---

## 6. API 坐标与计费约定(测试断言依据)

- **坐标**:API/网页输入为 **1-based inclusive**;内部模型 0-based half-open。适配层 `start_0 = max(0, start_1 - 1)`;染色体别名归一化(`chr01` → `Chr1`)。
- **`window_len`** = 32768(`MAX_SEQ_LEN`)。
- **计费**:`prompt_tokens` = 窗口碱基数 × `DCS_PROMPT_TOKEN_MULTIPLIER`(默认 1);`completion_tokens` = 输出数组元素总数 × `DCS_COMPLETION_TOKEN_MULTIPLIER`(默认 1);SNV 为 ref+mut 双轨元素合计。
- **output_format**:`full`(逐碱基数组)/ `mean`(标量)/ `downsample`(默认 `max_points=1024`)。
- **错误码**:参数校验错误(缺 `start`、非法 `output_format`、非法 `snv_base`、非法 JSON)→ **400**;未知基因组 / 执行错误 → **500**。

---

## 7. curl API 调用示例(命令行用法)

启动 `dcs_adapter.py`(方式 A)后**不需要网页前端**,模型加载完(约 30s)health 返回 `ok` 即可直接通过 curl 调 API 推理。以下命令与 `test/test_dcs_api.sh` 中实测通过的 18 项完全一致。

> **鉴权头(部署到 DCS 平台时必读)**:在 `rice-mut/.env` 配置 `DCS_API_KEY` 后,
> 所有 POST 请求需带 `Authorization: Bearer <DCS_API_KEY>`(也兼容 `X-API-Key: <DCS_API_KEY>`);
> 本地未配置 `DCS_API_KEY`(留空)时**无需该 header**,`GET /health` 始终免鉴权。
> 平台部署时 URL 由网关转发为 `https://test.dcs.cloud/api/aigress/openai/<PATH>`,
> 请求头改为平台的 `Authorization: Bearer <YOUR_API_KEY>`。
>
> 鉴权失败返回 `HTTP 401`(而非 400/500):
> ```bash
> curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
>   -H "Content-Type: application/json" \
>   -d '{"genome":"osa1_r7","start":1000}'
> # {"usage":{"prompt_tokens":0,"completion_tokens":0},"status":401,"message":"无效或缺失的 API Key","result":null}
> ```

### 7.1 启动服务

```bash
cd /mnt/rice/default/Workspace/yangdong/rice_reg/rice-server/rice-mut
BACKEND_PORT=8001 /root/miniconda3/envs/vllm/bin/python backend/dcs_adapter.py
```

> 若提示 `address already in use`,说明 8001 已有服务,先清理再启动:`fuser -k 8001/tcp` 或用 `ss -tlnp | grep 8001` 找到 PID 后 `kill <pid>`。

### 7.2 健康检查

```bash
curl -s http://127.0.0.1:8001/api/aigress/openai/health
# {"status":"ok","predictor_initialized":true,"genomes":["osa1_r7"]}
```

`status` 为 `ok` 且 `predictor_initialized` 为 `true` 才可推理。

### 7.3 参考序列表达预测(核心推理)

```bash
# full 格式(逐碱基数组,约 286KB 响应,默认 output_format)
# 配置了 DCS_API_KEY 时加 -H "Authorization: Bearer <DCS_API_KEY>",本地留空可省略
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}'
```

```bash
# mean 格式(每轨道标量均值,响应小、最常用)
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}'
# 实测:mean = 0.599956
```

```bash
# downsample 格式(均匀降采样到 max_points 点,默认 1024)
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}'
```

### 7.4 SNV 变异预测(双轨 ref+mut 对比)

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut/snv \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DCS_API_KEY>" \
  -d '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T"}'
```

### 7.5 响应结构说明

```json
{
  "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
  "status": 200,
  "message": "参考序列表达预测成功",
  "result": {
    "model": "rice-mut",
    "genome": "osa1_r7",
    "chromosome": "Chr1",
    "position_1based": {"start": 20716774, "end": 20749541},
    "window_len": 32768,
    "output_format": "full",
    "values": {"total_RNA-seq_+": {"NIP_leaf": [0.008057, 0.007874, ...]}}
  }
}
```

- `usage`:`prompt_tokens` = 窗口碱基数,`completion_tokens` = 输出数组元素总数(SNV 为 ref+mut 双轨合计,实测 65536)。
- `values`:轨道名(`total_RNA-seq_+` 等)→ 组织名(`NIP_leaf` 等)→ 数值数组。

### 7.6 参数速查表

| 参数 | 必填 | 说明 |
|---|---|---|
| `start` | ✅ | 1-based 起始位点(缺失返回 400) |
| `end` | ❌ | 缺省自动取 32768 窗口 |
| `chromosome` | ❌ | 默认 `chr01`,`chr01`/`Chr1` 均可 |
| `genome` | ❌ | 默认唯一基因组 `osa1_r7` |
| `output_format` | ❌ | `full`(默认)/ `mean` / `downsample` |
| `max_points` | ❌ | downsample 目标点数,默认 1024 |
| `snv_index` + `snv_base` | SNV 时 ✅ | 1-based 变异位点 + 目标碱基 A/C/G/T/N |

错误语义:参数校验错误(缺 `start`、非法 `output_format`、非法 `snv_base`、非法 JSON)→ **400**;未知基因组 / 执行错误 → **500**。

### 7.7 实用技巧

```bash
# 1) 保存完整响应到文件(避免终端刷屏),再提取 JSON 字段
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}' \
  -o result.json
python3 -c "import json; d=json.load(open('result.json')); print(d['result']['values'])"

# 2) 用 jq 直接提取 mean 值(未安装 jq 时用上一条 python 方式)
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"output_format":"mean"}' \
  | jq .result.values

# 3) 只输出 HTTP 状态码,快速验证可用性
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8001/api/aigress/openai/health

# 4) 设置超时(首次推理约 7s,加载约 30s)
curl -s --max-time 60 -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"output_format":"mean"}'

# 5) 批量推理:把请求体写入 JSON 文件,循环调用
for i in 20716774 20731844 20749541; do
  curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"rice-mut\",\"chromosome\":\"chr01\",\"start\":$i,\"output_format\":\"mean\"}"
done
```

---

## 8. 常见问题与调试技巧

### 8.1 大 JSON 与 `json_field`
- full 格式 32768 数组响应约 286KB,`python3 -c "... $RESP"` 以 argv 传参会触发 **`Argument list too long`**(单参数 ~128KB 上限)。
- 因此 `json_field` 采用 **stdin 传 JSON**:`printf '%s' "$1" | python3 -c "d=json.load(sys.stdin)" "$2"`。
- **不要在会被 `$(...)` 命令替换捕获的函数内 echo 描述文本**,否则会污染 JSON 导致解析失败(描述文本放在调用处)。

### 8.2 测试输出过大
- API 测试的 full 响应会打印完整数组(写入临时文件或 `> /tmp/xxx.log`),用 `grep -aE "PASS|FAIL"` / `awk '{print NR": "substr($0,1,130)}'` 提取摘要即可。

### 8.3 服务未就绪
- dcs_adapter 启动后模型加载约 30s;health 端点返回 `{"status":"ok",...}` 才算就绪。
- 测试脚本自带 60s 就绪等待,超时会提示"请先启动服务"并以退出码 1 结束。

### 8.4 Flash Attention dtype 警告
- 启动日志中的 `Flash Attention 2 only supports torch.float16/bfloat16...` 为**良性警告**,不影响测试。

### 8.5 修改服务代码后
- **必须重启服务进程**再跑 API 测试(新代码需要新进程加载)。
- 单测(第 4.1 节)不受服务状态影响,可随时快速回归。

---

## 9. 测试通过标准(实测基线 2026-08-19)

| 套件 | 结果 |
|---|---|
| `test_dcs_adapter.py` | 25/25 PASS(含 `TestApiKey` 鉴权 5 项) |
| `test_dcs_api.sh` | 18/18 PASS(默认无 key);`--auth` 模式 24/24 PASS |
| `test_frontend.sh` | 12/12 PASS |
| 全量 | `All test suites passed.` |

代表性断言值:full `status=200`、usage `prompt=32768/completion=32768`、`window_len=32768`、`Chr1:20716774-20749541`;mean 标量 `0.599956`;downsample 长度 `1024`;SNV 双轨 completion `65536`;鉴权失败 `status=401`。
