# Rice-Introgression Server（粳/籼渗入分析 Web 工具）

把 `20.introgression_analysis` 的离线渗入分析流程移植为可交互的 Web 应用：
前后端分离（.env 配置 + FastAPI 后端 + Gradio 前端），推理逻辑与离线脚本
**逐位对齐**，可视化用 Plotly（离线 JS，交互式）。

## 目录结构

```
OGR_Introgression/
├── .env                  # 全部配置（端口/模型/参数/基因组路径）
├── .env.example          # 配置模板
├── requirements.txt      # 后端依赖清单（vllm 环境）
├── start.sh              # 统一启动：all/backend/stop/status/restart
├── backend/
│   ├── rice_introgression/
│   │   ├── main.py            # uvicorn 入口（加载 .env）
│   │   ├── api.py             # FastAPI 路由：网页路由 + DCS 子路径（同一进程）
│   │   ├── model.py           # 模型结构（与离线 models/model.py 一致）
│   │   ├── predictor.py       # 单例预测器 + 8k 片段切分 + GPU 推理
│   │   ├── analysis.py        # 窗口网格 / top-k 聚合 / 双阈值分组 / 区域融合
│   │   ├── prediction_service.py  # 核心编排（单窗/整染色体）+ CACHE_MODE
│   │   ├── genome_service.py  # 基因组注册（managed）、染色体列表、长度
│   │   ├── progress.py        # 分析进度（可供 /progress 查询）
│   │   └── cache_service.py   # 内容寻址 LRU 缓存（CACHE_MODE=false 不落盘）
│   ├── run_backend.sh / stop_backend.sh
│   ├── scripts/           # 自测脚本（如 selftest_max_windows.py）
│   └── logs/
├── frontend/
│   ├── config.py          # 前端配置（读 .env，扫描 GENOME_*_FASTA）
│   ├── app.py             # Gradio UI（含 API Key 输入框）+ /backend/* 反代 + Plotly iframe
│   ├── run_frontend.sh / stop_frontend.sh
│   ├── static/plotly.min.js   # 离线 Plotly.js（前端 iframe 用）
│   └── logs/
├── tools/startup_self_check.sh
└── cache/                 # 运行时缓存（上传 FASTA 等）
```

## 快速开始

```bash
# 1. 配置（复制模板后按实际修改）
cp .env.example .env
#   关键项：
#   BASE_MODEL_PATH / CHECKPOINT_PATH —— 基座与 LoRA+头 权重
#   GENOME_YF47_FASTA                —— 目标基因组 FASTA（支持 .gz）
#   BACKEND_PYTHON_BIN / FRONTEND_PYTHON_BIN —— 前后端各自 Python 环境
#   CACHE_MODE                       —— true=全量缓存；false=轻量模式（每次重跑 GPU，
#                                        上传按本地 path 懒注册零拷贝）

# 2. 自检
bash tools/startup_self_check.sh

# 3. 启动（统一命令：后端+前端 一套）
bash start.sh              # 启动后端(API :5001) + 前端(网页 :5000)
#    可选子命令：backend（只起后端）/ stop / status / restart
```

- **前端**（Gradio UI）：浏览器访问 `http://<服务器IP>:5000`
- **后端**（FastAPI）：`http://<服务器IP>:5001`（`/health` 健康检查）
- 端口由 `.env` 的 `FRONTEND_PORT` / `BACKEND_PORT` 控制（默认 5000 / 5001）；`BACKEND_API_URL` 为前端进程内部调用后端地址。

停止：`bash start.sh stop`

## 网页端与纯 API 调用（同一进程、同一子路径、同一返回）

`backend/rice_introgression/main.py` 同时承载网页路由与 DCS 风格子路径，网页端与
纯 API（curl）调用者**共用同一进程、同一推理核心、同一 DCS 子路径、同一返回结构**：

- **网页端（Gradio :5000）与纯 API / DCS 网关都调用**
  `POST :5001/api/aigress/openai/OGR-Introgression/Prediction`
  （camelCase 键、1-based 坐标；兼容旧短路径 `…/predict`）。
- 统一返回信封：`{model_name, usage, token_cnt, status, message, result}`；
  `result` 同时含窗口数据（`windows`/`segments`/`regions`）与全基因组展示骨架
  （`chromosomes` / `chromosome_lengths` / `elapsed_seconds` / `cached`）——
  **网页端 Plotly 渲染完全来自该返回值**（前端就是“填好参数的 curl”）。
- 网页端辅助查询仍走内部端点：基因组/染色体列表 `GET /genomes…`、进度 `GET /progress`；
  自定义基因组注册走 DCS `RegisterGenome` 子路径。

```bash
# 启动统一后端（纯 API 场景，不拉网页）
bash start.sh backend
```

- 调用文档：`API.md`（请求/返回结构、DCS 网关入口、本机 curl 示例）
- 配置：`.env` 的 `DCS_API_KEY`（鉴权，可选）、`DCS_*_MULTIPLIER`（计费倍数）
- 接入网关：`dcs_gateway` 的 `BACKENDS` 注册 `ogr_introgression`（`RICE_INTRO_PORT` 默认 5001）

## 使用方式（前端）

1. **基因组**下拉：内置基因组（来自 `.env` 的 `GENOME_{名称}_FASTA`），或选「自定义」后填**后端机器上的 FASTA 绝对路径**（`CACHE_MODE=false` 为路径文本框、零拷贝注册，非文件上传）。
2. **染色体**下拉：加载自 FASTA 实际染色体名（如 `GWHBKAR00000001`，不归一化）。
3. **Start / End**：
   - End 为空 → 恒 1 个窗口 = `[Start, Start+256k)`
   - Start + End → 重叠滑窗（起点 = Start, 步长 64k），起点 > End 停止，
     最后一个起点 ≤ End 的窗口保留（尾部可超出 End）；窗口数受 `MAX_NUMBER_256W` 限制
   - Start 必填（不支持整条染色体模式）
   - 第 1 个窗口超染色体末端 → 返回空结果提示（警示）
4. 点 **Predict**，等待推理完成，下方 iframe 渲染 Plotly 图（数据来自 `Prediction` 子路径返回的 `result`）：
   - **全基因组三 track 图**：所有染色体车道（Chr12 顶 → Chr01 底），每条染色体
     3 个紧密 track（**Jap 蓝 / Ind 橘 / uncertain 绿**，块内无行距），未推理区
     gray 底色；x 轴 = 全基因组范围（Mb），hover 显示类别、区间、`n_windows`、
     top-k 均值。

## Web 查询语义（start 锚定，2026-11 重构）

Web 窗口**不再吸附固定网格**，而是严格按用户 Start 锚定；**Start 必填**：
- **窗口模式**（Start 必填）：
  - 只填 start → 恒 1 个窗口 `[Start, Start+256k)`
  - start+end → 重叠滑窗起点 = start, start+64k, ...（step=WINDOW_STEP）；
    起点 > end 停止；最后一个起点 ≤ end 的窗口即使尾部 > end 也保留输出；
    窗口数 > `MAX_NUMBER_256W` 取前 N 个。
  - 第 1 个窗口尾部 > 染色体末端 → 返回空（200 + message）；后续窗口尾部
    > 染色体末端 → 丢弃该窗口及之后（break）。
- **整条模式已废弃**（Start 缺失 → 400 报错）。

因此 **Web 结果与离线网格分析不再逐位一致**（起点由用户 start 决定，任意非网格坐标）。
返回坐标统一 **1-based inclusive**（`win_start` = 请求 Start 本身）。

### 参数默认值（可在 .env 覆盖）

| 参数 | 值 | 说明 |
|---|---|---|
| SEGMENT_SIZE | 8000 | 片段长度（bp） |
| WINDOW_SIZE | 256000 | 窗口长度（bp） |
| WINDOW_STEP | 64000 | 窗口步长（bp） |
| TOP_K | 10 | 窗口内 top-k 片段均值聚合 |
| THRESHOLD_JAP | 0.55519 | 粳阈值 |
| THRESHOLD_IND | 0.53473 | 籼阈值 |

分组规则（与离线一致）：
- `Jap`: `prob_jap ≥ thr_jap` 且 `prob_ind < thr_ind`
- `Ind`: `prob_jap < thr_jap` 且 `prob_ind ≥ thr_ind`
- 否则 `uncertain`

## API（后端 :5001，网页与纯 API 同一进程、同一子路径）

**推荐（网页端与 curl 都走这里）**：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/aigress/openai/OGR-Introgression/Prediction`（兼容 `…/predict`） | 渗入分析（camelCase + 1-based，`Start` 必填） |
| POST | `/api/aigress/openai/OGR-Introgression/Genome` | 基因组列表 |
| POST | `/api/aigress/openai/OGR-Introgression/Chromosome` | 染色体列表 |
| POST | `/api/aigress/openai/OGR-Introgression/RegisterGenome` | 按本机路径注册自定义 FASTA |
| GET/POST | `/health`、`…/OGR-Introgression/health` | 健康检查（免鉴权） |

**网页端辅助内部端点**（不用于纯 API 调用）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/genomes` | 基因组列表 |
| GET | `/genomes/{id}/chromosomes` | 染色体列表（FASTA 实际命名） |
| GET | `/genomes/{id}/chromosomes/{chrom}/length` | 染色体长度 |
| GET | `/progress` | 推理进度（前端轮询） |
| POST | `/uploadFasta` | 上传自定义 FASTA（multipart `file`；CACHE_MODE=false 时可带 `path` 表单字段零拷贝懒注册） |
| POST | `/analyze` | 网页内部：渗入分析（snake_case 方言；已被前端弃用，前端现在直连 DCS `Prediction`） |
| POST | `/analyze-genome` | 网页内部：全基因组逐染色体分析 |

DCS 子路径请求示例（网页端与 curl 同方言、同返回）：

```bash
curl -s -X POST http://127.0.0.1:5001/api/aigress/openai/OGR-Introgression/Prediction \
  -H 'Content-Type: application/json' \
  -d '{"Genome": "YF47", "Chromosome": "GWHBKAR00000001", "Start": 100001, "End": 356001}'
```

响应 `result` 含：
- `segments`：8k 片段级 `{start, end, prob_jap, prob_ind}`（1-based）
- `windows`：窗口级 `{win_start, win_end, center, n_segments, topk_mean_jap, topk_mean_ind, group}`（1-based）
- `regions`：`{Ind, Jap, uncertain}` 连续区域（带 `chromosome` 字段）
- `chromosomes` / `chromosome_lengths`：全基因组 12 染色体车道骨架（前端 Plotly 渲染数据）
- `message` / `cached` / `elapsed_seconds` / `params` / `threshold_rule`

## 环境

| 组件 | 环境 | 依赖 |
|---|---|---|
| 后端 | `vllm` (py3.12) | torch, transformers, peft, safetensors, pyfaidx, fastapi, uvicorn, pandas, numpy |
| 前端 | `rice_reg` (py3.10) | gradio, httpx, fastapi（**无模型、无 plotly**，前端 iframe 用静态 plotly.min.js） |

前后端环境独立：后端负责模型推理，前端只做 UI + HTTP 转发 + Plotly 渲染。