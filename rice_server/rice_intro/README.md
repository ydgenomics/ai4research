# Rice-Introgression Server（粳/籼渗入分析 Web 工具）

把 `20.introgression_analysis` 的离线渗入分析流程移植为可交互的 Web 应用：
前后端分离（.env 配置 + FastAPI 后端 + Gradio 前端），推理逻辑与离线脚本
**逐位对齐**，可视化用 Plotly（离线 JS，交互式）。

## 目录结构

```
rice_intro/
├── .env                  # 全部配置（端口/模型/参数/基因组路径）
├── .env.example          # 配置模板
├── requirements.txt      # 后端依赖清单（vllm 环境）
├── backend/
│   ├── rice_introgression/
│   │   ├── main.py            # uvicorn 入口（加载 .env）
│   │   ├── api.py             # FastAPI 路由（/analyze 等）
│   │   ├── model.py           # 模型结构（与离线 models/model.py 一致）
│   │   ├── predictor.py       # 单例预测器 + 8k 片段切分 + GPU 推理
│   │   ├── analysis.py        # 窗口网格 / top-k 聚合 / 双阈值分组 / 区域融合
│   │   ├── prediction_service.py  # 核心编排（单窗/整染色体）
│   │   ├── genome_service.py  # 基因组注册、染色体列表、长度
│   │   └── cache_service.py   # 内容寻址 LRU 缓存
│   ├── dcs_adapter.py          # ★ DCS 适配层（单入口 + mode 分发，独立进程）
│   ├── run_backend.sh / stop_backend.sh
│   ├── run_dcs_adapter.sh / stop_dcs_adapter.sh
│   └── logs/
├── frontend/
│   ├── config.py          # 前端配置（读 .env，扫描 GENOME_*_FASTA）
│   ├── app.py             # Gradio UI + /backend/* 反向代理 + Plotly iframe
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

# 2. 自检
bash tools/startup_self_check.sh

# 3. 启动（后端先，前端后）
bash backend/run_backend.sh && bash frontend/run_frontend.sh
```

- **前端**（Gradio UI）：浏览器访问 `http://<服务器IP>:5000`
- **后端**（FastAPI）：`http://<服务器IP>:5001`（`/health` 健康检查）
- 端口由 `.env` 的 `FRONTEND_PORT` / `BACKEND_PORT` 控制（默认 5000 / 5001）；`BACKEND_API_URL` 为前端进程内部调用后端地址。

停止：`bash backend/stop_backend.sh && bash frontend/stop_frontend.sh`

## DCS 适配层（对接 DCS 平台，独立进程）

`backend/dcs_adapter.py` 提供 OpenAI 风格 HTTP API（`POST /api/aigress/openai/rice_intro`），由 DCS 统一网关转发。**独立于网页版后端进程**，DCS 部署约定监听 **5001**（平台注入 `PORT` 优先，回退 `BACKEND_PORT`；与网页版是不同机器）。

```bash
# 启动 DCS 适配层（监听 5001）
bash backend/run_dcs_adapter.sh
# 停止
bash backend/stop_dcs_adapter.sh
```

- 调用文档：`API.md`（请求/返回结构、DCS 网关入口、本机 curl 示例）
- 配置：`.env` 的 `DCS_API_KEY`（鉴权，可选）、`DCS_*_MULTIPLIER`（计费倍数）
- 接入网关：`dcs_gateway` 的 `BACKENDS` 注册 `rice_intro`（`RICE_INTRO_PORT` 默认 5001）

## 使用方式（前端）

1. **基因组**下拉：内置基因组（来自 `.env` 的 `GENOME_{名称}_FASTA`），或上传自定义 FASTA。
2. **染色体**下拉：加载自 FASTA 实际染色体名（如 `GWHBKAR00000001`，不归一化）。
3. **Start / End**：
   - End 为空 → 窗口 = `[Start, Start+256k]`
   - Start/End 都空 → 整条染色体（所有滑动窗口）推理
4. 点 **Predict**，等待推理完成，下方 iframe 渲染 Plotly 图：
   - **上图**：染色体全景轨道 —— 每条窗口的 group 色带、Ind/Jap 连续区域、查询范围红色虚线框。
   - **下图**：匹配窗口内 8k 片段级 `P(Jap)` / `P(Ind)` 双曲线 + 分类底色 + 双阈值虚线。
   - 图例单击隐藏 / 双击隔离 —— 可只显示 Ind、只显示 Jap 或全部（**零自定义按钮**）。

## Web 与离线结果一致性保证

Web 并非对用户任意区间"重新切窗"，而是在**固定滑动窗口网格**上计算：
- 离线 `4.run_analysis.aggregate_windows` 的窗口起点 = 片段网格 `starts[idx]`
  （trim 后 0 起点、8000 步长；`idx` 网格 0, 8, 16, ..., n-32）。
- Web 实现同样的网格：`standard_window_starts(params, chrom_len, trim_start, n_segments)`。
- 用户输入通过**最大覆盖度匹配**（`match_window_to_grid`）落到最近的网格窗口，
  对该窗口内相邻 32 个 8k 片段做 top-k 聚合 —— 与离线同一窗口完全一致。
- 因此任意查询返回的都是"离线分析中存在的一个窗口的精确结果"。

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

## API（后端 :5001）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 + 模型状态 + 基因组列表 |
| GET | `/genomes` | 基因组列表 |
| GET | `/genomes/{id}/chromosomes` | 染色体列表（FASTA 实际命名） |
| GET | `/genomes/{id}/chromosomes/{chrom}/length` | 染色体长度 |
| POST | `/uploadFasta` | 上传自定义 FASTA（multipart `file`） |
| POST | `/analyze` | 渗入分析（JSON：`genome, chromosome, start?, end?`） |

`/analyze` 请求示例：

```bash
curl -X POST http://127.0.0.1:5001/analyze \
  -H 'Content-Type: application/json' \
  -d '{"genome": "YF47", "chromosome": "GWHBKAR00000001", "start": 100000, "end": 356000}'
```

响应含：
- `segments`：8k 片段级 `{start, end, prob_jap, prob_ind}`
- `windows`：窗口级 `{win_start, win_end, center, n_segments, topk_mean_jap, topk_mean_ind, group}`
- `regions`：`{Ind, Jap, uncertain}` 连续区域（overlap 融合）

## 环境

| 组件 | 环境 | 依赖 |
|---|---|---|
| 后端 | `vllm` (py3.12) | torch, transformers, peft, safetensors, pyfaidx, fastapi, uvicorn, pandas, numpy |
| 前端 | `rice_reg` (py3.10) | gradio, httpx, fastapi（**无模型、无 plotly**，前端 iframe 用静态 plotly.min.js） |

前后端环境独立：后端负责模型推理，前端只做 UI + HTTP 转发 + Plotly 渲染。