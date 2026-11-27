# Rice-Intro Backend — 计算逻辑与开发指南

> 本文档面向**在本后端上开发/排障**的协作者，说明 `rice_intro/backend` 的**计算逻辑**（如何从 FASTA + 模型权重得到渗入分析结果）、文件职责、配置与约束。
> 我是把离线流水线 `20.introgression_analysis`（`scripts/0~4` + `models/model.py` + `configs/config.yaml`）移植为 FastAPI 在线服务的实现，**计算逻辑与离线脚本逐位对齐**。

## OVERVIEW

在线粳稻/籼稻血缘渗入（introgression）Web 服务后端。用户在前端选择基因组/染色体/区间（Start 必填），后端**只对请求区域做 GPU 推理**，返回片段级概率、窗口级聚合、区域级融合，并在响应中附带全基因组展示骨架（12 条染色体）。前端（Gradio）与纯 API（curl）**走同一 DCS 子路径**（`…/OGR-Introgression/Prediction`），Plotly 全基因组三 track 图的渲染数据**完全来自该返回值**（前端 = 填好参数的 curl）。

- 模型：`rice_1B_stage2_8k_hf`（Mixtral-1B DNA 基座）+ LoRA（`q_proj/v_proj`，r=16）+ 投影头（1024→512→128，L2 归一化）+ 双标签分类头（Jap/Ind）。
- 关键参数（与离线 config 一致，`.env` 可覆盖）：`SEGMENT_SIZE=8000`、`WINDOW_SIZE=256000`、`WINDOW_STEP=64000`、`TOP_K=10`、`THRESHOLD_JAP=0.55519`、`THRESHOLD_IND=0.53473`。

## STRUCTURE

```
backend/
├── run_backend.sh / stop_backend.sh   # 统一后端 启动/停止（nohup + PID）
├── logs/                              # 运行日志
└── rice_introgression/                # 有效 Python 包
    ├── main.py             # uvicorn 入口（加载 .env 环境变量）
    ├── api.py              # FastAPI 路由：网页路由 + DCS 子路径（同一进程）
    ├── model.py            # FullModel（与离线 models/model.py 逐位一致）
    ├── predictor.py        # 单例预测器：FASTA 读取、trim、8k 片段切分、GPU 推理、全局推理锁
    ├── analysis.py         # 核心算法：标准窗口网格、top-k 聚合、双阈值分组、区域融合
    ├── prediction_service.py  # 编排：请求解析 → trim 缓存 → 切片段 → 推理 → 聚合 → 区域 → payload
    ├── genome_service.py   # 基因组注册（内置 .env + 上传自定义）、.fai 索引、染色体列表/长度
    ├── cache_service.py    # 内容寻址 LRU + TTL + 磁盘持久化预测缓存
    └── progress.py         # 单槽进度跟踪（/progress 轮询）
```

> **网页端与纯 API 调用是同一进程、同一子路径、同一返回**（2026-11 重构）：
> 预测统一走 `…/OGR-Introgression/Prediction`（前端 `frontend/app.py` 直连该子路径，
> 读信封 `result` 渲染 Plotly；不再用 `/analyze`）。网页路由（`/analyze`、`/uploadFasta` 等）
> 仍保留作内部辅助端点；共用同一推理核心与统一返回结构
> `{model_name, usage, token_cnt, status, message, result}`。
> 独立进程 `dcs_adapter.py` 已删除；统一启动见项目根 `start.sh`，网关入口见 `API.md`。

## CORE COMPUTATION（核心计算逻辑）

### 数据流总览

```
请求 (genome, chrom, start?, end?)
  │ 0. 坐标约定：API 层 1-based → 内部统一 0-based half-open（start_0b = start_1b - 1，end_0b = end_1b）
  ▼
genome_service.resolve_genome_config    上传自定义 > 内置 GENOME_*_FASTA；.fai 显式/自动构建（支持 gzip）→ 染色体列表 + 长度
  ▼
get_chromosome_trim（缓存）             读整条染色体 → trim_n 去头尾 N → [trim_start, trim_end)
  ▼
模式分发（Start 必填，2026-11 用户确认；不再支持整条模式）：
  窗口模式 start 必填（end 空=start+256k）——「start 锚定，取消网格吸附」：
    只填 start（end 空）→ 恒 1 个窗口 [start, start+256k)（MAX_NUMBER_256W 不参与）
    start/end 都填      → 重叠滑窗起点 = start, start+64k, ...（step=WINDOW_STEP）
                          起点 > end 停止生成；最后一个起点 ≤ end 的窗口即使尾部
                          > end 也保留输出（"若窗口>end，则输出一个"）
    超限语义：第 1 个窗口尾部 > chrom_len → 整体返回空（200 + message，不报 400）；
              后续窗口尾部 > chrom_len → 丢弃该窗口及之后（break），保留前面的
    MAX_NUMBER_256W → 窗口数 > N 取前 N 个（无 end 时不参与）
    多窗统一 _segments_window_grid：按绝对坐标 [w, w+256k) 切 8k 片段（不重复 trim，
    窗口尾部 ≤ chrom_len 故无需 N 补齐）
  Start 缺失 → 400（整条染色体模式已废弃）
  ▼
predictor.predict_segments              tokenize(max_length=8000) → FullModel 前向 → sigmoid → prob_jap / prob_ind
                                        （权重 fp32 + torch.autocast fp16；全局 _INFER_LOCK 串行 GPU；batch 回调进度）
  ▼
聚合 + 分组（analysis.py）               top-k 聚合 → 双阈值分组
  ▼
区域融合 call_group_regions             相邻同组窗口 overlap 合并 → regions {Ind, Jap, uncertain}
  ▼
inject_genome_context                   附加全基因组骨架：chromosomes + chromosome_lengths（未推理区由前端标记 uninferenced）
  ▼
前端 Gradio 与 curl 同走 …/OGR-Introgression/Prediction → Plotly 全基因组三 track 图（所有染色体车道 × 每染色体 Jap 蓝 / Ind 橘 / uncertain 绿 3 轨，未推理 gray）
```

### 关键算法细节（改动前必读）

1. **trim 依据**：仅裁染色体头尾 N（装配未解析区，无真实碱基信息），保留内部 N 以对齐参考坐标；片段绝对坐标 = `trim_start + i*8000`。离离线流程是「先 trim 整条染色体再切」，故后端**缓存染色体级 trim 偏移**并用于网格生成，保证 Web 查询与离线窗口一一对应。末端 N 与 `read_fasta_sequence` 的 `seq.ljust(end-start, "N")` 补齐（离线 `win_end=win_start+window_size` 不裁剪末端，需补齐以切满 32 段）。
2. **标准窗口网格**（`standard_window_starts`）：离线 `aggregate_windows` 的窗口起点 = 片段数组切片索引 `idx`（`0, 8, 16, …, n-32`，步长 `window_step/segment_size`）；Web 网格起点 = `trim_start + i*segment_size`（i 同上）。**2026-11 重构后 Start 必填，仅窗口模式**；`match_window_to_grid` / `_select_max_region_starts` / 整条模式路径不再被调用。

6. **单请求窗口数上限**（`MAX_NUMBER_256W`，`.env`——见 `.env` 表）：
   - 空 / 0 / 负数 / 非数字 → `None`（不限制）；正整数 `N` → 每请求至多推理 `N` 个 256k 窗口。
   - 只填 start（end 空）：恒 1 个窗口 `[start, start+256k)`——MAX 不参与。
   - start/end 都填 + N：重叠滑窗生成全部候选（起点 ≤ end），窗口数 > N → 取前 N 个。
   - 窗口模式 + N 时多窗口各自独立切片段/推理/聚合（`_aggregate_window_slice`），`segments` 合并、`windows` 为 N 行；进度总量=各窗片段数之和（上界近似）。
3. **top-k 聚合**（`aggregate_one_window` / `_aggregate_window_slice`）：窗口内 32 个 8k 片段，对 `prob_jap`、`prob_ind` **各自独立**取最高的 `top_k=10` 个求均值 → `topk_mean_jap` / `topk_mean_ind`。整条染色体的 `aggregate_windows_for_chrom` 与离线 `4.run_analysis.aggregate_windows` 逐位一致（片段按 start 排序、`win_end=win_start+window_size` 不裁剪）。
4. **双阈值分组**（`assign_dual_group`，与离线完全一致）：
   - `Jap`：`topk_mean_jap ≥ THRESHOLD_JAP(0.55519)` 且 `topk_mean_ind < THRESHOLD_IND(0.53473)`
   - `Ind`：`topk_mean_jap < THRESHOLD_JAP` 且 `topk_mean_ind ≥ THRESHOLD_IND`
   - `uncertain`：其余
5. **区域融合**（`call_regions` / `call_group_regions`）：染色体上相邻且同组窗口合并为连续区间（overlap 并集），记录区间均分 `topk_mean_*` 与 `n_windows`。

### 模式差异

| 模式 | 推理量 | 聚合路径 |
|---|---|---|
| 窗口（MAX 空） | 1 个 256k 窗 = 32 片段（约 4 个 batch） | `_aggregate_window_slice`（单窗口） |
| 窗口（MAX=N，只填 start） | 1 个 256k 窗（start 即目标位点，忽略 MAX） | `_aggregate_window_slice`（单窗口） |
| 窗口（MAX=N，start/end 都填） | ≤N 个 256k 窗（每窗 32 片段；窗间可能重叠） | 逐窗 `_aggregate_window_slice`，`windows` 多行 |

## CONVENTIONS

- **配置优先级**：`.env` > 代码默认值；全部参数从环境变量读取（`predictor.py` / `prediction_service.py` / `cache_service.py` 均有 `_env_*` 助手）。
- **坐标约定**：API 层 1-based 闭区间（`start_1b - 1` → 0-based），内部计算统一 0-based half-open；**返回 payload 坐标统一 1-based inclusive**（`win_start` = 请求 Start 本身，`win_end` = 窗口闭区间末尾，`segments/regions.start` 同理）。不要混用。
- **染色体名不做归一化**：FASTA 实际命名原样透传（YF47 为 `GWHBKAR00000001`…），与 `rice_mut`/`rice_reg` 的 `normalize_chromosome()` 不同。
- **GPU 推理串行**：`_INFER_LOCK`（`predictor.py`）保护所有前向；单例 `_PREDICTOR`（网页与 API 共用同一进程，只有一个模型实例）。
- **进度单槽**：`progress.py` 单 uvicorn worker 内仅维护「当前活动任务」一个槽；后到请求会抢占进度槽，但阻塞等待 GPU 期间不会回调，语义自洽。
- **进程管理**：统一 `bash start.sh [all|backend|stop|status|restart]`（nohup + PID，日志 `logs/backend.nohup.log` / `logs/frontend.nohup.log`）；或直接 `bash backend/run_backend.sh` / `bash frontend/run_frontend.sh`。

## CACHE（缓存机制）

`cache_service.PredictionResultCache`——内容寻址 LRU + TTL + 磁盘持久化，受 `CACHE_MODE` 控制：

- **CACHE_MODE=true（默认）**：
  - **键**：`build_key("intro", genome=…, chromosome=…, start=…, end=…)`（参数排序后 `|` 拼接，确定性内容寻址，跨用户共享）。
  - **命中**：直接返回 payload（`cached: true`），跳过 GPU。
  - **未命中**：推理 → `put()`（内存 OrderedDict + 落盘 `<BACKEND_PREDICTION_CACHE>/<sanitized_key>.json`）。
  - **TTL / 上限**：`PREDICTION_CACHE_TTL_SECONDS`（默认 3600s）、`PREDICTION_CACHE_MAX_ENTRIES`（默认 512），LRU 淘汰时同步删盘；冷数据可回读磁盘。
- **CACHE_MODE=false（懒加载部署）**：`prediction_cache` 磁盘目录置空、`run_introgression` 跳过读写——每次请求重跑 GPU，磁盘不积累。
- **启动策略（三服务统一，固定行为）**：每次启动 `prediction_cache.clear()`（清空内存；CACHE_MODE=true 时同时删除磁盘全部 `*.json`/`*.tmp`），**不做预热**；上载目录 `cache/uploaded_fasta` 总是先清后建。
- **自定义基因组 .fai**：原生 `.fai` 缺失/过期时，**不**写用户目录，而是软链接 FASTA 到 `cache/uploaded_fasta/<时间戳ns>_<文件名>` 再在该目录建 `.fai`（规避原目录无写权限）；临时 id 统一 `<时间戳ns>_<文件名>`。`managed=False` 时清理只删 cache 软链接/索引，不删用户原始 FASTA。
- 已见缓存示例：`cache/predictions/intro_chromosome_Chr01_end_256000_genome_YF47_start_0.json`。

## .env 关键项（见项目根 `.env`）

| 配置 | 说明 |
|---|---|
| `BACKEND_PYTHON_BIN` | 后端解释器（当前 `/root/miniconda3/envs/vllm/bin/python`，py3.12） |
| `BASE_MODEL_PATH` | 基座 `source/rice_mut/rice_1B_stage2_8k_hf` |
| `CHECKPOINT_PATH` | 微调权重 `source/rice_intro/model.safetensors`（目录会自动拼 `model.safetensors`） |
| `DEVICE` / `MODEL_TORCH_DTYPE` | `cuda:1` / `float32`（权重 fp32；`MODEL_USE_FP16=true` 时前向 autocast fp16，与离线 Trainer 数值一致；**不要 `model.to(fp16)`**） |
| `INFERENCE_BATCH_SIZE` | 8（覆盖 256k 窗的 32 段） |
| `LORA_*` / `PROJ_DIMS` / `POOLING_STRATEGY` | 与训练一致（r=16 α=32，q/v_proj；1024,512,128；masked_mean） |
| `SEGMENT_SIZE` / `WINDOW_SIZE` / `WINDOW_STEP` / `TOP_K` / `THRESHOLD_*` | 渗入分析参数（勿与离线 config 背离） |
| `MAX_NUMBER_256W` | 单请求至多推理的 256k 窗口数上限；空=不限制，正整数 N=至多 N 个（start+end 都填取前 N 个；只填 start 恒 1 窗忽略本上限） |
| `GENOME_<ID>_FASTA` | 内置基因组（多个 `GENOME_*_FASTA` 自动进下拉） |
| `BACKEND_PREDICTION_CACHE` / `BACKEND_UPLOADED_FASTA` | 缓存目录（绝对路径） |
| `CACHE_MODE` | 只控制懒加载+内容寻址缓存：`true`=启用内容寻址缓存（落盘+复用）；`false`=懒加载部署（不缓存每次重跑，自定义基因组按 path 懒注册零拷贝，TTL 不删外部文件）。两种模式重启都清空预测缓存 |
| `PREDICTION_CACHE_TTL_SECONDS` / `PREDICTION_CACHE_MAX_ENTRIES` | 预测缓存 TTL / 最大条目（默认 3600s / 512） |
| `UPLOADED_GENOMES_TTL_HOURS` / `_CLEANUP_INTERVAL` | 上传基因组闲置清理（默认 1h / 300s） |
| `DCS_API_KEY` | DCS 子路径鉴权（留空=不鉴权；配置后除 health 外需 `Authorization: Bearer <key>` / `X-API-Key`） |
| `DCS_PROMPT_TOKEN_MULTIPLIER` / `DCS_COMPLETION_TOKEN_MULTIPLIER` | 计费倍数：`prompt_tokens`=片段数×8000×倍数；`completion_tokens`=窗口数×倍数（默认 1） |

## DCS 子路径（并入统一后端，同一进程）

`api.py` 内嵌 DCS 风格子路径（与网页路由同进程/端口，**无独立进程**）：

- **入口**：`POST /api/aigress/openai/OGR-Introgression/{Prediction,Genome,Chromosome,RegisterGenome}`
  （兼容旧短路径 `/OGR-Introgression/*` 与 `…/predict`）；body 无 `mode` 分发，camelCase + 1-based 坐标。
- **监听**：`PORT`（DCS 平台注入）优先，回退 `BACKEND_PORT`（部署约定 5001）。
- **复用**：同一 `run_introgression` / `init_predictor` / `list_genomes`，故与网页端（同一 `Prediction` 子路径）**逐位一致**（含 `MAX_NUMBER_256W`、CACHE_MODE、阈值）。
- **计费**：`prompt_tokens = 片段数 × segment_size(8000)`；`completion_tokens = 窗口数`；受 `DCS_*_MULTIPLIER` 缩放。
- **鉴权**：`DCS_API_KEY` 配置后，除 health 外需 `Authorization: Bearer <key>` / `X-API-Key`；前端网页端 API Key 由页面输入框提供（不读 .env）。
- **懒注册**：请求 body 带 `fasta_path` 即现场注册（managed=False，零拷贝，TTL 不删外部文件）。
- **返回**：`{model_name, usage, token_cnt, status, message, result}`；`result` 含 `windows[]`（每窗 1-based 坐标 / center / n_segments / topk_mean_jap / topk_mean_ind / group）+ `params` + `threshold_rule`。
- **调用文档**：项目根 `API.md`。

## API 速查

| 方法/路径 | 说明 |
|---|---|
| `GET /health` | 健康 + 模型状态 + 基因组列表 + 当前参数 |
| `GET /genomes` | 基因组列表（内置 + 已上传） |
| `GET /genomes/{id}/chromosomes` | 染色体列表（FASTA 实际命名） |
| `GET /genomes/{id}/chromosomes/{chrom}/length` | 染色体长度（.fai 优先，pyfaidx 兜底） |
| `POST /uploadFasta` | 上传自定义 FASTA（≤640MB，自动 .fai，注册 `custom_<ts>`，优先于内置） |
| `POST /analyze` | 渗入分析；`start` 必填（1-based）；`end` 空 → `start+256k`；`MAX_NUMBER_256W=N` 时单请求至多 N 窗（见上）；返回 segments/windows/regions（坐标 1-based）+ 展示骨架 |
| `POST /analyze-genome` | 全基因组逐染色体分析（逐条走缓存） |
| `GET /progress` | 推理进度（前端每 2s 轮询；无任务返回 idle） |

统一启动：`bash start.sh [all|backend|stop|status|restart]`（先保证 `.env` 且 `tools/startup_self_check.sh` 通过）。
网页端/纯 API 共用同一后端进程；调用文档 `API.md`。

## NOTES / ANTI-PATTERNS

- **不要修改 `backend/` 下已与离线对齐的数值路径**（top-k 排序、`win_end` 不裁剪、双阈值比较符），改动会破坏在线=离线一致性；改参数走 `.env`，改算法需同时评估离线 `4.run_analysis.py`。
- **坐标/网格敏感**：`match_window_to_grid`、`standard_window_starts`、`_segments_window_grid` 三者必须保持同一套网格语义；新增裁剪/补齐逻辑前先读 `predictor.read_fasta_sequence` 的 N 补齐行为。
- **重叠计算必须用 `window_size`**：`_select_max_region_starts` 的 overlap 宽度 = `window_size`（256k），不是网格步长 `window_step`（64k）；改错会让 `MAX_NUMBER_256W` 的“覆盖度最大 N 窗”选择失真。
- **`MAX_NUMBER_256W` 是每请求上限**：改 `.env` 后需重启后端（`load_dotenv` 在进程启动时读取）；置空即恢复旧版单窗行为。
- **权重精度**：checkpoint 以 fp32 加载（与训练一致），前向 autocast fp16，不要改 `model.to(fp16)`。
- **GPU 由锁串行**：不要为「提速」移除 `_INFER_LOCK`（并发调用同一模型 OOM/态崩）。
- **缓存即权威**：命中缓存时返回 `cached:true`，排障时先看请求是否命中（省 GPU 不等于 bug）。
- **这是第四个子项目**：`rice_server/` 下另有 `rice_mut` / `rice_reg` / `rice_OGR`，风格相近但**本服务不做染色体归一化**，复制代码时勿引入 `normalize_chromosome`。
- **CACHE_MODE=false 不落盘**：轻量模式下 `prediction_cache` 磁盘目录置空且 `run_introgression` 跳过读写，每次请求重跑 GPU；`cache/uploaded_fasta` 仍先清后建。
- **懒注册文件不删**：`managed=False` 的基因组（`fasta_path` / uploadFasta `path` 字段）TTL 清理只移除注册表条目，绝不删除底层用户文件。