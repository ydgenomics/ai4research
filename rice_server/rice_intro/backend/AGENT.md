# Rice-Intro Backend — 计算逻辑与开发指南

> 本文档面向**在本后端上开发/排障**的协作者，说明 `rice_intro/backend` 的**计算逻辑**（如何从 FASTA + 模型权重得到渗入分析结果）、文件职责、配置与约束。
> 我是把离线流水线 `20.introgression_analysis`（`scripts/0~4` + `models/model.py` + `configs/config.yaml`）移植为 FastAPI 在线服务的实现，**计算逻辑与离线脚本逐位对齐**。

## OVERVIEW

在线粳稻/籼稻血缘渗入（introgression）Web 服务后端。用户在前端选择基因组/染色体/区间，后端**只对请求区域做 GPU 推理**，返回片段级概率、窗口级聚合、区域级融合，并在响应中附带全基因组展示骨架（12 条染色体）。前端（Gradio）消费 `/analyze` payload 渲染 Plotly 车道图。

- 模型：`rice_1B_stage2_8k_hf`（Mixtral-1B DNA 基座）+ LoRA（`q_proj/v_proj`，r=16）+ 投影头（1024→512→128，L2 归一化）+ 双标签分类头（Jap/Ind）。
- 关键参数（与离线 config 一致，`.env` 可覆盖）：`SEGMENT_SIZE=8000`、`WINDOW_SIZE=256000`、`WINDOW_STEP=64000`、`TOP_K=10`、`THRESHOLD_JAP=0.55519`、`THRESHOLD_IND=0.53473`。

## STRUCTURE

```
backend/
├── run_backend.sh / stop_backend.sh   # 网页版后端 启动/停止（nohup + PID）
├── run_dcs_adapter.sh / stop_dcs_adapter.sh  # DCS 适配层（独立进程，监听 5001）
├── dcs_adapter.py        # ★ DCS 适配层：OpenAI 风格单入口（POST /api/aigress/openai/rice_intro）
├── logs/                              # 运行日志
└── rice_introgression/                # 有效 Python 包
    ├── main.py             # uvicorn 入口（加载 .env 环境变量）
    ├── api.py              # FastAPI 路由：/health /genomes /uploadFasta /analyze /analyze-genome /progress
    ├── model.py            # FullModel（与离线 models/model.py 逐位一致）
    ├── predictor.py        # 单例预测器：FASTA 读取、trim、8k 片段切分、GPU 推理、全局推理锁
    ├── analysis.py         # 核心算法：标准窗口网格、top-k 聚合、双阈值分组、区域融合
    ├── prediction_service.py  # 编排：请求解析 → trim 缓存 → 切片段 → 推理 → 聚合 → 区域 → payload
    ├── genome_service.py   # 基因组注册（内置 .env + 上传自定义）、.fai 索引、染色体列表/长度
    ├── cache_service.py    # 内容寻址 LRU + TTL + 磁盘持久化预测缓存
    └── progress.py         # 单槽进度跟踪（/progress 轮询）
```

> **DCS 适配层**：`dcs_adapter.py` 是**独立进程**（不挂载到网页版 `api.py`），复用 `rice_introgression` 包的 `run_introgression`/`init_predictor`/`list_genomes`，与网页版 5001 是**不同机器/部署**。DCS 监听约定 5001（`PORT` 优先，回退 `BACKEND_PORT`）；调用文档见 `API.md`。

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
模式分发（受 MAX_NUMBER_256W 限制，见下）：
  窗口模式 start/end 非空（end 空=start+256k）：
    只填 start（end 空）→ 始终 1 个「最大覆盖度」网格窗（start 即用户显式指定
                          目标位点，忽略 MAX_NUMBER_256W）
    start/end 都填：
      max_windows is None  → ana.match_window_to_grid → 1 个「最大覆盖度」网格窗
      max_windows = N      → select_windows → 与用户区间重叠最大的 N 个网格窗（多窗）
    多窗/单窗统一 _segments_window_grid：按绝对坐标网格切 8k 片段（不重复 trim）
  整条模式 start/end 都空：
    max_windows is None  → _segments_chromosome：读全染色 → trim → 按 8000 无重叠切全部片段（与离线完全一致）
    max_windows = N      → 只取标准网格前 N 个 256k 窗（染色体 5' 端），走窗口模式路径
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
前端 Gradio /analyze → Plotly 4 色车道图（Ind 橙 / Jap 蓝 / uncertain 灰 / uninferenced 浅灰）
```

### 关键算法细节（改动前必读）

1. **trim 依据**：仅裁染色体头尾 N（装配未解析区，无真实碱基信息），保留内部 N 以对齐参考坐标；片段绝对坐标 = `trim_start + i*8000`。离离线流程是「先 trim 整条染色体再切」，故后端**缓存染色体级 trim 偏移**并用于网格生成，保证 Web 查询与离线窗口一一对应。末端 N 与 `read_fasta_sequence` 的 `seq.ljust(end-start, "N")` 补齐（离线 `win_end=win_start+window_size` 不裁剪末端，需补齐以切满 32 段）。
2. **标准窗口网格**（`standard_window_starts`）：离线 `aggregate_windows` 的窗口起点 = 片段数组切片索引 `idx`（`0, 8, 16, …, n-32`，步长 `window_step/segment_size`）；Web 网格起点 = `trim_start + i*segment_size`（i 同上）。用户输入经**最大覆盖度**匹配（`match_window_to_grid`，重叠并列取更早窗口，确定性）落到该网格——这是「在线=离线一致」的核心。

6. **单请求窗口数上限**（`MAX_NUMBER_256W`，`.env`——见 `.env` 表）：
   - 空 / 0 / 负数 / 非数字 → `None`（不限制，行为与旧版一致）；正整数 `N` → 每请求至多推理 `N` 个 256k 网格窗口。
   - 整条模式（start/end 空）+ N：取标准网格**前 N 个**窗口（染色体 5' 端，`grid_starts[:N]`），region 缩为 `[wins[0], wins[-1]+window_size]`，`mode="window"`。
   - 只填 start（end 空）：无论 MAX 是否设置都**只推 1 个**「最大覆盖度」网格窗口——start 是用户显式指定的目标位点，不做多窗扩算。
   - start/end 都填 + N：对所有网格起点算与用户区间的**重叠长度** `overlap=max(0, min(gs+window_size,user_end)-max(gs,user_start))`，按 overlap 降序取前 N 个；**平局取起点更早者**（确定性）；无正重叠 → 兜底前 N 个网格起点。注意重叠必须用 `window_size`（256k）计算，不是网格步长（64k）。
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
| 整染色体（MAX 空） | trim 后全部片段（YF47 Chr01 ≈ 5,488 段 ≈ 686 batch） | `aggregate_windows_for_chrom`（滑窗网格，与离线一致） |
| 整条 5' 端（MAX=N） | 前 N 个 256k 窗 = 前 N*32 片段 | 窗口模式路径（同窗口 MAX=N） |
| 全基因组 | 逐染色体走整条模式，缓存命中跳过 GPU | `run_genome_introgression` 拼装 |

## CONVENTIONS

- **配置优先级**：`.env` > 代码默认值；全部参数从环境变量读取（`predictor.py` / `prediction_service.py` / `cache_service.py` 均有 `_env_*` 助手）。
- **坐标约定**：API 层 1-based 闭区间（`start_1b - 1` → 0-based），内部统一 0-based half-open；不要混用。
- **染色体名不做归一化**：FASTA 实际命名原样透传（YF47 为 `GWHBKAR00000001`…），与 `rice_mut`/`rice_reg` 的 `normalize_chromosome()` 不同。
- **GPU 推理串行**：`_INFER_LOCK`（`predictor.py`）保护所有前向；单例 `_PREDICTOR`（注意：DCS 适配层与网页版后端是**两个独立进程**，各自加载模型/锁/缓存）。
- **进度单槽**：`progress.py` 单 uvicorn worker 内仅维护「当前活动任务」一个槽；后到请求会抢占进度槽，但阻塞等待 GPU 期间不会回调，语义自洽。
- **进程管理**：网页版 `run_backend.sh` / `stop_backend.sh`；DCS 适配层 `run_dcs_adapter.sh` / `stop_dcs_adapter.sh`（nohup + PID，日志 `logs/dcs_adapter.nohup.log`）。

## CACHE（缓存机制）

`cache_service.PredictionResultCache`——内容寻址 LRU + TTL + 磁盘持久化：

- **键**：`build_key("intro", genome=…, chromosome=…, start=…, end=…)`（参数排序后 `|` 拼接，确定性内容寻址，跨用户共享）。
- **命中**：直接返回 payload（`cached: true`），跳过 GPU。
- **未命中**：推理 → `put()`（内存 OrderedDict + 落盘 `<BACKEND_PREDICTION_CACHE>/<sanitized_key>.json`）。
- **TTL / 上限**：`PREDICTION_CACHE_TTL_SECONDS`（默认 3600s）、`PREDICTION_CACHE_MAX_ENTRIES`（默认 512），LRU 淘汰时同步删盘；冷数据可回读磁盘。
- **启动策略**（`.env` 可配，见 `.env` 表）：
  - `CLEAR_CACHE_ON_STARTUP=true` → 启动即 `prediction_cache.clear()`（清空内存 + 删除磁盘全部 `*.json`/`*.tmp`），之后重新推理；上载目录 `cache/uploaded_fasta` 总是先清后建。
  - 默认/其他 → `warm_prediction_cache()` 从磁盘加载之前推理过的窗口（重启不重算 GPU）。
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
| `MAX_NUMBER_256W` | 单请求至多推理的 256k 窗口数上限；空=不限制，正整数 N=至多 N 个（未填 start 取前 N 个窗口；start+end 都填取覆盖度最大 N 个） |
| `GENOME_<ID>_FASTA` | 内置基因组（多个 `GENOME_*_FASTA` 自动进下拉） |
| `BACKEND_PREDICTION_CACHE` / `BACKEND_UPLOADED_FASTA` | 缓存目录（绝对路径） |
| `CLEAR_CACHE_ON_STARTUP` | 重启服务时自动清理预测缓存：`true`=启动清空内存+磁盘缓存重建；留空/其他=预热磁盘缓存 |
| `PREDICTION_CACHE_TTL_SECONDS` / `PREDICTION_CACHE_MAX_ENTRIES` | 预测缓存 TTL / 最大条目（默认 3600s / 512） |
| `UPLOADED_GENOMES_TTL_HOURS` / `_CLEANUP_INTERVAL` | 上传基因组闲置清理（默认 1h / 300s） |
| `DCS_API_KEY` | DCS 适配层鉴权（留空=不鉴权；配置后除 health 外需 `Authorization: Bearer <key>` / `X-API-Key`） |
| `DCS_PROMPT_TOKEN_MULTIPLIER` / `DCS_COMPLETION_TOKEN_MULTIPLIER` | 计费倍数：`prompt_tokens`=片段数×8000×倍数；`completion_tokens`=窗口数×倍数（默认 1） |

## DCS 适配层

`dcs_adapter.py` 提供 DCS 平台接入（**独立进程**，不挂载到网页版 `api.py`）：

- **入口**：`POST /api/aigress/openai/rice_intro`（简写 `/rice_intro`）+ body `mode` 分发；`healthy`/空 body → health，`predict`/`analyze`/`intro`/其余 → predict。
- **监听**：`PORT`（DCS 平台注入）优先，回退 `BACKEND_PORT`；**部署约定 5001**，与网页版是不同机器。启动：`bash backend/run_dcs_adapter.sh`，停止：`bash backend/stop_dcs_adapter.sh`（日志 `logs/dcs_adapter.nohup.log`）。
- **复用**：`run_introgression` / `init_predictor` / `list_genomes`（`rice_introgression` 包），故与网页版 `/analyze` **逐位一致**（含 `MAX_NUMBER_256W`、缓存、阈值）。
- **计费**：`prompt_tokens = 片段数 × segment_size(8000)`；`completion_tokens = 窗口数`；受 `DCS_*_MULTIPLIER` 缩放。
- **返回**：`{usage, status, message, result}`；`result` 含 `windows[]`（每窗 1-based 坐标 / center / n_segments / topk_mean_jap / topk_mean_ind / group）+ `params` + `threshold_rule`。
- **调用文档**：项目根 `API.md`。

## API 速查

| 方法/路径 | 说明 |
|---|---|
| `GET /health` | 健康 + 模型状态 + 基因组列表 + 当前参数 |
| `GET /genomes` | 基因组列表（内置 + 已上传） |
| `GET /genomes/{id}/chromosomes` | 染色体列表（FASTA 实际命名） |
| `GET /genomes/{id}/chromosomes/{chrom}/length` | 染色体长度（.fai 优先，pyfaidx 兜底） |
| `POST /uploadFasta` | 上传自定义 FASTA（≤640MB，自动 .fai，注册 `custom_<ts>`，优先于内置） |
| `POST /analyze` | 渗入分析；`start` 空 → 整条染色体；`end` 空 → `start+256k`；`MAX_NUMBER_256W=N` 时单请求至多 N 窗（见上）；返回 segments/windows/regions + 全基因组骨架 |
| `POST /analyze-genome` | 全基因组逐染色体分析（逐条走缓存） |
| `GET /progress` | 推理进度（前端每 2s 轮询；无任务返回 idle） |

网页版启动：`bash backend/run_backend.sh`（先保证 `.env` 且 `tools/startup_self_check.sh` 通过）。
DCS 适配层启动：`bash backend/run_dcs_adapter.sh`（见上文「DCS 适配层」；调用文档 `API.md`）。

## NOTES / ANTI-PATTERNS

- **不要修改 `backend/` 下已与离线对齐的数值路径**（top-k 排序、`win_end` 不裁剪、双阈值比较符），改动会破坏在线=离线一致性；改参数走 `.env`，改算法需同时评估离线 `4.run_analysis.py`。
- **坐标/网格敏感**：`match_window_to_grid`、`standard_window_starts`、`_segments_window_grid` 三者必须保持同一套网格语义；新增裁剪/补齐逻辑前先读 `predictor.read_fasta_sequence` 的 N 补齐行为。
- **重叠计算必须用 `window_size`**：`_select_max_region_starts` 的 overlap 宽度 = `window_size`（256k），不是网格步长 `window_step`（64k）；改错会让 `MAX_NUMBER_256W` 的“覆盖度最大 N 窗”选择失真。
- **`MAX_NUMBER_256W` 是每请求上限**：改 `.env` 后需重启后端（`load_dotenv` 在进程启动时读取）；置空即恢复旧版单窗/整条行为。
- **权重精度**：checkpoint 以 fp32 加载（与训练一致），前向 autocast fp16，不要改 `model.to(fp16)`。
- **GPU 由锁串行**：不要为「提速」移除 `_INFER_LOCK`（并发调用同一模型 OOM/态崩）。
- **缓存即权威**：命中缓存时返回 `cached:true`，排障时先看请求是否命中（省 GPU 不等于 bug）。
- **这是第四个子项目**：`rice_server/` 下另有 `rice_mut` / `rice_reg` / `rice_OGR`，风格相近但**本服务不做染色体归一化**，复制代码时勿引入 `normalize_chromosome`。
- **DCS 适配层勿与网页版同端口同时起**：两者都默认监听 5001（但 DCS 部署是不同机器/容器，仅本机联调时避免端口冲突）；DCS 配 `PORT` 环境变量可改。