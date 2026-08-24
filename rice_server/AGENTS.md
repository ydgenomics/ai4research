# Rice-Server — 项目总览与部署指南

> 本文档面向**部署/交付人员**，介绍本目录下两个水稻组学预测服务（`rice-mut`、`rice-reg`）的文件结构、设计思路与部署方法。代码相关的开发细节见各子项目 README 与 `docs/`。

---

## 1. 目录总览

```
rice-server/
├── AGENTS.md              # 本文档
├── README-yd.md           # rice-mut 的需求变更记录（历史）
├── rice-mut/              # ★ 项目一：水稻 DNA → 多组学表达预测（变异对比）
│   ├── .env / .env.example    # 环境配置（部署时必改）
│   ├── requirements.txt       # 统一依赖（前后端共用）
│   ├── README.md              # 完整使用文档
│   ├── backend/               # FastAPI 后端（端口 8001）
│   │   ├── run_backend.sh / stop_backend.sh
│   │   ├── rice_mutation/     # 应用代码（main.py 入口 / api.py 路由 / prediction_service.py / igv_payload.py / cache_service.py）
│   │   ├── core/predictor.py  # ★ RiceMutationPredictor 核心推理类
│   │   ├── src/               # ★ 模型定义（GenOmics UNet 等，直接复用）
│   │   └── inference.ipynb / TAC1_inference.ipynb  # 原始推理原型（保留参考）
│   ├── frontend/              # Gradio 前端（端口 8000）
│   │   ├── app.py / config.py / igv_payload.py
│   │   └── static/igv.min.js  # IGV.js 静态资源（本地化，无需外网）
│   ├── tools/startup_self_check.sh   # 启动自检
│   └── test/test_api.sh              # API 冒烟测试
│
├── rice-reg/               # ★ 项目二：水稻 ATAC-seq → RNA-seq 表达预测
│   ├── .env / .env.example     # 环境配置（部署时必改）
│   ├── requirements.txt        # 统一依赖（前后端共用）
│   ├── README.md               # 使用文档（含配置说明）
│   ├── WORK-yd.md              # 需求变更记录（历史）
│   ├── backend/                # FastAPI 后端（端口 7001）
│   │   ├── requirements.txt    # 后端依赖
│   │   ├── run_backend.sh / stop_backend.sh
│   │   └── rice_reg/           # 应用代码（main.py / api.py / prediction_service.py / igv_payload.py / cache_service.py）
│   │       └── core/           # rice_reg.py（★ RiceRegPredictor）/ scaling.py / model/（pipeline、encoder、predictor）
│   ├── frontend/               # Gradio 前端（端口 7000）
│   │   ├── requirements.txt
│   │   ├── app.py / config.py / igv_payload.py
│   │   └── static/igv.min.js
│   ├── tools/startup_self_check.sh
│   ├── docs/                   # IMPLEMENTATION_PLAN.md（设计决策）/ DEBUG_LOG.md
│   └── test/test_api.sh
│
└── source/                 # ★ 模型与基因组数据（约 17G，交付必带）
    ├── rice-mut/
    │   ├── rice_1B_stage2_8k_hf/     # 基础模型（HuggingFace 格式，4.7G）
    │   ├── csq-5_model.safetensors   # 微调 checkpoint（2.6G）
    │   ├── index_stat.json           # assay/biosample 输出头索引
    │   ├── osa1_r7.asm.ch.fa(.fai)   # 参考基因组（osa1_r7）
    │   └── osa1_r7.all_models.gff3   # 注释
    └── rice-reg/
        ├── rice_1B_32k_hf/           # 基础模型（HuggingFace 格式，4.7G）
        ├── model.safetensors         # 微调 checkpoint（2.6G）
        ├── genome/                   # MH63.fa / NIP.fa + .fai + GFF（1.1G）
        └── ATAC/                     # 内置 ATAC bigWig（SAM2_MH63_1 / SAM2_NIP_1）
```

---

## 2. 两个项目的区别与设计

| 维度 | rice-mut（变异预测） | rice-reg（ATAC 条件预测） |
|---|---|---|
| 模型输入 | **仅 DNA 序列** | DNA + **ATAC bigWig 信号** |
| 基础模型 | `rice_1B_stage2_8k_hf` | `rice_1B_32k_hf` |
| 输出头 | `GenOmics`（UNet，assay × biosample 多维） | fusion predictor（RNA-seq ± 两通道） |
| 反归一化 | 模型内部完成 | 服务端 `scaling.py` |
| 核心场景 | **参考 vs 单碱基突变（SNV）双轨对比** | 不同 ATAC 条件下的表达预测 |
| 序列长度 | 32768（超出截断） | 32678（固定窗口） |
| 端口（前端/后端） | **8000 / 8001** | **7000 / 7001** |

> ⚠️ 两项目可部署在同一台机器，端口互不冲突；`rice-mut` 用 8000/8001，`rice-reg` 用 7000/7001。

### 架构设计（两项目一致）

```
浏览器 (Gradio 前端 :8000/:7000)
   │  HTTP (httpx)
   ▼
FastAPI 后端 (:8001/:7001)
   ├── prediction_service.py   单例预测器 + 核心推理 + bigWig 写盘
   ├── cache_service.py        内容寻址预测缓存（LRU+TTL）+ bigWig 后台清理
   ├── igv_payload.py          IGV.js 渲染所需 payload（reference + tracks）
   └── core/                   模型加载与推理（RiceMutationPredictor / RiceRegPredictor）
   │
   ▼
IGV.js（前端 iframe 内）通过后端 /static-files 静态服务加载 bigWig / FASTA / GFF
```

- 预测结果写成 **bigWig** 临时文件，由 `cache_service` 后台线程定期清理（默认 TTL 1800s）。
- 支持**上传自定义基因组**（FASTA 自动建 `.fai` 并注册为 `custom_<ts>`，可附加 GFF）；空闲超过 `UPLOADED_GENOMES_TTL_HOURS`（默认 0.5h）自动清理。
- 前端染色体统一显示 `chr01`–`chr12`，后端 `normalize_chromosome()` 自动通配到实际 FASTA 命名（`chr01`→`Chr1`、`1`、`ChrUn` 等兼容）。
- SNV 对比：`/predict/snv` 生成 `snv_id`，`/predict/snv/stat` 计算 `(result1−result2)/result1` 区域差异统计；前端用双指针滑动选区域。

### 2.1 并发机制

后端是 FastAPI 异步服务，多个请求可同时进入，但**GPU 推理被严格串行化**：

- **推理锁 `_infer_lock`**（`threading.Lock`）：两个项目的预测器内都持有该锁，GPU 前向传播（`model.predict(...)`）整体在锁内执行。原因是单常驻模型不能从多个线程并发调用（GPU 争抢 / OOM）。
- **故意不调用 `torch.cuda.empty_cache()`**：它是全局 GPU 同步点，逐请求调用会严重损害并发吞吐；显存复用交给 PyTorch 缓存分配器自己管理。
- **in-flight merge（缓存未命中合并，rice-reg 独有）**：多个并发请求同时 miss 同一缓存键时，只有第一个请求成为 *leader* 执行 GPU 推理，其余请求通过 `threading.Event` 等待 leader 结果（超时 600s）；若 leader 失败或超时，follower 会自己算一次（每个等待者至多计算一次，无递归）。这样热点位点在并发下也只跑一次 GPU。
- **注册表锁 `_REGISTRY_LOCK`**：上传基因组的注册 / id 生成 / 读写均加锁，避免并发上传时互相覆盖。
- **`INFERENCE_BATCH_SIZE`**：单次模型前向的批大小（rice-mut 默认 1，rice-reg 默认 8）。当前 API 为单窗口请求，预留批量推理能力；并发下 GPU 仍由 `_infer_lock` 串行，该值不改变并发度。
- 上传文件的**清理线程**（bigWig 清理、上传基因组 TTL 清理）均为 daemon 线程，独立于请求处理运行。

### 2.2 缓存机制（用户提交文件如何缓存、如何控制内存）

#### 预测缓存：内容寻址 LRU + TTL（`cache_service.py`）

- **缓存键 = 规范化输入参数的确定性拼接**（`build_key`）：
  - rice-mut：`genome` + `chromosome` + **窗口规范化后**的 `start/end`（`_adjust_window` 先把任意窗口中心对齐到 32768，使等价请求共享同一缓存条目）+ `biosample` 列表；SNV 请求额外含突变位置与碱基。
  - rice-reg：`genome` + `chromosome` + 规范化 `start/end` + **ATAC 文件路径**（内置或上传的路径都是键的一部分）。
- **跨用户共享**：缓存是全局单例、所有用户共享（内容寻址），多用户请求同一热门位点时只推理一次。
- **命中时**：直接返回缓存的 `igv_payload`，**跳过 GPU**（响应 message 标记 `cached`）。
- **未命中时**：执行推理 → 结果写为 **bigWig 文件落盘**（缓存目录）→ 内存只存 `igv_payload`（内含 bigWig 的 HTTP URL 引用）+ 元数据。
- **rice-mut 的值数组缓存**：`_REF_CACHE` / `_SNV_CACHE` 按 `prediction_id` 在内存中保留推理出的数值数组，供 `/predict/bar`（柱状图区域均值）与 `/predict/snv/stat`（差异统计）直接查询，**无需重跑模型**；上限 `_CACHE_ARRAY_MAX = 256` 条，超出逐出最旧。

#### 内存如何受控（磁盘 / 内存分层）

1. **大数组不常驻内存**：一份预测是 32768 长度 × 多 assay × 多 biosample 的 float 数组，全部落盘为 bigWig；内存中只保留小体积的 payload / URL 引用 / 元数据。这是控制内存的关键设计。
2. **内容缓存上限**：`PredictionResultCache(max_entries=128, ttl_seconds=1800)` —— 内存缓存最多 128 条，LRU 超出即逐出；条目 TTL 30 分钟，过期惰性删除。
3. **磁盘 bigWig 后台清理**：`start_bigwig_cleanup` 启动 daemon 线程，每 600s 扫描一次缓存目录，删除 mtime 超过 `TTL + 600s` 余量的 `.bw` 文件（保守余量保证没有活的缓存条目仍引用它）。
4. **上传文件 TTL 清理**：`start_uploaded_genome_cleanup` 每 `UPLOADED_GENOMES_CLEANUP_INTERVAL`（默认 300s）扫描，删除闲置超过 `UPLOADED_GENOMES_TTL_HOURS`（默认 0.5h）的自定义基因组（FASTA + `.fai` + GFF 全删）。
5. **启动时清空缓存目录**：后端启动事件会对所有缓存目录 `rmtree` 后重建，避免上次运行的残留文件堆积。

> 说明：缓存条目数（128）、TTL（1800s）、清理间隔（600s）目前是代码内常量，未暴露为环境变量；可配置的见下方 3.3 表格。

---

## 3. 部署指南

### 3.1 硬件与环境要求

- **NVIDIA GPU**（两个 1B 模型同时部署建议显存 ≥ 24G，至少 16G；`DEVICE=cuda:0` 可在 `.env` 调整）
- Python 3.10+（建议独立 conda 环境，前后端共用同一环境即可）
- CUDA 工具链（`flash-attn` 需要；若安装困难，rice-mut 可将 `USE_FLASH_ATTN=false` 关闭）
- 磁盘：模型 + 基因组约 17G，预测缓存另计

### 3.2 安装依赖

```bash
# 如果同时部署在一台机器上，环境可以共用
# rice-mut（依赖统一在根目录）
cd rice-mut && pip install -r requirements.txt

# rice-reg（前端/后端各有 requirements.txt，装根目录那份即可覆盖）
cd rice-reg && pip install -r requirements.txt
```

> 交付时建议用 `pip freeze` 锁定实际验证过的版本（当前 requirements 用 `>=`，新环境可能装到不兼容的新版本）。

### 3.3 配置 .env（部署必改）

每个项目根目录执行 `cp .env.example .env`，然后修改以下路径类配置（**必须指向本机实际路径**）：

| 配置项 | 说明 |
|---|---|
| `BACKEND_PYTHON_BIN` / `FRONTEND_PYTHON_BIN` | Python 解释器绝对路径（可同一环境） |
| `BASE_MODEL_PATH` / `CHECKPOINT_PATH` / `INDEX_STAT_PATH` | 指向 `source/<项目>/` 下的模型文件 |
| `GENOME_<ID>_FASTA/FAI/GFF` | 参考基因组文件 |
| `ATAC_PATH_<ID>` + `ATAC_GENOME_MAP_<GENOME>`（rice-reg） | 内置 ATAC bigWig 与基因组的映射 |
| `BACKEND_UPLOADED_*` / `BACKEND_PREDICTION_CACHE` | 缓存目录（**必须绝对路径**，静态文件服务按绝对路径解析） |
| `BACKEND_API_URL` | 前端调后端地址，默认本机回环，一般无需改 |

**缓存与并发相关配置项**（可调项一览）：

| 配置项 | 默认值 | 作用 |
|---|---|---|
| `BACKEND_PREDICTION_CACHE` | `<项目>/cache/predictions` | 预测 bigWig 缓存目录（磁盘，后台线程按 TTL 清理） |
| `BACKEND_UPLOADED_FASTA`（rice-mut） | `<项目>/cache/uploaded_fasta` | 用户上传的自定义基因组 FASTA 目录 |
| `BACKEND_UPLOADED_GENOMES`（rice-reg） | `<项目>/cache/uploaded_genomes` | 用户上传基因组/注释目录 |
| `BACKEND_UPLOADED_ATAC`（rice-reg） | `<项目>/cache/uploaded_atac` | 用户上传的 ATAC bigWig 目录 |
| `UPLOADED_GENOMES_TTL_HOURS` | `0.5` | 闲置上传基因组的存活时长（小时），超时自动删除 FASTA+FAI+GFF |
| `UPLOADED_GENOMES_CLEANUP_INTERVAL` | `300` | 上传基因组清理线程的扫描间隔（秒） |
| `MAX_UPLOAD_MB` | `640` | 上传 FASTA / GFF 的大小上限（MB） |
| `MAX_ATAC_UPLOAD_MB`（rice-reg） | `10240` | 上传 ATAC bigWig 大小上限（MB，默认 10 GB） |
| `INFERENCE_BATCH_SIZE` | rice-mut `1` / rice-reg `8` | 单次模型前向的批大小（预留批量推理；并发下 GPU 仍由锁串行） |
| `DISPLAY_HEADS` / `DISPLAY_BIOSAMPLES`（rice-mut） | `RNA-seq` / `Leaf` | 前端/IGV 展示名覆盖（推理仍用 `index_stat` 内部键） |
| `USE_FLASH_ATTN`（rice-mut） | `true` | 是否启用 FlashAttention；装不上时可改 `false` |
| `MODEL_TORCH_DTYPE` / `DEVICE` | `bfloat16` / `cuda:0` | 模型精度与推理设备（rice-reg 另有 `CUDA_VISIBLE_DEVICES`） |

> 注意：缓存条数（128）、预测 TTL（1800s）、bigWig 清理间隔（600s）为代码内常量，未开放配置；如需调整需改 `cache_service.py`。

端口（默认已规划，一般无需改）：
- rice-mut：`FRONTEND_PORT=8000`、`BACKEND_PORT=8001`
- rice-reg：`FRONTEND_PORT=7000`、`BACKEND_PORT=7001`

### 3.4 启动 / 停止 / 日志

```bash
# 以 rice-mut 为例（rice-reg 同样操作）
cd rice-mut

# 1) 自检（推荐）：检查 .env、端口占用、Python 解释器、模型/基因组文件
bash tools/startup_self_check.sh

# 2) 启动后端（后台运行，写 pid + nohup 日志）
bash backend/run_backend.sh

# 3) 启动前端
bash frontend/run_frontend.sh

# 停止
bash backend/stop_backend.sh
bash frontend/stop_frontend.sh

# 日志
tail -f backend/logs/backend.nohup.log
tail -f frontend/logs/frontend.nohup.log
```

浏览器访问：rice-mut → `http://<host>:8000`，rice-reg → `http://<host>:7000`。

### 3.5 验证

```bash
curl http://127.0.0.1:8001/health   # rice-mut 后端
curl http://127.0.0.1:7001/health   # rice-reg 后端
bash test/test_api.sh               # 完整 API 冒烟测试（需后端已启动）
```

### 3.6 常用 API（两项目）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 + 模型元信息 |
| `/genomes` | GET | 基因组列表（内置 + 上传） |
| `/genomes/{id}/chromosomes` | GET | 染色体列表（chrNN 风格） |
| `/uploadFasta` | POST | 上传自定义基因组 FASTA（自动建 .fai） |
| `/uploadGff` | POST | 上传注释 GFF 并附加到自定义基因组 |
| `/predict` | POST | 参考序列预测 |
| `/predict/snv`、`/predict/snv/stat` | POST | rice-mut：单碱基突变对比 + 区域差异统计 |
| `/predict/bar` | POST | rice-mut：区域平均表达量（前端柱状图） |
| `/predict/rice-reg` | POST | rice-reg：ATAC 条件预测 |

请求/响应示例见各项目 README。

---

## 4. 交付注意事项（打包前必读）

1. **安全风险**：两个后端均将整个根文件系统挂载为静态服务
   `app.mount("/static-files", StaticFiles(directory="/"))` —— 这是为了让 IGV.js 能加载绝对路径下的 bigWig/FASTA。**部署时若端口暴露到公网，任何访问者可通过 `/static-files/etc/passwd`、`/static-files/.../rice-mut/.env` 下载服务器任意文件**。上线前应改为白名单目录挂载或加访问鉴权。
2. **打包清单**：`source/`（模型 + 基因组，约 17G）必须随包交付，无法从网络重新获取；`cache/`、`logs/`、`*.pid`、`.nfs*` 属运行时产物，打包前应删除（启动时会自动重建）。
3. **路径耦合**：`.env` 中所有路径为开发机绝对路径，部署机必须逐项修改（见 3.3），建议提前准备路径对照表。
4. **版本锁定**：`requirements.txt` 使用 `>=`，交付前建议 `pip freeze > requirements-lock.txt` 固化验证过的环境。

---

## 5. 常见问题（FAQ）

- **端口被占用 / 与另一项目冲突**：确认 rice-mut 用 8000/8001、rice-reg 用 7000/7001；`tools/startup_self_check.sh` 会检测端口占用。
- **前端页面能开但 IGV 无数据**：检查后端是否启动、`BACKEND_API_URL` 是否正确、防火墙是否放行后端端口。
- **`flash-attn` 安装失败**：`USE_FLASH_ATTN=false` 可跳过（rice-mut），或按官方文档先装好 CUDA 工具链。
- **上传文件很快被清理**：`UPLOADED_GENOMES_TTL_HOURS` 默认 0.5 小时，闲置上传基因组会被自动删除，属正常行为。
- **前端染色体命名对不上**：前端统一 `chr01`–`chr12`，后端会自动通配实际 FASTA 命名，无需改前端。
