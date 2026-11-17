# 水稻 ATAC 预测服务（Rice-Reg Server）执行方案

## 1. 项目概述

基于 Genos-Reg Server 的架构，构建水稻（Rice）ATAC-seq → RNA-seq 表达预测服务。核心推理逻辑遵循 `inference2.ipynb`，前后端架构参考 `genos-reg-server`。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     rice-reg-server2/                        │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Frontend    │    │   Backend    │    │   Model      │   │
│  │  (Gradio +    │───▶│  (FastAPI)   │───▶│  (PyTorch)   │   │
│  │   FastAPI)    │    │              │    │              │   │
│  │   port 8000   │◀───│   port 8001  │    │  inference2  │   │
│  └──────────────┘    └──────────────┘    │  .ipynb 核心  │   │
│                                          └──────────────┘   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  .env (所有配置)                                      │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 前端框架 | Gradio + FastAPI | 与 genos-reg-server 一致 |
| 后端框架 | FastAPI + uvicorn | 与 genos-reg-server 一致 |
| 模型推理 | PyTorch + safetensors | 按 inference2.ipynb 实现 |
| 可视化 | IGV.js (igv 2.x) | 展示预测 bigWig 轨道 |
| 数据格式 | bigWig / FASTA / GFF3 | 水稻参考基因组和 ATAC 数据 |
| Python 解释器 | `/root/miniconda3/envs/rice_reg/bin/python` | 通过 .env 指定 |

---

## 3. 项目目录结构

```
rice-reg-server2/
├── .env                          # 所有配置（模型路径、ATAC路径、基因组路径等）
├── .env.example                  # 配置模板
├── README.md                     # 项目文档
├── backend/
│   ├── run_backend.sh            # 后端启动脚本
│   ├── stop_backend.sh           # 后端停止脚本
│   ├── requirements.txt          # Python 依赖
│   └── rice_reg/
│       ├── __init__.py
│       ├── main.py               # 后端入口（uvicorn）
│       ├── api.py                # API 路由定义
│       ├── prediction_service.py # 预测核心逻辑
│       ├── igv_payload.py        # IGV payload 构建
│       └── core/
│           ├── __init__.py
│           ├── rice_reg.py       # ★ 模型定义 + 推理逻辑（从 inference2.ipynb 移植）
│           └── scaling.py        # 反归一化工具函数
├── frontend/
│   ├── run_frontend.sh           # 前端启动脚本
│   ├── stop_frontend.sh          # 前端停止脚本
│   ├── requirements.txt          # 前端依赖
│   ├── config.py                 # 前端配置（从 .env 读取）
│   ├── app.py                    # Gradio 界面定义
│   ├── igv_payload.py            # IGV payload 构建
│   └── static/                   # 静态资源
│       └── user.svg
├── tools/
│   └── startup_self_check.sh     # 启动自检脚本
├── docs/
│   └── IMPLEMENTATION_PLAN.md    # 本执行方案
└── test/
    └── test_api.sh               # API 测试脚本
```

---

## 4. 详细实施步骤

### Step 1: 移植模型推理核心（`backend/rice_reg/core/rice_reg.py`）

**目标**：将 `inference2.ipynb` 中的模型定义和推理逻辑封装为可复用的 Python 类。

**关键类**：

1. **`ATAC_Encoder`** — ATAC 信号 1D CNN 编码器（与 inference2.ipynb 一致）
2. **`InferenceMultiModalPredictor`** — 多模态融合预测模型（与 inference2.ipynb 一致）
3. **`RiceRegPredictor`** — 高层封装类，提供 `predict()` 接口

**`RiceRegPredictor.predict()` 接口**：

```python
def predict(self, chrom: str, start: int, end: int, atac_path: str) -> dict:
    """
    对指定窗口执行推理
    
    Args:
        chrom: 染色体名称 (如 "chr1")
        start: 起始位置 (0-based)
        end: 结束位置
        atac_path: ATAC bigWig 文件路径
    
    Returns:
        {
            "sequence": str,
            "position": (chrom, start, end),
            "values": {
                "RNA-seq_+": {sample_name: np.ndarray},
                "RNA-seq_-": {sample_name: np.ndarray},
            }
        }
    """
```

**关键参数**（从 inference2.ipynb 提取）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | 来自 .env | HuggingFace 基础模型目录 |
| `CKPT_PATH` | 来自 .env | 训练好的 checkpoint |
| `PREDICTOR_TYPE` | `fusion` | 预测器类型 |
| `DATASET_TYPE` | `v0` | 数据集类型 |
| `ATAC_ENCODER_OUTPUT_DIM` | 1024 | ATAC 编码器输出维度 |
| `TARGET_LEN` | 32000 | 窗口大小 |
| `TRACK_MEAN_PLUS` | 2.83 | 反归一化固定均值 |
| `TRACK_MEAN_MINUS` | 2.85 | 反归一化固定均值 |
| `MODEL_TORCH_DTYPE` | `bfloat16` | 模型精度 |

### Step 2: 构建后端 API（`backend/rice_reg/`）

参考 `genos-reg-server/backend/genos_reg/` 的结构：

#### 2.1 `main.py` — 后端入口

- 加载 `.env` 配置
- 启动时初始化 `RiceRegPredictor`
- 启动 uvicorn

#### 2.2 `api.py` — API 路由

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/uploadFile` | POST | 上传 ATAC bigWig 文件 |
| `/predict` | POST | 执行预测（接收 genome, atac_inputs_state, locus, left/right bp） |
| `/predict/rice-reg` | POST | 同上，显式路由 |

**`PredictRequest` 模型**：

```python
class PredictRequest(BaseModel):
    genome: str                        # 基因组名称（如 "MH63RS3"）
    chromosome: str                    # 染色体（如 "chr1"）
    start: int                         # 起始位置
    end: Optional[int] = None          # 结束位置（可选，留空自动计算）
    atac_source: Optional[str] = None  # 内置 ATAC ID（如 "SAM2_MH63_1"），与 uploaded_atac 二选一
    uploaded_atac: Optional[str] = None  # 用户上传的 bigWig 后端路径，与 atac_source 二选一
```

#### 2.3 `prediction_service.py` — 预测逻辑

- `init_predictor()` — 初始化模型（启动时调用）
- `require_predictor()` — 获取已初始化的 predictor
- `release_predictor()` — 释放模型和 GPU 缓存
- `run_prediction_core()` — 核心预测流程：
  1. 解析 ATAC 源（`uploaded_atac` 优先，否则用 `atac_source`）
  2. 校验 ATAC bigWig 文件存在且可读
  3. 计算模型输入窗口（基于 start/end 和 TARGET_LEN）
  4. 读取 FASTA 序列 + ATAC 信号
  5. 调用 `predictor.predict()` 执行推理
  6. 将预测结果保存为 bigWig 文件（+链和-链各一个）
  7. 返回 bigWig 文件路径供 IGV 加载

#### 2.4 `igv_payload.py` — IGV 数据构建

- `build_prediction_payload()` — 构建 IGV.js 可渲染的 payload
- `signal_to_features()` — 将 numpy 数组转为 IGV bedgraph features
- `read_bw_as_features()` — 读取 bigWig 转为 features（用于 ATAC 轨道展示）

### Step 3: 构建前端界面（`frontend/`）

参考 `genos-reg-server/frontend/` 的结构。

#### 3.1 用户交互界面（Gradio）

**布局**：

```
┌──────────────────────────────────────────────────────┐
│  Rice-Reg: ATAC-conditioned RNA Expression Prediction │
├──────────────────────────────────────────────────────┤
│  Genome:        [下拉单选: MH63RS3 / NIP / ...]       │
│  Chromosome:    [下拉单选: chr1 ~ chr12]              │
│  Start:         [输入框]                              │
│  End:           [输入框] (可选，留空自动计算)           │
│  ATAC-seq File: [下拉单选: 内置 ATAC 列表]             │
│                  (随 Genome 切换自动过滤)               │
│  Upload ATAC:   [单文件上传: .bw/.bigWig]             │
│                  (上传后自动替换下拉选择)               │
│                                                      │
│  [Predict RNA-seq Coverage Tracks]                   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           IGV.js 可视化面板                    │   │
│  │  - 参考基因组轨道                              │   │
│  │  - ATAC 信号轨道                               │   │
│  │  - 预测 RNA-seq (+链) 轨道                     │   │
│  │  - 预测 RNA-seq (-链) 轨道                     │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

#### 3.2 前端配置（`config.py`）

从 `.env` 读取：

```python
# 基因组配置（每个基因组关联其 FASTA/FAI/GFF 路径）
GENOME_CONFIGS = {
    "MH63RS3": {
        "fasta": "/mnt/rice/default/Workspace/yangdong/xuyu/atac/ref/MH63RS3/MH63.fa",
        "fai": "/mnt/rice/default/Workspace/xuyu/atac/ref/MH63RS3/MH63.fa.fai",
        "gff": "/mnt/rice/default/Workspace/xuyu/atac/ref/MH63RS3/MH63.gff3",
        # 该基因组兼容的内置 ATAC 列表
        "atac_options": ["SAM2_MH63_1"],
    },
    "NIP": {
        "fasta": "/mnt/rice/default/Workspace/xuyu/atac/ref/NIP/NIP.fa",
        "fai": "/mnt/rice/default/Workspace/xuyu/atac/ref/NIP/NIP.fa.fai",
        "gff": "/mnt/rice/default/Workspace/yangdong/rice_reg/ref/GCF_034140825.1.gff",
        "atac_options": ["SAM2_NIP_1"],
    },
}

# ATAC 内置路径（从 .env 的 ATAC_PATH_* 读取）
ATAC_SIGNAL_PATHS = {
    "SAM2_MH63_1": "/mnt/rice/default/Workspace/yangdong/xuyu/atac/ATAC/10.process/ATAC_SAM2_MH63_1/ATAC_SAM2_MH63_1.MH63RS2.q30.bin5.CPM.bw",
    "SAM2_NIP_1": "/mnt/rice/default/Workspace/yangdong/xuyu/atac/ATAC/10.process/ATAC_SAM2_NIP_1/ATAC_SAM2_NIP_1.NIP.q30.bin5.CPM.bw",
}
```

#### 3.3 前端交互逻辑

1. **基因组选择**（下拉单选）→ 切换时自动更新：
   - IGV 参考基因组配置
   - ATAC 内置下拉列表（只显示与该基因组兼容的选项）
   - 清空已选的 ATAC 文件（包括下拉选择和上传文件）
2. **染色体选择**（下拉单选）→ chr1 ~ chr12
3. **起始/结束位置** → 用户输入，结束可选（留空自动计算）
4. **ATAC 文件选择**（下拉单选 + 上传互斥）→ 详见下方 3.4
5. **Predict 按钮** → 调用后端 API → 渲染 IGV

#### 3.4 ATAC 源选择逻辑（核心设计）

**基本原则**：内置 ATAC 下拉选择 与 用户上传 bigWig **互斥**，任何时候只有一个 ATAC 源参与预测。

**状态定义**：

```
ATAC 源状态 = 以下三种之一：
  A) 未选择任何 ATAC（初始状态）
  B) 选择了内置 ATAC（下拉单选，如 "SAM2_MH63_1"）
  C) 上传了用户 bigWig（单文件上传）
```

**交互规则**：

| 用户操作 | 触发行为 |
|----------|----------|
| 切换基因组 | 清空 ATAC 选择状态 → 回到 A |
| 下拉选择内置 ATAC | 如果之前有上传文件，自动清除上传 → 进入 B |
| 上传 bigWig 文件 | 如果之前有下拉选择，自动清除选择 → 进入 C |
| 点击 Predict | 根据当前 ATAC 源状态决定使用哪个文件 |

**前端 UI 行为**：

- 下拉单选组件：`value` 在用户上传时自动置空
- 上传组件：`value` 在下拉选择时自动置空
- 状态提示：显示当前 ATAC 源类型（"内置: SAM2_MH63_1" 或 "用户上传: xxx.bw"）

**后端接收的请求格式**：

```python
class PredictRequest(BaseModel):
    genome: str
    chromosome: str
    start: int
    end: Optional[int] = None
    atac_source: Optional[str] = None       # 内置 ATAC ID（如 "SAM2_MH63_1"），与 uploaded_atac 二选一
    uploaded_atac: Optional[str] = None      # 用户上传的 bigWig 后端路径，与 atac_source 二选一
```

**后端 ATAC 解析逻辑**：

```python
def resolve_atac_path(
    atac_source: Optional[str],
    uploaded_atac: Optional[str],
    atac_signal_paths: dict,
) -> str:
    """
    解析最终使用的 ATAC bigWig 路径。
    优先级: uploaded_atac > atac_source
    """
    if uploaded_atac:
        # 用户上传优先
        if not os.path.exists(uploaded_atac):
            raise FileNotFoundError(f"Uploaded ATAC not found: {uploaded_atac}")
        return uploaded_atac
    
    if atac_source:
        # 内置 ATAC
        path = atac_signal_paths.get(atac_source)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Built-in ATAC '{atac_source}' not found")
        return path
    
    raise ValueError("No ATAC source provided (either atac_source or uploaded_atac required)")
```

**基因组与 ATAC 的兼容性校验**：

```
用户选择 Genome=MH63RS3 时:
  - 内置 ATAC 下拉只显示: ["SAM2_MH63_1"]（因为 SAM2_MH63_1 是 MH63 的 ATAC）
  - 用户上传 bigWig: 不做限制，但后端会在预测时校验 bigWig 的染色体名与 FASTA 一致

用户选择 Genome=NIP 时:
  - 内置 ATAC 下拉只显示: ["SAM2_NIP_1"]
  - 用户上传 bigWig: 同上
```

### Step 4: 配置 `.env` 文件

```bash
# ===== 服务配置 =====
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=8000
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8001
BACKEND_API_URL=http://127.0.0.1:8001

# ===== Python 解释器 =====
BACKEND_PYTHON_BIN=/root/miniconda3/envs/rice_reg/bin/python
FRONTEND_PYTHON_BIN=/root/miniconda3/envs/rice_reg/bin/python

# ===== 模型路径 =====
BASE_MODEL_PATH=/mnt/rice/default/Workspace/yangdong/xuyu/atac/foundation/rice_1B_32k_hf
CHECKPOINT_PATH=/mnt/rice/default/Workspace/xuxiaolong/RNAprediction/xuY/model/checkpoint-733600/model.safetensors

# ===== 推理参数 =====
PREDICTOR_TYPE=fusion
DATASET_TYPE=v0
ATAC_ENCODER_OUTPUT_DIM=1024
TARGET_LEN=32000
FIXED_TRACK_MEAN_PLUS=2.83
FIXED_TRACK_MEAN_MINUS=2.85
MODEL_TORCH_DTYPE=bfloat16

# ===== 基因组配置 =====
# 格式: GENOME_<ID>_FASTA, GENOME_<ID>_FAI, GENOME_<ID>_GFF
GENOME_MH63RS3_FASTA=/mnt/rice/default/Workspace/yangdong/xuyu/atac/ref/MH63RS3/MH63.fa
GENOME_MH63RS3_FAI=/mnt/rice/default/Workspace/xuyu/atac/ref/MH63RS3/MH63.fa.fai
GENOME_MH63RS3_GFF=/mnt/rice/default/Workspace/xuyu/atac/ref/MH63RS3/MH63.gff3
GENOME_NIP_FASTA=/mnt/rice/default/Workspace/xuyu/atac/ref/NIP/NIP.fa
GENOME_NIP_FAI=/mnt/rice/default/Workspace/xuyu/atac/ref/NIP/NIP.fa.fai
GENOME_NIP_GFF=/mnt/rice/default/Workspace/yangdong/rice_reg/ref/GCF_034140825.1.gff

# ===== 内置 ATAC 路径 =====
# 格式: ATAC_PATH_<ID>，ID 与 GENOME_CONFIGS 中 atac_options 对应
ATAC_PATH_SAM2_MH63_1=/mnt/rice/default/Workspace/yangdong/xuyu/atac/ATAC/10.process/ATAC_SAM2_MH63_1/ATAC_SAM2_MH63_1.MH63RS2.q30.bin5.CPM.bw
ATAC_PATH_SAM2_NIP_1=/mnt/rice/default/Workspace/yangdong/xuyu/atac/ATAC/10.process/ATAC_SAM2_NIP_1/ATAC_SAM2_NIP_1.NIP.q30.bin5.CPM.bw

# ===== 缓存目录 =====
BACKEND_UPLOADED_ATAC=/mnt/rice/default/Workspace/yangdong/rice_reg/rice-reg-server2/cache/uploaded_atac
BACKEND_PREDICTION_CACHE=/mnt/rice/default/Workspace/yangdong/rice_reg/rice-reg-server2/cache/predictions
```

### Step 5: 启动/停止脚本与日志

参考 genos-reg-server 的设计，前后端各有独立的 `run_*.sh` / `stop_*.sh` 脚本，日志统一写入 `logs/` 目录。

#### 日志目录结构

```
backend/logs/
├── backend.nohup.log    # 后端运行日志（stdout + stderr）
└── backend.pid          # 后端进程 PID

frontend/logs/
├── frontend.nohup.log   # 前端运行日志（stdout + stderr）
└── frontend.pid         # 前端进程 PID
```

#### `backend/run_backend.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

# 加载 .env
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

LOG_DIR="$ROOT_DIR/backend/logs"
LOG_FILE="$LOG_DIR/backend.nohup.log"
PID_FILE="$LOG_DIR/backend.pid"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

# 检查是否已在运行
if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Backend already running (PID=$OLD_PID). Log: $LOG_FILE"
        exit 0
    fi
fi

# 启动后端
nohup "$BACKEND_PYTHON_BIN" backend/rice_reg/main.py >> "$LOG_FILE" 2>&1 &
PID=$!

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[ERROR] Backend failed to start. Check log: $LOG_FILE"
    exit 1
fi

echo "$PID" > "$PID_FILE"
echo "Backend started (PID=$PID). Log: $LOG_FILE"
```

#### `backend/stop_backend.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/backend/logs"
LOG_FILE="$LOG_DIR/backend.nohup.log"
PID_FILE="$LOG_DIR/backend.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "Backend not running (no PID file)."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
    rm -f "$PID_FILE"
    exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping backend (PID=$PID)..."
    kill "$PID"
    # 等待优雅退出（最多 10 秒）
    for _ in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Grace period exceeded, sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Backend stopped."
else
    echo "Backend not running (stale PID)."
fi
rm -f "$PID_FILE"
```

#### `frontend/run_frontend.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

LOG_DIR="$ROOT_DIR/frontend/logs"
LOG_FILE="$LOG_DIR/frontend.nohup.log"
PID_FILE="$LOG_DIR/frontend.pid"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Frontend already running (PID=$OLD_PID). Log: $LOG_FILE"
        exit 0
    fi
fi

nohup "$FRONTEND_PYTHON_BIN" frontend/app.py >> "$LOG_FILE" 2>&1 &
PID=$!

sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[ERROR] Frontend failed to start. Check log: $LOG_FILE"
    exit 1
fi

echo "$PID" > "$PID_FILE"
echo "Frontend started (PID=$PID). Log: $LOG_FILE"
```

#### `frontend/stop_frontend.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/frontend/logs"
LOG_FILE="$LOG_DIR/frontend.nohup.log"
PID_FILE="$LOG_DIR/frontend.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "Frontend not running (no PID file)."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
    rm -f "$PID_FILE"
    exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping frontend (PID=$PID)..."
    kill "$PID"
    for _ in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Grace period exceeded, sending SIGKILL..."
        kill -9 "$PID"
    fi
    echo "Frontend stopped."
else
    echo "Frontend not running (stale PID)."
fi
rm -f "$PID_FILE"
```

#### 日常使用方式

```bash
# 启动（先后端，再前端）
cd /mnt/rice/default/Workspace/yangdong/rice_reg/rice-reg-server2
bash backend/run_backend.sh      # 启动后端 → backend/logs/backend.nohup.log
bash frontend/run_frontend.sh    # 启动前端 → frontend/logs/frontend.nohup.log

# 查看日志
tail -f backend/logs/backend.nohup.log
tail -f frontend/logs/frontend.nohup.log

# 停止（先前端，再后端）
bash frontend/stop_frontend.sh
bash backend/stop_backend.sh

# 重启
bash backend/stop_backend.sh && bash backend/run_backend.sh
```

### Step 6: 测试脚本

#### `tools/startup_self_check.sh`

检查项：
- 端口 8000/8001 是否可用
- 模型文件是否存在
- 参考基因组文件是否存在
- ATAC bigWig 文件是否存在
- Python 解释器是否可用

#### `test/test_api.sh`

测试 API 端点：
- `GET /health` — 健康检查
- `POST /predict/rice-reg` — 预测请求

---

## 5. 与 inference2.ipynb 的关键差异

| 项目 | inference2.ipynb | Rice-Reg Server |
|------|------------------|-----------------|
| 窗口生成 | 全基因组滑动窗口 | 从前端获取具体 locus |
| 输出格式 | CSV / Pickle | bigWig 格式（IGV 展示） |
| 输入方式 | 代码中硬编码路径 | 前端交互选择 |
| 运行模式 | 一次性脚本 | 持久化服务 |
| 多 ATAC 支持 | 单一样本 | 多 ATAC 源并行预测 |

### 输出格式改造

推理结果需要转为 bigWig 格式以便 IGV 展示：

```python
def save_prediction_as_bigwig(
    chrom: str, start: int, end: int,
    pred_values: np.ndarray,
    output_path: str
):
    """将预测结果保存为 bigWig 文件"""
    import pyBigWig
    bw = pyBigWig.open(output_path, "w")
    # 定义染色体长度（从参考基因组获取）
    chrom_lengths = get_chrom_lengths(genome_fasta)
    bw.addHeader(list(chrom_lengths.items()))
    # 写入预测值
    bw.addEntries([chrom], [start], values=pred_values, span=1, step=1)
    bw.close()
```

---

## 6. 已确认的设计决策

以下是根据你的反馈确认后的最终设计：

### ✅ 6.1 基因组与 ATAC 对齐

**决策**：一次只选一个基因组，ATAC 源也一次只有一个（内置下拉单选 与 用户上传互斥）。

- 前端：Genome 下拉单选，ATAC-seq File 下拉单选，Upload ATAC 单文件上传
- 基因组切换时，ATAC 内置列表自动过滤为与该基因组兼容的选项
- **ATAC 源互斥规则**：
  - 用户下拉选择内置 ATAC → 自动清除已上传的文件
  - 用户上传 bigWig → 自动清除下拉选择
  - 任何时候只有一个 ATAC 源参与预测
  - 后端解析优先级：`uploaded_atac` > `atac_source`
- 后端根据所选基因组 ID 查找对应的 FASTA 文件进行序列读取

### ✅ 6.2 染色体名称格式

**决策**：前端统一显示 `chr1 ~ chr12`，后端根据 FASTA 实际名称映射。

- 前端染色体下拉固定为 `["chr1", "chr2", ..., "chr12"]`
- 后端在读取 FASTA 时，如果 FASTA 中实际名称为 `1`、`Chr1` 等，自动做标准化映射
- 映射逻辑：去掉 `chr`/`Chr`/`CHR` 前缀后比较数字部分

### ✅ 6.3 结束位置自动计算

**决策**：留空则默认 `start + TARGET_LEN`（32000 bp）。

- 结束位置为空 → 显示窗口 = `[start, start + TARGET_LEN]`
- 结束位置 < start + TARGET_LEN → 自动扩展到 TARGET_LEN
- 结束位置 - start > PREDICTION_MAX_SPAN（500000 bp）→ 自动裁剪

### ✅ 6.4 输出 bigWig 文件管理

**决策**：缓存到 `cache/predictions/`，启动时清理。

- 预测结果保存为 bigWig 文件，路径：`cache/predictions/{sample}_{chrom}_{start}_{end}_{strand}.bw`
- 后端启动时清空 `cache/predictions/` 目录
- 前端 IGV 通过 `/gradio_api/file=` URL 直接加载这些临时 bigWig 文件

### ✅ 6.5 模型文件拷贝

**决策**：直接拷贝 `test/model/` 到 `backend/rice_reg/core/model/`，**不做任何修改**。

- 将 `/mnt/rice/default/Workspace/yangdong/rice_reg/test/model/` 整个目录拷贝到 `backend/rice_reg/core/model/`
- 保持所有文件内容、import 路径、类名、函数签名完全不变
- 后续如果需要修改模型参数或逻辑，必须先和用户确认

---

## 7. 开发路线图

| 阶段 | 任务 | 预计工时 |
|------|------|----------|
| **Phase 1** | 移植模型推理核心（rice_reg.py） | 2-3天 |
| **Phase 2** | 构建后端 API（main.py, api.py, prediction_service.py） | 2天 |
| **Phase 3** | 构建前端界面（app.py, config.py） | 2-3天 |
| **Phase 4** | IGV 集成与 bigWig 输出 | 1-2天 |
| **Phase 5** | 配置与启动脚本（.env, run_*.sh） | 0.5天 |
| **Phase 6** | 测试与文档 | 1天 |
| **合计** | | **~10天** |

---

## 8. 风险与注意事项

1. **GPU 显存**：模型推理需要 GPU，确保服务器有足够的显存（建议 24GB+）
2. **模型加载时间**：1B 参数模型加载约需 30-60 秒，启动时预加载
3. **并发请求**：当前设计为单例 predictor，并发请求需排队
4. **文件路径**：所有路径通过 `.env` 配置，避免硬编码
5. **IGV.js 版本**：使用 igv 2.x 版本，与 genos-reg-server 保持一致
