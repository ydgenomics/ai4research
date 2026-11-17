# Debug Log

本文件记录项目开发过程中遇到的问题和解决方案，供后续参考。

---

## 1. ModuleNotFoundError: No module named 'rice_reg'

**日期**: 2026-06-12

**问题**: 启动后端时 `from rice_reg.api import app` 报错 `ModuleNotFoundError: No module named 'rice_reg'`

**原因**: `main.py` 中将项目根目录（`ROOT_DIR`）加入 `sys.path`，但 `rice_reg` 包位于 `backend/rice_reg/`，需要的是 `backend/` 在路径中。

**修复**: 将 `backend/` 目录加入 `sys.path` 而非项目根目录。

```python
# 修改前 (main.py)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# 修改后
BACKEND_DIR = Path(__file__).resolve().parent  # backend/rice_reg/
backend_parent = str(BACKEND_DIR.parent)       # backend/
if backend_parent not in sys.path:
    sys.path.insert(0, backend_parent)
```

**涉及文件**: `backend/rice_reg/main.py`

---

## 2. ModuleNotFoundError: No module named 'model'

**日期**: 2026-06-12

**问题**: 修复问题 1 后，`core/rice_reg.py` 中 `from model.config import effective_inference_batch_size` 报错 `ModuleNotFoundError: No module named 'model'`

**原因**: `model` 包位于 `backend/rice_reg/core/model/`，其父目录 `core/` 不在 `sys.path` 中。原始 `inference2.ipynb` 在 notebook 环境中运行，当前工作目录就是 `model/` 所在目录，所以能直接 `import model`。但作为包运行时路径不同。

**修复**: 在 `core/__init__.py` 中将 `core/` 目录加入 `sys.path`。

```python
# 修改前 (core/__init__.py)
from .rice_reg import RiceRegPredictor

# 修改后
import sys
from pathlib import Path
_core_dir = Path(__file__).resolve().parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))
from .rice_reg import RiceRegPredictor
```

**涉及文件**: `backend/rice_reg/core/__init__.py`

**经验**: 当从 notebook 移植代码到包结构时，注意 `sys.path` 的差异。Notebook 的当前工作目录通常就是模型文件所在目录，但包结构需要显式添加路径。

---

## 3. ImportError: FlashAttention2 not installed

**日期**: 2026-06-12

**问题**: 模型加载时报错 `ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed.`

**原因**: `load_pretrained.py` 中硬编码了 `attn_implementation="flash_attention_2"`，但环境中未安装 `flash-attn` 包。

**修复**: 将 `attn_implementation` 改为 `"eager"`，使用 PyTorch 原生 attention 实现。

```python
# 修改前 (load_pretrained.py)
model_kwargs = dict(
    trust_remote_code=True,
    revision="main",
    attn_implementation="flash_attention_2",
)

# 修改后
model_kwargs = dict(
    trust_remote_code=True,
    revision="main",
    attn_implementation="eager",
)
```

**涉及文件**: `backend/rice_reg/core/model/load_pretrained.py`

**备选方案**: 安装 `flash-attn`（`pip install flash-attn --no-build-isolation`），但需要编译，耗时较长。使用 `eager` 模式在推理场景下功能完全等价，只是速度略慢。

---

## 4. 日志文件未清空导致误判

**日期**: 2026-06-12

**问题**: 重启后端后查看日志，看到的仍然是旧错误信息，误以为修复未生效。

**原因**: `run_backend.sh` 使用 `>>` 追加模式写入日志，不会清空旧内容。

**修复**: 重启前手动清空日志文件。

```bash
> backend/logs/backend.nohup.log   # 清空日志
bash backend/run_backend.sh        # 重新启动
```

**经验**: 调试时养成先清空日志再启动的习惯，或修改启动脚本在启动时自动清空日志。

---

## 5. 启动脚本日志追加 vs 覆盖

**日期**: 2026-06-12

**问题**: 多次重启后日志文件混杂了多次运行的输出，难以区分哪次是最近的。

**修复建议**: 考虑在 `run_backend.sh` 和 `run_frontend.sh` 中启动前先清空日志文件：

```bash
# 在 nohup 命令前添加
> "$LOG_FILE"
```

**涉及文件**: `backend/run_backend.sh`, `frontend/run_frontend.sh`
