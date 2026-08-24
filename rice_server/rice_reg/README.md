# Rice-Reg Server

水稻 ATAC-seq 基因表达预测服务。基于深度学习模型，利用 ATAC-seq 信号预测 RNA-seq 表达谱，并通过 IGV.js 可视化展示。

## 项目结构

```
rice-reg-server2/
├── .env                    # 环境配置（基因组、ATAC、模型路径等）
├── .env.example            # 环境配置模板
├── requirements.txt        # 统一依赖清单（前后端共用）
├── README.md               # 本文档
├── backend/
│   ├── main.py             # FastAPI 应用入口
│   ├── api.py              # API 路由定义
│   ├── prediction_service.py  # 预测服务层
│   ├── igv_payload.py      # IGV 数据构建
│   ├── rice_reg/
│   │   ├── core/
│   │   │   ├── rice_reg.py     # 核心推理类 (RiceRegPredictor)
│   │   │   ├── scaling.py      # 数据标准化
│   │   │   ├── model/          # 模型定义（pipeline, encoder, predictor 等）
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── requirements.txt     # 后端依赖
│   ├── run_backend.sh       # 后端启动脚本
│   └── stop_backend.sh      # 后端停止脚本
├── frontend/
│   ├── app.py               # Gradio 前端界面
│   ├── config.py            # 前端配置
│   ├── igv_payload.py       # IGV 辅助函数
│   ├── requirements.txt     # 前端依赖
│   ├── run_frontend.sh      # 前端启动脚本
│   └── stop_frontend.sh     # 前端停止脚本
├── tools/
│   └── startup_self_check.sh # 启动自检脚本
├── docs/
│   ├── IMPLEMENTATION_PLAN.md # 执行方案与设计决策
│   └── DEBUG_LOG.md           # Debug 记录与解决方案
└── test/
    └── test_api.sh           # API 测试脚本
```

## 快速开始

### 1. 环境配置

复制环境变量模板并根据实际路径修改：

```bash
cp .env.example .env
# 编辑 .env 文件，设置正确的路径
```

### 2. 安装依赖

前后端共用同一个 Python 环境，安装根目录的统一依赖清单即可：

```bash
pip install -r requirements.txt
```

### 3. 启动服务

**启动自检**（推荐先运行）：

```bash
bash tools/startup_self_check.sh
```

**启动后端**（端口 7001）：
**启动前端**（端口 7000）：

```bash
bash backend/run_backend.sh
bash frontend/run_frontend.sh
```

### 4. 停止服务

```bash
bash backend/stop_backend.sh
bash frontend/stop_frontend.sh
```

### 5. 查看日志

```bash
tail -f backend/logs/backend.nohup.log
tail -f frontend/logs/frontend.nohup.log
```

## 配置说明

关键配置项（`.env`）：
- `BACKEND_PYTHON_BIN` / `FRONTEND_PYTHON_BIN`: Python 解释器路径（可指向同一个环境）
- `BASE_MODEL_PATH`: 基础模型路径（HuggingFace 格式）
- `CHECKPOINT_PATH`: 模型 checkpoint (.safetensors)
- `GENOME_*_FASTA` / `GENOME_*_FAI` / `GENOME_*_GFF`: 参考基因组文件
- `ATAC_PATH_*`: 内置 ATAC bigWig 文件路径

### 2. 启动服务

**启动自检**（推荐先运行）：

```bash
bash tools/startup_self_check.sh
```

**启动后端**（端口 7001）：
**启动前端**（端口 7000）：

```bash
bash backend/run_backend.sh
bash frontend/run_frontend.sh
```

### 3. 停止服务

```bash
bash backend/stop_backend.sh
bash frontend/stop_frontend.sh
```

### 4. 查看日志

```bash
tail -f backend/logs/backend.nohup.log
tail -f frontend/logs/frontend.nohup.log
```

## API 文档

### `GET /health`

健康检查。

**响应**: `{"status": "ok"}`

### `POST /uploadFile`

上传 ATAC bigWig 文件。

**请求**: `multipart/form-data`，字段名 `file`

**响应**: `{"filename": "...", "path": "..."}`

### `POST /predict/rice-reg`

执行基因表达预测。

**请求体**:

```json
{
  "genome": "MH63RS3",
  "chromosome": "chr1",
  "start": 0,
  "end": 32678,
  "atac_source": "SAM2_MH63_1",
  "uploaded_atac": null
}
```

**参数说明**:
- `genome`: 基因组名称（如 MH63RS3, NIP）
- `chromosome`: 染色体名称（如 chr1-chr12）
- `start`: 起始位置（bp）
- `end`: 结束位置（bp，可选；前端固定窗口模式自动设为 `start + 32678`，留空由后端按 `TARGET_LEN` 自动计算）
- `atac_source`: 内置 ATAC 源名称（与 uploaded_atac 二选一）
- `uploaded_atac`: 用户上传的 ATAC 文件路径（与 atac_source 二选一）

**响应**: IGV.js 可用的 JSON payload，包含 reference、locus 和 tracks（预测结果 bigWig URL）。

## 前端使用

1. 打开浏览器访问 `http://<host>:7000`
2. 选择基因组（MH63RS3 / NIP）
3. 选择染色体（chr1-chr12）
4. 输入起始位置（窗口长度固定为 32678 bp，结束位置自动计算）
5. 选择内置 ATAC 文件或上传自己的 bigWig 文件（二选一）
6. 点击 "Predict" 按钮
7. 在 IGV 浏览器中查看预测结果

## 测试

```bash
# 确保后端已启动
bash backend/run_backend.sh

# 运行 API 测试
bash test/test_api.sh
```

## 依赖

所有依赖统一声明在根目录的 `requirements.txt` 中，前后端共用：

```bash
pip install -r requirements.txt
```

主要依赖：
- Python 3.10+
- PyTorch 2.1+
- FastAPI + Uvicorn
- Gradio 5+
- pyBigWig, pyfaidx, safetensors, transformers

## 架构说明

- **后端**: FastAPI 服务，负责模型加载、推理计算、bigWig 文件生成
- **前端**: Gradio 应用，提供用户交互界面和 IGV.js 可视化
- **推理核心**: 移植自 `inference2.ipynb`，使用 `RiceRegPredictor` 封装
- **模型**: 基于 rice_1B_32k_hf 基础模型 + ATAC encoder，窗口大小 32kb
- **可视化**: IGV.js 2.x，通过 bigWig 文件展示预测的 RNA-seq +/- 链信号

## 扩展指南：添加新基因组和 ATAC 文件

系统支持通过 `.env` 配置灵活添加新的基因组和 ATAC bigWig 文件，无需修改代码。

### 添加新基因组

在 `.env` 中添加三行配置（`FASTA` + `FAI` + `GFF`），`<ID>` 为自定义的基因组标识符：

```bash
# 格式: GENOME_<ID>_FASTA, GENOME_<ID>_FAI, GENOME_<ID>_GFF
GENOME_MYGENOME_FASTA=/path/to/genome.fa
GENOME_MYGENOME_FAI=/path/to/genome.fa.fai
GENOME_MYGENOME_GFF=/path/to/annotation.gff3
```

**说明**：
- `<ID>` 会作为前端下拉框的选项名称显示（如 `MH63RS3`、`NIP`）
- `FASTA`：参考基因组序列文件（必需）
- `FAI`：FASTA 索引文件（推荐，由 `samtools faidx` 生成）
- `GFF`：基因注释文件（可选，用于 IGV 显示基因 track）

**准备 FAI 索引**（如没有）：
```bash
samtools faidx /path/to/genome.fa
```

### 添加内置 ATAC bigWig 文件

在 `.env` 中添加一行配置，`<ID>` 为自定义的 ATAC 标识符：

```bash
# 格式: ATAC_PATH_<ID>
ATAC_PATH_MYATAC_1=/path/to/atac_signal.bw
```

**说明**：
- `<ID>` 会作为前端下拉框的选项名称显示（如 `SAM2_MH63_1`、`SAM2_NIP_1`）
- 支持 `.bw` / `.bigWig` 格式
- 添加后无需任何代码修改，重启前后端即可生效

### 完整示例：同时添加基因组和对应 ATAC

假设要添加一个名为 `ZS97RS3` 的基因组及其 ATAC 数据：

```bash
# 1. 基因组配置
GENOME_ZS97RS3_FASTA=/path/to/ZS97RS3/ZS97.fa
GENOME_ZS97RS3_FAI=/path/to/ZS97RS3/ZS97.fa.fai
GENOME_ZS97RS3_GFF=/path/to/ZS97RS3/ZS97.gff3

# 2. ATAC 配置
ATAC_PATH_SAM2_ZS97RS3_1=/path/to/ATAC_SAM2_ZS97RS3_1.bw
```

添加后重启服务即可在界面中看到新的选项。

### 注意事项

1. **染色体命名一致性**：GFF 注释文件和 ATAC bigWig 中的染色体名称必须与 FASTA 一致（例如都用 `chr1` 或都用 `NC_089035.1`）。不一致会导致 IGV 无法显示基因 track。
2. **ATAC 与基因组的兼容性**：前端会显示所有 ATAC 选项，不限制基因组匹配。预测时后端会验证 ATAC bigWig 中是否包含所选染色体，不兼容时会报错。
3. **重启生效**：修改 `.env` 后需要重启前后端服务：
   ```bash
   bash backend/stop_backend.sh && bash backend/run_backend.sh
   bash frontend/stop_frontend.sh && bash frontend/run_frontend.sh
   ```

## 常见问题

### 启动报错 `ModuleNotFoundError: No module named 'rice_reg'`

**原因**: Python 模块搜索路径未包含 `backend/` 目录。

**解决**: 已修复，`main.py` 会自动将 `backend/` 加入 `sys.path`。如果手动运行 Python 命令，需确保在项目根目录执行，或设置 `PYTHONPATH`。

### 启动报错 `ModuleNotFoundError: No module named 'model'`

**原因**: `core/rice_reg.py` 中 `from model.config import ...` 需要 `core/` 在路径中。

**解决**: 已修复，`core/__init__.py` 会自动处理路径。详情见 `docs/DEBUG_LOG.md`。

### 启动报错 `ImportError: FlashAttention2 ... not installed`

**原因**: 模型加载配置中启用了 Flash Attention 2，但环境中未安装 `flash-attn`。

**解决**: 已修改为使用 PyTorch 原生 attention（`attn_implementation="eager"`）。如需安装 flash-attn：`pip install flash-attn --no-build-isolation`。

### 日志中看到旧错误信息

**原因**: 启动脚本使用 `>>` 追加模式，不会清空旧日志。

**解决**: 重启前手动清空日志：`> backend/logs/backend.nohup.log`，或参考 `docs/DEBUG_LOG.md` 中的详细说明。


