# Rice-Reg Server

水稻 ATAC-seq 基因表达预测服务。基于深度学习模型，利用 ATAC-seq 信号预测 RNA-seq 表达谱，并通过 IGV.js 可视化展示。

> **API 调用（本机 + DCS）见 [API.md](API.md)**。

## 项目结构

```
rice-reg-server2/
├── .env                    # 环境配置（基因组、ATAC、模型路径等）
├── .env.example            # 环境配置模板
├── requirements.txt        # 统一依赖清单（前后端共用）
├── README.md               # 本文档
├── API.md                  # ★ API 调用文档（本机 + DCS 测试代码）
├── backend/
│   ├── main.py             # FastAPI 应用入口
│   ├── api.py              # API 路由定义（网页版）
│   ├── dcs_adapter.py      # ★ DCS 适配层（单入口 + mode 分发）
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

**启动网页版后端 + 前端**（后端 7001 / 前端 7000）：

```bash
bash backend/run_backend.sh
bash frontend/run_frontend.sh
```

**启动 DCS 适配层**（供 DCS 网关转发，独立于网页版进程；建议用 `.env` 的 `BACKEND_PYTHON_BIN` 解释器）：

```bash
cd backend && python dcs_adapter.py
```

### 4. 停止服务

```bash
bash backend/stop_backend.sh && bash frontend/stop_frontend.sh
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

## API 文档

完整 API 调用说明（本机 + DCS 两套测试代码）见 **[API.md](API.md)**。

网页版后端简要接口一览：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/uploadFile` | POST | 上传 ATAC bigWig 文件（multipart，字段名 `file`） |
| `/predict/rice-reg` | POST | 执行基因表达预测 |

> 网页版请求/响应示例见 `test/test_api.sh`；DCS 适配层（单入口 + mode 分发）的完整调用见 [API.md](API.md)。

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
