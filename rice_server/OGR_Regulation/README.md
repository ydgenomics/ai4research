# Rice-Reg Server

水稻 ATAC-seq 基因表达预测服务。基于深度学习模型，利用 ATAC-seq 信号预测 RNA-seq 表达谱，并通过 IGV.js 可视化展示。

> **API 调用（本机 + DCS）见 [API.md](API.md)**。

## 项目结构

```
OGR_Regulation/
├── .env                    # 环境配置（基因组、ATAC、模型路径、端口、CACHE_MODE）
├── .env.example            # 环境配置模板
├── requirements.txt        # 统一依赖清单（前后端共用）
├── README.md               # 本文档
├── API.md                  # ★ API 调用文档（本机 + DCS 测试代码）
├── backend/
│   ├── rice_reg/main.py    # FastAPI 入口（网页端 + DCS 子路径同一进程）
│   ├── rice_reg/api.py     # API 路由（网页端内部端点 + DCS 子路径）
│   ├── rice_reg/prediction_service.py  # 预测服务层
│   ├── rice_reg/igv_payload.py  # IGV 数据构建
│   ├── rice_reg/cache_service.py    # 内容寻址缓存 / 自定义基因组注册
│   ├── rice_reg/core/
│   │   ├── rice_reg.py     # 核心推理类 (RiceRegPredictor)
│   │   ├── scaling.py      # 数据标准化
│   │   ├── model/          # 模型定义（pipeline, encoder, predictor 等）
│   │   └── __init__.py
│   ├── requirements.txt     # 后端依赖
│   ├── run_backend.sh       # 后端启动脚本
│   └── stop_backend.sh      # 后端停止脚本
├── frontend/
│   ├── app.py               # Gradio 前端界面（API Key 输入框 + /backend/* 反代）
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

**DCS 子路径**已并入统一后端（`rice_reg/main.py`），网页端与纯 API 同一进程/端口，
**不再需要单独启动 `dcs_adapter.py`**。网关/纯 API 直接调用后端的
`/api/aigress/openai/OGR-Regulation/{Genome,Chromosome,Prediction,RegisterGenome}`。

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
- `BACKEND_PYTHON_BIN` / `FRONTEND_PYTHON_BIN`: Python 解释器路径（可指向同一个环境，部署均为 vllm）
- `BACKEND_PORT` / `FRONTEND_PORT`: 后端 7001 / 前端 7000
- `BASE_MODEL_PATH`: 基础模型路径（HuggingFace 格式，`rice_1B_32k_hf`）
- `CHECKPOINT_PATH`: 模型 checkpoint (`three_sam2.model.safetensors`)
- `GENOME_*_FASTA` / `GENOME_*_FAI` / `GENOME_*_GFF`: 参考基因组文件（MH63 / NIP）
- `ATAC_PATH_*`: 内置 ATAC bigWig 文件路径（`SAM2_MH63_1` / `SAM2_NIP_1`）
- `TARGET_LEN`: 预测窗口长度（默认 32768）
- `CACHE_MODE`: `true` = 懒加载 + 内容寻址缓存（相同窗口复用、跳过 GPU）；`false`（当前部署）= 每次重跑 GPU；两种模式中间 bigWig 均写 `BACKEND_PREDICTION_CACHE`，服务重启自动清空

## API 文档

完整 API 调用说明（本机 + DCS 测试代码）见 **[API.md](API.md)**。

接口一览（网页端前端与纯 API **走同一套 DCS 子路径**，camelCase + 1-based）：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/aigress/openai/OGR-Regulation/Prediction` | POST | ATAC→RNA-seq 表达预测（网页端与纯 API 共用） |
| `/api/aigress/openai/OGR-Regulation/Genome` | POST | 基因组列表 |
| `/api/aigress/openai/OGR-Regulation/Chromosome` | POST | 某基因组染色体列表 |
| `/api/aigress/openai/OGR-Regulation/RegisterGenome` | POST | 按本机路径注册 FASTA / 挂载 GFF |
| `/health` | GET | 健康检查（免鉴权） |
| `/uploadFile` | POST | 上传 ATAC bigWig（multipart，字段名 `file`；网页端内部端点） |

> 网页 Gradio 前端通过 `_call_dcs()` 直接调用上述 DCS 子路径（`/backend/*` 反代），浏览器只连前端端口；
> 纯 API 调用方拿到**同一份**返回（`{model_name, usage, token_cnt, status, message, result}`）。
> 旧版独立 `dcs_adapter.py` 与 body `mode` 分发均已废弃。

## 前端使用

1. 打开浏览器访问 `http://<host>:7000`
2. 在 **「API Key（必填）」** 输入框填写 DCS API Key（后端启用鉴权时必填；前端只把它作为 `Authorization: Bearer <key>` 发送，不读 `.env`）
3. 选择基因组（`MH63` / `NIP`）
4. 选择染色体（chr01–chr12）
5. 输入起始位置（1-based，窗口长度固定 32768 bp，结束位置自动计算）
6. 选择内置 ATAC 文件或上传自己的 bigWig 文件（二选一；两者都给时上传优先）
7. 点击 "Predict" 按钮
8. 在 IGV 浏览器中查看预测结果（浏览器只连前端端口，IGV/静态资源经 `/backend/*` 反代）

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
