# Genos 适配
## 1. 简介
Genos 是一个专为 DNA 序列分析设计的深度学习模型。目前针对华为昇腾 NPU 和沐曦 MetaX 加速器进行了优化，通过 RESTful API 提供高性能的 DNA 序列嵌入提取和核苷酸级别碱基预测服务。

目录结构如下：

```
Adaptation/
├── Dockerfile.npu      # NPU 版本的 Docker 镜像配置
├── Dockerfile.metax    # MetaX 版本的 Docker 镜像配置
├── genos_server.py     # 主服务器程序，提供 API 接口
└── README.md           # 本文档
```

## 2. 功能特性

- **DNA 序列嵌入提取**：支持多种池化方法（mean、max、last、none）
- **DNA 核苷酸预测**：预测 DNA 序列的下游核苷酸
- **多设备支持**：优先使用 NPU，同时支持 GPU 和 CPU
- **多模型支持**：默认支持 1.2B 和 10B 模型
- **RESTful API**：提供简单易用的 HTTP 接口


## 3. 华为昇腾 NPU

###  环境要求

- 华为昇腾 NPU 设备（如 Atlas 系列）
- Docker 环境
- NPU 驱动已正确安装

###  构建 Docker 镜像

进入包含 Dockerfile 的目录，执行以下命令构建镜像：

```bash
cd Adaptation/
docker build --network=host -t genos-npu-image -f Dockerfile.npu .
```

###  运行 Docker 容器

构建完成后，使用以下命令运行容器：

```bash
docker run --rm -d \
--name genos-npu-server \
--network=host \
--device=/dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--shm-size=2g \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /Path/To/Model/:/AI_models/zhejianglab/ \
-p 8000:8000 \
-it genos-npu-image
```

**参数说明**
- `--device=/dev/davinci0`：指定使用的 NPU 设备（根据实际情况调整）
- `-v /Path/To/Model/:/AI_models/zhejianglab/`：将模型文件目录挂载到容器中
- `-p 8000:8000`：端口映射。如果主机端口 8000 被占用，可调整，例如 `-p 8888:8000`
- `-d`：以后台模式运行容器


## 4. 沐曦 MetaX

###  环境要求
- 沐曦 GPU 设备（如曦云 C500 系列）  
- Docker 环境  
- GPU 驱动已正确安装

###  构建 Docker 镜像

进入包含 Dockerfile 的目录，执行以下命令构建镜像：

```bash
cd Adaptation/
docker build --network=host -t genos-metax-image -f Dockerfile.metax .
```

###  运行 Docker 容器

构建完成后，使用以下命令运行容器：
```bash
docker run --rm -it -d \
--name genos-metax-server \
--network=host \
--device=/dev/dri \
--device=/dev/mxcd \
--privileged=true \
--group-add video \
--device=/dev/mem \
--device=/dev/infiniband \
--security-opt seccomp=unconfined \
--security-opt apparmor=unconfined \
--shm-size '100gb' \
--ulimit memlock=-1 \
-v /Path/To/Model/:/AI_models/zhejianglab/ \
-p 8000:8000 \
genos-metax-image
```
**参数说明**
- `--device=/dev/mxcd --device=/dev/dri`：指定沐曦 GPU 设备（根据实际情况调整）
- `-v /Path/To/Model/:/AI_models/zhejianglab/`：将模型文件目录挂载到容器中
- `-p 8000:8000`：端口映射。如果主机端口 8000 被占用，可调整，例如 `-p 8888:8000`
- `-d`：以后台模式运行容器

## 5. 调用 API 接口

容器启动后，可以通过 HTTP API 调用服务。

### 提取 DNA 序列嵌入

```bash
curl -X POST http://localhost:8000/extract \
-H "Content-Type: application/json" \
-d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "pooling_method": "max"}'
```

### 预测 DNA 序列的下游核苷酸

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "predict_length": 25}'
```

## 6. API 接口文档

###  嵌入提取接口

**URL**：`/extract`  
**方法**：`POST`  
**Content-Type**：`application/json`

**请求参数**：
- `sequence`（必需）：输入的 DNA 序列
- `model_name`（必需）：模型名称（"1.2B" 或 "10B"）
- `pooling_method`（可选）：池化方法（"mean"、"max"、"last"、"none"，默认值："mean"）

**响应示例**：
```json
{
  "success": true,
  "message": "Sequence embedding extraction successful",
  "result": {
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "sequence_length": 24,
    "token_count": 5,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "max",
    "model_type": "flash",
    "device": "npu:0",
    "embedding": [0.123, 0.456, ...]
  }
}
```

###  核苷酸预测接口

**URL**：`/predict`  
**方法**：`POST`  
**Content-Type**：`application/json`  

**请求参数**：
- `sequence`（必需）：输入的 DNA 序列
- `model_name`（必需）：模型名称（"1.2B" 或 "10B"）
- `predict_length`（可选）：要预测的核苷酸数量（默认值：10，最大值：1000）

**响应示例**：
```json
{
  "success": true,
  "message": "Nucleotide prediction successful",
  "result": {
    "original_sequence": "GGATCCGGATCCGGATCCGGATCC",
    "predicted_sequence": "GGATCCGGATCCGGATCCGGATCCATCGATCGATCGATCGAT",
    "predicted_bases": "ATCGATCGATCGATCGAT",
    "predict_length": 16,
    "total_length": 40
  }
}
```

## 7. 自定义配置

### 命令行参数

`genos_server.py` 支持以下命令行参数：

| 参数 | 说明 | 默认值 |
|----------|-------------|---------------|
| `--host` | 服务器监听地址 | 0.0.0.0 |
| `--port` | 服务器监听端口 | 8000 |
| `--force_cpu` | 强制使用 CPU | False |
| `--device` | 指定运行设备（单设备：npu:0、cuda:0、cpu；多设备：逗号分隔，如 "cuda:0,cuda:1" 或 "npu:0,npu:1"） | None |
| `--device_map` | 设备映射策略（auto、balanced、sequential） | None |
| `--memory_ratio` | 内存分配比例 | 0.9 |
| `--model_path_prefix` | 模型存储路径前缀 | /AI_models/zhejianglab/ |
| `--log_level` | 日志级别 | INFO |

### 示例：自定义路径和设备

```bash
docker run --rm -d \
--name genos-npu-server \
--network=host \
--device=/dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--shm-size=2g \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /DW/AI_models/modelscope/hub/models/zhejianglab/:/AI_models/zhejianglab/ \
-it genos-npu-image \
python genos_server.py --device npu:0
```


## 8. 查看日志

可以使用以下命令查看容器日志：

```bash
docker logs ${server_name}
```

## 9. 停止服务

使用以下命令停止并删除容器：

```bash
docker stop ${server_name}
```

## 10. 技术支持

如有任何问题或建议，请提交 issue 或通过 genos@genomics.cn 联系我们。

---