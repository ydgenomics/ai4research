# OGR-Prediction API 测试指南

统一后端 `backend/rice_mutation/main.py`（:8001）同时承载网页内部端点与 DCS 子路径，
模型加载完（约 30–60s）`/health` 返回 `ok` 即可调用；无需网页前端，直接 curl 即可。

---

## 1. 启动服务

```bash
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/OGR_Prediction
/root/miniconda3/envs/vllm/bin/python backend/rice_mutation/main.py      # :8001
# 或一键（含网关）：bash ../dcs_gateway/start_all.sh ogr_prediction
```

### 1.1 DCS 平台冒烟

```bash
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/OGR/health | python3 -m json.tool
curl -s -X POST "https://www.dcs.cloud/api/aigress/openai/OGR/OGR-Prediction/Prediction" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{"Genome":"osa1_r7","Chromosome":"chr01","Start":20716774,"Format":"Mean"}'
```

> 若提示 `address already in use`，说明 8001 已有服务，先 `kill $(cat backend/logs/backend.pid)` 或用 `fuser -k 8001/tcp` 清理。

---

## 2. 健康检查

```bash
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
```

返回示例：
```json
{
    "status": "ok",
    "predictor_initialized": true,
    "genomes": ["osa1_r7"]
}
```

---

## 3. 参考序列表达预测（核心推理）

### 3.1 Full 格式（逐碱基数组，默认）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Prediction \
  -H "Content-Type: application/json" \
  -d '{"Genome":"osa1_r7","Chromosome":"chr01","Start":20716774}' | python3 -m json.tool | head -30
```

### 3.2 Mean 格式（每轨道标量均值，响应最小、最常用）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/Prediction \
  -H "Content-Type: application/json" \
  -d '{"Genome":"osa1_r7","Chromosome":"chr01","Start":20716774,"Format":"Mean"}' | python3 -m json.tool
```

实测响应（节选）：
```json
{
    "model_name": "OGR",
    "usage": {"prompt_tokens": 32768, "completion_tokens": 1},
    "token_cnt": 32769,
    "status": 200,
    "message": "Prediction completed in 2.7s",
    "result": {
        "model": "OGR-Prediction",
        "Genome": "osa1_r7",
        "Chromosome": "Chr1",
        "Start": 20716774,
        "window_start": 20716773,
        "window_len": 32768,
        "Format": "Mean",
        "values": {"RNA-seq": {"Leaf": 0.599956}},
        "igv_payload": {"...": "..."},
        "ref_id": "ref_1730000000000"
    }
}
```

> 不再有 `downsample` / `end` 参数：窗口由 `Start` 居中到 32768 bp；`Format` 仅 `Full` / `Mean`。

---

## 4. SNV 变异预测（双轨对比）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/OGR-Prediction/SNV \
  -H "Content-Type: application/json" \
  -d '{"Genome":"osa1_r7","Chromosome":"chr01","Start":20716774,"SNV_position":20731844,"SNV_base":"T","Format":"Mean"}' | python3 -m json.tool | head -50
```

> 响应 `result` 含 `ref_values` / `mut_values` 双轨 + `SNV_position` / `ref_base` / `SNV_base` / `snv_id`。

---

## 5. 响应结构说明

```json
{
    "model_name": "OGR",
    "usage": {
        "prompt_tokens": 32768,      // 计费用：输入窗口碱基数
        "completion_tokens": 32768   // 计费用：输出数组元素总数
    },
    "token_cnt": 65536,              // prompt + completion
    "status": 200,
    "message": "Prediction completed in 2.7s",
    "result": {
        "model": "OGR-Prediction",
        "Genome": "osa1_r7",
        "Chromosome": "Chr1",
        "Start": 20716774,           // 1-based
        "window_start": 20716773,    // 0-based
        "window_len": 32768,
        "Format": "Full",           // Full | Mean
        "values": {
            "RNA-seq": {
                "Leaf": [0.008057, 0.007874, ...]  // Full 为逐碱基数组；Mean 为标量
            }
        },
        "igv_payload": {"...": "..."},
        "ref_id": "ref_1730000000000"
    }
}
```

## 6. 参数速查

| 参数 | 必填 | 说明 |
|------|------|------|
| `Start` | ✅ | 1-based 起始位点（缺失返回 400），后端居中到 32768 bp 窗口 |
| `Format` | ❌ | `Full`（默认）/ `Mean` |
| `Chromosome` | ❌ | 默认 `chr01`，`chr01`/`Chr1`/`1` 均可（自动通配 FASTA 实际名） |
| `Genome` | ❌ | 默认唯一基因组 `osa1_r7` |
| `biosample_names` | ❌ | 逗号分隔或数组；须用**内部名**（如 `NIP_leaf`），非展示名 |
| `fasta_path` / `gff_path` | ❌ | 自定义基因组：传本机绝对路径即时懒注册（无需先 RegisterGenome） |
| `SNV_position` | SNV 时 ✅ | 1-based 变异位点（须在窗口内） |
| `SNV_base` | SNV 时 ✅ | 目标碱基 `A`/`C`/`G`/`T` |

**错误语义**：参数校验错误 → **400**（英文 usage）；鉴权失败 → **401**；文件不存在 → **404**；执行错误 → **500**。

---

## 7. 运行自动化测试

项目附带 API 自动化测试脚本：

```bash
# 网页/后端内部端点测试（/health、/genomes、/predict、/predict/snv、/predict/bar 等）
bash test/test_api.sh

# 前端测试
bash test/test_frontend.sh

# 运行全部测试
bash test/run_all_tests.sh
```

> **注意**：统一后端（`backend/rice_mutation/main.py`，默认 `:8001`）同时承载网页内部端点与 DCS 子路径。
> DCS 风格端点为 `/api/aigress/openai/OGR-Prediction/{Prediction,SNV,Genome,Chromosome,RegisterGenome}`（camelCase、1-based）。
>
> `test_dcs_api.sh` / `test_dcs_adapter.py` 针对**已废弃**的独立 `backend/dcs_adapter.py`（旧 `rice-mut` 路由），
> 仅作历史参考、不再适用于当前服务；如需验证 DCS 子路径，请直接使用 [API.md](../API.md) 或
> [dcs_gateway/API.md](../../dcs_gateway/API.md) 中的 curl 示例。