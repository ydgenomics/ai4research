# Rice-Mutation DCS Adapter API 测试指南

启动 `dcs_adapter.py` 后不需要网页前端，直接通过 curl 即可调用 API 推理。模型加载完（约 30s）health 返回 `ok` 即可调用。

---

## 1. 启动服务

```bash
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/rice_mut
BACKEND_PORT=8001 python backend/dcs_adapter.py
```

dcs
```bash
curl -X POST http://www.dcs.cloud/api/aigress/openai/health
curl -X POST "https://www.dcs.cloud/api/aigress/openai/rice_mut" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{"model":"rice_mut6","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}'
```

> 若提示 `address already in use`，说明 8001 已有服务，先 `kill $(cat backend/logs/backend.pid)` 或用 `fuser -k 8001/tcp` 清理。

---

## 2. 健康检查

```bash
curl -s http://127.0.0.1:8001/api/aigress/openai/health | python3 -m json.tool
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

### 3.1 full 格式（逐碱基数组，约 286KB 响应）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice_mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}' | python3 -m json.tool | head -30
```

### 3.2 mean 格式（每轨道标量均值，响应最小、最常用）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice_mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}' | python3 -m json.tool
```

实测响应：
```json
{
    "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
    "status": 200,
    "message": "参考序列表达预测成功",
    "result": {
        "model": "rice_mut",
        "genome": "osa1_r7",
        "chromosome": "Chr1",
        "position_1based": {"start": 20716774, "end": 20749541},
        "window_len": 32768,
        "output_format": "mean",
        "values": {"RNA-seq": {"Leaf": 0.599956}}
    }
}
```

### 3.3 downsample 格式（均匀降采样到指定点数）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice_mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}' | python3 -m json.tool | head -30
```

---

## 4. SNV 变异预测（双轨对比）

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice_mut/snv \
  -H "Content-Type: application/json" \
  -d '{"model":"rice_mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T"}' | python3 -m json.tool | head -50
```

---

## 5. 响应结构说明

```json
{
    "usage": {
        "prompt_tokens": 32768,      // 计费用：输入窗口碱基数
        "completion_tokens": 32768   // 计费用：输出数组元素总数
    },
    "status": 200,
    "message": "参考序列表达预测成功",
    "result": {
        "model": "rice_mut",
        "genome": "osa1_r7",
        "chromosome": "Chr1",
        "position_1based": {
            "start": 20716774,       // 1-based inclusive
            "end": 20749541          // 1-based inclusive
        },
        "window_len": 32768,
        "output_format": "full",     // full | mean | downsample
        "values": {
            "RNA-seq": {
                "Leaf": [0.008057, 0.007874, ...]  // 逐碱基预测值（full 格式）
            }
        }
    }
}
```

## 6. 参数速查

| 参数 | 必填 | 说明 |
|------|------|------|
| `start` | ✅ | 1-based 起始位点（缺失返回 400） |
| `end` | ❌ | 缺省自动取 32768 窗口 |
| `chromosome` | ❌ | 默认 `chr01`，`chr01`/`Chr1`/`1` 均可 |
| `genome` | ❌ | 默认唯一基因组 `osa1_r7` |
| `output_format` | ❌ | `full`（默认）/ `mean` / `downsample` |
| `max_points` | ❌ | downsample 目标点数，默认 1024 |
| `snv_index` | SNV 时 ✅ | 1-based 变异位点（须在窗口内） |
| `snv_base` | SNV 时 ✅ | 目标碱基 `A`/`C`/`G`/`T`/`N` |

**错误语义**：参数校验错误 → **400**；未知基因组/执行错误 → **500**。

---

## 7. 运行自动化测试

项目附带完整的 API 自动化测试脚本：

```bash
# 测试常规 API 端点（18 项测试）
bash test/test_api.sh

# 测试 DCS 适配层 API（18 项测试）
bash test/test_dcs_api.sh

# 测试前端
bash test/test_frontend.sh

# 运行全部测试
bash test/run_all_tests.sh
```

> **注意**：路由命名使用**下划线** `rice_mut`，而非连字符 `rice-mut`。代码中 `dcs_adapter.py` 定义的路由为 `/api/aigress/openai/rice_mut` 和 `/api/aigress/openai/rice_mut/snv`。