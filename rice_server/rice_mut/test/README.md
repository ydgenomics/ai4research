是的,**启动 dcs_adapter.py 后不需要网页前端,直接就能通过 curl 调 API 推理**。模型加载完(约 30s)health 返回 `ok` 即可调用。

## 1. 启动服务

```bash
cd /mnt/rice/default/Workspace/yangdong/rice_reg/rice-server/rice-mut
BACKEND_PORT=8001 /root/miniconda3/envs/vllm/bin/python backend/dcs_adapter.py
```

> 若提示 `address already in use`,说明 8001 已有服务,先 `kill $(cat backend/logs/backend.pid)` 或用 `fuser -k 8001/tcp` 清理。

## 2. 健康检查

```bash
curl -s http://127.0.0.1:8001/api/aigress/openai/health
# {"status":"ok","predictor_initialized":true,"genomes":["osa1_r7"]}
```

## 3. 参考序列表达预测(核心推理)

```bash
curl --location 'http://127.0.0.1:8001/api/aigress/openai/rice-mut' \
--header 'Authorization: Bearer hello' \
--header 'Content-Type: application/json' \
-d '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}'
```

```bash
curl -X POST "https://www.dcs.cloud/api/aigress/openai/v1" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-BxVGisztHyXMivKtO-NiZsQ0RfYxD-GUZpCN8f7vHvz1QZuu" \
  -d '{"model":"org-mut", "genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}'


curl -X POST "https://www.dcs.cloud/api/aigress/openai/v1" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-BxVGisztHyXMivKtO-NiZsQ0RfYxD-GUZpCN8f7vHvz1QZuu" \
  -d '{"model":"OGR-Mutation", "genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}'

```

```bash
# full 格式(逐碱基数组,约 286KB 响应)
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541}'
```

```bash
# mean 格式(每轨道标量均值,响应小、最常用)
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}'
# 实测:mean = 0.599956
```

```bash
# downsample 格式(均匀降采样到 1024 点)
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}'
```

## 4. SNV 变异预测(双轨对比)

```bash
curl -s -X POST http://127.0.0.1:8001/api/aigress/openai/rice-mut/snv \
  -H "Content-Type: application/json" \
  -d '{"model":"rice-mut","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T"}'
```

## 5. 响应结构说明

```json
{
  "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},  // 计费用
  "status": 200,
  "message": "参考序列表达预测成功",
  "result": {
    "model": "rice-mut",
    "genome": "osa1_r7",
    "chromosome": "Chr1",
    "position_1based": {"start": 20716774, "end": 20749541},  // 1-based inclusive
    "window_len": 32768,
    "output_format": "full",
    "values": {"total_RNA-seq_+": {"NIP_leaf": [0.008057, 0.007874, ...]}}
  }
}
```

## 6. 参数速查

| 参数 | 必填 | 说明 |
|---|---|---|
| `start` | ✅ | 1-based 起始位点(缺失返回 400) |
| `end` | ❌ | 缺省自动取 32768 窗口 |
| `chromosome` | ❌ | 默认 `chr01`,`chr01`/`Chr1` 均可 |
| `genome` | ❌ | 默认唯一基因组 `osa1_r7` |
| `output_format` | ❌ | `full`(默认)/ `mean` / `downsample` |
| `max_points` | ❌ | downsample 目标点数,默认 1024 |
| `snv_index`+`snv_base` | SNV 时✅ | 1-based 变异位点 + 目标碱基 A/C/G/T/N |

错误语义:参数校验错误 → **400**;未知基因组/执行错误 → **500**。

这套 curl 与 `test/test_dcs_api.sh` 里实测通过的 18 项完全一致,可直接复制使用。需要我把这段 curl 示例补进 `test/AGENT.md` 吗?