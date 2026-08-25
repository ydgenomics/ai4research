# DCS API 快速上手（DCS_API_quick_start）

> 一行命令搞定健康检查 / 参考预测 / SNV 预测，并保存为 JSON 文件。
> 完整参数说明见 [DCS_API.md](DCS_API.md)。

## 0. 准备工作

```bash
# ① 服务已启动且模型加载完成(本地调试)
curl -s http://127.0.0.1:8001/api/aigress/openai/health | python3 -m json.tool

# ② 定义公共变量(改这里即可切换环境)
BASE="http://127.0.0.1:8001/api/aigress/openai/rice_mut"     # 本地
# BASE="https://www.dcs.cloud/api/aigress/openai/rice_mut"   # DCS 线上
KEY="<DCS_API_KEY>"                                           # 服务端未启用鉴权可留空
AUTH=(); [[ -n "$KEY" ]] && AUTH=(-H "Authorization: Bearer $KEY")
```

---

## 1. 健康检查

### 1.1 直接查看

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"health"}' | python3 -m json.tool
```

### 1.2 保存为 JSON 文件

```bash
# 方式 A:curl -o 直接保存原始响应
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"health"}' -o health.json && python3 -m json.tool health.json

# 方式 B:管道重定向(效果同上)
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"health"}' > health.json

# 方式 C:只提取关键字段再保存(推荐,文件更小)
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"health"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); open('health.summary.json','w').write(json.dumps({'predictor_initialized': d['predictor_initialized'], 'genomes': d['genomes'], 'actual_port': d['diagnostics']['listen']['actual_port']}, ensure_ascii=False, indent=2))"
```

---

## 2. 参考序列表达预测

### 2.1 mean 格式(每轨道标量,最常用)

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"predict","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}' \
  -o predict_mean.json && python3 -m json.tool predict_mean.json
```

### 2.2 full 格式(逐碱基数组,约 286KB,适合落盘分析)

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"predict","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"full"}' \
  -o predict_full.json
ls -lh predict_full.json   # 确认文件大小
python3 -c "import json; d=json.load(open('predict_full.json')); print('保存成功, 染色体:', d['result']['chromosome'], '| 窗口长度:', d['result']['window_len'])"
```

### 2.3 downsample 格式(降采样到 1024 点,兼顾大小与形状)

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"predict","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"downsample","max_points":1024}' \
  -o predict_downsample.json && python3 -m json.tool predict_downsample.json | head -30
```

### 2.4 自动推断(不写 mode,现有调用零改动)

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"output_format":"mean"}' \
  -o predict_auto.json
```

---

## 3. SNV 变异预测(双轨对比)

### 3.1 显式 mode=snv

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"snv","genome":"osa1_r7","chromosome":"chr01","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T","output_format":"mean"}' \
  -o snv_result.json && python3 -m json.tool snv_result.json
```

### 3.2 自动识别(带 snv_index 即走 SNV)

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"genome":"osa1_r7","chromosome":"chr09","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T","output_format":"mean"}' \
  -o snv_auto.json
```

### 3.3 同时保存参考/突变两轨道(解析后拆分)

```bash
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"snv","genome":"osa1_r7","chromosome":"chr09","start":20716774,"end":20749541,"snv_index":20731844,"snv_base":"T","output_format":"downsample","max_points":256}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
json.dump(d['ref_values'], open('snv.ref.json','w'), ensure_ascii=False, indent=2)
json.dump(d['mut_values'], open('snv.mut.json','w'), ensure_ascii=False, indent=2)
json.dump({'snv_index': d['snv_index_1based'], 'ref_base': d['ref_base'], 'snv_base': d['snv_base']}, open('snv.meta.json','w'), ensure_ascii=False, indent=2)
print('已保存: snv.ref.json / snv.mut.json / snv.meta.json')
"
```

---

## 4. 批量落盘案例(Python 脚本,循环调用保存)

```bash
cat > batch_predict.py <<'EOF'
import json, time, urllib.request

BASE = "http://127.0.0.1:8001/api/aigress/openai/rice_mut"
API_KEY = ""  # 服务端启用鉴权时填入

def call(payload):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)

# 场景1:多个区间的参考预测,逐个保存
regions = [
    {"chromosome": "chr01", "start": 20716774, "end": 20749541},
    {"chromosome": "chr01", "start": 20800000, "end": 20832767},
    {"chromosome": "chr02", "start": 1000000,  "end": 1032767},
]
for i, r in enumerate(regions, 1):
    out = call({**r, "mode": "predict", "output_format": "mean"})
    fname = f"region_{i}.json"
    with open(fname, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[{i}] 保存 {fname} -> {out['result']['chromosome']}:{out['result']['position_1based']}")

# 场景2:同一区间多个碱基的 SNV,结果合并到一个文件
snv_results = []
for base in ["A", "C", "G", "T"]:
    out = call({"mode": "snv", "chromosome": "chr01", "start": 20716774,
                "end": 20749541, "snv_index": 20731844, "snv_base": base,
                "output_format": "mean"})
    r = out["result"]
    snv_results.append({"snv_base": base, "ref_base": r["ref_base"],
                        "ref_value": r["ref_values"]["RNA-seq"]["Leaf"],
                        "mut_value": r["mut_values"]["RNA-seq"]["Leaf"]})
    time.sleep(0.1)
with open("snv_all_bases.json", "w") as f:
    json.dump(snv_results, f, ensure_ascii=False, indent=2)
print("已保存 snv_all_bases.json:", snv_results)
EOF
python3 batch_predict.py
```

---

## 5. 错误场景保存

```bash
# 非法参数(400):保存错误响应,便于复现
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"predict","start":20716774,"output_format":"bad"}' \
  -o error_400.json
cat error_400.json

# 未知 mode(400)
curl -s -X POST "$BASE" -H "Content-Type: application/json" "${AUTH[@]}" \
  -d '{"mode":"xxx"}' | tee error_mode.json
```

---

## 6. 常用命令速查表

| 目的 | 命令要点 |
|---|---|
| health 并保存 | `-d '{"mode":"health"}' -o health.json` |
| 参考预测 mean | `-d '{"mode":"predict",...,"output_format":"mean"}'` |
| 参考预测 full | `-d '{"mode":"predict",...,"output_format":"full"}' -o predict_full.json` |
| 参考预测 downsample | `-d '{"mode":"predict",...,"output_format":"downsample","max_points":1024}'` |
| SNV 显式模式 | `-d '{"mode":"snv",...,"snv_index":N,"snv_base":"T"}'` |
| SNV 自动识别 | body 带 `snv_index` 即可,无需 mode |
| 提取字段保存 | 管道 `python3 -c "import json,sys; ..."` |
| 批量落盘 | 见第 4 节 `batch_predict.py` |

> 坐标约定:`start`/`end`/`snv_index` 均为 **1-based inclusive**。
> 详细参数与响应结构见 [DCS_API.md](DCS_API.md)。