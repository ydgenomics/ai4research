# 水稻模型OGR及其应用模型的网页/API部署

网页部署需要分前端（frontend）展示和后端（backend）计算，以及前后端的联动API。网页部署完全参考Genos的部署，前端使用Python包Gradio，后端即模型推理，打包为一个docker（ydgenomics/org_web:sanic）。

## 1. 模型使用

水稻泛基因组(水稻+野生稻)DNA模型OneGenomeRice基于MoE架构(Decoder-only，next token)学到了生命的”语法“，one-hot编码即一个碱基一个token，具有长文本能力(1 M)。基于基模衍生其下游应用，例如接入下游应用架构（U-net）的RNA表达预测(单碱基分辨率，32k窗口)，一个是直接从序列到表达 sequence-to-expression (OGR-Mutation)；另一个是多模态的表达预测 DNA+ATAC->RNA (OGR-Reg)。

### 1.1 OGR

水稻基模提供提取embedding后和下游碱基预测的服务，API使用默认最大可提取32678长的序列。

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/aigress/openai/rice_ogr`、`/rice_ogr` | POST | 单入口，按 `mode` 分发 `dna_embedding` / `predict` |
| `/api/aigress/openai/rice_ogr/dna_embedding` | POST | 子路径方式(等价 `mode=dna_embedding`) |
| `/api/aigress/openai/rice_ogr/predict` | POST | 子路径方式(等价 `mode=predict`) |

参数：
- `sequence`（必填）：DNA 序列
- `model_name`（必填）：注册表模型名，如 `1B_8k` / `1B_32k`
- `pooling_method`（可选）：`mean`（默认）/ `max` / `last` / `none`

> `model` 为**服务名**(`rice_ogr`,与入口 path 末段一致);实际**模型名**放在 `model_name`
> (如 `1B_8k` / `1B_32k`)。向后兼容:`model` 不等于 `rice_ogr` 时仍视为模型名。


```bash
# mode=dna_embedding(默认,--data 中可省略 mode)
curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_ogr/dna_embedding \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "pooling_method": "mean"
  }'

# mode=predict(下游碱基预测)
curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_ogr \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "mode": "predict",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'

curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_ogr/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'
```

<details> <summary> demo </summary>

```shell
[19:46:02] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl --location 'https://www.dcs.cloud/api/aigress/openai/rice_ogr'   --header 'Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx'   --header 'Content-Type: application/json'   --data '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "mode": "predict",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'
{"usage":{"prompt_tokens":16,"completion_tokens":10},"status":200,"message":"下游碱基预测成功","result":{"model":"rice_ogr","model_name":"1B_8k","mode":"predict","original_sequence":"ACGTTGCATGCAACGT","predicted_sequence":"ACGTTGCATGCAACGTACAACAAAAA","predicted_bases":"ACAACAAAAA","predict_length":10,"total_length":26,"elapsed_seconds":0.3191}}


[20:17:14] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_ogr/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_ogr",
    "model_name": "1B_8k",
    "sequence": "ACGTTGCATGCAACGT",
    "predict_length": 10
  }'
{"usage":{"prompt_tokens":16,"completion_tokens":10},"status":200,"message":"下游碱基预测成功","result":{"model":"rice_ogr","model_name":"1B_8k","mode":"predict","original_sequence":"ACGTTGCATGCAACGT","predicted_sequence":"ACGTTGCATGCAACGTACAACAAAAA","predicted_bases":"ACAACAAAAA","predict_length":10,"total_length":26,"elapsed_seconds":0.3227}}
```

</details>


### 1.2 OGR-Mutation

水稻RNA表达预测模型（在4个品种的配对的基因组和叶转录组数据训练），具有单碱基分辨率RAN表达预测能力，每次预测输入32k窗口（不足则padding到32k，超过则被修剪到32k），输出对应32k窗口的表达峰图；支持单碱基突变。

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | 参考序列表达预测 | `chromosome` + `start` |
| `snv` | 单突变位点的窗口轨迹对比 | `chromosome` + `start` + `snv_index` + `snv_base` |


参数：
- `pooling_method`（可选）：`mean`（默认）/ `max` / `last` / `none`

```shell
curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_mut",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'

# mode=snv：chr09:20731844 位点 A→T 的 ref/mut 轨迹对比
curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_mut",
    "mode": "snv",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }'
```

<details> <summary> demo </summary>

```shell
[20:01:34] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \ https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_mut",
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'
{"usage":{"prompt_tokens":32768,"completion_tokens":32768},"status":200,"message":"参考序列表达预测成功","result":{"model":"rice_mut","genome":"osa1_r7","chromosome":"Chr9","position_1based":{"start":20716774,"end":20749541},"window_len":32768,"output_format":"mean","values":{"RNA-seq":{"Leaf":0.430133}}}}

[20:01:59] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_mut",
    "mode": "snv",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "snv_index": 20731844,
    "snv_base": "T",
    "output_format": "mean"
  }'
{"usage":{"prompt_tokens":32768,"completion_tokens":65536},"status":200,"message":"SNV 预测成功 (ref C → T)","result":{"model":"rice_mut","genome":"osa1_r7","chromosome":"Chr9","position_1based":{"start":20716774,"end":20749541},"window_len":32768,"snv_index_1based":20731844,"ref_base":"C","snv_base":"T","output_format":"mean","ref_values":{"RNA-seq":{"Leaf":0.430133}},"mut_values":{"RNA-seq":{"Leaf":0.43156}}}}
```

</details>


### 1.3 OGR-Reg

水稻多模态RNA表达预测模型（在2个品种的配对的基因组、茎尖分生组织的ATAC和RNA数据训练），具有单碱基分辨率RAN表达预测能力，每次预测输入32k窗口的DNA序列和ATAC信号（不足则padding到32k，超过则被修剪到32k），输出对应32k窗口的RNA表达峰图（区分正负链）。

| mode | 功能 | 必填参数 |
|---|---|---|
| `predict` | 序列表达预测 | `genome` + `chromosome` + `start` + `atac_source` |

参数：
- `genome`（必填）：基因组 `MH63RS3` / `NIP`
- `output_format`（可选）：`full`（默认）/ `mean` / `downsample`

```shell
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/rice_reg \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{
    "model": "rice_reg",
    "mode": "predict",
    "genome": "MH63RS3",
    "chromosome": "chr01",
    "start": 20716774,
    "atac_source": "SAM2_MH63_1",
    "output_format": "mean"
  }'
```

<details> <summary> demo </summary>

```shell
[20:01:34] root@iZ1pp00klhtaeu6pj9ib01Z:/# curl -X POST https://www.dcs.cloud/api/aigress/openai/rice_mut \ https://www.dcs.cloud/api/aigress/openai/rice_mut \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rice_mut",
    "mode": "predict",
    "genome": "osa1_r7",
    "chromosome": "chr09",
    "start": 20716774,
    "end": 20749541,
    "output_format": "mean"
  }'
{"usage":{"prompt_tokens":32768,"completion_tokens":32768},"status":200,"message":"参考序列表达预测成功","result":{"model":"rice_mut","genome":"osa1_r7","chromosome":"Chr9","position_1based":{"start":20716774,"end":20749541},"window_len":32768,"output_format":"mean","values":{"RNA-seq":{"Leaf":0.430133}}}}

```

</details>


## 模型测试


## 模型部署心得
- 地址不要包含-
- 使用tosutil传到云平台
- 基因组文件fa和fa.fai，一定要先传fa，再传fai