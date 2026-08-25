# Rice-Mutation Server

水稻 DNA 序列 → 多组学表达预测服务。基于深度学习模型(GenOmics: rice_1B_stage2_8k_hf 基础模型 + UNet 输出头),输入 DNA 序列预测 RNA-seq 表达谱,支持**参考 vs 突变**双轨道对比,通过 IGV.js 可视化展示。

## 与 rice-reg-server2 的区别

| 维度 | rice-reg-server2 | rice-mutation-server(本项目) |
|---|---|---|
| 模型输入 | DNA + ATAC bigWig | **仅 DNA 序列** |
| 模型 | rice_1B_32k_hf + fusion predictor | rice_1B_stage2_8k_hf + `GenOmics`(UNet) |
| 输出 | RNA-seq +/− 两通道 | **assay × biosample 多维**(如 `total_RNA-seq_+` × `NIP_CSQ`) |
| 反归一化 | 服务端 `LabelScaler` | **模型内部已完成**(`predictions_scaling_torch`) |
| 核心场景 | ATAC 条件预测 | **变异效应对比(ref vs mut)** |
| 序列长度 | 32000 | 32768(max_length,超出截断) |

## 项目结构

```
rice-mutation/
├── .env / .env.example          # 环境配置(模型、基因组、推理参数)
├── requirements.txt             # 统一依赖清单
├── backend/
│   ├── inference.ipynb          # 原始推理原型(保留)
│   ├── src/                     # ★ 模型定义(直接复用,GenOmics 等)
│   │   ├── model.py  dataset.py  viewer.py  util.py ...
│   ├── run_backend.sh / stop_backend.sh
│   └── rice_mutation/
│       ├── main.py              # uvicorn 入口(加载 .env + sys.path)
│       ├── api.py               # FastAPI 路由
│       ├── prediction_service.py# 单例预测器 + 核心预测逻辑 + bigWig 写盘
│       ├── igv_payload.py       # IGV payload 构建(ref/mut 双轨)
│       ├── cache_service.py     # 内容寻址预测缓存(LRU+TTL) + bigWig 后台清理
│       └── core/
│           └── predictor.py     # ★ RiceMutationPredictor(移植自 inference.ipynb)
├── frontend/
│   ├── app.py                   # Gradio 界面
│   ├── config.py                # 前端配置(从 .env 读取)
│   ├── run_frontend.sh / stop_frontend.sh
│   └── static/igv.min.js        # IGV.js 静态资源
├── tools/startup_self_check.sh  # 启动自检
├── cache/predictions/           # 预测 bigWig 缓存
└── test/test_api.sh             # API 测试
```

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 编辑 .env,设置模型/基因组/推理参数
# 注意:缓存目录请用绝对路径(静态文件服务按绝对路径解析)
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 自检(推荐)
bash tools/startup_self_check.sh

bash backend/run_backend.sh && bash frontend/run_frontend.sh
```

### 4. 停止 / 日志

```bash
bash backend/stop_backend.sh && bash frontend/stop_frontend.sh

tail -f backend/logs/backend.nohup.log
tail -f frontend/logs/frontend.nohup.log
```

## 前端使用

1. 打开浏览器访问 `http://<host>:8000`
2. 在左侧「Required parameters」选择**基因组**(如 osa1_r7)、**染色体**(chr01-chr12,前端统一命名;后端自动通配到实际 FASTA 命名,如 `chr01`→`Chr1`,也兼容 `chr1`/`1`/`ChrUn` 等写法)、**起始位置**(预测窗口 = 从该 1-based 位置起 32 kb / 32768 bp)
3. 点击 **🚀 Predict** 预测
4. 在 IGV 中查看预测轨道:**灰色 = 参考表达量,其他颜色 = 模型预测**
5. 右侧柱状图显示当前 IGV 可见窗口内各轨道平均表达量(随视图缩放/平移自动刷新)

### 单碱基突变对比(SNV,可选)

在「Single-nucleotide variant (optional)」填写 **SNV position**(1-based bp,须落在预测窗口内)与 **SNV base**(A/C/G/T/N),再点 **🚀 Predict** — 同一按钮自动改为「参考 vs 突变」对比:

- **result1 (ref)**:无突变参考序列(灰色 `#6b7280`)
- **result2 (mut @C16000>A)**:单碱基突变序列(彩色,首个 `#d62728`)

两条轨道叠加在 IGV 中显示,右侧柱状图并排显示 result1/result2 平均表达量。两个字段留空则仅做参考预测。

> 注意:SNV position 为窗口内的**绝对坐标(1-based)**;窗口 = `MAX_SEQ_LEN`(32768)与输入区间的中心对齐结果。

### 自定义基因组 + 注释 GFF

1. 在**基因组下拉**选 **📤 Custom Genome** — **Genome 一行**自动拆分为三个等宽列:**Genome | Upload Genome FASTA | Upload Annotation GFF**;其余参数(染色体/起始位置/SNV)保持全宽
2. 在 **Upload Genome FASTA** 上传 **Genome FASTA**(.fa/.fasta/.fna,≤`MAX_UPLOAD_MB`MB,默认 640),后端自动建 `.fai` 索引并注册为 `custom_<ts>` 基因组,下拉自动切换到该 id、染色体列表按实际 FASTA 刷新
3. (可选)在 **Upload Annotation GFF** 上传 **GFF**(.gff/.gff3/.gtf,可 .gz),IGV 即显示该自定义基因组的 Genes 轨道
4. 之后即可按该自定义基因组预测(优先级高于内嵌基因组);切换回内置基因组时 Genome 行恢复全宽、上传列隐藏

> 提示:若 FASTA 较大、GFF 较小,先选好 GFF 时(FASTA 尚未上传完)**不会报错** —— GFF 会排队等待,FASTA 上传完成注册成功后自动附加(GFF 已就绪状态 → 自动附加成功)。

#### 上传文件要求(命名 / 内容)

- **FASTA 扩展名(硬性)**:必须是 `.fa` / `.fasta` / `.fna`(大小写不敏感,后端报错 `Only .fa / .fasta accepted`);文件名本身无限制,可任意命名。后端存储时自动重命名为 `<时间戳ns>_<原文件名>` 并注册为 `custom_<ts>`,与文件名无关
- **FASTA 内容**:至少含 1 条序列记录(否则报 `No sequence records found in FASTA`),且为 pyfaidx 可解析的标准 FASTA;序列名(header)即染色体名,纯数字 / `Chr1` / `chr01` 等统一显示为 `chr01`–`chr12`,非数字的(如 `ChrUn`、`scaffold_1`)保留原名
- **GFF 扩展名(硬性)**:`.gff` / `.gff3` / `.gtf`,或压缩版 `.gff.gz` / `.gff3.gz` / `.gtf.gz`(大小写不敏感)
- **GFF 格式按扩展名自动识别**:文件名以 `.gtf` / `.gtf.gz` 结尾按 **GTF** 解析,否则按 **GFF3** 解析 —— GTF 文件必须用 `.gtf` 结尾,否则 Genes 轨道可能显示异常
- **GFF 其它**:只能附加到**已上传的自定义基因组**(内置基因组不能附加);内容不做格式校验,但建议第 1 列 seqid 与 FASTA 染色体名一致,Genes 轨道才会落在对应染色体
- **大小**:FASTA / GFF 均 ≤ `MAX_UPLOAD_MB`(默认 640MB)

### IGV 内功能(工具栏)

- **Save SVG**(原生):下载当前视图为 SVG
- **Save PNG**:下载当前视图为 PNG(基于 IGV 原生 `browser.toSVG()` 输出光栅化,与 Save SVG 内容一致)
- **信号轨道右键菜单**:
  - **Line chart / Bar chart**:随时切换预测信号轨道的图形模式(当前模式带 ✓ 标记),切换后立即重绘

## API 文档

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 + 模型元信息 |
| `/genomes` | GET | 已配置基因组列表(内置 + 上传) |
| `/genomes/{id}/chromosomes` | GET | 某基因组的染色体列表(chrNN 风格) |
| `/assays` | GET | assay 列表(来自 index_stat) |
| `/biosamples` | GET | biosample 列表(来自 index_stat) |
| `/uploadFasta` | POST | 上传**自定义基因组** FASTA(≤`MAX_UPLOAD_MB`MB,默认 640,自动建 .fai 并注册) |
| `/uploadGff` | POST | 上传**注释 GFF**(.gff/.gff3/.gtf,可 .gz)并附加到指定自定义基因组(`genome` + `file` 表单) |
| `/predict` | POST | 参考序列表达预测(内容寻址缓存,相同输入命中直接返回) |
| `/predict/snv` | POST | 单碱基突变对比(result1 ref / result2 mut),返回 `snv_id`(内容寻址缓存) |
| `/predict/snv/stat` | POST | 按 `snv_id` 对窗口内区域计算 (result1−result2)/result1 差异统计 |
| `/predict/bar` | POST | 按 `prediction_id` 计算当前区域各轨道平均表达量(前端柱状图数据) |

### 请求示例

```bash
# 参考预测
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"genome":"osa1_r7","chromosome":"chr01","start":0,"end":32000,
       "biosample_names":["NIP_CSQ"]}'

# 上传自定义基因组(自动建 .fai 并注册,返回 genome id + 染色体列表)
curl -X POST -F "file=@/path/to/custom_genome.fa" http://127.0.0.1:8001/uploadFasta
# → {"success":true, "genome":"custom_<ts>", "chromosomes":["chr01", ...], ...}

# 上传注释 GFF(附加到上面注册的自定义基因组,IGV 将显示 Genes 轨道)
curl -X POST -F "genome=custom_<ts>" -F "file=@/path/to/annotation.gff3" http://127.0.0.1:8001/uploadGff
# → {"success":true, "genome":"custom_<ts>", "gff_path":"...", ...}

# 用自定义基因组预测(优先级高于内嵌)
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"genome":"custom_<ts>","chromosome":"chr01","start":0,"end":32000,
       "biosample_names":["NIP_CSQ"]}'

# 单碱基突变对比(窗口内第 16000 个碱基 C→A)
curl -X POST http://127.0.0.1:8001/predict/snv \
  -H "Content-Type: application/json" \
  -d '{"genome":"osa1_r7","chromosome":"chr01","start":0,"end":32000,
       "biosample_names":["NIP_CSQ"],"snv_index":16000,"snv_base":"A"}'
# → {"success":true, "metadata":{"snv_id":"snv_...", "ref_base":"C", "snv_base":"A",
#     "snv_index":16000, "window_len":32768, "window_start":0}, "igv_payload":{...}}

# 区域差异统计(窗口内 5000–20000)
curl -X POST http://127.0.0.1:8001/predict/snv/stat \
  -H "Content-Type: application/json" \
  -d '{"snv_id":"snv_...","region_start":5000,"region_end":20000}'
# → {"success":true, "region":[5000,20000],
#     "stats":{"total_diff_pct":-0.56, "mean_diff_pct":-0.56,
#               "max_diff_index":16159, "max_abs_diff":-0.2344,
#               "max_diff_pct":-8.47, "region_len":15000}, ...}
```

### 响应结构

```json
{
  "success": true,
  "message": "Prediction completed in 1.6s",
  "elapsed_seconds": 1.59,
  "igv_payload": {
    "reference": {"id": "osa1_r7", "fastaURL": "http://.../static-files/...", "tracks": [...]},
    "locus": "Chr1:0-32,768",
    "tracks": [
      {"name": "NIP_CSQ total_RNA-seq_+", "url": "http://.../xxx_ref_xxx.bw", ...}
    ]
  }
}
```

## 测试

```bash
bash test/test_api.sh
```

## 关键配置(`.env`)

- `BASE_MODEL_PATH` / `CHECKPOINT_PATH` / `INDEX_STAT_PATH`:基础模型(HF)、权重、index_stat
- `USE_FLASH_ATTN=true`、`MODEL_TORCH_DTYPE=bfloat16`、`DEVICE=cuda:0`
- `PROJ_DIM=1024`、`NUM_DOWNSAMPLES=4`、`BOTTLENECK_DIM=1536`(GenOmics 参数,须与 checkpoint 训练一致)
- `MAX_SEQ_LEN=32768`
- `GENOME_<ID>_FASTA/FAI/GFF`:参考基因组
- `MAX_UPLOAD_MB`:自定义基因组 FASTA/GFF 上传大小上限(单位 MB,默认 640)
- 缓存路径**必须用绝对路径**

## 已知限制

- 整段突变序列对比(MVP 旧接口)已移除;现在仅支持**单碱基突变**(`/predict/snv`),原始 mut 序列仅用于内部计算,不直接作为模型输入
- 预测结果缓存:内容寻址 LRU(128 条)+ TTL 30 分钟;命中直接返回(`message="cached"`),不重复推理;过期条目及其 bigWig 由后台线程自动清理
- SNV/参考数组保存在后端内存缓存(`_SNV_CACHE`/`_REF_CACHE`,最多 256 条,最旧淘汰);服务重启后 `prediction_id` 失效,需重新预测
- 上传的自定义基因组仅本次服务运行有效(内存注册表,重启后需重新上传;`cache/uploaded_genomes/` 启动时清理)
- 预测窗口中心对齐到 `MAX_SEQ_LEN`(32768),参考序列超出截断;SNV position 为窗口内绝对坐标(1-based)
- 模型输出由 `GenOmics.forward` 内部完成反归一化,服务端直接写 bigWig
- GPU 推理为单实例 + 全局锁串行化(并发安全);动态批处理等吞吐优化为后续迭代项
