# Rice-Mutation Server

水稻 DNA 序列 → 多组学表达预测服务。基于深度学习模型(GenOmics: rice_1B_stage2_8k_hf 基础模型 + UNet 输出头),输入 DNA 序列预测 RNA-seq 表达谱,支持**参考 vs 突变**双轨道对比,通过 IGV.js 可视化展示。

> **API 调用（本机 + DCS）见 [API.md](API.md)**。

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
OGR_Prediction/
├── .env / .env.example          # 环境配置(模型、基因组、推理参数、CACHE_MODE)
├── requirements.txt             # 统一依赖清单
├── backend/
│   ├── inference.ipynb          # 原始推理原型(保留)
│   ├── src/                     # ★ 模型定义(直接复用,GenOmics 等)
│   │   ├── model.py  dataset.py  viewer.py  util.py ...
│   ├── run_backend.sh / stop_backend.sh
│   └── rice_mutation/
│       ├── main.py              # uvicorn 入口(加载 .env + sys.path)
│       ├── api.py               # FastAPI 路由(网页端点 + DCS 子路径)
│       ├── prediction_service.py# 单例预测器 + 核心预测逻辑 + bigWig 写盘
│       ├── igv_payload.py       # IGV payload 构建(ref/mut 双轨)
│       ├── plot_service.py      # 柱状图/统计等绘图服务
│       ├── cache_service.py     # 内容寻址预测缓存(LRU+TTL) + 上传注册表
│       └── core/
│           └── predictor.py     # ★ RiceMutationPredictor(移植自 inference.ipynb)
├── frontend/
│   ├── app.py                   # Gradio 界面
│   ├── config.py                # 前端配置(从 .env 读取)
│   ├── run_frontend.sh / stop_frontend.sh
│   └── static/igv.min.js        # IGV.js 静态资源
├── tools/startup_self_check.sh  # 启动自检
├── OGR/                     # 内置数据 + 运行时缓存
│   ├── builtin/             # osa1_r7 参考基因组（fa/fai/gff3/gvlfa/mmi）
│   ├── predictions/         # 预测 bigWig 缓存
│   └── uploaded_fasta/ uploaded_genomes/   # 上传缓存
└── test/test_api.sh         # API 测试
```

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 编辑 .env,设置模型/基因组/推理参数
# 注意:缓存目录请用绝对路径(静态文件服务按绝对路径解析)
```

### 1.1 缓存模式（`CACHE_MODE`）

部署前先选好缓存策略（`.env` 的 `CACHE_MODE`，默认 `true`）：

| 模式 | 上传自定义基因组 | 预测 bigWig | 内容寻址缓存 |
|---|---|---|---|
| `CACHE_MODE=true`（默认，全量缓存） | 复制到 `OGR/OGR_Prediction/uploaded_fasta`（服务托管，TTL 可清理） | 持久写入 `BACKEND_PREDICTION_CACHE`（长 TTL） | 开启：相同 (genome, window, SNV) 请求直接命中，跳过 GPU |
| `CACHE_MODE=false`（轻量，推荐部署） | **只注册本机路径，不复制**（前端会把 Gradio 临时文件路径传给后端，后端原地注册） | 写入临时目录 `BACKEND_PREDICTION_TMP`（默认 `/tmp/rice_prediction_tmp`，短 TTL 自动清理） | 关闭：每次预测重跑 GPU |

> - 轻量模式下上传文件**零拷贝**（前端/后端同机时前端直接把文件路径发给后端注册），预测 bigWig 仍会生成（IGV 渲染必需）但落在临时目录、由后台线程按 `TMP_BIGWIG_TTL_SECONDS`（默认 900s）自动清理，不长期占用磁盘。
> - 轻量模式下 TTL 清理**只移除注册表、不删除外部原文件**（`managed=False`），用户文件不受影响。
> - 两种模式下前端展示 / API 返回结构完全一致（`igv_payload` + `ref_id`/`snv_id`），切换模式无需改前端。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务（统一启动，网页端 + 纯 API 一套命令）

```bash
# 自检(推荐)
bash tools/startup_self_check.sh

# ★ 统一启动：一个命令拉起【网页端(:8000) + API 后端(:8001)】
bash start.sh all
# 也可分别控制：
bash start.sh backend   # 只启动 API 后端（纯 API 场景，不拉网页）
bash start.sh status    # 查看状态
bash start.sh stop      # 停止
```

> **统一架构说明**：网页端与纯 API 调用是**同一套后端进程**（`rice_mutation/main.py`，:8001）。
> 网页 Gradio(:8000) 通过 `/backend/*` 反代调用后端；纯 API 调用方 / DCS 网关直接调用后端
> 的 DCS 风格子路径 `/api/aigress/openai/OGR-Prediction/{Prediction,SNV,Genome,Chromosome,RegisterGenome}`。
> 两条路共用同一推理核心、同一统一返回结构
> `{model_name, usage, token_cnt, status, message, result{values, igv_payload}}` ——
> 前端 JS 就是"填好参数的 curl"，拿到返回后直接渲染；纯 API 调用方拿到同样内容。
> （不再需要单独启动 `dcs_adapter.py`，其功能已并入后端。）

### 4. 停止 / 日志

```bash
bash start.sh stop

tail -f logs/backend.nohup.log
tail -f logs/frontend.nohup.log
```

## 前端使用

1. 打开浏览器访问 `http://<host>:8000`
2. 在左侧「Required parameters」选择**基因组**(如 osa1_r7)、**染色体**(chr01-chr12,前端统一命名;后端自动通配到实际 FASTA 命名,如 `chr01`→`Chr1`,也兼容 `chr1`/`1`/`ChrUn` 等写法)、**起始位置**(预测窗口 = 从该 1-based 位置起 32 kb / 32768 bp)
3. 点击 **🚀 Predict** 预测
4. 在 IGV 中查看预测轨道:**灰色 = 参考表达量,其他颜色 = 模型预测**
5. 右侧柱状图显示当前 IGV 可见窗口内各轨道平均表达量(随视图缩放/平移自动刷新)

### 单碱基突变对比(SNV,可选)

在「Single-nucleotide variant (optional)」填写 **SNV position**(1-based bp,须落在预测窗口内)与 **SNV base**(A/C/G/T),再点 **🚀 Predict** — 同一按钮自动改为「参考 vs 突变」对比:

- **result1 (ref)**:无突变参考序列(灰色 `#6b7280`)
- **result2 (mut @C16000>A)**:单碱基突变序列(彩色,首个 `#d62728`)

两条轨道叠加在 IGV 中显示,右侧柱状图并排显示 result1/result2 平均表达量。两个字段留空则仅做参考预测。

> 注意:SNV position 为窗口内的**绝对坐标(1-based)**;窗口 = `MAX_SEQ_LEN`(32768)与输入区间的中心对齐结果。

### 自定义基因组 + 注释 GFF

1. 在**基因组下拉**选 **📤 Custom Genome** — **Genome 一行**拆分为三个等宽列:**Genome | Genome FASTA | Annotation GFF**;其余参数(染色体/起始位置/SNV)保持全宽
2. 在 **Genome FASTA** 填**后端机器上的 FASTA 绝对路径**(`.fa`/`.fasta`/`.fna`, ≤`MAX_UPLOAD_MB`MB,默认 640),后端自动注册(自动建 `.fai` 索引)并为该基因组生成 id,下拉自动切换到该 id、染色体列表按实际 FASTA 刷新
3. (可选)在 **Annotation GFF** 填**GFF 绝对路径**(`.gff`/`.gff3`/`.gtf`,可 `.gz`),IGV 即显示该自定义基因组的 Genes 轨道
4. 之后即可按该自定义基因组预测(优先级高于内嵌基因组);切换回内置基因组时 Genome 行恢复全宽、自定义列隐藏

> 提示:若先填 GFF 而 FASTA 尚未填好**不会报错** —— GFF 会排队等待,FASTA 注册成功后自动附加。
>
> 提示(`CACHE_MODE=false` 轻量模式):两个自定义文件列均为**路径文本框(非文件上传)**,后端把该路径直接注册为自定义基因组(零拷贝)。请确保路径在后端机器上存在且可读;如需长期稳定引用,建议使用部署机上的固定目录。
> 若使用 `CACHE_MODE=true` 全量模式,则为真正的文件上传(上传后复制到 `OGR/OGR_Prediction/uploaded_fasta`,由服务托管并可按 TTL 清理)。

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

完整 API 调用说明（本机 + DCS 两套测试代码）见 **[API.md](API.md)**。

**统一返回结构**：所有预测接口（网页端小写路径 + DCS 风格子路径）返回同一外层包装：

```json
{
  "model_name": "OGR",
  "usage": {"prompt_tokens": 32768, "completion_tokens": 32768},
  "token_cnt": 65536,
  "status": 200,
  "message": "Prediction completed in 3.5s",
  "result": {
    "model": "OGR-Prediction", "Genome": "osa1_r7", "Chromosome": "Chr9",
    "Start": 20716773, "window_len": 32768, "Format": "Full|Mean",
    "values": {...},          // 参考预测数值（Full 数组 / Mean 标量）
    "igv_payload": {...}      // 可视化：reference{fastaURL,indexURL,Genes}, locus, tracks[{bw url}]
    // SNV 另有 ref_values / mut_values 双轨
  }
}
```

> 网页端与纯 API 拿到**同一份**返回：`result.igv_payload` 可直接喂给 IGV 渲染，`result.values` 是数值结果。
> 网页 Gradio 就是"填好参数的 curl"——调用 API 后把返回内容填回页面展示。

**接口一览**（网页端路径 :8001 / DCS 风格子路径 `/api/aigress/openai/OGR-Prediction/...`）：

| 网页端路径 | DCS 子路径 | 方法 | 说明 |
|---|---|---|---|
| `/health` | `/health` | GET | 健康检查 + 模型元信息 |
| `/genomes` | `.../Genome` | GET/POST | 已配置基因组列表(内置 + 上传) |
| `/genomes/{id}/chromosomes` | `.../Chromosome` | GET/POST | 某基因组的染色体列表(chrNN 风格) |
| `/assays` | — | GET | assay 列表(来自 index_stat) |
| `/biosamples` | — | GET | biosample 列表(来自 index_stat) |
| `/uploadFasta` | — | POST | 上传自定义基因组 FASTA(轻量模式零拷贝注册) |
| `/uploadGff` | — | POST | 上传注释 GFF 并附加到自定义基因组 |
| `/predict` | `.../Prediction` | POST | 参考序列表达预测(内容寻址缓存仅在 `CACHE_MODE=true` 时生效) |
| `/predict/snv` | `.../SNV` | POST | 单碱基突变对比(result1 ref / result2 mut),返回 `snv_id` |
| `/predict/snv/stat` | — | POST | 按 `snv_id` 对窗口内区域计算差异统计 |
| `/predict/bar` | — | POST | 按 `prediction_id` 计算当前区域各轨道平均表达量(前端柱状图) |
| — | `.../RegisterGenome` | POST | 传本地 `fasta_path` 懒注册自定义基因组(纯 API) |

> 网页版请求/响应示例见 `test/test_api.sh`；纯 API 调用（curl）见 [API.md](API.md)。
