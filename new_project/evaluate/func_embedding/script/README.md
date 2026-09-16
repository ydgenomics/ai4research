# 多模型基因区域 Embedding 提取

从 **GFF3 + FASTA** 提取基因区域序列，用**多个基因组大模型**各自前向推理，
得到逐基因的 mask-aware 池化 embedding（池化方式可在 yaml 中选，见 §4.3）。
所有模型共享同一份基因序列与顺序，
产物按模型分目录存放，可跨模型按行（index）直接对齐比较。

> 上游工具：此目录配套的单模型脚本见 `../extract_gene_embeddings.py`；
> 下游评测（分类/回归 head）参考 `benchmarks_botanic` 框架的思路。

---

## 1. 设计要点（为什么这样做）

| 问题 | 做法 |
|---|---|
| batch 内张量等长 | `padding=True` + `attention_mask` |
| padding 污染结果 | mask-aware 池化 `mean = Σ(hs·mask) / Σ(mask)`，pad 位置权重为 0 |
| 长序列求和溢出（fp16） | 池化求和/求均值统一**提升到 fp32** 进行（hidden 通常为 fp16，数万 token 同号累加会超 fp16 上限 65504 溢出为 ±inf，见 §9） |
| 序列长度混杂 | 池化方式可配（`mean`/`max`/`first`/`last`/`mean_last_k`/`mean_trim`，见 §4.3）+ 可选 `l2_normalize`；同时记录每条序列的真实长度/是否截断 |
| 显存浪费 / OOM | 长度降序 + **每批总 token ≤ max_tokens** 动态打包（非固定 batch_size） |
| 不同 tokenizer / 架构差异 | **adapter 模式**：每种 tokenizer 一个预处理 + tokenize 参数实现 |
| 超长序列（超过模型上下文） | 按各模型 `max_context` 右截断（`max_len: 0` 跟随各模型上限）；`meta.tsv` 记录 `truncated` |
| 跨模型对齐 | GFF 只解析一次、序列只取一次，所有模型按同一顺序写入同名列 `gene_names.txt` |
| 断点续跑 | 已存在 `gene_embeddings.npy` 的模型自动跳过（`resume: true`） |
| 打包方式 | `batch_size>0` 固定条数打包；`batch_size=0` 用各模型 `max_tokens` 动态打包（二选一） |
| 多卡 | `gpus: [0,1]` 支持：多卡**串行**（模型依次放不同卡）或多卡**并行**（每卡一进程） |
| 测试子集 | `--max-genes N` / `--gene-frac 0.1`（或 run yaml `max_genes`/`gene_frac`）只跑 GFF 前 N 条/前 10% |
| 数据质量 | 落盘前 `output.py` 断言 embedding 全有限（±inf / NaN 直接报错并指出异常行），杜绝脏数据流入下游 |

---

## 2. 目录结构

```
script/
├── main.py                    # 唯一入口
├── configs/
│   ├── models.yaml            # 模型注册表（新增模型只改这里）
│   └── run_example.yaml       # 一次运行的配置（选模型 / 数据 / 打包参数）
├── ogr_extract/               # 逻辑包
│   ├── __init__.py
│   ├── registry.py            # models.yaml -> ModelEntry（校验 + config.json 兜底推断）
│   ├── loader.py              # 按 arch 加载 model + tokenizer（causal→AutoModelForCausalLM）
│   ├── genome.py              # GFF 解析 + pyfaidx 取序列（负链反互补、flank）
│   ├── packing.py             # 长度降序贪心打包 + padding 率计算
│   ├── embedder.py            # 前向 + 可配置 mask-aware 池化（mean/last/…）+ 显存日志
│   ├── output.py              # 统一落盘 + 断点判断 is_done()
│   └── adapter/               # tokenizer 适配器（序列预处理 + tokenize 参数）
│       ├── base.py            # SequenceAdapter 抽象 + get_adapter 工厂
│       ├── singlebase.py      # 单碱基 1bp=1tok（rice/muxi）
│       ├── kmer.py            # kmer3(OryzaG3) / kmer6(GENERator 补A) / kmer6_cls(Botanic/AgroNT)
│       └── char.py            # 字符级（NTv3 / PlantCAD2）
└── output/                    # 运行生成（路径见 run yaml output_dir）
    └── {model_name}/
        ├── gene_embeddings.npy   # [N, H] float32，行序 = GFF 基因顺序
        ├── gene_names.txt        # gene_id 逐行，与 npy 行序一致（跨模型对齐键）
        ├── meta.tsv              # gene_id chr start end strand seq_len token_len truncated
        └── model_info.json       # hidden/layers/context/layer/耗时/截断统计（溯源）
```

---

## 3. 快速开始

```bash
# 1) 按 run_example.yaml 跑（其中列出的模型）
python main.py --run configs/run_example.yaml

# 2) 覆盖 yaml，只跑指定模型
python main.py --run configs/run_example.yaml --models rice_1B_stage2_8k NTv3_650M_pre

# 3) 跑注册表里全部模型
#    把 run yaml 中 models 留空即可

# 4) 冒烟测试：只取前 500 条 / 前 10% 基因（产物只含这些基因）
python main.py --run configs/run_example.yaml --max-genes 500
python main.py --run configs/run_example.yaml --gene-frac 0.1

# 5) 多卡
python main.py --run configs/run_example.yaml --gpus 0 1             # 多卡串行(模型轮流放不同卡)
python main.py --run configs/run_example.yaml --gpus 0 1 --parallel  # 多卡并行(每卡一进程)
python main.py --run configs/run_example.yaml --serial               # 强制单卡串行
```

### 运行前必改
`configs/run_example.yaml` 中的 `gff` / `fasta` / 模型 `path`（在
`configs/models.yaml` 内）需指向你机器上实际存在、且为 HF 格式的目录。

### 冒烟建议
正式全量前先小批量验证：挑 1–2 个模型，确认产物
`output/{model}/gene_embeddings.npy` 形状为 `[N, hidden]` 且 `hidden`
与预期一致（rice=1024、NTv3=1536、PlantCAD2-L=3072、GENERator-1.2b=2048…）。

---

## 4. 配置说明

### `configs/models.yaml` —— 模型注册表
新增模型 = 追加一条配置。字段：

| 字段 | 取值 | 说明 |
|---|---|---|
| `path` | 目录 | HF 格式模型目录（**必填**） |
| `arch` | `causal` / `encoder` | decoder LM 用 `AutoModelForCausalLM`；否则 `AutoModel` |
| `tokenizer` | `singlebase` / `kmer3` / `kmer6` / `kmer6_cls` / `char` | 见 §5 |
| `max_context` | int / `null` | 上下文 token 上限；`null` = 无硬上限不截断 |
| `layer` | int / 省略 | 取第几层 hidden；缺省跟随 run yaml 的 `layer`（-1=最后一层） |
| `pooling` | 同 run yaml / 省略 | 覆盖 run 级池化；建议 causal→`last`、`kmer6_cls`→`first`；省略则跟随 run yaml |
| `max_tokens` | int | 每批总 token 限额（按显存/注意力复杂度调整） |
| `flash_attention` | bool | 是否用 `flash_attention_2`（仅支持的架构） |

> `hidden_size` / `num_layers` / `max_context` 若未配置，会从模型目录
> `config.json` 自动兜底推断（含 RCPS 的 `d_model*2`）。

### `configs/run_example.yaml` —— 一次运行
| 字段 | 说明 |
|---|---|
| `models` | 要跑的模型名列表；留空/省略 = 全部 |
| `models_registry` | 模型注册表路径（相对 run yaml） |
| `gff` / `fasta` | 基因组注释 / 参考序列 |
| `flank` | 基因两侧侧翼 bp |
| `max_len` | `0` = 跟随各模型上下文；`>0` = 全局统一截断（所有模型一致） |
| `layer` | 默认层；可被 `models.yaml` 中单模型 `layer` 覆盖 |
| `pooling` | 池化方式（见 §4.3）；可被 `models.yaml` 中单模型 `pooling` 覆盖 |
| `pool_k` | 仅 `pooling: mean_last_k` 生效：末尾取多少 token（默认 64） |
| `pool_trim` | 仅 `pooling: mean_trim` 生效：首尾各去掉的比例（默认 0.25，须 `0 <= v < 0.5`） |
| `l2_normalize` | `true` = 池化后按行 L2 归一化（只保留方向，丢掉常随长度变化的模长） |
| `output_dir` | 产物根目录（相对 run yaml 解析） |
| `resume` | 产物已存在**且 `pooling`/`layer` 等配置一致**则跳过；配置变了会自动重跑 |
| `gpus` | GPU id 列表（如 `[0,1]`）；多卡时默认**串行**轮流放不同卡 |
| `parallel` | `true` 时多卡**并行**（每卡一进程，同时跑不同模型；需显存足够） |
| `device` | 单卡设备：`cuda` / `cuda:0` / `auto`（单模型跨多卡）/ `cpu` |
| `batch_size` | `>0` 固定每批条数（忽略 max_tokens）；`0` 用各模型 `max_tokens` 动态打包 |
| `max_genes` / `gene_frac` | 只跑 GFF 前 N 条 / 前 fraction（冒烟测试用，可被 CLI `--max-genes`/`--gene-frac` 覆盖） |

### 4.3 池化方式（`pooling`）

一个区间有 $T$ 个 token，需要聚合成一个向量。`embedder.py` 支持：

| `pooling` | 取哪些 token | 适用场景 |
|---|---|---|
| `mean`（默认） | 全部真实 token 的均值 | 旧行为；**causal 模型下会把长度写进 PC1** |
| `max` | 全部真实 token 的最大值 | 关心显著 motif 时 |
| `first` | 第一个真实 token | `kmer6_cls`（ESM 风格）的 `<cls>` |
| `last` | 最后一个真实 token | causal/decoder（末位已通过注意力看到全部上文） |
| `mean_last_k` | 末尾 `pool_k` 个真实 token 的均值 | 保留结尾上下文，同时弱化长度 |
| `mean_trim` | 去掉首尾各 `pool_trim` 比例后的均值 | 缓解首尾边界效应 |

实现要点：
- 选位置用 **token 序号**（`mask.cumsum`）而非张量下标，因此**与 padding side 无关**
  （HF 部分 causal 模型默认左 padding）；
- 池化前 hidden 统一升 fp32（见 §9）；
- `max` 用 `masked_fill(-inf)` 后再取 max，保证 pad 不参与。

> **为什么要能多选**：全 token 均值池化会把序列长度编码进表征。实测
> `rice_1B_stage2_8k` 上 PC1 与 $\log_{10}(\text{seq\_len})$ 的 Spearman 达 **0.957**、
> 方差占比 **72.7%**，UMAP1 几乎就是长度轴。causal 模型改用 `last` 是首选修法
> —— 比“把输入截断到固定 100bp”好：不丢内容、不引入分布漂移。

运行时会把实际使用的池化写入 `model_info.json`；`resume` 会比对配置，
**改了池化不会被静默跳过**，而是自动重跑。

---

## 5. Tokenizer 适配器（adapter）

| adapter | 对应模型 | 关键行为 |
|---|---|---|
| `singlebase` | rice/muxi 系列 | 1 bp = 1 token，`add_special_tokens=False` |
| `kmer3` | OryzaG3 | 非重叠 3-mer |
| `kmer6` | GENERator v2 | 非重叠 6-mer；**输入长度必须为 6 的倍数**，否则尾部追加 `<oov>` → 自动左侧补 `A` |
| `kmer6_cls` | Botanic0 / AgroNT | 6-mer + 自动前置 `<cls>`（ESM 风格），`add_special_tokens=True` |
| `char` | NTv3 / PlantCAD2 | 字符级（含 N），1 碱基 1 token |

每个 adapter 负责两件事：
1. `prepare_seq(seq, max_len)`：大小写、k-mer 补齐、全局 max_len 截断回退；
2. `tokenize_kwargs()`：返回 tokenize 时的特殊参数。

公共方法 `truncate_to_context(seq)`：按该模型 `max_context` 右截断并标记。

---

## 6. 输出格式

- **`gene_embeddings.npy`**：`[N, hidden]` float32。`N` = 基因数；行 i 对应
  `gene_names.txt` 第 i 行；各模型同一行 index = 同一基因 → **按 index 对齐即跨模型对齐**。
- **`gene_names.txt`**：gene_id，每行一个（GFF 原始顺序）。
- **`meta.tsv`**：`gene_id chr start end strand seq_len truncated`；
  `seq_len` 是该模型**实际看到的**（预处理/截断后）序列长度（bp）。
  单碱基模型下 token_len = seq_len；k-mer 模型请按 tokenizer 换算。
- **`model_info.json`**：模型元信息与运行统计（含 `pooling` / `pool_k` / `pool_trim` /
  `l2_normalize` / `layer` / `n_truncated` 等），用于实验溯源；`resume` 依赖这些字段判断是否重跑。

---

## 7. 故障排查

| 现象 | 处理 |
|---|---|
| 某模型报错、其它正常 | 设计如此：单模型失败不阻断；看日志堆栈 |
| flash_attention 加载失败 | loader 自动回退默认实现（仅 causal + 开启时） |
| OOM | 降低该模型在 `models.yaml` 的 `max_tokens`（全注意力短上下文模型建议 32768） |
| 需要 `trust_remote_code` | loader 已统一开启；首次会拉取仓库内 modeling 文件，需联网 |
| 序列超长被截断 | 属预期；见 `meta.tsv` 的 `truncated` 列统计 |
| 报错 "embedding 含非有限值(±inf/NaN)" | 池化长序列时 fp16 求和溢出所致；已内置 fp32 修复 + 落盘校验（见 §9）。若仍触发，检查该模型 `hidden_size`/`max_tokens` 是否配置错误 |

---

## 9. fp16 求和溢出（已修复）与数据校验

### 背景
模型以 fp16 加载时，`output_hidden_states` 的 hidden states 也是 **fp16**。
若直接在其精度下沿 token 轴做均值池化求和 `(hs·mask).sum(1)`，当序列很长
（数万 token，如超长 Repeat/基因区域）时，**部分同号维度的累计和会超出
fp16 上限 65504，溢出为 `±inf`**。实测 32k token 序列在 dim 319/835/956 等
维复现 `-inf`；hidden 本身全有限（abs max ≈ 7.7），纯属求和阶段溢出。
单条前向（序列短 / 无长序列）不会触发，因此易被忽略。

### 修复
`embedder.py` 的 `embed_batch` 中，**求和 / 求均值统一提升到 fp32**：

```python
hs = out.hidden_states[layer].float()        # [B, T, H] 先升 fp32
emb = pool_hidden(hs, mask, pooling, pool_k, pool_trim)   # 池化全程 fp32
```

验证：同样 batch 重跑后 embedding 全有限（范围约 [-2.5, 2.3]），且与原实现
未溢出维度的数值仅差 ~1e-3（fp16 舍入级）——只修正了溢出维度，其余不变。

### 落盘校验（防御性兜底）
`output.py` 的 `save_run` 在写入前调用 `_check_finite`：
`np.isfinite(embs)` 不全为 True 时**直接抛错**，并报出涉及的行（gene_id、
行号、非有限维度数）与总个数，避免脏数据流入下游评测。因此若某模型报
`ValueError: embedding 含非有限值...`，即为该模型数值异常，需按上述排查。

### 提醒
2026-09 之前用旧代码生成、已含 `-inf` 的 `gene_embeddings.npy`（如
`output/IRGSP/rice_1B_stage2_8k/`）属脏数据，**需删除重跑**后再用于下游。

---

## 10. 与配套项目的关系

- `../extract_gene_embeddings.py`：**单模型**（OGR rice_1B）版脚本，本目录是其多模型泛化；
- `benchmarks_botanic`（另一仓库）：多模型**下游评测**框架（MLP/XGB/RF + 报告），
  本目录产物（按模型分目录的 `.npy`）可与它对接做下游任务；
- 跨模型比较注意：encoder(MLM 双向) 与 decoder(CLM 因果) 表示空间不同，
  建议用"统一下游分类器"而非直接裸 cos 相似度比较。
