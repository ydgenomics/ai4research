# omics2phenotype：大模型建模多组学数据赋能水稻代谢物预测

> 修订版说明：在原始 README 基础上，补充了问题定义（任务分层）、方法论路线、数据资源、评估基线等结构性内容，供团队讨论与迭代。

---

## 一、背景与动机

水稻智能育种的核心诉求，是建模宏观（表型）与微观（分子）之间的关系，回答"表型为什么如此"的问题——这需要深入分子层面寻找答案与机制。

- 现有 **sequence2function** 模型（如 Enformer、GenOmics 及基于 DNA 基模的逐碱基表达预测）主要建模配对的 DNA–RNA 数据，类似的传统生信分析是 eQTL 与 TWAS，回答的是"序列变化与表达变化的关系"，定位到的是 eQTL 位点与 eGene。
- 在此框架下，**表达变化是表型变化的"果"（下游结果）**，eQTL/TWAS 提供的只是 potential（遗传决定潜力）。从 DNA 到表型历经复杂的生物学过程：**RNA 有差异，蛋白质/代谢物未必有差异**（转录-表型解耦：翻译调控、蛋白稳态、代谢网络缓冲），因此我们期望建模更深层的结果。
- gradient: 基因型 → 转录（RNA）→ 翻译（蛋白）→ 代谢 → 表型，逐层解耦、逐层衰减。

## 二、为什么选代谢组

- **代谢物是最接近表型的组学**：代谢物含量本身就是一种表型（如中药中具有药用价值的代谢物），定位到影响代谢物含量的关键位点，对改良高产优质中药/稻米品质意义重大。
- **代谢物的数据获取成本远高于测序**：若能由序列直接预测，可实现数据的 **imputation**，极大降低育种与筛选成本。
- **注意：代谢物可预测性并非均一**——只有遗传力较高的代谢物才值得建模。建议以"遗传力分层"或"已报道 mQTL 位点比例"作为可行性分诊依据，优先建模遗传决定强的代谢物。

## 三、任务定义（关键决策）

需要明确区分两个场景，二者共享同一基模与数据管线：

| 场景 | 输入 | 输出 | 定位 |
|------|------|------|------|
| **A：机制研究** | DNA + RNA（配对样本） | 代谢物含量 | 探索转录→代谢的解耦机制，RNA 作为输入 |
| **B：育种/应用** | 仅 DNA | 代谢物含量 | imputation 价值所在（推理时拿不到 RNA），RNA 仅作为训练时的中间监督或知识蒸馏桥梁 |

**推荐架构：级联 + 多任务**
1. 复用已训练的 DNA→RNA 预测头（如 GenOmics，单碱基 PCC ≈ 0.94）作为**中间监督**；
2. 在其之上增加 RNA→代谢物、或 DNA→代谢物端到端分支；
3. 两个场景共享基模与数据准备，评估时分别报告。

**关于表达数据的定位（贯穿全项目）**：基模预测的 RNA 本质是 **genetically-determined expression（遗传决定的表达）**，是"潜力"而非实测值。它与真实 RNA-seq 的差异（残差）恰好承载了环境/状态信息——这一视角可作为后续机制研究的切入点。

## 四、多组学建模的可行性论证

从宏观来看，表型由基因型与环境共同决定（进一步可细分为田间管理等三个因素）：

- **DNA 代表基因型**（潜力/限定条件）；
- **RNA 是 DNA 转录的产物**，既反映基因型，也受环境影响——可视为**同时携带基因型信息与环境信息**的统一载体；
- 因此，基于 DNA + RNA 建模表型/代谢物在信息论上是完备的：`表型 ≈ f(DNA, 环境) ≈ g(DNA, RNA)`。

**尚未解决的关键问题：环境编码（G×E）**
- RNA 隐含环境信息，但显式建模环境（如低氮 CK/ST、盐胁迫处理）需要条件编码器——现有 Enformer 式架构中 biosample 用 embedding/多通道表达，**环境状态需要类似的条件注入机制**，这是超出当前 U-Net 架构的新课题；
- "表型 = 基因型 + 环境"不是简单加法，而是 **G×E 互作**，需要模型显式学习交互项。

## 五、技术路线：从序列到代谢物的聚合

代谢物是多基因共同作用的产物（多基因/网络调控），因此必须引入**基因网络或基因集**的先验：

```
DNA 窗口 (逐碱基) 
  → 基模编码 (OGR / GenOmics，1.25B MoE，32K 上下文)
  → U-Net 解码 → 逐碱基多组学信号 (RNA / ATAC / 甲基化)
  → 基因级聚合层 (逐碱基 → 基因表达量)
  → 先验网络读出层 (GNN / 图注意力，网络约束而非后验富集)
  → 代谢物含量预测
```

- **基因级聚合**：把逐碱基输出聚合到基因（如外显子/UTR 区间均值），这是"序列窗口 → 全基因组代谢物"跳跃的必要桥梁；
- **网络/基因集来源**（现成的、可落地）：
  - 共表达网络：可从 404 样本根转录组自建（WGCNA）；
  - 已知基因集：盐胁迫响应基因、氮代谢基因（商连光团队数据中已有清单）；
  - 通路数据库：KEGG、MapMan；
  - 已报道基因：mGWAS 定位已报道基因（可用作验证集）；
- **结构建议**：网络作为**先验约束**（结构归纳偏置）嵌入读出层，而非仅作后验富集分析。

## 六、数据资源

### 6.1 自有/在库数据（可直接起步）

| 数据 | 规模 | 用途 |
|------|------|------|
| 商连光团队 NH 材料组学数据 | 251 品种 SNP/SV VCF、NH 代谢组表型、404 样本根转录组（CK/ST）、19 份非洲稻 WGBS、mGWAS 结果、142 个表型性状 GWAS | 本项目核心起步数据 |
| 422 水稻泛基因组（OGR 预训练语料） | 栽培稻 + 野生稻 | 基模预训练 |
| 多品种多组织 RNA-seq BigWig（GenOmics demo） | 品种 × 组织逐碱基轨道 | DNA→RNA 预训练/验证 |

### 6.2 外部公开群体数据集（作为补充/交叉验证）

| 数据集 | 规模 | 组学配对 | 登录号/来源 | 文献 |
|--------|------|----------|-------------|------|
| 华中农大 529 份栽培稻群体 | 529 品种 1:1:1:1 配对 | 基因组 + 幼叶转录组 + 非靶向代谢组 | PRJNA517013 / PRJNA342647；RIGW (ncpgr.cn) / RiceOmics 平台可免比对下载矩阵 | Genome Biology 2019; Nat. Commun. 2024 ("Multi-omics networks governing complex traits in rice populations") |
| 韩国 475 份世界水稻收集 | 421 基因组-代谢组、279 基因组-转录组、64 基因组-蛋白质组 | 基因组 + 转录组 + 蛋白质组 + 挥发性代谢组 | PRJNA744112；蛋白/代谢矩阵见文章补充材料 | J. Adv. Res. 2022 (水稻香味调控) |
| 中科院 ~260 份 RIL 根际群体 | ~260 份宿主-根际 1:1 配对 | 宿主基因组 + 根转录组 + 根际宏基因组/16S | CRA012431 / PRJNA1021456 | Nat. Plants 2025 |
| 籼稻 ZS97/MH63/SY63 全生命周期 | 39 组织配对 | 基因组 + 转录组 + 代谢组(>2500 注释) + PPI | RIGW CREP (crep.ncpgr.cn) | NAR / RIGW 平台 |
| 单细胞多组学图谱 | 116,564 单细胞 | scRNA + scATAC 同细胞配对 | Rice-SCMR (elabcaas.cn/scmr) | Nature 2025 |
| 日本晴翻译组+转录组 | 12 节点 36 重复配对 | RNA-seq + Ribo-seq | CNCB-GSA | Plant Biotech. J. 2021 |
| 幼穗时空表观-转录 | 72 组时空配对 | ChIP-seq/ATAC-seq + RNA-seq | — | Plant Cell 2023 |
| 籼粳逆境表观群体 | Nip/93-11 × 多胁迫 | WGBS + RNA-seq + 图像表型 | CNGBdb / GEO | Cell Reports 2026 |

**数据获取建议**：优先从 RIGW / RiceOmics / CNCB-NGDC 平台下载官方清洗后的矩阵（VCF、表达矩阵、代谢物定量表），跳过原始 Fastq 比对，直接开展下游建模。

## 七、评估与基线

### 7.1 基线

- 代谢物 BLUP / 多性状 GWAS 预测得分；
- 传统机器学习：XGBoost / 线性模型（以基因型或基因表达为输入）；
- 与序列模型的对比：随机 DNA 窗口基线、参考基因组 vs 个体基因组。

### 7.2 指标

- 代谢物层面：预测-实测相关（Pearson/Spearman）、R²；
- **遗传力分层**：按代谢物遗传力分桶报告指标（证明"可预测性"随遗传力变化）；
- 已报道基因重叠：预测聚焦位点是否命中 mGWAS 已报道基因；
- 可解释性：突变模拟（in silico perturbation，如 TAC1 式分析）、归因分析（saliency/ISM）。

### 7.3 与序列模型既有成果的衔接

- 已有 GenOmics 单碱基 PCC ≈ 0.94（DNA→RNA）；
- 下一步里程碑：**DNA→代谢物**（场景 B）与 **DNA+RNA→代谢物**（场景 A）均超越上述基线；
- 需要确认自有数据的**配对性**：251 VCF / NH 代谢组 / 404 根转录组是否同一批材料一一配对——这是项目成立的前提，建议文档中显式标注并以清单核对。

## 八、里程碑规划

1. **数据盘点**:核实 251 品种 VCF / NH 代谢组 / 根转录组的配对关系，锁定基因网络来源；
2. **基线建立**:代谢物 BLUP、XGBoost 基线 + 遗传力分层图谱；
3. **中间监督**:DNA→RNA 头（GenOmics）在群体数据上验证/微调，建立基因级聚合层；
4. **代谢物预测**:场景 A（DNA+RNA）与场景 B（DNA-only）双路线模型与评估；
5. **可解释性与验证**:突变模拟 + 与已报道 mQTL/mGWAS 基因的交叉验证。

---

## 参考文献

- Avsec, Ž. et al. Effective gene expression prediction from sequence by integrating long-range interactions. *Nat. Methods* (2021).(Enformer)
- Rastogi, R., Reddy, A. J., Chung, R., Ioannidis, N. M. Fine-tuning sequence-to-expression models on personal genome and transcriptome data. *bioRxiv* (2024).(ni-lab/finetuning-enformer)
- Chen, W. et al. Large-scale multi-omics genetic analyses in rice. *Genome Biology* (2019).(529 群体)
- Multi-omics networks governing complex traits in rice populations. *Nat. Commun.* (2024).
- Multi-omics analysis reveals the genetic basis of rice fragrance regulation. *J. Adv. Res.* (2022).(韩国 475 群体)
- Genomic and transcriptomic determinants of the rice rhizosphere microbiome assembly. *Nat. Plants* (2025).
- A single-cell multi-omics atlas of rice. *Nature* (2025).
- Comprehensive translatomic and transcriptomic profiling reveals the post-transcriptional landscape of rice development. *Plant Biotech. J.* (2021).
- Spatiotemporal multi-omics profiling reveals epigenetic regulations during rice panicle development. *Plant Cell* (2023).
- Multi-omics integration maps CHH methylation and gene regulatory network. *Cell Reports* (2026).
- Liu, T. et al. Pre-training Genomic Language Model with Variants for Better Modeling Functional Genomics. *bioRxiv* (2025).
- Lin, J. et al. EPInformer: scalable and integrative prediction of gene expression from promoter-enhancer sequences with multimodal epigenomic profiles. *Nat. Commun.* (2026).