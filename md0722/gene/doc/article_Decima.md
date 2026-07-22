# 学术报告：Decima 深度学习模型解析

---

## 摘要 (Executive Summary)

**Decima** 是由 Genentech 研究团队发表于 *Nature Methods* 的新型 sequence-to-function 深度学习模型。传统的基因表达预测模型（如 Enformer、Borzoi）主要基于批量（Bulk）数据或特定细胞系的表观基因组轨道（Assay Tracks），无法直接预测细胞类型（Cell type）**及**疾病状态（Disease state）特异性的 mRNA 表达水平。

Decima 通过从超过 **2,200 万个单细胞/单细胞核 RNA 测序（sc/snRNA-seq）数据**中提取伪批量（Pseudobulk）表达矩阵，以单基因周围的 DNA 序列为输入，直接预测该基因在数百种细胞类型和疾病状态下的表达量向量，实现了跨基因、跨细胞类型及疾病条件的零样本（Zero-shot）泛化预测。

---

## 一、 模型架构与核心参数

Decima 改变了以往以固定基因组窗口（如 100 kb/1 Mb）作为连续轨道输出的模式，采用**以基因 TSS（转录起始位点）为中心**的单基因输入/多状态输出架构。

```
    DNA 序列 (A, C, G, T)  [长度: L bp, 以 TSS 为中心]
                    │
                    ▼
       ┌───────────────────────────┐
       │ 1D 卷积层 (1D Conv Blocks) │ ── 局部顺式调控基元特征提取
       └───────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │   Transformer 编码器层    │ ── 长距离增强子-启动子 (E-P) 相互作用建模
       └───────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │ 池化与全连接层 / 输出头    │ ── 映射至多维细胞/疾病状态空间
       └───────────────────────────┘
                    │
                    ▼
   预测表达量向量 y_hat [维度: K 个 Pseudobulk 状态/轨道]

```

### 1. 模型架构细节

* **输入 (Input)**：以目标基因 TSS 为中心延伸的 DNA 序列（编码为 $4 \times L$ 的 One-hot 矩阵）。输入序列同时包含启动子及近端/远端顺式调控元件（cREs）。
* **主干网络 (Backbone)**：结合 **1D 卷积神经网络 (1D CNN)** 抽取局部 Motif 特征，以及 **Transformer 编码器 (Transformer Encoder)** 捕捉跨越数万个碱基的长距离增强子-启动子（Enhancer-Promoter）相互作用。
* **输出 (Output)**：一个长度为 $K$ 的一维表达向量（$K$ 代表预先定义的 Pseudobulk 细胞状态数，如 201 种细胞类型与 82 种疾病条件的组合）。

### 2. 关键参数与特征维度

* **模型参数量**：根据开源代码与预训练权重（如 Zenodo 托管的 4 个 Replicates），Decima 的参数量约为 **1.2 亿至 1.5 亿（~120M–150M parameters）**。
* **输出分辨率**：**基因水平（Gene-level）**。Decima 并不在全基因组 128 bp 掩码窗口上预测基因组 Track，而是直接针对具体基因在各细胞类型/状态下的**定量标量表达值**（经过 CPM 标准化及 $\log(1+x)$ 变换的 Pseudobulk 值）进行直接回归。

---

## 二、 数据集构建：训练集、验证集与测试集设计

为了验证模型真正学到了“顺式调控密码（cis-regulatory syntax）”而非单纯记忆基因身份，作者设计了严格的**跨染色体基因划分策略（Cross-chromosome Split）**。

```
                      2,200万+ 单细胞 RNA-seq 表达谱
                                   │
                                   ▼
                      构建 Pseudobulk 细胞状态矩阵
                         (K 个状态 × N 个基因)
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
   训练/验证集基因 (N - 1,811)                       测试集基因 (1,811)
[位于训练染色体, 如 chr1-19, 21, 22]               [严格限定于特定染色体, 如 chr8, chr9]
             │                                           │
             ▼                                           ▼
       模型参数训练                                零样本 (Unseen Genes) 评估

```

### 1. 数据集构建

* **语料库规模**：整合来自 Human Cell Atlas 等项目的 **2200 万+ 单细胞/单细胞核 RNA-seq（sc/snRNA-seq）数据**。
* **状态定义 ($K$)**：按照 `Study × Organ/Tissue × Cell Type × Disease` 进行聚合，生成包含 **201 种细胞类型、271 种组织、82 种疾病状态**的 Pseudobulk 表达矩阵。
* **数据预处理**：对每个 Pseudobulk 样本计算 CPM（Counts Per Million），并进行 $\log(1+\text{CPM})$ 方差稳定化变换。

### 2. 测试集与染色体隔离

* **未见基因测试集 (Unseen Genes Test Set)**：评估集合包含 **1,811 个测试基因**，这些基因位于被严格保留（Held-out）的染色体上（如 chr8 和 chr9）。
* **评价逻辑**：在训练阶段，模型**从未接触过这 1,811 个基因的任何 DNA 序列**。测试时，仅输入这 1,811 个基因周围的 DNA 序列，要求模型预测其在 $K$ 个 Pseudobulk 状态下的表达分布。

---

## 三、 评估指标与对比方法

### 1. 评估指标设计 (Evaluation Metrics)

作者设计了两类维度的统计学评估指标：

* **跨基因相关性 (Cross-gene Correlation / Across-gene R)**：
在某一个特定的细胞类型/状态 $k$ 下，计算所有测试基因的预测表达量与真实表达量之间的相关性。评估模型能否分清“在当前细胞中，哪个基因表达高，哪个基因表达低”。
* **跨细胞类型/状态相关性 (Per-gene / Cross-condition Correlation)**：
对某一个特定的测试基因 $g$，提取其在 $K$ 个 Pseudobulk 状态下的预测向量与实际测量向量，计算 **皮尔逊相关系数 (Pearson Correlation, $r$)** 与 **斯皮尔曼秩相关系数 (Spearman Correlation, $\rho$)**。

$$\text{Pearson } r_g = \frac{\sum_{k=1}^K (\hat{y}_{g,k} - \bar{\hat{y}}_g)(y_{g,k} - \bar{y}_g)}{\sqrt{\sum_{k=1}^K (\hat{y}_{g,k} - \bar{\hat{y}}_g)^2} \sqrt{\sum_{k=1}^K (y_{g,k} - \bar{y}_g)^2}}$$



这是论文最核心的指标，代表模型捕获**细胞类型特异性**和**疾病差异表达**的能力。

### 2. 基线对比方法 (Baseline Methods)

Decima 与以下几类主流模型和基线进行了系统性对比：

1. **SOTA 全基因组 sequence-to-function 模型**：
* **Enformer**（DeepMind 提出的基于 Transformer 的 100 kb 跨度表达量/表观预测模型）。
* **Borzoi**（基于 Conv-Transformer 的 1 Mb 跨度全基因组及 RNA-seq 轨道预测模型）。


2. **基线模型/简单均值模型 (Mean / Median Baselines)**：
* **Gene Average/Tissue Average**：直接使用训练集中该基因在其他组织/细胞类型的平均表达量作为预测值。



---

## 四、 具体评估性能数值与范围 (Benchmark Results)

在 **1,811 个未见基因（Unseen Test Genes）** 的 **Per-gene Cross-condition 相关性** 评测中，各方法的定量表现如下：

### 1. 跨细胞类型/条件基因表达预测性能对比 (Per-gene Pearson $r$)

| 方法/模型 | 模型类型 / 训练输入 | 1,811 未见基因的中位数 Pearson $r$ | Pearson $r$ 数值分布范围 | 细胞类型特异性预测能力 |
| --- | --- | --- | --- | --- |
| **Decima** | 单细胞 Pseudobulk Sequence-to-Expression | **0.45 – 0.58**（中位数 $\approx \mathbf{0.52}$） | **-0.10 到 0.88**（主体集中在 0.30–0.75） | **极高**（能精准预测特异细胞类型中的高表达） |
| **Borzoi** | Bulk RNA-seq / CAGE / ChIP-seq Track | 0.22 – 0.35 | -0.20 到 0.60 | 中等（受限于 Bulk 混合信号） |
| **Enformer** | CAGE / DNase / ChIP-seq Track | 0.15 – 0.28 | -0.25 到 0.55 | 较低（难以泛化到细分单细胞类型） |
| **Mean Baseline** | 训练集细胞类型均值标量 | ~ 0.00（无跨条件变异度） | 0.00 | 无（无法捕获基因调控动态） |

> **性能解析**：
> 1. **为什么传统的 Enformer/Borzoi 得分较低**：传统模型针对 Bulk CAGE 或 Bulk RNA-seq 轨道进行训练，细胞类型消融在混合组织中；当将其预测输出映射到单细胞级别的 Pseudobulk 细胞类型时，出现严重的“平滑效应（Smoothing）”，对细胞类型特异性表达变异的解释力较弱。
> 2. **Decima 的优势**：在 1,811 个测试基因中，有超过 60% 的基因其 $r > 0.40$；对于高度细胞类型特异性的标志基因（Marker Genes），Decima 的预测相关性可达 **$r > 0.75$**。
> 
> 

---

## 五、 Decima 的核心应用与创新点总结

1. **零样本未见基因表达预测 (Zero-shot Gene Expression Prediction)**
仅凭 DNA 序列，精准预测未见基因在不同细胞类型和疾病状态下的表达量。
2. **疾病与组织驻留状态变异解码 (Disease State & Cell State Decoding)**
能敏锐捕捉同一细胞类型在“健康 vs 疾病”（例如阿尔茨海默病、炎症状态）条件下的顺式调控差异，定位驱动差异表达的关键 TF 结合位点。
3. **非编码区突变效应预测 (Non-coding Variant Effect Prediction, In Silico Mutagenesis)**
可在单细胞分辨率下评估非编码区 SNP 或突变对特定细胞类型表达量的破坏性或激活效应。
4. **从头设计细胞类型特异性合成 DNA (De Novo Design of Cell-type-specific Promoters/Enhancers)**
通过反向传播与梯度优化，从头生成能够在特定细胞类型/疾病条件下特异性驱动基因表达的合成调控元件。