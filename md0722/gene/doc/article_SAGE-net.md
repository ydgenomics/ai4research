基于华盛顿大学/斯坦福大学 Sara Mostafavi 实验室（Spiro et al., 2025/2026）的论文与开源成果，将针对 **SAGE-net** 整理一份**系统、严谨且符合计算生物学规范的学术技术报告**。

为了便于学术交流与文档归档，下方系统整理了该框架的核心机制与定量 Benchmark：

---

# SAGE-net：面向个人基因组的大规模可扩展 S2F 预测框架技术报告

> **论文全称**：*A scalable approach to investigating sequence-to-function predictions from personal genomes*
> **算法全称**：**SAGE-net** (*Small And Good Enough CNN / Sequence-to-Activity Genome Evaluation network*)
> **核心突破**：解决了传统参考基因组模型（如 Enformer）在输入个人 WGS 序列时“无法捕获个体间表达差异（Inter-individual variation）”的根本瓶颈，并通过数据流优化实现了 TB 级个人基因组的大规模训练。

---

## 一、 工作概述与设计动机

### 1.1 核心痛点与问题背景

现有的基因组深度学习模型（Sequence-to-Function, S2F）主要面临以下挑战：

* **参考基因组模型（如 Enformer, Borzoi）的局限**：模型在训练阶段仅观察单一参考基因组（Reference Genome），擅长预测不同基因位点之间的平均表达强弱（Across-loci/Inter-gene variation），但在直接推理包含个人 SNP/Indel 的单倍体序列时，**预测出的个体间表达差异相关性近乎为零 ($R_{\text{indiv}} \approx 0$)**。
* **高昂的数据计算与存储瓶颈**：跨数百至数千个体的全基因组测序（WGS）数据涉及复杂的单倍体相位（Phasing）。如果将每个个体的全基因组预编码为庞大的深度学习矩阵，会产生巨大的存储与 I/O 开销，阻碍了端到端的个人基因组训练。

### 1.2 SAGE-net 的核心解决方案

SAGE-net 提出个人基因组训练（Personal Genome Training）范式，包含两大核心贡献：

1. **即时动态数据管道（On-the-fly Data Pipeline）**：在 DataLoader 阶段，基于 VCF 文件将个体独特的变异实时注入参考基因组并进行 One-Hot 编码，摆脱了预存庞大矩阵的限制。
2. **解耦预测与对比学习架构**：将“基因均值”与“个人残差”剥离，结合对比学习提取微小的顺式调控变异特征。

---

## 二、 模型架构、参数量与输入输出规范

### 2.1 模型架构设计

SAGE-net 采用了模块化的双分支/解耦架构，分为 **r-SAGE-net**（Reference 分支）与 **p-SAGE-net**（Personal 残差分支）：

* **Backbone 结构**：以多层一维卷积神经网络（1D-CNN）为基础，搭配池化（Pooling）与非线性激活函数，专门提取启动子/增强子区域的 Motif 局部特征。
* **解耦双路径推理**：
* **均值分支 ($f_{\text{mean}}$)**：仅以参考基因组序列为输入，预测群体平均表达量。
* **个人残差分支 ($f_{\text{residual}}$)**：同时输入参考序列与个体的两个单倍体序列（Haplotype 1 和 Haplotype 2），通过对比特征编码提取由于个人变异引起的偏差。



```
                  ┌──────────────────────────────┐
                  │ Reference Genome (1 x 40 kb) │
                  └──────────────┬───────────────┘
                                 │
                             r-SAGE-net
                                 │
                                 ▼
                    Mean Expression (群体平均表达)
                                 
───────────────────────────────────────────────────────────────────

  ┌──────────────────────────┐  ┌──────────────────────────┐
  │ Haplotype 1 (1 x 40 kb)  │  │ Haplotype 2 (1 x 40 kb)  │ (Personal WGS)
  └─────────────┬────────────┘  └─────────────┬────────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                           p-SAGE-net
            (Contrastive / Residual Feature Encoder)
                               │
                               ▼
               Personal Residual Deviation (个体表达偏差)

```

### 2.2 参数量与计算效率

* **模型参数量**：轻量级设计，参数量级通常在 **数百万（< 10M）** 左右，远小于 Enformer（~2.5 亿参数）。
* **推理与训练效率**：相比 Enformer，SAGE-net 的推理速度提升了 **~70 倍**，使得在大规模群体 WGS 队列（如 ROSMAP、GTEx）上开展全基因组级别的微调和变异图谱分析成为可能。

### 2.3 输入与输出规范

| 维度 | 规格定义 |
| --- | --- |
| **输入序列长度** | 以基因转录起始位点（TSS）为中心，上下游各延伸 20 kb，总长度为 **40 kb** ($40,000 \text{ bp}$)。 |
| **输入数据形状** | • **r-SAGE-net**: $[B, 4, 40000]$ (4 通道 One-Hot 编码: A, C, G, T)<br>

<br>• **p-SAGE-net**: $[B, 3, 4, 40000]$ (包括 1 个 Reference 序列 + 2 个 Phased Haplotypes) |
| **输出分辨率** | **基因水平（Gene-level）** 或 **区域窗口水平（Bin-level, 如 DNA 甲基化）**。 |
| **输出张量形式** | 连续标量回归输出：<br>

<br>$$\hat{y}_{\text{final}} = \hat{y}_{\text{mean}} + \hat{y}_{\text{residual}}$$

 |

---

## 三、 数据集划分、评估指标与基线方法

### 3.1 训练集与测试集设计（二维交叉分割）

为了严谨评估模型对“新个体”与“新基因”的泛化能力，SAGE-net 采用了二维交叉测试划分策略：

1. **场景 A：未见过个体（Held-out Individuals, Seen Loci）**
* **设计**：训练集与测试集包含相同的基因位点，但个体完全不重叠（如用 80% 人群训练，在剩余 20% 人群上预测）。
* **评估目的**：测试模型能否根据**新个体独特的 SNP/Indel 组合**预测该个体的表达偏差。


2. **场景 B：双重未见过（Held-out Individuals, Held-out Loci）**
* **设计**：测试集使用未见过的基因位点和未见过的个体（对齐 Enformer 的染色体分割策略，如 Hold-out 染色体上的基因）。
* **评估目的**：评估模型是否学习到了跨位点通用的“顺式调控语法（Regulatory Grammar）”。



### 3.2 评估指标设计

* **跨个体相关系数 ($R_{\text{indiv}}$ / Per-gene Pearson $R$)**：
固定单个基因 $g$，计算测试集所有个体中的预测表达量与实测表达量的相关性：

$$R_{\text{indiv}}(g) = \text{Corr}\left( \{\hat{y}_{g, i}\}_{i=1}^N, \{y_{g, i}\}_{i=1}^N \right)$$



*直观含义：排除基因间表达强弱影响，纯粹评估“人与人之间预测准确度”。*
* **跨位点相关系数 ($R_{\text{loci}}$ / Across-loci Pearson $R$)**：
固定单个个体 $i$，计算其全基因组范围内所有测试基因预测均值与实际表达量的相关性。
* **残差预测 MSE / $R^2$**：针对剔除群体均值后的表达残差（Residuals）进行直接比对。

### 3.3 对比基线（Baselines）

1. **参考基因组深度学习模型**：Enformer、Borzoi（零样本推理与微调版）。
2. **统计遗传学 / TWAS 模型**：PrediXcan（基于 Elastic Net / Lasso 的线性回归模型）。
3. **消融控制组**：r-SAGE-net（仅参考序列均值模型）、无对比学习分支的纯 CNN。

---

## 四、 核心评测结果与定量数值对比

在 ROSMAP（人脑前额叶皮层 WGS + RNA-seq/DNAm 队列）、GTEx 及 Geuvadis 数据集上的实际定量表现总结如下：

### 4.1 基因表达预测（RNA-seq）定量性能比对

| 模型方法 | 架构类型 | 跨个体相关性中位数 ($R_{\text{indiv}}$) | 跨位点相关性 ($R_{\text{loci}}$) | 推理相对耗时 |
| --- | --- | --- | --- | --- |
| **Enformer (Zero-shot)** | Transformer (2.5 亿参数) | **$\approx 0.00 \sim 0.05$** | **$0.75 \sim 0.85$** | $70\times$ ( Baseline ) |
| **Fine-tuned Enformer** | Transformer (微调版) | $0.15 \sim 0.25$ | $0.75 \sim 0.85$ | $70\times$ |
| **PrediXcan / TWAS** | 统计线性回归 (Elastic Net) | $0.20 \sim 0.32$ | — | $< 0.01\times$ |
| **r-SAGE-net (Mean)** | 轻量 1D-CNN (参考序列) | $0.00$ *(输出常数)* | $0.65 \sim 0.75$ | **$1\times$** |
| **p-SAGE-net (Proposed)** | 轻量 1D-CNN (个人基因组) | **$0.22 \sim 0.32$** | $0.65 \sim 0.75$ | **$1.2\times$** |

### 4.2 重要学术发现与规律

1. **针对“未见过个体”与“未见过基因”的泛化差异**：
* 在未见过个体（Held-out Individuals）上，个人基因组训练显著提升了 $R_{\text{indiv}}$（从 ~0 升至 0.30+）。
* 在未见过基因（Held-out Loci）上，RNA 表达预测的泛化提升较为有限。原位突变分析（In-silico Mutagenesis, ISM）表明，模型在表达预测任务中倾向于“记忆”具体的预测性突变，而非完全掌握了通用的顺式调控语法。


2. **表观遗传学（DNA 甲基化 DNAm）的突破性表现**：
* 与 RNA-seq 不同，当 SAGE-net 用于预测 **DNA 甲基化（DNAm）** 时，在**未见过个体和未见过基因位点**上均实现了显著的泛化性能（$R_{\text{indiv}}$ 中位数达到 **$0.40 \sim 0.55$**）。
* **结论**：DNA 甲基化等表观遗传标记在序列层面的调控语法比基因表达量更容易被深度学习模型捕获和通用化。



---

## 五、 总结与学术评价

SAGE-net 厘清了基因组深度学习领域长期存在的一个误区——**高 $R_{\text{loci}}$ 不等于能预测个人基因组差异**。其主要学术价值体现在：

1. **工程创新**：提供了开源的高效 Python/PyTorch 框架 `SAGEnet`，解决了个人基因组大规模训练的数据流 bottlenecks。
2. **范式转变**：证明了通过“解耦均值与残差”以及“对比学习”，轻量化 CNN 即可在预测个人表达差异上达到甚至超越重型 Transformer 模型（如 Enformer）微调后的效果，同时计算开销降低了 70 倍。
3. **机制洞察**：指出了当前 S2F 模型预测基因表达的局限（偏向特定变异记忆），并证明了表观遗传学预测才是解码通用顺式调控语法的关键突破口。

---