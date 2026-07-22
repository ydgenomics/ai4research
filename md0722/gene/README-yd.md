



我们把个体基因组（按基因TSS上下padding 500kb）输入genos的基模，提取最后两层的embedding，作为后续建模个体差异的输入

## 参考
面向个人基因组的大规模可扩展框架——SAGE-net
Nature Methods 文章《Decoding sequence determinants of gene expression in diverse cellular and disease states》提出了 Decima，一个从基因周围 DNA 序列预测细胞类型和疾病状态特异性基因表达的 sequence-to-function 模型。

先了解这个工作，概述其工作
模型架构是什么，参数量是多少，输入和输出是什么？
基因水平还是碱基水平输出？
如何设计的训练集和测试集？评测的指标如何设计的？评测了那些其它方法？
给具体的不同方法的评测指标，数值范围

提示词优化，最终希望AI给出学术性的文档

一句话总结包括架构，输出，指标，概述

- [2026|Nat. Methods | 解读个体遗传变异的功能影响: 一种可扩展的序列–功能预测方法](https://mp.weixin.qq.com/s/Q-vMqcDEuYo0Xh0LPij-zg)
  - SAGE-net 采用双分支轻量级 1D-CNN 架构与动态数据(40 kb)管道，通过解耦预测输出基因水平的表达量及其个体残差，在跨个体相关系数（$R_{\text{indiv}}$）上实现了从零到 0.3+ 的突破，攻克了传统参考模型无法预测个人基因组表达差异的计算与建模瓶颈。
  - 对比学习、两个hap、基因水平、个体表达与参考的平均表达
- [2026|同一段 DNA，为什么只在某些细胞里“亮起来”？Nature Methods 发布 Decima](https://mp.weixin.qq.com/s/KpVSIWVy13ELrJ2ETUyoCQ?scene=1)
  - Decima 是一个基于 1D-CNN 和 Transformer 的单基因 sequence-to-function 模型，输入以 TSS 为中心的 DNA 序列512 Kb，在基因水平直接预测其在 200+ 种单细胞及疾病状态下的 Pseudobulk 表达向量，并在跨染色体未见基因（Unseen Genes）的测试集上实现了跨条件中位数 Pearson 相关系数 $r \approx 0.52$ 的零样本预测能力。