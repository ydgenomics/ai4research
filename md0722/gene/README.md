- 先把基因水平的个体差异先建模好，一方面是后面主要看基因水平，另一个层面是基因水平的噪声相比单碱基水平要小
- 400人的monocyte的单细胞数据已经pseudobulk为个体x基因的矩阵，从caiting的elasticnet的结果筛选比较好从序列建模个体差异的基因作为训练基因

我们现在有400人的个人基因组文件，400人的monocyte的单细胞数据已经pseudobulk为个体x基因的矩阵，elasticnet的结果筛选比较好从序列建模个体差异的基因作为训练基因，如何建模个体差异表达？

梁健强
```
genos小模型生产的高斯加权后的embedding位置：/mnt/a100-nas-new/peixunban/tanxinjiang/13.SNPbag.pre_exp/model_training/embeddings_gaussian_sigma15.0
查询指定染色体和指定位置变异embedding的参考脚本：/mnt/genos100-new/peixunban/smk/tasks/varformer_modify/scripts/main/query_embedding.py

@徐小龙 @叶程 人群变异子模型的embedding跑完了，由于区间内的SNP数量大部分都大于1024个，所以用了8192length的模型跑了embedding，结果路径
/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_res_260708/    ← 28.7 GB
│
├── inference_manifest.json           ← 全局清单 (窗口数、耗时等)
│
├── ACAP3_ENST00000492936.5_win_1/    ← 窗口1 (chr1:805928-1805928, 1995 SNPs)
│   ├── features.pt                   ← 共享: ref_vecs/alt_vecs[1995,512] + genotypes[101,1995]
│   ├── meta.json                     ← 窗口元信息 (SNP数、1KGP交集率等)
│   ├── CIMA-H005_CIMA-H005.vcf.pt    ← 样本1 ┐
│   ├── CIMA-H009_CIMA-H009.vcf.pt    ← 样本2  │
│   ├── ...                           ← ...    │ 每样本 2-6 MB
│   └── CIMA-H324_CIMA-H324.vcf.pt    ← 样本101┘
│
├── SMIM1_ENST00000642557.4_win_2/    ← 窗口2 (2541 SNPs)
│   ├── ...
│   └── ... (101个样本.pt + features.pt + meta.json)
│
├── ...
│
└── MAPK12_ENST00000497036.5_win_100/ ← 窗口97 (2007 SNPs)
    └── ...

每个PT文件结构：
hidden_states	[L, 1024]	float16	hap0/hap1 池化后的上下文嵌入
logits	[2, L, 2]	float16	每个 haplotype 的 [REF, ALT] 分对数
alt_probability	[L]	float16	替代等位基因概率（haploid 平均）
positions	[L]	int64	基因组坐标
variant_keys	[L]	list	如 chr1_805928_G_A

一共有97个窗口是成功的，有3个正好在缺失的那个chr7中
```

可用数据信息
```
1024 SNPs 输入模型：模型文件及使用教程。
建模目标与实验设计：建议以目标基因的转录起始位点（TSS）为中心（对齐不同基因间输入的结构），向上下游延伸截取，确保覆盖 1024 个 SNP 位点作为模型输入，以此预测目标基因的表达量。在此基础上，对比评估Genos模型的Embedding和SNP模型的Embedding的效果。



计划的数据，可以先使用测试数据进行测试：
101个人vcf：/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/101vcf
测试个体：/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/101vcf/CIMA-H005_CIMA-H005.vcf.gz
100个基因窗口信息：/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/top100_robust_cv_16k_windows.tss_1mb.bed
测试用10个基因：/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/test.10gene.bed

人群变异子模型的embedding跑完了，由于区间内的SNP数量大部分都大于1024个，所以用了8192length的模型跑了embedding，结果路径/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_res_260708/

RNA-seq:
● Expression matrix with annotations: /mnt/genos100-new/peixunban/yecheng/data/CIMA/Monocyte_matrix_log2_TPM_annot.tsv.gz
● README: /mnt/genos100-new/peixunban/yecheng/data/CIMA/Monocyte_matrix_log2_TPM_annot.README.md

Personal Genome:
● VCF: /mnt/genos100-new/peixunban/chengming/CIMA_413_maf01_cohort_phasing_1kgp/per_sample_nonref_vcfs_plugin/
● FASTA:/mnt/genos100-new/Public/CIMA/fasta_snps_413/

基因筛选：
● 千人基因组欧洲人群具有显著cis-eQTL的基因列表（3259）：/mnt/zzbnew/peixunban/yancaiting/WorkSpace_GenOmics/eur_egene_list.txt，映射到CIMA表达量矩阵的gene_id（2805）：/mnt/genos100-new/peixunban/yecheng/data/CIMA/eur_egene_CIMA_gene_id_list.txt
● TSS±100kb 5-fold ElasticNetCV for raw Count，test R2 从高到低排序 top1000基因列表：/mnt/zzbnew/peixunban/yancaiting/WorkSpace_GenOmics/result/CIMA_100k_ElasticNet_cv/raw/genelist_Elasticnet_cv_top1000.txt ，其中top250的基因R2>0.2。

在对基因水平表达量建模时可以使用表达量矩阵中的数值做标签，表达量矩阵已做log2(TPM+0.01)变换，共17811 genes，406 individuals，含5列注释信息（gene_id, chr, tss, gene_length_bp, strand）。可以先挑ElasticNetCV测试R2>0.2的基因取部分来做测试。
```

- 可参考的xxl的脚本/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/run_multi_gene_models.ipynb


方案二：利用SNP Embedding预测表达量
```
CIMA vcf
↓
Genos 提取以基因为中心32K窗口内
 SNP Embedding
↓
Padding到1024bp
Concat到一起
特征：每位置 1,024 维

下游模型
MLP / CNN / Transformer+CNN
基因级任务通过池化输出标量。
```

## plan
- 输入/mnt/genos100-new/peixunban/yecheng/data/CIMA/Monocyte_matrix_log2_TPM_annot.tsv.gz的师兄对应的100人和/mnt/zzbnew/peixunban/yancaiting/WorkSpace_GenOmics/result/CIMA_100k_ElasticNet_cv/raw/genelist_Elasticnet_cv_top1000.txt中top250的基因，重新提取对应基因的embedding拿到embedding的subset数据

最终的输入是一个embedding+个体x基因的表达矩阵

提取100人对应100基因snp在foundation model的embedding之后，配上配对的人x基因的表达矩阵，建模个体基因表达差异
todo
- 选人，选对应基因，得到个人x基因的表达矩阵
- 选的人对应的vcf和fa文件提取foundation model的embedding。拿到 人数 x 基因数 x snp数(不足则padding到1024) x 维度(1014)
- ？给每个 SNP 追加一维 distance_to_TSS，成本几乎为零，收益明确


## ref
- 2026|genome biology Variformer
- biorxiv GenomicVariExpress


## modeling difference基线指标
模型方案,跨个体平均 PCC (r),核心特点与瓶颈
Enformer (Out-of-the-box),≈0.03∼0.08,直接拿预训练大模型去零样本（Zero-shot）预测个体突变效应，几乎完全失效，甚至有 30% 左右的基因预测方向是反的（负相关）。
传统线性模型 (PrediXcan/Elastic Net),≈0.25∼0.32,目前最难击败的经典 Baseline。直接基于 VCF 矩阵和表达量做弹性网络回归，简单粗暴但极为稳健。
大模型 + 简单微调/MLP,≈0.20∼0.28,往往由于过拟合，在 unseen 个体上性能低于传统线性模型。
大模型 + 配对损失 + 协变量 (本项目),≈0.30∼0.38,预期的最优表现。 引入配对损失后能显著纠正变异效应方向（Direction），在含有高遗传度（high-heritability）eQTL 的基因上，PCC 可以冲到 0.40∼0.60。


为什么从snp学习到表达差异很难，有几个方面，一是表达差异的原因不仅仅来源与序列层面，还有表观，环境等；表达差异还可能来自于测序和实验误差，表达差异来自于序列层面的话也收到序列长度的影响，我们默认是基因上下游8k，但其实有更远的调控。这些问题共同导致了很难仅基于现有计划实现个体差异表达的modeling


参考文献，不同人群，不同模型，不同线性模型的能力如何，以及在我们这个数据集中应该实现的能力