https://alidocs.dingtalk.com/i/nodes/MNDoBb60VLrdNBAoFm4evgz28lemrZQ3?iframeQuery=utm_source%3Dportal%26utm_medium%3Dportal_recent&rnd=0.8738508474765427


这是一个我也不知道要做到什么程度的项目
- 群体变异数据的引入
- 建模个体差异
- 多组织或跨组织
- 多组学（甲基化、代谢组）
- 表型定位与功能基因组


- 水稻模型建模个体表达差异的能力到底如何？


变异模型本身是面向DNA模型的，期望的是具有变异感知的模型其在下游应用会更好，其评价指标分两大类，一是本身的评价（例如下一个token预测的准确性，基因功能的表征能力）；二是接入下游任务后模型的能力（例如RNA表达预测）
如果研究的是一个品种的群体，可以用群体的数据先微调，然后再接入下游头表达预测
- 方案一：基模（不冻）+下游头
- 方案二：基模（微调后冻住）+下游头
- 方案三：基模（微调后不冻）+下游头

## 数据统计
口径	密度	含义
群体 union 位点密度	33,928,212 记录 / ~373 Mbp（12 条主染色体）≈ 1/11 bp	251 个样本合并后的多态位点集
按唯一位置去重后	约 29.3 M 位置 / 373 Mbp ≈ 1/13 bp	多等位拆分只占约 13.5% 重叠
单个体相对于参考	~13.8 万差异位点（117k SNP + 21k indel）/ 373 Mbp ≈ 1/2,700 bp	关键！每个个体 99.96% 位置与参考一致
结论：你说的"1/15"在群体 union 口径下基本正确（实际 1/11~1/13）。但注入 OGR 时真正要处理的是第二行——个体特异差异在单个窗口内高度稀疏。这决定了注入方式不能是"密集矩阵"，而必须走**"参考序列 + 稀疏变异叠加"**的思路。


## 测试代码走通
先用4个样本的数据做CPT
- japonica: NH001, NH002
- indica: NH006, NH007

怎么做
- 确定参考基因组：/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.asm.ch.fa
- 确定vcf文件：/mnt/rice/data/Lianguang_shang/251.SNP.vcf.gz，按样本subset之后可能数据会小一点
- 怎么制作样本CPT到OGR里面？
- 不同类别的序列提取embedding后umap，评价其模型的能力

## 参考文献
- Liu, T., Zhang, X., Lin, J., Pinello, L., Ying, R., & Zhao, H. (2025). Pre-training Genomic Language Model with Variants for Better Modeling Functional Genomics. bioRxiv : the preprint server for biology, 2025.02.26.640468. https://doi.org/10.1101/2025.02.26.640468
- Lin, J. et al. EPInformer: scalable and integrative prediction of gene expression from promoter-enhancer sequences with multimodal epigenomic profiles. Nat. Commun. https://doi.org/10.1038/s41467026-70535-8 (2026).