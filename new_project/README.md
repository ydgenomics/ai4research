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


不同类别的序列提取embedding后umap，评价其模型的能力

参考文献
- Liu, T., Zhang, X., Lin, J., Pinello, L., Ying, R., & Zhao, H. (2025). Pre-training Genomic Language Model with Variants for Better Modeling Functional Genomics. bioRxiv : the preprint server for biology, 2025.02.26.640468. https://doi.org/10.1101/2025.02.26.640468
- Lin, J. et al. EPInformer: scalable and integrative prediction of gene expression from promoter-enhancer sequences with multimodal epigenomic profiles. Nat. Commun. https://doi.org/10.1038/s41467026-70535-8 (2026).