script2/
├── main.ipynb                          # 唯一入口 notebook
├── src/                                # 模块化源码
│   ├── __init__.py                     # 统一导出
│   ├── models.py                       # SpecificSNPRegressor, SpecificSNPTransformerCNN, PositionBinnedRegressor
│   ├── preprocessing.py                # center_and_pad, normalize_positions, fit_snp_centers
│   ├── losses.py                       # pairwise_difference_loss, mixed_loss, SameGenePairBatchSampler
│   ├── metrics.py                      # per_gene_metrics, macro_gene_metrics, safe_pearson/r2
│   ├── scaling.py                      # per-gene Z-score scaling
│   ├── data.py                         # GeneRecord, BigWig targets, SpecificSNPDataset, build_multi_gene_data
│   ├── training.py                     # Trainer, predict_all, summarize, run_experiment
│   └── checkpoint.py                   # build_checkpoint, load_checkpoint
├── configs/
│   ├── 101samples.name.txt
│   ├── test.10gene.bed
│   └── top100_robust_cv_16k_windows.tss_1mb.bed
├── extract_top250_tss_embeddings.py     # 保留: embedding 提取脚本
└── analyze_snp_embedding_variance.py    # 保留: 方差分析脚本