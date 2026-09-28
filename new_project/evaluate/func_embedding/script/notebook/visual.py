import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import os


base = "/mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/func_embedding/script/output/IRGSP/rice_1B_stage2_8k"
output_dir = "/mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/func_embedding/script/output"

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

os.chdir(output_dir)

# 1. 读取嵌入矩阵
embeddings = np.load(f"{base}/gene_embeddings.npy")   # (50, 1024)
print("embeddings shape:", embeddings.shape)

# 2. 读取基因名（第一列作为 obs 索引）
gene_names = pd.read_csv(f"{base}/gene_names.txt", header=None, sep="\t")
print("gene_names shape:", gene_names.shape)
obs_names = gene_names.iloc[:, 0].astype(str).values

# 3. 读取 meta（作为 obs 的内容）
meta = pd.read_csv(f"{base}/meta.tsv", sep="\t")
print("meta shape:", meta.shape)
print(meta.head())

# 4. 检查行数一致性
assert embeddings.shape[0] == len(obs_names) == meta.shape[0], \
    f"行数不一致: emb={embeddings.shape[0]}, names={len(obs_names)}, meta={meta.shape[0]}"

# 5. 构建 AnnData（把 embedding 当作 X，基因名作 obs 索引，meta 合并进 obs）
adata = ad.AnnData(
    X=embeddings.astype(np.float32),
    obs=meta.copy(),
)
adata.obs_names = obs_names
# 若 meta 自带索引列，可改成 adata.obs = meta.set_index(...)，按需处理
adata.obs_names_make_unique()

print(adata)

adata.obs['label'] = adata.obs['gene_id'].str.split('_').str[0]

# 1. 基本统计量（整体）
print("shape:", X.shape)
print("min :", X.min())
print("max :", X.max())
print("mean:", X.mean())
print("std :", X.std())

# 2. 按行统计（每个基因/样本的分布）
row_mean = X.mean(axis=1)
row_std  = X.std(axis=1)
print("\nrow mean  : min=%.4f  max=%.4f  mean=%.4f" % (row_mean.min(), row_mean.max(), row_mean.mean()))
print("row std   : min=%.4f  max=%.4f  mean=%.4f" % (row_std.min(), row_std.max(), row_std.mean()))

# 3. 按列统计（每个 embedding 维度）
col_mean = X.mean(axis=0)
col_std  = X.std(axis=0)
print("\ncol mean  : min=%.4f  max=%.4f  mean=%.4f" % (col_mean.min(), col_mean.max(), col_mean.mean()))
print("col std   : min=%.4f  max=%.4f  mean=%.4f" % (col_std.min(), col_std.max(), col_std.mean()))

# 4. 分位数
qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
print("\npercentiles:", np.percentile(X, qs))
print("values     :", np.round(np.percentile(X, qs), 4))

# 5. 直方图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(X.ravel(), bins=100)
axes[0].set_title("All values")
axes[1].hist(row_std, bins=50)
axes[1].set_title("Per-row std")
plt.tight_layout()
plt.savefig(output_dir / "x_distribution.png", dpi=150)
plt.show()

adata.obs['log_seq_len'] = np.log10(adata.obs['seq_len'])
sc.pl.pca(
    adata,
    color=["label", "log_seq_len"],
    components=["1,2", "2,3", "3,4", "4,5"],
    show=True,
)


# ========== 2. 邻居图 ==========
# 50 样本，n_neighbors 调小；n_pcs 看累计方差挑
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)

# ========== 3. UMAP / t-SNE ==========
sc.tl.umap(adata)
# sc.tl.tsne(adata, n_pcs=15, perplexity=10)   # 样本少，perplexity 调小

# ========== 4. 聚类 ==========
sc.tl.leiden(adata, resolution=0.1, key_added="leiden")

sc.pl.pca(adata, color=["leiden", "label"], show=True, save="_pca_leiden.png")

# 1. 绘制原始 UMAP
sc.pl.umap(adata, color=["leiden", "label"], show=True)


sc.pl.umap(adata, color=["seq_len", "log_seq_len"], color_map="viridis", show=True, save="_umap_seq_len.png")

# 3. 计算 UMAP 坐标与序列长度的 Spearman 相关系数
from scipy.stats import spearmanr
umap_coords = adata.obsm['X_umap']
corr1, p1 = spearmanr(umap_coords[:, 0], adata.obs['seq_len'])
corr2, p2 = spearmanr(umap_coords[:, 1], adata.obs['seq_len'])
print(f"UMAP1 与 seq_len 的 Spearman 相关系数: {corr1:.4f} (p={p1:.4e})")
print(f"UMAP2 与 seq_len 的 Spearman 相关系数: {corr2:.4f} (p={p2:.4e})")