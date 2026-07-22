# Chat 历史记录：模型架构对比与个体表达差异建模优化

> 日期：2026-07-23

---

## 一、 对话背景

基于三篇文档进行模型架构对比分析：

1. **xxl SNP Embedding 模型**（`interpetation_xxl.md`）— 基于 Genos SNP embedding 预测个体表达差异
2. **SAGE-net**（`article_SAGE-net.md`）— 解耦双分支 1D-CNN，个人基因组 S2F 预测
3. **Decima**（`article_Decima.md`）— CNN+Transformer，单细胞 Pseudobulk 多状态表达预测

核心 Source Code 文件：

- `ai4research/md0722/gene/script/specific_snp_model.py` — SpecificSNPRegressor, SpecificSNPTransformerCNN, PositionBinnedRegressor
- `ai4research/md0722/gene/script/position_binned_model.py` — 分箱预处理 + Conv1D 模型
- `ai4research/md0722/gene/script/run_multi_gene_models.ipynb` — 训练循环（executed 版本含 pairwise loss）
- `ai4research/md0722/gene/script/run_multi_gene_models.executed.ipynb` — 实际执行版（含训练 log）

---


## 二、 聚焦方向：建模个体表达差异

最终确定焦点为 **R_indiv（跨个体相关性）**——即"为什么同基因在不同个体中表达不同"。

### 2.1 关键训练结果（从 executed notebook 提取）

SpecificSNPRegressor 在 97 基因 × 101 个体（81 train / 10 val / 10 test）上的训练曲线：

```text
Epoch  1: train_pearson=0.12, val_pearson=-0.14
Epoch 10: train_pearson=0.37, val_pearson=0.06
Epoch 20: train_pearson=0.55, val_pearson=0.13
Epoch 30: train_pearson=0.80, val_pearson=-0.08
Epoch 40: train_pearson=0.93, val_pearson=0.05
```

**核心结论**：训练集 Pearson 涨到 0.93，但验证集始终 ~0.05 → 严重过拟合，模型未学到泛化的个体差异信号。

### 2.2 过拟合的根因分析

| 原因 | 细节 |
|------|------|
| 样本量小 | 81 train × 97 genes = 7857 样本，per-gene 仅 81 人 |
| 信号瓶颈 | delta = hidden_i - mean(hidden_train)，大部分维度可能是噪声 |
| 模型能力 | 35K 参数在 7857 样本上已过剩 |
| loss 结构 | pairwaise_loss 与 absolute_loss 下降幅度相同（均降 84%），未提供额外约束 |

---

## 三、 三种模型架构对比（个体差异视角）

### 3.1 SpecificSNPRegressor（注意力池化基线）

```text
delta [SNPs, 1024] → LayerNorm → Linear(1024→64) → + 位置投影
→ Tanh Attention (concat gene_emb) → masked softmax → weighted sum
→ concat(gene_emb) → LayerNorm → MLP Head → 标量
```

| ✅ 优势 | ❌ 劣势 |
|---------|---------|
| 极轻量（~35K），不易过拟合 | 每 SNP 独立投影，无局部交互（丢失 epistasis） |
| 显式基因条件化嵌入 | 单 head attention，所有 SNP 竞争单一权重 |
| `pairwise_difference_loss` 对比损失 | 64 维 projection 是信息瓶颈 |
| `SameGenePairBatchSampler` 成对采样 | 无参考基线，均值/残差耦合 |

### 3.2 SpecificSNPTransformerCNN（增强版，代码已存在）

```text
delta → projection(1024→64) → Sinusoidal PE
  ├→ Transformer Encoder (2层, 4头)    长程交互
  ├→ MultiScale CNN (k=3,5,9,15)       局部 motif
  └→ shortcut (残差)
→ Fusion → masked_mean + masked_max → concat(gene_emb) → head
```

| ✅ 优势 | ❌ 劣势 |
|---------|---------|
| CNN (k=15) 捕获 ~150bp 局部调控模块 | 参数增至 ~200K，过拟合风险更高 |
| Transformer 捕获远端 SNP 协同 | 计算复杂度 O(n²) |
| max pooling 保留极端 SNP 效应 | 需加强正则化（dropout, weight_decay） |
| 残差融合，梯度稳定 | |

### 3.3 PositionBinnedRegressor（分箱卷积版）

```text
delta → 32 bin (mean+max+log1p count) → [32, 2049]
→ LayerNorm → Linear(2049→128) → Conv1D × 2 残差块
→ masked_mean + masked_max → concat(gene_emb) → head
```

| ✅ 优势 | ❌ 劣势 |
|---------|---------|
| 硬编码分箱，对齐不同基因 SNP 分布 | 32 bin 粒度太粗，丢失精确定位 |
| log1p(count) 编码 SNP 密度 | 边界效应 |
| 残差卷积，训练稳定 | 不适合需要精确定位的调控分析 |

### 3.4 SAGE-net（参考框架）

- **核心创新**：解耦均值/残差双分支 + 对比学习
- **个体差异链路**：y = y_mean(Ref DNA) + y_residual([Ref, Hap1, Hap2])
- **性能**：R_indiv ≈ 0.22-0.32（ROSMAP ~700 人）
- **启示**：解耦是解决 R_indiv ≈ 0 瓶颈的关键设计模式

### 3.5 Decima（不适用于个体差异建模）

- 参考基因组模型，无法感知个体间遗传变异
- 评估的是跨细胞类型差异（per-gene cross-condition R），非跨个体差异
- **排除**

---

## 四、 优化方向（按优先级）

### ⭐⭐⭐ 方向 1：解耦均值与残差（仿 SAGE-net，立即实施）

**思路**：双分支架构，分离"基因表达量级"和"个体差异"。

```text
Branch_ref:   0 delta (或基因ID+位置) → y_ref（群体均值）
Branch_indiv: 实际 delta → y_residual（个体偏差）
y_pred = y_ref + y_residual
Loss = Huber(y_pred, y_true) + β · Huber(y_residual, y_true - y_mean_train)
```

- 代码改动 ~20 行
- 预期 val_pearson 从 0.05 → **0.15-0.20**

### ⭐⭐⭐ 方向 2：切换到 SpecificSNPTransformerCNN（代码已存在）

- 只需在 notebook 切换 model_cls
- 需要加强正则化应对过拟合

### ⭐⭐ 方向 3：Multi-head Attention

- 多个 head 关注不同调控区域（TSS 近端、增强子、剪接位点等）
- 将 `Linear(projection+gene→1)` 改为 `Linear(projection+gene→n_heads)`

### ⭐⭐ 方向 4：加权 Pairwise Loss

- 当前对所有同基因对等权平均，差异大的对贡献被稀释
- 改进：按 `|truth_diff|` 加权，或使用 triplet loss

### ⭐ 方向 5：数据增强

- delta 加高斯噪声生成虚拟个体
- 随机 mask 部分 SNP

### ⭐ 方向 6：辅助 SNP 级预测头

- 需要额外 GWAS/eQTL summary stats 作为弱监督

---

## 五、 代码关键位置速查

| 文件 | 关键内容 |
|------|---------|
| `specific_snp_model.py:212` | `class SpecificSNPRegressor` — 主模型 |
| `specific_snp_model.py:317` | `class SpecificSNPTransformerCNN` — 增强版 |
| `specific_snp_model.py:278` | `class MultiScaleConvBlock` — 多尺度 CNN |
| `specific_snp_model.py:263` | `class SinusoidalPositionEncoding` — 位置编码 |
| `position_binned_model.py:47` | `class PositionBinnedRegressor` — 分箱模型 |
| `position_binned_model.py:13` | `def bin_snp_hidden_states` — 分箱预处理 |
| `specific_snp_model.py:21` | `def pairwise_difference_loss` — 对比损失 |
| `specific_snp_model.py:87` | `class SameGenePairBatchSampler` — 成对采样器 |
| `specific_snp_model.py:130` | `def center_and_pad_snp_hidden` — 减中心+padding |
| `specific_snp_model.py:177` | `def normalize_snp_positions` — 位置归一化 |
| `position_binned_model.py:69` | `def fit_gene_target_scalers` — per-gene Z-score |
| `run_multi_gene_models.ipynb` | 训练循环与评估 |

## 六、 未解决问题 / 待探索

- [ ] delta 的信噪比是多少？需要实证检查 `‖delta‖ / ‖hidden‖` 的分布
- [ ] 不同基因的 R_indiv 差异很大，哪些基因学得好/差？
- [ ] 用 SpecificSNPTransformerCNN 的实际训练结果如何？
- [ ] pairwise_difference_loss 与 absolute_loss 的权重比调优
