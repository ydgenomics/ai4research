让我先确认所有关键维度的来源。 

Read [](file:///mnt/rice/default/Workspace/yangdong/modeling_difference/gene-level/script2/src/models.py#1-1), lines 1 to 30

Read [](file:///mnt/rice/default/Workspace/yangdong/modeling_difference/gene-level/script2/src/preprocessing.py#80-80), lines 80 to 160

Read [](file:///mnt/rice/default/Workspace/yangdong/modeling_difference/gene-level/script2/src/data.py#100-100), lines 100 to 310

---

## Script2 模型架构完整总结

### 一、数据来源

```
磁盘上的文件:
  EMBEDDING_ROOT/{gene}/CIMA-{H005}_CIMA-{H005}.vcf.pt     ← 每个基因×每个人的 embedding
  内含: hidden_states [n_snps, 1024], positions [n_snps], variant_keys [n_snps]
  
  表达矩阵: Monocyte_matrix_log2_TPM_annot.tsv.gz
  内含: 基因×个体 的 log2(TPM+0.01) 值
```

---

### 二、从磁盘 → 训练输入的完整变换

#### Step 0: 原始数据

```
磁盘: .vcf.pt 文件
  hidden_states:     [n_snps_i, 1024]    ← 基因i在某个体中的SNP hidden states
  positions:         [n_snps_i]          ← 1-based 基因组坐标
```

#### Step 1: 减 Center（去背景）

```python
# 对基因 i, 用 train 个体的均值作为 center
center_i = mean_of_train_individuals(hidden_states)    # [n_snps_i, 1024]

# 每个个体的 delta = hidden_states - center
delta_i = hidden_states - center_i                     # [n_snps_i, 1024]
```

**意义**: 去掉所有个体共有的序列上下文，只保留个体特异性信号。

#### Step 2: Padding + Mask

```python
max_snps = max(n_snps across all genes)                # 例如 2541

output[gene][ind, :n_snps] = delta                     # [n_genes, n_ind, max_snps, 1024]
mask[gene, :n_snps] = True                             # [n_genes, max_snps]
```

**意义**: 不同基因 SNP 数不同，统一到 `max_snps`，padding 位填 0 + mask=False。

#### Step 3: 位置归一化

```python
relative_pos[gene, :n_snps] = (pos - 1 - start) / (end - start)   # [n_genes, max_snps]
```

**意义**: 基因组坐标 → [0, 1]，padding 位填 0。

#### Step 4: 标签

```
表达矩阵:  expr_matrix[gene_symbol, "CIMA-H056"] = log2(TPM+0.01)  标量

标签 y[gene, ind] = 该基因该个体的表达值   # 不做差分，直接用绝对值
```

#### Step 5: Per-gene Scaling

```python
# 仅用 train 个体计算
mean_g = mean(y[gene, train_ind])         # 标量
std_g  = std(y[gene, train_ind])          # 标量

y_scaled[gene, ind] = (y - mean_g) / std_g
```

**意义**: 让不同表达量级的基因在同一尺度上训练。推理时 inverse 回去。

---

### 三、Dataset 输出 → 模型输入

`SpecificSNPDataset.collate()` 的输出（一次 batch）：

```
batch 结构:  B = 2 (batch_size)

  x:                  [B, max_snps, 1024]    float16   SNP delta (已减 center)
  snp_mask:           [B, max_snps]           bool      True=真实SNP
  relative_positions:  [B, max_snps]          float32   [0,1] 归一化位置
  gene_ids:           [B]                     int64     基因索引 (0~96)
  targets:            [B, 1]                  float32   y_scaled (标准化后标签)
  row_indices:        [B]                     int64     原始行号
```

---

### 四、前向传播（`SpecificSNPRegressor.forward`）

```
输入:
  x                  [B, max_snps, 1024]
  snp_mask           [B, max_snps]
  relative_positions  [B, max_snps]
  gene_ids           [B]

────────────────────────────────────────────────────
Step A: SNP 投影 + 位置注入
────────────────────────────────────────────────────

  ┌─ snp_projection ────────────────────────────┐
  │ LayerNorm(1024) → Linear(1024→64) → GELU    │
  │ 输入:  [B, max_snps, 1024]                  │
  │ 输出:  [B, max_snps, 64]                    │
  └──────────────────────────────────────────────┘
                        +
  ┌─ position_projection ───────────────────────┐
  │ Linear(1→64) → GELU → Linear(64→64)         │
  │ 输入:  relative_positions [B, max_snps, 1]  │
  │ 输出:  [B, max_snps, 64]                    │
  └──────────────────────────────────────────────┘
                        ↓
          hidden = snp_proj + pos_proj
          hidden: [B, max_snps, 64]

────────────────────────────────────────────────────
Step B: 基因条件化
────────────────────────────────────────────────────

  ┌─ gene_embedding ────────────────────────────┐
  │ Embedding(97, 32)                            │
  │ 输入:  gene_ids [B]                         │
  │ 输出:  gene_vector [B, 32]                  │
  └──────────────────────────────────────────────┘
                        ↓
          gene_per_snp = expand(gene_vector, max_snps)
          gene_per_snp: [B, max_snps, 32]

────────────────────────────────────────────────────
Step C: Attention 池化
────────────────────────────────────────────────────

  concat(hidden, gene_per_snp) → [B, max_snps, 96]
                        ↓
  ┌─ attention ─────────────────────────────────┐
  │ Linear(96→64) → Tanh → Linear(64→1)          │
  │ 输出:  [B, max_snps, 1]                     │
  └──────────────────────────────────────────────┘
                        ↓
          masked_fill(padding → -inf)
          softmax(dim=1) → attention_weights [B, max_snps, 1]
                        ↓
          pooled = sum(attention_weights * hidden, dim=1)
          pooled: [B, 64]

────────────────────────────────────────────────────
Step D: 输出头
────────────────────────────────────────────────────

  concat(pooled, gene_vector) → [B, 96]
                        ↓
  ┌─ head ──────────────────────────────────────┐
  │ LayerNorm → Linear(96→64) → GELU → Dropout   │
  │ → Linear(64→1)                               │
  │ 输出:  [B, 1]    (scaled 预测值)             │
  └──────────────────────────────────────────────┘
```

---

### 五、反向传播（Loss → 梯度流）

```
损失:
  pred_scaled [B, 1]   vs   y_scaled [B, 1]
  ↓
  HuberLoss(delta=1.0) → 标量 loss
  或
  pairwise_difference_loss(pred, y, gene_ids) → 标量 loss

梯度流:
  loss → head (Linear 96→64→1)
       → attention (Linear 96→64→1, 影响 SNP 权重分布)
       → gene_embedding (影响基因条件化向量的学习)
       → snp_projection (影响 SNP 特征压缩方式)
       → position_projection (影响位置注入方式)
```

梯度**不流向** Genos embedding（embedding 是预提取的、冻结的）。

---

### 六、参数量

```
snp_projection:     1024×64 + 64        = 65,600
position_projection: 1×64 + 64×64       ≈  4,160
gene_embedding:      97×32              =  3,104
attention:          (64+32)×64 + 64×1   =  6,208
head:               (64+32)×64 + 64×1   =  6,208
────────────────────────────────────────────
总计:                                    ≈ 85,280
```

---

### 七、维度速查表

| 阶段 | 张量 | 形状 |
|------|------|------|
| 磁盘 .vcf.pt | `hidden_states` | `[n_snps, 1024]` |
| 减 center | `delta` | `[n_genes, n_ind, max_snps, 1024]` |
| mask | — | `[n_genes, max_snps]` |
| 位置 | `relative_positions` | `[n_genes, max_snps]` |
| 标签 | `y` / `y_scaled` | `[n_genes × n_ind]` / 同 |
| batch 输入 | `x` | `[2, max_snps, 1024]` |
| 投影后 | `hidden` | `[2, max_snps, 64]` |
| 基因向量 | `gene_vector` | `[2, 32]` |
| attention权重 | `weights` | `[2, max_snps, 1]` |
| 池化后 | `pooled` | `[2, 64]` |
| 最终输出 | 标量 | `[2, 1]` |

**实际数值**（基于 97 基因 × 101 人测试）：`hidden_dim=1024`, `max_snps≈2000-2500`, `n_genes=97`, `B=2`, 模型 ~85K 参数。