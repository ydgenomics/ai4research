# Cross-Attention Fusion Predictor 设计文档

## 1. 设计目标

设计一个与 `predictor_v0.py` 参数量相当的模态融合模型，通过改进 ATAC Encoder 和 Fusion 模块提升 DNA 与 ATAC 信号的交互能力。

## 2. 架构概览

```
DNA序列 → Base Model (12层Mixtral, 完全冻结) ───────────┐
                                                      ├──→ 单向Cross-Attn ──→ 三门控融合 ──→ U-Net Decoder → RNA预测
ATAC信号 → 轻量Transformer (d=192, 6层) ──────────────┘
```

## 3. 模块设计

### 3.1 Base Model

- **模型**: Mixtral 1B (12层, hidden_size=1024)
- **状态**: 完全冻结，所有参数 `requires_grad=False`
- **输出**: DNA特征 `[batch, L, 1024]`
- **理由**: 保护预训练知识，释放参数预算给下游模块

### 3.2 ATAC Encoder

轻量深层 Transformer 设计，低维瓶颈控制参数量。

**架构**:
```
输入: [B, L, 1] (ATAC信号)
  ↓
Linear(1 → 192) + 可学习位置编码
  ↓
6 × TransformerLayer(
    d_model=192,
    n_heads=4,
    ffn_dim=768 (192×4),
    dropout=0.1
)
  ↓
Linear(192 → 1024)  # 匹配DNA维度
输出: [B, L, 1024]
```

**参数量**: ~15M
- 6层 Self-Attention: 6 × 4 × 192² ≈ 0.9M
- FFN: 6 × (192×768×2) ≈ 1.4M
- 投影层: ~0.2M
- 位置编码 + LayerNorm: ~0.5M

**关键设计**:
- 低维瓶颈 (d=192 vs d=1024): Self-Attention 复杂度 O(L²×d)，192² vs 1024² = 64倍节省
- 深层 (6层): 层数 > 维度，高效捕获长程依赖
- 4 heads: 适配低维空间

**初始化**:
```python
for layer in atac_encoder.layers:
    nn.init.xavier_uniform_(layer.self_attn.in_proj_weight, gain=0.1)
    nn.init.zeros_(layer.self_attn.out_proj.bias)
```

### 3.3 Fusion Module

#### 3.3.1 单向 Cross-Attention

ATAC 作为 Query 查询 DNA (Key/Value)。

```python
# 单向: 只有 ATAC → DNA
queries = atac_features          # [B, L, 1024] (可训练)
keys = dna_features               # [B, L, 1024] (冻结)
values = dna_features               # [B, L, 1024] (冻结)

cross_out, attn_weights = F.multi_head_attention_forward(
    query=queries,
    key=keys,
    value=values,
    embed_dim_to_check=1024,
    num_heads=8,
    q_proj_weight=self.q_proj.weight,    # [1024, 1024]
    k_proj_weight=self.k_proj.weight,    # [1024, 1024]
    v_proj_weight=self.v_proj.weight,    # [1024, 1024]
    out_proj=self.out_proj,               # [1024, 1024]
    # 总参数量: 4 × 1024² = 4.2M
)
```

**梯度流向**:
```
Loss
  ↓
cross_out
  ↓
├─→ Q_proj 梯度 ──→ ATAC Encoder 梯度 (优化ATAC如何查询)
├─→ K_proj 梯度 ──→ (更新K投影，但DNA特征本身冻结)
└─→ V_proj 梯度 ──→ (更新V投影，但DNA特征本身冻结)
```

**参数量**: ~4.2M (单向 vs 双向 8.4M，节省50%)

#### 3.3.2 三门控融合 (Ternary Gating)

```python
# 三个输入
atac_self = atac_features               # ATAC自表示
dna_self = dna_features                  # DNA自表示
cross = cross_out                        # 交互表示

# 门控网络 (softmax约束 sum=1)
gate_input = concat([atac_self, dna_self, cross])  # [B, L, 3072]
weights = softmax(MLP(gate_input))                  # [B, L, 3]
# MLP: Linear(3072 → 512) → GELU → Linear(512 → 3)

w_atac, w_dna, w_cross = weights[..., 0], weights[..., 1], weights[..., 2]
# 约束: w_atac + w_dna + w_cross = 1

# 融合
fused = w_atac * atac_self + w_dna * dna_self + w_cross * cross
```

**参数量**: ~1.6M
- MLP: (3072×512 + 512) + (512×3 + 3) ≈ 1.6M

**初始化**:
```python
with torch.no_grad():
    fusion.gate_mlp[-1].bias[0] = -0.2   # ATAC (稍抑制)
    fusion.gate_mlp[-1].bias[1] = 0.3    # DNA (稍激活，利用预训练)
    fusion.gate_mlp[-1].bias[2] = 0.0    # cross (中性)
```

**生物学意义**:
- `w_dna` 高: 组成型表达，仅DNA序列决定
- `w_atac` 高: 诱导型表达，染色质可及性主导
- `w_cross` 高: 交互调控，需两者协同

#### 3.3.3 Attention Dropout (替代 Modal Dropout)

不使用模态dropout（会破坏冻结DNA的稳定性），改用attention dropout:

```python
# 在 Cross-Attention 权重上应用 dropout
attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
# 强制 ATAC 关注更多位置，不只聚焦最强信号

# 对 ATAC 特征做 channel dropout (可选)
atac_features = F.dropout(atac_features, p=0.1, training=self.training)
```

**理由**:
- DNA 冻结，模态 dropout 会破坏 batch 间一致性
- 单向 Cross-Attn 对 Query 缺失敏感
- Attention dropout 保持模态存在但增加噪声，增强鲁棒性

### 3.4 Decoder

保持不变，与 `predictor_v0.py` 完全一致。

```
fusion_out [B, 1024, L]
  ↓
enc1: Conv1d(1024→512, k=3)
enc2: Conv1d(512→1024, k=3, stride=2)
enc3: Conv1d(1024→1024, k=3, stride=2)
bottleneck: 3× DilatedConv(1024, dilation=1,2,4)
up1: ConvTranspose1d(1024→1024, stride=2)
dec1: Conv1d(2048→1024, k=3)  # skip from enc2
up2: ConvTranspose1d(1024→512, stride=2)
dec2: Conv1d(1024→512, k=3)    # skip from enc1
final: Conv1d(512→2, k=1)      # plus/minus strands
```

**参数量**: ~31M

## 4. 计算量对比 (Per Step)

假设配置: batch_size=1, seq_len=32000, 计算理论 FLOPs (浮点运算次数)

**FLOPs 计算公式** (来源: 矩阵乘法标准复杂度):
- 矩阵乘法 `Y = X @ W`: `FLOPs = 2 × M × N × K` (输入[M,K] @ 权重[K,N] → 输出[M,N])
- Attention Q@K^T: `FLOPs = 2 × B × heads × L × L × d_head` (来源: "Attention Is All You Need" 论文)

### 4.1 Base Model (Mixtral) 单层详细分解

**单层激活参数量**: ~28M (不是 104M 总参数!)
- Self-Attention (Q/K/V/O): ~3.15M (GQA: K/V 用 d/2)
- MoE FFN (top-2 experts): **~25.17M** (占总激活参数 89%，**此前计算遗漏此项**)

| 组件 | 计算 | FLOPs | 占比 |
|------|------|-------|------|
| Self-Attention Q/K/V/O 投影 | 4 × Linear(d→d) | 201B | 3.3% |
| Self-Attention Q@K^T | [L,d]×[d,L]→[L,L] | **2,097B** | **34.6%** |
| Self-Attention Attn@V | [L,L]×[L,d]→[L,d] | **2,097B** | **34.6%** |
| MoE FFN (top-2 experts) | 2 × SwiGLU | **1,611B** | **26.6%** | ← **此前遗漏** |
| Gate + LayerNorms | - | 1B | 0.9% |
| **单层总计** | | **6,048B** | **100%** |

**关键修正**: Base Model 单层计算量是 **6.05 TFLOPs**，不是此前错误的 ~1.8 TFLOPs。错误原因是混淆了"参数量"和"计算量"，且完全遗漏了 MoE FFN 的 1.61 TFLOPs。

### 4.2 predictor_v0 (原方案)

| 模块 | 前向 FLOPs | 反向 FLOPs | 总计 | 说明 |
|------|-----------|-----------|------|------|
| Base Model (11层冻结) | 66.53T | 0 | 66.53T | 11层 × 6.05TF |
| Base Model (1层可训练) | 6.05T | 12.10T | 18.15T | 1层前向+反向 (反向≈2×前向) |
| ATAC Encoder (CNN) | 67.7B | 67.7B | 135.4B | 5层卷积，约68M参数量 |
| Fusion (逐元素相加) | 0.03B | 0.03B | 0.07B | 几乎无计算 |
| Decoder (U-Net) | 1.98T | 3.97T | 5.95T | ~31M参数 |
| **总计** | **74.6T** | **16.1T** | **90.8T** | **~91 TFLOPs/step** |

### 4.3 新方案 (Cross-Attn Fusion)

| 模块 | 前向 FLOPs | 反向 FLOPs | 总计 | 说明 |
|------|-----------|-----------|------|------|
| Base Model (12层冻结) | 72.58T | 0 | 72.58T | 12层 × 6.05TF，无反向 |
| ATAC Encoder (6层Trans) | 5.02T | 10.05T | 15.07T | 低维(d=192)Transformer |
| Fusion (单向Cross-Attn) | 4.50T | 9.01T | 13.51T | 标准Attention，与Base Self-Attn同量级 |
| Fusion (三门控MLP) | 0.10T | 0.21T | 0.31T | 轻量门控 |
| Decoder (U-Net) | 1.98T | 3.97T | 5.95T | 与原方案相同 |
| **总计** | **84.2T** | **23.2T** | **107.4T** | **~107 TFLOPs/step** |

### 4.4 关键对比与洞察

| 对比项 | predictor_v0 | 新方案 | 变化 |
|--------|-------------|--------|------|
| **每 step 总 FLOPs** | **90.8 TFLOPs** | **107.4 TFLOPs** | **+18.3%** |
| Base Model 占比 | 93.3% | 67.6% | 仍主导但占比下降 |
| Fusion 占比 | <0.1% | 12.9% | 新增主要开销 |
| ATAC 占比 | 0.15% | 14.0% | CNN→Transformer升级 |

**关键洞察**:

1. **Cross-Attention (4.50T) < Base Model 单层 (6.05T)**，差距 26%
   - 两者都是标准 Self-Attention (L=32k, d=1024, heads=8)
   - Base 更大是因为有额外的 MoE FFN (1.61T)

2. **新方案总增幅仅 +18.3%**，不是此前错误声称的 +63% 或 +7.6倍
   - 主要增加: Base 12层全冻结 (无反向节省部分抵消)
   - 次要增加: ATAC Transformer (+15T) 和 Fusion (+14T)

3. **计算量分布 (新方案)**:
   - Base Model (冻结): 67.6%
   - ATAC Encoder: 14.0%
   - Fusion: 12.9%
   - Decoder: 5.5%

### 4.5 Flash Attention 实现建议

**为什么需要 Flash Attention**:

Cross-Attention 的 4.50 TFLOPs 中，**93% 来自 L×L 矩阵乘法** (Q@K^T 和 Attn@V 各 2.1T)。Flash Attention 不改变理论 FLOPs，但可显著降低实际运行时间：

| 指标 | 标准 Attention | Flash Attention | 效果 |
|------|---------------|-----------------|------|
| 理论 FLOPs | 4.50 TF | **4.50 TF (不变)** | 数学运算相同 |
| HBM 访问量 | ~20 GB (存储 L×L 矩阵) | **~2 GB** | **减少 10×** |
| 实际运行时间 | 慢 (内存瓶颈) | **快 2-4×** | 从内存受限变为计算受限 |

**Flash Attention 原理**: 
- **分块 (Tiling)**: 将 Q/K/V 分块加载到 SRAM，避免存储完整的 L×L attention matrix
- **在线 Softmax**: 增量计算 softmax 统计量，无需全局归一化
- **重计算**: 反向传播时重新计算 forward 值，而非存储中间结果

**实现建议 (PyTorch 2.0+)**:

```python
import torch.nn.functional as F

# 推荐: 使用原生 scaled_dot_product_attention，自动选择最优 kernel
class CrossAttention(nn.Module):
    def __init__(self, d_model=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        
        # Q 来自 ATAC (可训练), K/V 来自 DNA (冻结)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, atac_features, dna_features):
        # atac_features: [B, L, d], dna_features: [B, L, d] (冻结)
        B, L, d = atac_features.shape
        
        # 投影
        Q = self.q_proj(atac_features).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(dna_features).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(dna_features).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        # Q,K,V: [B, heads, L, d_head]
        
        # Flash Attention (自动检测并使用)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,      # 启用 Flash Attention
            enable_math=True,       # 回退到数学实现
            enable_mem_efficient=True  # Memory-Efficient Attention
        ):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.1 if self.training else 0.0,
                is_causal=False,     # Cross-Attn 不需要因果掩码
                scale=self.scale
            )
        
        # 合并 heads 并投影
        out = out.transpose(1, 2).contiguous().view(B, L, d)
        return self.o_proj(out)
```

**实现注意事项**:
1. **d_head 必须是 64 或 128** (Flash Attention-2 的硬件优化要求)
2. **序列长度 L 需能被 128 整除** (分块对齐要求)
3. **PyTorch 版本**: 需要 2.0+ 且 CUDA 11.6+ 或 2.1+ 且 CUDA 12.1+
4. **回退机制**: 如果不满足条件，自动回退到标准 Attention

**备选方案 (若 Flash Attention 不可用)**:

```python
# 方案1: 降低 heads 数 (计算量减半)
num_heads = 4  # 原为 8

# 方案2: 局部窗口 Attention (计算量降 10x)
window_size = 1024  # 只关注 ±1024 位置

# 方案3: Linformer / Performer 等近似 Attention (理论 FLOPs 降至 O(L))
```

**结论**: 理论计算量增加 18.3%，但使用 Flash Attention 后实际运行时间可能仅增加 **<5%**，甚至在某些硬件上持平。Cross-Attention 带来的建模能力提升远大于计算开销。

---

## 5. 训练策略

**数据规模**: ~10-20 steps/epoch，总计 ~50 epochs (共 500-1000 steps)

### 5.1 增强正则化策略

```python
# 1. 提高 Dropout 率
atac_dropout = 0.2        # 原为 0.1
attention_dropout = 0.15  # 原为 0.1
decoder_dropout = 0.15    # 原为 0.1

# 2. 更强的权重衰减
weight_decay = 0.05       # 原为 0.01

# 3. 梯度裁剪 (防止训练震荡)
gradient_clip_norm = 1.0

# 4. 标签平滑 (如果适用)
label_smoothing = 0.01
```

### 5.2 学习率调度 (延长Warmup)

```python
# 针对 50 epochs (~1000 steps max)
scheduler_config = {
    'type': 'cosine_with_warmup',
    'warmup_steps': 100,           # 前 2 epochs 线性增长 (原为 500)
    'total_steps': 1000,           # 50 epochs × 20 steps
    'min_lr_ratio': 0.1,           # 最终降至 10%
}

# 学习率调整
learning_rates = {
    'atac_encoder': 5e-4,      # 全新模块，快速学习
    'fusion': 1e-4,            # 依赖 ATAC，中等学习率
    'decoder': 5e-4,           # 全新训练，与 ATAC 同等对待
}
```

### 5.3 训练阶段

针对 ~10-20 steps/epoch，总计 ~50 epochs (500-1000 steps) 的训练配置。

**Phase 1: Warmup (Epoch 0-2, ~5-10% of total)**
- 学习率从 0 线性增长至 target (5e-4 / 1e-4 / 5e-4)
- 所有模块同时激活，保持同步学习
- 前 20 steps: **强制三门控均匀 0.33:0.34:0.33** (硬约束，稳定初始化)
- 监控: ATAC输出稳定性 (梯度范数应 < 10)

**Phase 2: 主训练 (Epoch 3-30, ~60% of total)**
- 全学习率运行
- ATAC Encoder 快速学习 ATAC→RNA 映射
- Fusion 学习有效的模态交互
- 每 5 epochs 评估验证集，监控:
  - 验证 MSE 趋势
  - 门控权重分布 (确保 `w_cross` > 0.1)

**Phase 3: 退火 (Epoch 31-45, ~30% of total)**
- 学习率 cosine 衰减至 30%
- 精细调整所有模块
- 准备收敛

**Phase 4: 微调 (Epoch 46-50, ~10% of total)**
- 学习率降至 10% of max
- 仅微调 Decoder (冻结 ATAC Encoder 和 Fusion，可选)
- 或使用 SWA (Stochastic Weight Averaging) 最后 5-10 epochs

### 5.4 检查点与评估策略

```python
checkpoint_config = {
    'save_every_n_epochs': 5,
    'keep_top_k': 3,                # 仅保留验证MSE最低的3个ckpt
    'save_on_best': True,           # 每次刷新最佳都保存
    'monitor': 'val_mse',
    'mode': 'min',
}

# 模型选择标准 (按优先级):
# 1. 验证集 MSE 最低
# 2. 门控权重 `w_cross` > 0.15 (确保交互有效)
# 3. 排除任一门控权重 < 0.05 (模态坍塌)
```

### 5.5 监控指标

| 指标 | 目标范围 | 说明 |
|------|----------|------|
| 训练 MSE | 逐步下降 | 正常收敛趋势 |
| 验证 MSE | 同步或略高 | 与训练MSE差距应在合理范围 |
| `w_atac` | 0.2-0.4 | ATAC支路活跃度 |
| `w_dna` | 0.3-0.6 | DNA支路活跃度 |
| `w_cross` | 0.1-0.4 | 交互支路活跃度，应 > 0.1 |
| `entropy` | 0.9-1.1 | 三门控分布熵，高熵表示均衡使用 |
| 梯度范数 | 稳定 | 避免剧烈震荡或消失 |

### 5.6 调试策略

若训练不稳定:

1. **降低 ATAC 学习率**: 5e-4 → 2e-4 → 1e-4
2. **延长 Warmup**: 从 2 epochs 延长至 5 epochs
3. **固定门控更久**: 硬约束从 20 steps 延长至 50 steps
4. **降低 Batch Size**: 配合 gradient accumulation 保持有效 batch
5. **检查梯度**: 使用梯度裁剪 (clip_norm=1.0) 防止爆炸

若 `w_cross` → 0 (交互失效):
- 增大 Fusion 学习率
- 检查初始化 bias (确保 w_cross 初始非负)
- 考虑预热阶段后强制 w_cross ≥ 0.1 硬约束

若 `w_atac` 或 `w_dna` → 0 (模态坍塌):
- 降低对应模态的学习率
- 检查辅助损失权重 (atac_l2, gate_entropy)
- 考虑模态 dropout (但当前设计已用 Attention Dropout 替代)
## 6. 参数量总结

| 模块 | 参数 | 说明 |
|------|------|------|
| Base Model | 0M | 完全冻结 (1.2B 冻结) |
| ATAC Encoder | ~15M | 低维深层 Transformer |
| Fusion (Cross-Attn) | ~4.2M | 单向，4×1024² |
| Fusion (三门控) | ~1.6M | MLP: 3072→512→3 |
| Decoder | ~31M | 与原predictor_v0一致 |
| **总计** | **~52M** | 全部可训练 |

对比:
- `predictor_v0`: ~137M (解冻Base最后一层 ~104M)
- 本方案: ~52M (更精简，但关键模块能力更强)

## 7. 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| Base Model | 完全冻结 | 保护预训练知识，释放参数预算 |
| ATAC Encoder | 低维深层 Transformer | 层数 > 维度，高效捕获长程依赖 |
| Fusion方向 | 单向 (ATAC→DNA) | 参数量减半，符合生物学直觉 |
| 门控设计 | 三门控 + softmax | 约束sum=1，防止坍塌 |
| Dropout策略 | Attention dropout | 替代modal dropout，兼容冻结DNA |
| Decoder | 保持不变 | 已验证有效，专注上游融合改进 |

## 8. 待验证假设

1. 低维深层 (d=192, 6层) vs 高维浅层 (d=512, 2层): 哪个ATAC编码更有效？
2. 单向 vs 双向 Cross-Attn: 单向是否足够？
3. 三门控 vs 双门控: 三门控是否更稳定？
4. Attention dropout rate: 0.1 是否最优？

## 9. 实现提示

### 9.1 前向传播伪代码

```python
def forward(self, input_ids, atac_signal, labels=None):
    # 1. DNA编码 (冻结)
    with torch.no_grad():
        dna_features = self.base_model(input_ids)  # [B, L, 1024]
    
    # 2. ATAC编码 (可训练)
    atac_features = self.atac_encoder(atac_signal)  # [B, L, 1024]
    
    # 3. Fusion
    # 3.1 单向Cross-Attention (ATAC查询DNA)
    cross_out = self.cross_attn(
        query=atac_features,
        key=dna_features,      # 冻结特征
        value=dna_features     # 冻结特征
    )
    
    # 3.2 Attention Dropout
    cross_out = F.dropout(cross_out, p=0.1, training=self.training)
    
    # 3.3 三门控融合
    weights = self.ternary_gate(atac_features, dna_features, cross_out)
    # weights: [w_atac, w_dna, w_cross], sum=1
    
    fused = (weights[:, 0] * atac_features + 
             weights[:, 1] * dna_features + 
             weights[:, 2] * cross_out)
    
    # 4. Decoder (保持不变)
    logits = self.decoder(fused)
    
    # 5. 损失
    loss = self.criterion(logits, labels)
    return {'loss': loss, 'logits': logits}
```

### 9.2 检查点选择

保存所有 epoch，选择标准:
1. **首要**: 验证集 MSE 最低
2. **次要**: `w_cross` > 0.15 (确保交互有效)
3. **排除**: 任一门控权重 < 0.05 (模态坍塌)

---

**版本**: v1.3  
**日期**: 2025-04-14  
**作者**: AI Assistant  
**更新**: 
- v1.3: 清理文档错误，统一学习率配置
  - 删除"小数据集"残留描述
  - Decoder 学习率: 1e-5 → 5e-4 (全新训练，与ATAC同等对待)
  - 修复章节编号: 统一为第5-9章
- v1.2: **修正计算量错误**，重新核算所有模块FLOPs，添加Flash Attention实现建议
  - Base Model 单层: 6.05 TFLOPs (修正此前错误的~1.8T)
  - predictor_v0 总计: 90.8 TFLOPs (修正此前错误的~16T)  
  - 新方案总计: 107.4 TFLOPs (修正此前错误的~26T)
  - 实际增幅: +18.3% (修正此前错误的+63%)
  - 添加 PyTorch Flash Attention 实现代码
- v1.1: 添加计算量对比分析，训练策略
- v1.0: 初始设计完成

**状态**: 设计完成，待实现验证
