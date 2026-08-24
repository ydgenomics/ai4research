# Cross-Attention Fusion Predictor 设计文档

## 1. 设计目标

设计一个与 `predictor_v0.py` 参数量相当的模态融合模型，通过改进 ATAC Encoder 和 Fusion 模块提升 DNA 与 ATAC 信号的交互能力。

## 2. 架构概览

```
DNA encoder:  DNA序列 → Base Model (12层Mixtral, 完全冻结) → X_dna [B, L, 1024]
ATAC encoder: ATAC信号 → 轻量Transformer (d=192, 6层) → X_atac [B, L, 1024]

Fusion + U-Net decoder (operate at L/4)
  X_dna, X_atac
    ↓ Downsample (n=4)
  X_dna_ds [B, L/4, 1024], X_atac_ds [B, L/4, 1024]
    ↓ 全局双向 Cross-Attn (L/4)
    ↓ 四门控融合 (L/4)
    ↓ Post-fusion bottleneck (predictor_v0-style, L/4)
    ↓ U-Net Decoder
    ↓ Upsample ×4
  RNA预测 [B, 2, L]
    ↑ full-res dual-skip from X_dna + X_atac (learnable weights)
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
Linear(1 → 192) + 与 Base Model 一致的位置编码（共享 RoPE / rotary_emb）
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

**位置编码一致性（关键约束）**:
- Base Model 使用 RoPE（`rotary_emb`）进行位置交互与外推（32k 长度）。
- ATAC Encoder 不再引入独立的 learned absolute position embedding（避免两套位置系统不一致）。
- 实现上建议复用 Base Model 的 `rotary_emb`（或相同参数/相同 `rope_theta` 配置），并使用同一份 `position_ids`，确保 `X_atac` 与 `X_dna` 在 token 坐标系上严格对齐，便于后续 Cross-Attn 融合。

**参数量（重算，约）**: **~2.86M**
- 输入投影 `Linear(1→192)`: ~0.0004M
- 6× TransformerLayer (d=192, heads=4, ffn=768):
  - Self-Attn (Q/K/V/O): \(6 × 4 × 192^2\) ≈ 0.88M
  - FFN (192→768→192): \(6 × 2 × 192 × 768\) ≈ 1.77M
  - LayerNorm + bias: ~0.01M
- 输出投影 `Linear(192→1024)`: ~0.198M
- RoPE 本身 **无可学习参数**（复用/同配 `rope_theta`）

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

#### 3.3.0 Downsample / Upsample (n=4)

为在保持“全局 Cross-Attention”建模能力的同时显著降低计算量，本方案将 Cross-Attn 与 U-Net 的主体计算放在降采样后的序列上。

- **Downsample factor**: `n=4`
- **输入输出形状**:
  - `X_dna`: `[B, L, 1024]`
  - `X_atac`: `[B, L, 1024]`
  - `X_dna_ds`: `[B, L/4, 1024]`
  - `X_atac_ds`: `[B, L/4, 1024]`
- **约束**: 需要 `L % 4 == 0`（默认 `L=32000` 满足）

#### 3.3.1 双向 Cross-Attention

在降采样分辨率上做全局 Cross-Attention，采用双向交互：
- **ATAC → DNA**：ATAC 作为 Query 查询 DNA (Key/Value)
- **DNA → ATAC**：DNA 作为 Query 查询 ATAC (Key/Value)

```python
# 双向: ATAC ↔ DNA
# 方向1: ATAC → DNA
q_a = X_atac_ds                     # [B, L/4, 1024] (可训练)
k_d = X_dna_ds                      # [B, L/4, 1024] (冻结)
v_d = X_dna_ds                      # [B, L/4, 1024] (冻结)

cross_a2d, attn_a2d = F.multi_head_attention_forward(
    query=q_a,
    key=k_d,
    value=v_d,
    embed_dim_to_check=1024,
    num_heads=8,
    q_proj_weight=self.q_a2d.weight,      # [1024, 1024]
    k_proj_weight=self.k_a2d.weight,      # [1024, 1024]
    v_proj_weight=self.v_a2d.weight,      # [1024, 1024]
    out_proj=self.o_a2d,                  # [1024, 1024]
)

# 方向2: DNA → ATAC
q_d = X_dna_ds                      # [B, L/4, 1024] (冻结)
k_a = X_atac_ds                     # [B, L/4, 1024] (可训练)
v_a = X_atac_ds                     # [B, L/4, 1024] (可训练)

cross_d2a, attn_d2a = F.multi_head_attention_forward(
    query=q_d,
    key=k_a,
    value=v_a,
    embed_dim_to_check=1024,
    num_heads=8,
    q_proj_weight=self.q_d2a.weight,      # [1024, 1024]
    k_proj_weight=self.k_d2a.weight,      # [1024, 1024]
    v_proj_weight=self.v_d2a.weight,      # [1024, 1024]
    out_proj=self.o_d2a,                  # [1024, 1024]
)
```

**梯度流向**:
```
Loss
  ↓
cross_a2d, cross_d2a
  ↓
├─→ ATAC Encoder 梯度（来自 q_a / k_a / v_a 相关投影）
└─→ DNA 特征本身冻结（X_dna_ds 不回传梯度）
```

**参数量**: ~8.4M（双向，2 × 4 × 1024²）

#### 3.3.2 四门控融合 (Quaternary Gating)

```python
# 四个输入
atac_self = X_atac_ds                    # ATAC自表示 (L/4)
dna_self = X_dna_ds                      # DNA自表示 (L/4)
cross_a2d = cross_a2d                    # 交互表示: ATAC→DNA (L/4)
cross_d2a = cross_d2a                    # 交互表示: DNA→ATAC (L/4)

# 门控网络 (softmax约束 sum=1)
gate_input = concat([atac_self, dna_self, cross_a2d, cross_d2a])  # [B, L/4, 4096]
weights = softmax(MLP(gate_input))                                 # [B, L/4, 4]
# MLP: Linear(4096 → 512) → GELU → Linear(512 → 4)

w_atac, w_dna, w_a2d, w_d2a = weights[..., 0], weights[..., 1], weights[..., 2], weights[..., 3]
# 约束: w_atac + w_dna + w_a2d + w_d2a = 1

# 融合
fused = (
    w_atac * atac_self +
    w_dna * dna_self +
    w_a2d * cross_a2d +
    w_d2a * cross_d2a
)
```

**参数量**: ~2.1M
- MLP: (4096×512 + 512) + (512×4 + 4) ≈ 2.10M

**初始化**:
```python
with torch.no_grad():
    fusion.gate_mlp[-1].bias[0] = -0.2   # ATAC (稍抑制)
    fusion.gate_mlp[-1].bias[1] = 0.3    # DNA (稍激活，利用预训练)
    fusion.gate_mlp[-1].bias[2] = 0.0    # A→D (中性)
    fusion.gate_mlp[-1].bias[3] = 0.0    # D→A (中性)
```

**生物学意义**:
- `w_dna` 高: 组成型表达，仅DNA序列决定
- `w_atac` 高: 诱导型表达，染色质可及性主导
- `w_a2d` / `w_d2a` 高: 交互调控，需两者协同（分别代表“ATAC驱动的DNA读取”与“DNA驱动的ATAC读取”）

#### 3.3.3 Attention Dropout (替代 Modal Dropout)

不使用模态dropout（会破坏冻结DNA的稳定性），改用attention dropout:

```python
# 在 Cross-Attention 权重上应用 dropout（双向）
attn_a2d = F.dropout(attn_a2d, p=0.1, training=self.training)
attn_d2a = F.dropout(attn_d2a, p=0.1, training=self.training)
# 强制 ATAC 关注更多位置，不只聚焦最强信号

# 对 ATAC 特征做 channel dropout (可选)
atac_features = F.dropout(atac_features, p=0.1, training=self.training)
```

**理由**:
- DNA 冻结，模态 dropout 会破坏 batch 间一致性
- Cross-Attn 对 Query 缺失敏感（双向可一定程度缓解单方向偏置）
- Attention dropout 保持模态存在但增加噪声，增强鲁棒性

#### 3.3.4 Post-fusion Bottleneck（沿用 `predictor_v0.py` 的 bottleneck 思路）

将 `predictor_v0.py` 中验证有效的 bottleneck 卷积块放在 **Cross-Attn + 四门控融合之后**、解码上采样之前（均在 `L/4` 分辨率上）作为“后融合局部精炼”模块：

- **位置**: `cross-attn (L/4) → gate fuse (L/4) → bottleneck convs (L/4) → upsample decoder`
- **作用**: Cross-Attn 提供全局混合，但空间上可能较“粗/噪”；bottleneck 的局部卷积（可用 dilation 扩展感受野）可在上采样前做平滑与结构化重整，通常更利于恢复边界与局部形状。
- **推荐默认**: `2–3 × Conv1d(1024→1024, k=3)`，dilation 取 `1/2/4`，配合 GELU + dropout。

### 3.4 Decoder

核心卷积核设计保持不变（沿用 `predictor_v0.py` 的 U-Net 思路），但本方案将 U-Net 的主体计算放在 `L/4` 分辨率上，并在最后 `Upsample ×4` 回到 `L`。

**Skip connection（高分辨率细节恢复）**:
- 在上采样路径中引入 **full-res** 的 `X_dna` 与 `X_atac` 两个 skip（均为 `[B, L, 1024]`），并使用 **dual-skip gate** 学习两路 skip 的权重（**可学习**，建议 softmax 约束 `w_dna_skip + w_atac_skip = 1`，可做 per-position 或 per-sample）。
  - 该设计用于补偿 `L→L/4` 下采样带来的细粒度信息损失：`X_dna` 补充序列细节，`X_atac` 补充峰形边界与局部可及性形状。

**Dual-skip gate（推荐默认实现，权重可学习）**:

```python
# dec2: [B, 512, L] (最后一次上采样后的解码特征)
# dna_skip = proj_dna(X_dna):  [B, 512, L]
# atac_skip = proj_atac(X_atac): [B, 512, L]

gate_in = concat([dec2, dna_skip, atac_skip], dim=1)  # [B, 1536, L]
logits = conv1x1(gate_in)                             # [B, 2, L]
w = softmax(logits, dim=1)                            # [B, 2, L]
w_dna_skip, w_atac_skip = w[:, 0:1], w[:, 1:2]

skip_mix = w_dna_skip * dna_skip + w_atac_skip * atac_skip  # [B, 512, L]
merged = conv3x3(concat([dec2, skip_mix], dim=1))           # [B, 512, L]
```

```
ATAC conv pyramid (pre-fusion downsample)
  X_atac [B, 1024, L]
    ↓ enc1_atac: Conv1d(1024→512, k=3, stride=1)        → [B, 512, L]
    ↓ enc2_atac: Conv1d(512→1024, k=3, stride=2)        → [B, 1024, L/2]
    ↓ enc3_atac: Conv1d(1024→1024, k=3, stride=2)       → [B, 1024, L/4]  (= X_atac_ds)

DNA downsample (for cross-attn)
  X_dna [B, 1024, L]
    ↓ dna_ds: Conv1d(1024→1024, k=5, stride=4)          → [B, 1024, L/4]  (= X_dna_ds)

Fusion @ L/4
  X_atac_ds, X_dna_ds
    ↓ bidirectional cross-attn (L/4)
    ↓ 4-way gate fuse (L/4)                             → fused_ds [B, 1024, L/4]

Post-fusion refine + U-Net upsample
  fused_ds [B, 1024, L/4]
    ↓ post_fusion_bottleneck: 2–3× DilatedConv1d(1024, dil=1/2/4)
    ↓ up1: ConvTranspose1d(1024→1024, stride=2)          → [B, 1024, L/2]
    ↓ dec1: Conv1d(2048→1024, k=3) (skip from enc2_atac) → [B, 1024, L/2]
    ↓ up2: ConvTranspose1d(1024→512, stride=2)           → [B, 512, L]
    ↓ dec2: Conv1d(1024→512, k=3) (skip from enc1_atac)  → [B, 512, L]
    ↓ dual-skip gate: mix(proj(X_dna), proj(X_atac)) with learnable weights
    ↓ final: Conv1d(512→2, k=1)                          → [B, 2, L]
```

**参数量**: 卷积解码与 pyramid、门控、双 skip 等合计见 **§6**（与 `MultiModalPredictorFusion` 一致）。

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
| ATAC Encoder (CNN) | 67.7B | 67.7B | 135.4B | 5层卷积，约1.06M参数量 |
| Fusion (逐元素相加) | 0.03B | 0.03B | 0.07B | 几乎无计算 |
| Decoder (U-Net) | 1.98T | 3.97T | 5.95T | ~31M参数 |
| **总计** | **74.6T** | **16.1T** | **90.8T** | **~91 TFLOPs/step** |

### 4.3 新方案 (Cross-Attn Fusion)

| 模块 | 前向 FLOPs | 反向 FLOPs | 总计 | 说明 |
|------|-----------|-----------|------|------|
| Base Model (12层冻结) | 72.58T | 0 | 72.58T | 12层 × 6.05TF，无反向 |
| ATAC Encoder (6层Trans) | 4.87T | 9.74T | 14.61T | 低维(d=192)Transformer（全局 self-attn @ L） |
| Fusion (双向Cross-Attn, L/4) | 0.66T | 1.32T | 1.98T | 双向全局Attention，在降采样序列上计算（L² 为 1/16） |
| Fusion (四门控MLP, L/4) | 0.03T | 0.07T | 0.10T | token-wise MLP: 4096→512→4 |
| Conv/U-Net (含ATAC conv pyramid + post-fusion bottleneck + dual-skip gate) | 2.23T | 4.47T | 6.70T | 结构接近 predictor_v0 的 U-Net，额外包含 dna_ds + skip merge |
| **总计** | **80.37T** | **15.60T** | **95.96T** | **~96.0 TFLOPs/step** |

### 4.4 关键对比与洞察

| 对比项 | predictor_v0 | 新方案 | 变化 |
|--------|-------------|--------|------|
| **每 step 总 FLOPs** | **90.8 TFLOPs** | **96.0 TFLOPs** | **+5.7%** |
| Base Model 占比 | 93.3% | 75.6% | 仍主导但占比下降 |
| Fusion 占比 | <0.1% | ~2.1% | 双向全局Cross-Attn（在 L/4 上）+ 四门控 |
| ATAC 编码占比 | ~0.15% | ~15.2% | CNN→Transformer升级（主要新增开销） |

**关键洞察**:

1. **双向 Cross-Attention 在 L/4 上计算，前向约 0.66T**（相较 full-res 单向的 ~4.5T 仍显著降低）
   - 仍是“全局”注意力，但在更粗的 token 网格上完成全局交互
   - 通过解码端 **dual-skip** 引入 `X_dna` + `X_atac`（full-res）恢复细粒度信息

2. **新方案总增幅约 +5–6%（重算约 +5.7%）**
   - 主要新增开销来自 ATAC Transformer
   - 全局 Cross-Attn 在 L/4 上计算后不再是主要负担

3. **计算量分布 (新方案)**:
   - Base Model (冻结): 76.0%
   - ATAC Encoder: 15.8%
   - Fusion (Cross-Attn + Gate): ~2.1%
   - Decoder: 6.2%

### 4.5 Flash Attention 实现建议

**为什么需要 Flash Attention**:

Cross-Attention 的计算几乎全部来自 \(L \times L\) 矩阵乘法（Q@K^T 和 Attn@V）。本方案将注意力放在 `L/4` 分辨率后，理论 FLOPs 已降低 16×；Flash Attention 进一步降低实际运行中的 HBM 访问压力。

| 指标 | 标准 Attention | Flash Attention | 效果 |
|------|---------------|-----------------|------|
| 理论 FLOPs | (L/4)² 缩放 | **不变（相同公式）** | 数学运算相同，仅输入规模变小 |
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

**结论**: 将双向 Cross-Attn 放在 `L/4` 分辨率后，总 FLOPs 增幅约 **+5.2%**；配合 Flash Attention，Cross-Attn 的实际运行开销通常可控，建模能力提升有望显著超过计算代价。

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
- 前 20 steps: **强制四门控均匀 0.25:0.25:0.25:0.25** (硬约束，稳定初始化)
- 监控: ATAC输出稳定性 (梯度范数应 < 10)

**Phase 2: 主训练 (Epoch 3-30, ~60% of total)**
- 全学习率运行
- ATAC Encoder 快速学习 ATAC→RNA 映射
- Fusion 学习有效的模态交互
- 每 5 epochs 评估验证集，监控:
  - 验证 MSE 趋势
  - 门控权重分布 (确保 `w_a2d` 与 `w_d2a` 均不过小，例如 > 0.05)

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
# 2. 交互门控权重 `w_a2d` / `w_d2a` 不过小（例如 min(w_a2d, w_d2a) > 0.05）
# 3. 排除任一门控权重 < 0.05 (模态坍塌)
```

### 5.5 监控指标

| 指标 | 目标范围 | 说明 |
|------|----------|------|
| 训练 MSE | 逐步下降 | 正常收敛趋势 |
| 验证 MSE | 同步或略高 | 与训练MSE差距应在合理范围 |
| `w_atac` | 0.2-0.4 | ATAC支路活跃度 |
| `w_dna` | 0.3-0.6 | DNA支路活跃度 |
| `w_a2d` | 0.05-0.3 | 交互支路活跃度（ATAC→DNA） |
| `w_d2a` | 0.05-0.3 | 交互支路活跃度（DNA→ATAC） |
| `entropy` | 1.2-1.4 | 四门控分布熵，高熵表示均衡使用 |
| 梯度范数 | 稳定 | 避免剧烈震荡或消失 |

### 5.6 调试策略

若训练不稳定:

1. **降低 ATAC 学习率**: 5e-4 → 2e-4 → 1e-4
2. **延长 Warmup**: 从 2 epochs 延长至 5 epochs
3. **固定门控更久**: 硬约束从 20 steps 延长至 50 steps
4. **降低 Batch Size**: 配合 gradient accumulation 保持有效 batch
5. **检查梯度**: 使用梯度裁剪 (clip_norm=1.0) 防止爆炸

若 `w_a2d` 或 `w_d2a` → 0 (交互失效):
- 增大 Fusion 学习率
- 检查初始化 bias (确保交互门控初始非负)
- 考虑预热阶段后强制 min(w_a2d, w_d2a) ≥ 0.05 硬约束

若 `w_atac` 或 `w_dna` → 0 (模态坍塌):
- 降低对应模态的学习率
- 检查辅助损失权重 (atac_l2, gate_entropy)
- 考虑模态 dropout (但当前设计已用 Attention Dropout 替代)
## 6. 参数量总结

| 模块 | 参数 | 说明 |
|------|------|------|
| Base Model | 0M | 完全冻结 (1.2B 冻结) |
| ATAC Encoder (Transformer) | ~2.86M | d=192, 6层, heads=4, ffn=768, RoPE |
| Fusion (Cross-Attn) | ~8.4M | 双向，2 × 4×1024² |
| Fusion (四门控) | ~2.1M | MLP: 4096→512→4 |
| Conv/U-Net + dual-skip（不含 Cross-Attn/Gate） | ~38.57M | predictor_v0 U-Net (~31.75M) − fusion1×1 + dna_ds + dual-skip gate/merge |
| **总计** | **~52.0M** | 全部可训练 |

对比:
- `predictor_v0`: ~137M (解冻Base最后一层 ~104M)
- 本方案: ~52M (更精简，但关键模块能力更强)

## 7. 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| Base Model | 完全冻结 | 保护预训练知识，释放参数预算 |
| ATAC Encoder | 低维深层 Transformer | 层数 > 维度，高效捕获长程依赖 |
| Fusion方向 | 双向 (ATAC↔DNA) | 同时允许“ATAC驱动读取DNA”与“DNA驱动读取ATAC”，减少单方向偏置 |
| 门控设计 | 四门控 + softmax | 约束sum=1，防止坍塌，并区分两种交互方向 |
| Dropout策略 | Attention dropout | 替代modal dropout，兼容冻结DNA |
| Decoder | 保持不变 | 已验证有效，专注上游融合改进 |

## 8. 待验证假设

1. 低维深层 (d=192, 6层) vs 高维浅层 (d=512, 2层): 哪个ATAC编码更有效？
2. 双向 vs 单向 Cross-Attn: 双向是否带来稳定且可复现的提升？
3. 四门控 vs 简化门控: 是否需要显式区分 A→D 与 D→A 两个交互方向？
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
    
    # 3. Fusion (operate at L/4)
    X_dna = dna_features
    X_atac = atac_features
    X_dna_ds = downsample(X_dna, n=4)
    X_atac_ds = downsample(X_atac, n=4)

    # 3.1 Bidirectional Cross-Attention (global at L/4)
    cross_a2d = self.cross_a2d(query=X_atac_ds, key=X_dna_ds, value=X_dna_ds)
    cross_d2a = self.cross_d2a(query=X_dna_ds, key=X_atac_ds, value=X_atac_ds)

    # 3.2 Attention Dropout (bidirectional)
    cross_a2d = F.dropout(cross_a2d, p=0.1, training=self.training)
    cross_d2a = F.dropout(cross_d2a, p=0.1, training=self.training)

    # 3.3 Quaternary gating
    weights = self.quaternary_gate(X_atac_ds, X_dna_ds, cross_a2d, cross_d2a)
    # weights: [w_atac, w_dna, w_a2d, w_d2a], sum=1

    fused_ds = (
        weights[:, 0] * X_atac_ds +
        weights[:, 1] * X_dna_ds +
        weights[:, 2] * cross_a2d +
        weights[:, 3] * cross_d2a
    )
    
    # 4. Decoder (operate at L/4, then upsample ×4 to L; include full-res skips from X_dna + X_atac)
    logits = self.decoder(fused_ds, skip_dna_full_res=X_dna, skip_atac_full_res=X_atac)
    
    # 5. 损失
    loss = self.criterion(logits, labels)
    return {'loss': loss, 'logits': logits}
```

### 9.2 检查点选择

保存所有 epoch，选择标准:
1. **首要**: 验证集 MSE 最低
2. **次要**: 交互门控不过小（例如 min(w_a2d, w_d2a) > 0.05）
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
- v1.2: 修正计算量错误，重新核算所有模块FLOPs，添加Flash Attention实现建议
- v1.1: 添加计算量对比分析，训练策略
- v1.0: 初始设计完成

**状态**: 设计完成，待实现验证
