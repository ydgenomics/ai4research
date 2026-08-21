# Fusion vs v0 — Architecture / Params / FLOPs Comparison (Recomputed)

本文件基于 `model/fusion.md` 与实际代码实现（`model/predictor_v0.py`, `model/predictor_fusion.py`, `model/encoder.py`, `model/encoder_transformer.py`）重新计算并对比 **v0** 与 **fusion** 两种架构在各部分的 **参数规模（单位：M）** 与 **计算量（单位：TFLOPs）**，并给出各部分占比。

---

## Assumptions & Counting Rules（口径）

- **Sequence length**: \(L=32000\)，**batch**: \(B=1\)
- **Compute**: 统计 **forward / backward** 理论 FLOPs（不含 optimizer）
- **Backward 口径（近似）**:
  - 对 **可训练** 模块：\(\text{FLOPs}_\text{bwd} \approx 2 \times \text{FLOPs}_\text{fwd}\)
  - 对 **冻结** 模块：\(\text{FLOPs}_\text{bwd} = 0\)
- **MAC 口径**: 矩阵乘法/卷积的一个 multiply-add 计为 **2 FLOPs**
- **忽略项**（数量级远小于主项，且实现/硬件相关差异大）:
  - LayerNorm / BatchNorm（统计其参数，但忽略其 FLOPs）
  - GELU / SiLU / Softmax / Dropout / Softplus（忽略 FLOPs）
- **Base model**:
  - **v0**: 按 `predictor_v0` 默认实现，**仅最后一层解冻**（其余 11 层冻结）
  - **fusion**: base **全冻结**
  - 说明：**forward FLOPs 不因是否冻结而变化**，均按完整 12 层 forward 计入；backward FLOPs 仅对解冻层计入。
- **DNA encoder 两种统计口径（说明）**:
  - **Params (M)** 列始终统计 **base 全量参数**（此项不随“解冻/冻结”变化）。
  - **Trainable params (M)** 列仅统计 **可训练部分**（v0 / fusion_unfreeze 为 base 最后一层；fusion 为 0）。
- **模块划分（你指定的 4 部分）**:
  - **DNA encoder**: base model（Mixtral 1B, 12L）
  - **ATAC encoder**: v0 为 CNN；fusion 为低维 RoPE-aligned Transformer（d=192, 6L）
  - **Fusion**: v0 为 1×1 conv 融合块；fusion 为 dna_downsample + 双向 cross-attn(@L/4) + 4-way gate MLP
  - **U-Net**: 下采样 + bottleneck CNN + 上采样 + final head（本对比中 **合并为一行**）

---

## Base model architecture（写入：`architechture.detail.txt` 关键原文）

以下内容摘自 `/mnt/zzb/default/Workspace/xuyu/atac/foundation/architechture.detail.txt`（与本 repo 使用的 DNA base model 一致）。

```text
architectures: MixtralForCausalLM
hidden_size: 1024
intermediate_size: 4096
num_hidden_layers: 12
num_attention_heads: 16
num_key_value_heads: 8
num_local_experts: 8
num_experts_per_tok: 2
max_position_embeddings: 32768
rope_theta: 1000000
vocab_size: 128

AutoModel.from_pretrained(..., attn=sdpa, dtype=bfloat16)
parameters: total=1,245,963,264  trainable=1,245,963,264

MixtralDecoderLayer:
  self_attn:
    q_proj: Linear(1024 -> 1024, bias=False)
    k_proj: Linear(1024 -> 512,  bias=False)
    v_proj: Linear(1024 -> 512,  bias=False)
    o_proj: Linear(1024 -> 1024, bias=False)
  block_sparse_moe (Top-2):
    gate: Linear(1024 -> 8, bias=False)
    experts: 8 × (w1: 1024->4096, w3: 1024->4096, w2: 4096->1024)
```

---

## Architecture comparison（结构差异概览）

### v0（`model/predictor_v0.py`）

- **DNA encoder**: Mixtral 12-layer 输出（代码路径为手动 forward；**仅最后一层解冻**）
- **ATAC encoder**: `ATAC_Encoder` 5-layer 1D CNN（1→64→128→256→512→1024）
- **Fusion**: `inputs_embeds + atac_embeds`（逐元素相加）后接 `Conv1d(1024→1024, k=1)` + BN + GELU
- **U-Net**: downsample(×4 via 2×stride2) + 3×dilated bottleneck + upsample(×4 via 2×ConvTranspose) + final head

### fusion（`model/predictor_fusion.py`）

- **DNA encoder**: base 完全冻结（本对比口径）
- **ATAC encoder**: `ATAC_TransformerEncoder`（d_low=192, 6 layers, heads=4, ffn=768, RoPE aligned）
- **Fusion（@L/4）**:
  - 双向 `CrossAttentionSDPA(d=1024, heads=8)`：A→D 与 D→A（均在 \(L/4\)）
  - 4-way gate MLP：`Linear(4096→512)→GELU→Linear(512→4)`（token-wise softmax）
- **U-Net（@L/4 主体 + full-res dual-skip）**:
  - ATAC conv pyramid 产生 downsample skip（`enc1_atac/enc2_atac/enc3_atac`）
  - `dna_downsample: Conv1d(1024→1024, k=5, stride=4)` 将 DNA 特征下采样到 \(L/4\)（本对比将其计入 **U-Net**）
  - post-fusion bottleneck 与 v0 同形态（3×dilated conv）
  - upsample 回到 full-res 后注入 dual-skip（DNA/ATAC 两路 1×1 投影 + softmax gate + merge conv）

---

## Unified comparison table（Params + FLOPs in one table）

### Base model（Mixtral）单层 FLOPs（用于 v0 的“仅最后一层解冻”）

- **Per-layer forward**: **6.006 TFLOPs**
- **Per-layer backward (trainable)**: **12.012 TFLOPs**（按 \(2\times\) 近似）
- **12 layers forward total**: **72.075 TFLOPs**

> 说明：此处按 GQA（K/V dim=512）+ MoE top-2（每 token 激活 2 个 expert）推导，且 **L=32000**。

### v0（单位：Params=M；FLOPs=TFLOPs）

| Part | Params (M) | Trainable params (M) | FLOPs fwd (T) | FLOPs bwd (T) | FLOPs total (T) |
|---|---:|---:|---:|---:|---:|
| DNA encoder (base) | 1245.963 | 103.819 | 72.075 | 12.012 | 84.087 |
| ATAC encoder | 1.059 | 1.059 | 0.068 | 0.135 | 0.203 |
| Fusion | 1.052 | 1.052 | 0.067 | 0.134 | 0.201 |
| U-Net (all) | 30.697 | 30.697 | 0.973 | 1.946 | 2.919 |
| **TOTAL** | **1278.771** | **136.627** | **73.183** | **14.228** | **87.411** |

### fusion（单位：Params=M；FLOPs=TFLOPs）

| Part | Params (M) | Trainable params (M) | FLOPs fwd (T) | FLOPs bwd (T) | FLOPs total (T) |
|---|---:|---:|---:|---:|---:|
| DNA encoder (base) | 1245.963 | 0.000 | 72.075 | 0.000 | 72.075 |
| ATAC encoder | 2.867 | 2.867 | 4.901 | 9.802 | 14.703 |
| Fusion | 10.496 | 10.496 | 0.692 | 1.384 | 2.076 |
| U-Net (all) | 38.567 | 38.567 | 1.225 | 2.450 | 3.674 |
| **TOTAL** | **1297.894** | **51.930** | **78.893** | **13.636** | **92.529** |

### fusion_unfreeze（fusion 模型也解冻 base 最后一层；单位：Params=M；FLOPs=TFLOPs）

| Part | Params (M) | Trainable params (M) | FLOPs fwd (T) | FLOPs bwd (T) | FLOPs total (T) |
|---|---:|---:|---:|---:|---:|
| DNA encoder (base) | 1245.963 | 103.819 | 72.075 | 12.012 | 84.087 |
| ATAC encoder | 2.867 | 2.867 | 4.901 | 9.802 | 14.703 |
| Fusion | 10.496 | 10.496 | 0.692 | 1.384 | 2.076 |
| U-Net (all) | 38.567 | 38.567 | 1.225 | 2.450 | 3.674 |
| **TOTAL** | **1297.894** | **155.750** | **78.893** | **25.649** | **104.542** |

---

## v0 vs fusion — delta summary（关键差异）

### 参数量

- **Trainable params**:
  - v0（**仅 base 最后一层解冻**）: **136.627M**（其中 base last layer **103.819M**）
  - fusion（base 全冻结）: **51.930M**

### 计算量（forward + backward）

- **Forward total**: v0 **73.183T** → fusion **78.893T**（+5.710T，+7.80%）
- **(Forward+Backward) total**: v0 **87.411T** → fusion **92.529T**（+5.118T，+5.86%）

---

## Notes（实现对应关系）

- v0 对应文件：
  - ATAC encoder: `model/encoder.py::ATAC_Encoder`
  - Predictor: `model/predictor_v0.py::MultiModalPredictor`
- fusion 对应文件：
  - ATAC encoder: `model/encoder_transformer.py::ATAC_TransformerEncoder`
  - Predictor: `model/predictor_fusion.py::MultiModalPredictorFusion`
- 本文 “Fusion” 的参数统计包含 `MultiModalPredictorFusion` 中：
  - `cross_a2d` + `cross_d2a`
  - `gate_mlp`
  - 注：本对比将 `dna_downsample` 计入 **U-Net**（见上表）。

