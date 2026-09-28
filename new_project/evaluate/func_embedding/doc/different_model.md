我已通读各模型 config.json、tokenizer 与 README，以下是从 **token 方法 / 模型架构 / 最大上下文 / 参数量** 四个维度的总结。

## 总览表

| 模型 | Token 方法 | 词表 | 模型架构 | 预训练目标 | 最大上下文（token / bp） | 参数量 |
|---|---|---|---|---|---|---|
| **rice/muxi-1B 系列** | ✅ 单碱基 | 128 | **Mixtral** 稀疏 MoE（8 experts, top-2, 12 层, d=1024, RoPE θ=5e7） | CLM（+gengram/mtp/nloss 变体） | **1,048,576 token ≈ 1M bp** | ~1B |
| **NTv3-650M** | 字符级（A/C/G/T/N） | 11 | U-Net 卷积塔 ↓ → Transformer(12层,d=1536) → 反卷积塔 ↑ | MLM | **无硬上限**（编码器降采样，基准中不截断） | ~650M |
| **PlantCAD2** (S/M/L) | 字符级（A/C/G/T/N） | 8 | **Caduceus/Mamba2** 双向 SSM（RCPS：正反向拼接，d_model 翻倍；48 层） | MLM | **无硬上限**（SSM 线性复杂度，不截断） | S< L（d=768→1536） |
| **GENERator-v2-1.2b/3b** | 非重叠 **6-mer** | 4128 | **Llama** decoder-only（26~30 层, GQA, RoPE θ=5e5） | CLM | 16,384 token ≈ **98k bp** | 1.2b / 3b |
| **OryzaG3-8k / -32k** | 非重叠 **3-mer** | 96 | **Gemma3** 混合滑动窗口+全注意力（26 层, d=1152, sliding=512） | CLM | 32,768 token ≈ **98k bp**（8k/32k 仅训练长度） | ~700M |
| **Botanic0-S/M/L** | 非重叠 **6-mer** | 4105 | **ESM 式 encoder**（learned positional，FFN 5120, 20 heads） | MLM (15%) | **1,024 token ≈ 6,144 bp**（硬截断） | S(4层) < M(10层) < L(40层) ≈ 100M–1B |
| **agront_1b** (AgroNT) | 6-mer + 1-mer 兜底 | 4104 | **ESM 式 encoder**（learned positional，同 ESMC） | MLM (15%) | **1,024 token ≈ 6,144 bp**（硬截断） | ~1B |

## 逐模型要点

**rice/muxi-1B 系列（你们自己的）**
- 单碱基 tokenizer ⇒ **1 bp = 1 token**，无 k-mer 压缩，上下文最长（1M）。
- Mixtral MoE：每层 8 个 expert、路由 top-2，12 层。`rope_theta=5e7` 支持长程外推。
- 命名里的 8k/32k/128k/1M 是**训练**上下文（hf 版 config 统一写 1,048,576）。
- 训练变体：gengram / mtp / nloss / newsplit（数据配比）——就是 benchmark 里消融的那批。

**NTv3-650M（InstaDeep）**
- 字符级 tokenizer（11 个 token），**不是** k-mer。
- 独有结构：卷积下采样 7 次 → Transformer → 反卷积上采样（整段序列复制式上采样）+ skip connection；因此对长序列天然友好且无位置上限。
- ~650M（config `num_layers=12, d=1536`；目录名即规模）。

**PlantCAD2（Caduceus/Mamba2 系）**
- 字符级（每个碱基 1 token，vocab=8）。
- **SSM（Mamba2）+ 双向（RCPS）**，正反向输出拼接 ⇒ `hidden_size = 2 × d_model`（L: d=1536→3072, M: d=1024→2048, S: d=768→1536）。
- 线性注意力 → **无上下文上限**，长序列无 OOM 压力。

**GENERator-v2（1.2b/3b）**
- 非重叠 6-mer，**输入长度需为 6 的倍数**（README 明确要求 left pad/trunc，否则末尾追加 `<oov>`）。
- Llama decoder-only，GQA，RoPE θ=5e5；16,384 token 上下文 ≈ 98,304 bp。
- 1.2b：26 层 d=2048；3b：30 层 d=3072。

**OryzaG3（-8k/-32k）**
- 非重叠 **3-mer**（96 词，含 ATC 全部 3-mer）+ 物种限定（水稻泛基因组 149 系，CLM）。
- Gemma3 混合架构：26 层中 4 层 full attention + 22 层 sliding window(512)，`max_position_embeddings=32768`（两个版本相同，差在训练长度）。
- ~700M，README 自称在 polyA 等任务上 AUC≈0.970 超过 AgroNT。

**Botanic0 / agront（两个"短上下文"基线）**
- 都是 **6-mer + ESM 式 encoder + learned positional**，只训练 1024 token。
- **硬截断**：botanic0 截到 1024 token（≈6,144 bp），agront 截到 1025——这就是上一轮说的"2566 bp 序列会被截尾 1542 bp"的模型。
- Botanic0 家族 100M–1B（L=40 层 d=1500 ≈ 1B，M=10 层，S=4 层）；agront = 1B/40 层/d=1500。

## 对对比实验最相关的几点

1. **bp/上下文换算**：同样 16k "上下文"，GENERator 是 98k bp，OryzaG3 是 98k bp，而 Botanic0/agront 只有 6k bp——**比较时务必用 bp 而非 token**。
2. **Token 粒度**差异即信息压缩率：1bp(1:1) vs 3-mer(3:1) vs 6-mer(6:1)。相同 bp 的序列 token 数不同，池化后的 embedding 比较也受此影响。
3. **可对比层**：rice/muxi 12 层，NTv3 12 层；GENERator 26/30 层，OryzaG3 26 层；Botanic/agront 40 层。跨模型比"第 k 层"无意义，建议用各自最后一层。
4. **方向性**：encoder MLM（NTv3/PlantCAD2/Botanic/agront）为双向表示；decoder CLM（rice/muxi/GENERator/OryzaG3）为因果表示——语义空间本就不同，embedding 直接做相似度对比需额外校准（如 CCA / 后接分类器而非裸 cos 相似度）。