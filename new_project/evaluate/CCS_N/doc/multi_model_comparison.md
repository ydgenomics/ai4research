# CCS-N 变异感知评测: 多模型 Zero-Shot 对比统计说明

> 评测时间: 2026-09-17 | 数据: 251 水稻样本 VCF (NH001 样本, 1000 纯合 SNP)
> 结果文件: `output/hom_1000_multi7/ccs_n_summary.json` | 图: `output/hom_1000_multi7/ccs_n_compare.png`

## 1. 评测设置

- **样本**: NH001（野生型参考水稻），**1000 个纯合 (1/1) 双等位 SNP**
- **窗口**: 256 bp，变异位于窗口中心 (idx=127)
- **对照**: 每个真实变异位点生成随机替代 (随机碱基 ≠ ref) 作为假变异
- **指标**: CCS-N = s₀ + Σ(sᵢ⁺ + sᵢ⁻)，N 为累加半径；AUC = 区分真实变异 vs 随机对照
- **模型全部 zero-shot**（未针对该数据集微调）

## 2. 模型清单

| 模型 | 参数量 | 架构 | tokenizer | 加载方式 |
|---|---|---|---|---|
| rice_1B_stage2_8k | 1B | 因果 (decoder) | 纯单碱基 | AutoModelForCausalLM |
| variant_cpt_32k (final) | 1B | 因果 (decoder) | 纯单碱基 | AutoModelForCausalLM |
| AgriGenome 1.2b (32k CPT) | 1.2B | 因果 (decoder, Mixtral) | 纯单碱基 | AutoModelForCausalLM |
| agront_1b | 1B | ESM 双向 (encoder) | 6-mer 合并 | AutoModel |
| Botanic0-L | 1B | ESM 双向 (encoder) | 6-mer 合并 | AutoModel |
| **PlantCAD2-Large** | 1B | Caduceus (双向 SSM) | 纯单碱基 | AutoModelForMaskedLM |
| **NTv3_650M_pre** | 650M | Encoder + 卷积上下采样 | 纯单碱基 | NTv3PreTrained (自定义) |

## 3. 主结果: 欧氏距离 AUC（真实变异 vs 随机对照）

| N | rice_1B | variant_cpt | AgriGenome | agront_1b | Botanic0-L | PlantCAD2-L | NTv3_650M |
|---|---|---|---|---|---|---|---|
| 0 | 0.478 | 0.476 | 0.495 | 0.805 | 0.810 | 0.522 | 0.513 |
| 1 | 0.603 | 0.598 | 0.653 | 0.833 | 0.838 | 0.543 | 0.508 |
| 2 | 0.626 | 0.625 | 0.693 | 0.841 | 0.841 | 0.559 | 0.517 |
| 4 | 0.634 | 0.635 | 0.723 | 0.848 | 0.839 | 0.590 | 0.535 |
| 8 | 0.650 | 0.655 | 0.746 | 0.845 | 0.833 | 0.625 | 0.561 |
| 16 | **0.660** | **0.667** | **0.768** | **0.855** | **0.831** | **0.666** | **0.591** |

### 余弦距离 AUC

| N | rice_1B | variant_cpt | AgriGenome | agront_1b | Botanic0-L | PlantCAD2-L | NTv3_650M |
|---|---|---|---|---|---|---|---|
| 0 | 0.478 | 0.476 | 0.495 | 0.808 | 0.778 | 0.513 | 0.514 |
| 1 | 0.527 | 0.520 | 0.551 | 0.850 | 0.832 | 0.532 | 0.517 |
| 2 | 0.526 | 0.522 | 0.570 | 0.860 | 0.843 | 0.549 | 0.522 |
| 4 | 0.513 | 0.510 | 0.589 | 0.873 | 0.855 | 0.580 | 0.534 |
| 8 | 0.520 | 0.522 | 0.614 | 0.880 | 0.867 | 0.617 | 0.551 |
| 16 | **0.531** | **0.537** | **0.646** | **0.882** | **0.874** | **0.663** | **0.576** |

### 排名（N=16 欧氏 AUC）

1. **agront_1b: 0.855**（余弦 0.882 也是最高）
2. **Botanic0-L: 0.831**（余弦 0.874）
3. **AgriGenome 1.2b: 0.768**
4. **variant_cpt (ft): 0.667**
5. **PlantCAD2-Large: 0.666**（余弦 0.663，方向信号明显强于因果模型）
6. **rice_1B base: 0.660**
7. **NTv3_650M_pre: 0.591**

## 4. 关键发现

### 4.1 ESM 双向模型 (agront/Botanic) 变异感知远超因果模型
- agront_1b / Botanic0-L 的 AUC 在 **N=0 就高达 0.80+**（单点变异 token 距离即可区分），且余弦距离也很强（0.83-0.88）
- 原因: ESM 双向注意力让变异信息**全序列传播**；且 6-mer tokenizer 下单个碱基变异影响 token 较多

### 4.2 因果单碱基模型 (rice/AgriGenome) 变异感知较弱但随 N 累积增强
- N=0 时 AUC ≈ 0.48-0.50（单点感知弱，ref/alt 单 token 距离无区分度）
- 随 N 增大单调上升: rice_1B 0.60→0.66, AgriGenome 0.65→0.77
- **CCS-N 累积机制有效**: 因果模型对变异的感知主要通过**上下文 token 表征变化**体现，而非变异位点本身
- 余弦较弱 (0.51-0.65) → 因果模型的变异感知主要是**幅度信号**（欧氏），方向信号弱

### 4.3 AgriGenome 1.2b 显著强于 rice_1B（同架构对比）
- 同为单碱基因果模型，AgriGenome 全面领先: N=1 0.653 vs 0.603, N=16 0.768 vs 0.660
- 说明 AgriGenome 的预训练/CPT 使其对变异更敏感，是**同架构下更好的变异感知模型**

### 4.4 FT 对变异感知几乎无提升
- variant_cpt (32k CPT) vs base: AUC 差异 < 0.01（0.667 vs 0.660）
- 变异感知是模型预训练阶段的固有表征能力，下游 CPT 微调不显著改变

### 4.5 PlantCAD2 (Caduceus 双向 SSM): 幅度与方向信号兼顾
- 欧氏 N=16 = 0.666，与因果模型 rice/final 相当
- 但**余弦 N=16 = 0.663**，明显优于因果模型的 0.53，接近 agront/Botanic 的水平
- N=0 仅 0.522 → 单点感知弱，需 CCS-N 上下文累积（0.52→0.67，增益 +0.14，仅次于 agront）
- SSM 双向结构使其变异感知同时保留方向信息

### 4.6 NTv3_650M_pre: 逐位分辨能力最弱
- 欧氏 N=16 = 0.591，**7 个模型中最低**；余弦 0.576
- N=0 仅 0.513 → 单点变异几乎无区分度
- 但 N 累积增益相对明显（0.513→0.591，+0.078），说明变异信息主要靠上下文传播
- 距离量级巨大（mean_real 7904 vs rice 390），因 encoder 下采样 + deconv 上采样的卷积结构导致表征范数很大
- **结构原因**: 7 层卷积下采样把 256bp 压缩到 2 token，再上采样恢复，单碱基变异的逐位细节在压缩中损失

## 5. 方法学说明

- **single_base 模式**: 变异 token = 坐标映射 (pos-1-chrom_start)，精确。适用于 rice/ft/AgriGenome/PlantCAD2/NTv3
- **kmer_diff 模式** (agront/Botanic): 用 ref/alt token id 差分定位首个不同 token（256bp→46 tokens，单碱基变异影响恰好 1 个 token，已验证），近似但稳定
- **caduceus 模式**: PlantCAD2 用 `AutoModelForMaskedLM` + `trust_remote_code` 加载 Caduceus 主体（RCPS 双向 SSM）；其 forward 不接受 attention_mask，已包装丢弃
- **ntv3 模式**: NTv3 用自定义 `Ntv3PreTrainedConfig` + `NTv3PreTrained` 加载（内部管理 fp32 卷积精度）；tokenizer 不返回 attention_mask，用全 1 兜底；encoder 下采样后 deconv 上采样恢复 256 token 逐位分辨率（已验证变异位响应最大）
- 不同模型 tokenizer 与表征尺度不同，**绝对距离不可跨模型比较**，但 **AUC（相对区分度）可比较**
- agront/Botanic 距离量级 (8-1751)、NTv3 (321-7904)、rice (19-390) 差异巨大，反映各自隐藏层范数

### 工程注意
- 多模型顺序评测需**每个模型跑完释放显存**（`del model` + `gc.collect()` + `torch.cuda.empty_cache()`），否则累积 OOM
- PlantCAD2 的 Triton kernel (mamba_ssm) 会 crash，若当前 CUDA 设备显存不足 —— 需先 `torch.cuda.set_device(device_idx)` 让 autotuner 在目标卡 benchmark

## 6. 结论

> 不同 DNA 模型对相同变异的 zero-shot 表征能力排名:
> **agront_1b (0.855) > Botanic0-L (0.831) > AgriGenome 1.2b (0.768) > variant_cpt (0.667) ≈ PlantCAD2-L (0.666) ≈ rice_1B (0.660) > NTv3_650M (0.591)**
>
> - **ESM 双向模型** (agront/Botanic) 天然对点突变高度敏感（N=0 即 AUC>0.8），无需上下文累积
> - **Caduceus 双向 SSM** (PlantCAD2) 欧氏与因果模型相当，但**方向信号（余弦 0.663）明显更强**，是幅度+方向兼顾的模型
> - **因果模型** (rice/AgriGenome/ft) 需 CCS-N 累积上下文 token 变化才能感知变异，其中 AgriGenome 1.2b 显著优于 rice_1B
> - **卷积上下采样结构** (NTv3) 因逐位分辨率在下采样中损失，变异感知最弱
> - 变异感知在因果模型中主要是**幅度信号**（欧氏），方向信号（余弦）较弱

## 7. 变异位点统计分析与 CCS-N 热图（2026-09-18）

> 新增 per-site 分数持久化 (`run_ccs_n.py --save-per-site`)，输出:
> `output/hom_1000_multi7/per_site/<model>/ccs_n_per_site.npz` + `distance_profiles.npz`
> 配套脚本: `scripts/variant_stats.py` / `scripts/heatmap_analysis.py` / `scripts/corr_analysis.py`

### 7.1 变异位点统计 (`variant_stats.json`)

| 统计量 | 数值 |
|---|---|
| 位点数 | 1000 (纯合 1/1 双等位 SNP) |
| 染色体分布 | Chr6=131, Chr10=140 最多; Chr3=32, Chr5=35 最少; ChrUn=1 |
| Ti/Tv 比值 | **3.18** (Ti=761, Tv=239) —— 水稻自然变异典型比值 |
| SNP 主要类型 | G>A (207), T>C (194), A>G (185), C>T (175) —— 均为转换 |
| AF 分布 | mean=0.508, median=0.595; 低频(<0.05)=0, 中频(0.05-0.5)=426, 高频(>=0.5)=574 |
| 窗口内位置 | 全部位于中心 (rel_pos=0.5, std=0) |
| 窗口 GC 含量 | mean=0.436, std=0.118 —— 略低于水稻基因组均值 (~0.44) |

要点:
- 采样位点全部为**高频 SNP** (AF≥0.05, 无稀有变异)，因纯合位点在群体中本身多为高频
- Ti/Tv=3.18 符合水稻 SNP 期望 (转换>颠换)
- 窗口内变异严格居中 → 两侧上下文对称, 无位置偏差

### 7.2 CCS-N 热图解读

**图 `heatmap_model_site.png`** — 模型(行) × 1000 位点(列) per-site CCS-N (欧氏 N=16):
- 列按所有模型平均得分排序 (左→右: 变异效应小→大)
- 行内色块差异大 → **位点间变异效应异质性强**
- agront/Botanic 行几乎全亮 → 对所有位点均高敏感; NTv3/PlantCAD2 行偏暗 → 整体感知弱
- 因果模型 (rice/final/AgriGenome) 行呈渐变 → 对"简单"位点得分低、对"难"位点得分高

**图 `heatmap_model_site_disc.png`** — per-site 区分度 (real − rand):
- 正值 (红) = 该位点变异效应 > 随机替代效应 → 模型捕捉到真实变异特异性
- agront/Botanic 行大面积为正且幅度大 → 变异特异性感知最强
- NTv3 行为负的区域较多 → 真实变异甚至不如随机替代可区分 (与 AUC≈0.59 一致)

**图 `heatmap_distance_profile.png`** — 模型 × token 偏移的平均距离剖面 (R=16, 方案 B 归一化):
- **上图 `log2(real/rand)`**: 每个 token 位置真实变异 vs 随机对照的距离比值 (log2 刻度, 模型间可比)
  - 中心 (offset=0) 处 Botanic/agront 达 0.9-1.2 (≈2-2.3×) → 变异点本身即强区分
  - NTv3/PlantCAD2 中心仅 0.02-0.25, 两侧 0.5-0.8 → 点信号弱, 靠上下文累积
  - 因果模型 (rice/final/AgriGenome) 中心 ≈0, **上游被 log2 截断在 +2** (上游 rand≈0, 单向注意力看不到变异) → 因果模型对上游完全"失明", 信号只在下游
- **下图**: 绝对距离剖面 (原始尺度) — 展示各模型表征范数差异 (NTv3 200-330 vs PlantCAD2 1-6)
- 边缘截断 (+2/-2) 是真实架构特性, 不是数值伪影

**图 `heatmap_model_corr.png`** — 模型间 per-site 得分 Spearman 相关:
- agront 与 Botanic 相关最高 (同为 ESM 双向 6-mer)
- 因果模型之间相关中等 (rice/final/AgriGenome 两两相关 0.5-0.7)
- NTv3 与其他模型相关最低 → 其变异感知机制与其他架构差异最大

### 7.3 per-site 得分 × 位点属性相关 (`corr_analysis.py`)

| 模型 | AF 相关 | Ti/Tv 相关 |
|---|---|---|
| AgriGenome | -0.041 | -0.188 |
| Botanic0-L | -0.061 | -0.082 |
| NTv3 | -0.001 | -0.088 |
| PlantCAD2 | -0.077 | -0.139 |
| agront_1b | -0.014 | -0.100 |
| final | -0.068 | -0.147 |
| rice_1B | -0.071 | -0.146 |

要点:
- **所有模型与 AF 相关性≈0** → CCS-N 反映的是**序列/表征层面**的变异效应，与群体频率无关
- Ti/Tv 弱负相关 (r≈-0.1~-0.19) → 颠换位点得分略高，但效应很小
- **模型间分歧最大位点**均为 NTv3 得分异常高 (>68000) 的位点 → 这些是 NTv3 卷积结构的"共振"位点，非真实变异效应

### 7.4 纯 CCS-N 感知分析（无 rand 版本, `scripts/ccsn_pure_analysis.py`）

> 只使用 ref vs alt 的真实 CCS-N 分数 (`real_*` key), **完全不用 rand**。
> 核心指标: **位点级上下文增益** = CCS-N(N=16) / CCS-N(N=0) 的 per-site 比值。
> 意义: 变异中心感知为 1, 上下文累积倍数 = 模型把变异信号扩散到上下文的能力。
> 该比率为分子分母同尺度的相对量, **不受模型表征范数影响, 可跨模型直接比较**。

| 排名 | 模型 | ctx增益中位 | ctx增益均值 | top10% | CV |
|---|---|---|---|---|---|
| 1 | **NTv3_650M_pre** | **25.1** | 49.0 | 3.93 | 1.53 |
| 2 | variant_cpt (final) | 21.3 | 23.5 | 1.64 | 0.41 |
| 3 | rice_1B base | 21.1 | 23.5 | 1.65 | 0.42 |
| 4 | AgriGenome 1.2b | 18.2 | 19.6 | 1.54 | 0.34 |
| 5 | agront_1b | 16.0 | 29.3 | 1.44 | 0.30 |
| 6 | Botanic0-L | 8.6 | 11.3 | 1.65 | 0.41 |
| 7 | PlantCAD2-L | 8.6 | 9.2 | 1.77 | 0.50 |

**无 rand 视角的关键发现**:
- **因果模型 (NTv3/final/rice/AgriGenome) 上下文增益最高 (18-25×)**: 变异中心信号弱 (s0 仅 18-321), 但随 N 累积放大 20 倍以上 → 它们主要通过**上下文传播**感知变异
- **ESM 双向模型 (agront/Botanic) 上下文增益低 (8-16×)**: s0 本身强 (8-189), 变异中心已含大部分信息, 上下文累积倍数小 → 它们主要通过**变异点本身**感知

## 7.5 基于功能区分类（CDS / Intron / Intergenic）的 6-token 分箱热图

为了更深入地探究模型对不同基因组功能区变异的感知能力，我们将 1000 个纯合 SNP 位点与水稻官方注释文件 `osa1_r7.all_models.gff3` 进行重叠分析，将位点划分为三类：
- **CDS (编码区)**: 166 个位点
- **Intron (内含子区)**: 194 个位点（包含非编码转录区）
- **Intergenic (基因间隔区)**: 640 个位点

同时，我们将 33 个 token 偏移（-16 到 +16）按每 6 个 token 进行分箱（Binning），并对每个模型进行全局 95% 分位数归一化，得到了 21 行（7 模型 × 3 区域）× 5 列（5 个 6-token 分箱）的格子热图 `heatmap_by_functional_region.png`。

### 核心数值结果

| Model | Region | [-12, -7] | [-6, -1] | [0] (中心) | [+1, +6] | [+7, +12] |
|---|---|---|---|---|---|---|
| AgriGenome_4n8a | cds        | 0.28 | 0.29 | **0.82** | 0.70 | 0.62 |
| AgriGenome_4n8a | intron     | 0.30 | 0.32 | **0.84** | 0.75 | 0.68 |
| AgriGenome_4n8a | intergenic | 0.26 | 0.28 | **0.83** | 0.70 | 0.62 |
| Botanic0-L | cds        | 0.19 | 0.23 | **0.90** | 0.35 | 0.33 |
| Botanic0-L | intron     | 0.19 | 0.23 | **0.93** | 0.35 | 0.35 |
| Botanic0-L | intergenic | 0.16 | 0.22 | **0.91** | 0.32 | 0.31 |
| NTv3_650M_pre | cds        | 0.12 | 0.15 | **0.18** | 0.14 | 0.12 |
| NTv3_650M_pre | intron     | 0.23 | 0.27 | **0.32** | 0.27 | 0.25 |
| NTv3_650M_pre | intergenic | 0.20 | 0.24 | **0.30** | 0.25 | 0.21 |
| PlantCAD2-Large | cds        | 0.28 | 0.36 | **1.15** | 0.36 | 0.28 |
| PlantCAD2-Large | intron     | 0.26 | 0.33 | **1.08** | 0.33 | 0.25 |
| PlantCAD2-Large | intergenic | 0.26 | 0.35 | **1.13** | 0.35 | 0.28 |
| agront_1b | cds        | 0.35 | 0.47 | **0.83** | 0.59 | 0.53 |
| agront_1b | intron     | 0.35 | 0.47 | **0.84** | 0.61 | 0.57 |
| agront_1b | intergenic | 0.34 | 0.47 | **0.84** | 0.58 | 0.54 |
| final | cds        | 0.23 | 0.25 | **0.58** | 0.55 | 0.49 |
| final | intron     | 0.25 | 0.29 | **0.65** | 0.62 | 0.58 |
| final | intergenic | 0.23 | 0.26 | **0.66** | 0.61 | 0.56 |
| rice_1B_stage2_ | cds        | 0.23 | 0.25 | **0.59** | 0.55 | 0.49 |
| rice_1B_stage2_ | intron     | 0.25 | 0.29 | **0.64** | 0.61 | 0.56 |
| rice_1B_stage2_ | intergenic | 0.23 | 0.26 | **0.66** | 0.61 | 0.56 |

### 关键科学洞察

1. **变异中心 `[0]` 的强感知信号被成功突出**：
   - 在之前的 6-token 混合分箱中，变异中心被稀释在 `[-3, +2]` 这一大格中，导致中心感知强度被低估（例如 `PlantCAD2-L` 之前仅为 0.55）。
   - 将变异中心 `[0]` 独立分箱后，**所有模型的中心感知强度均大幅跃升**。例如 `PlantCAD2-L` 在 `cds` 区的中心感知达到了 **1.15**，`Botanic0-L` 达到了 **0.90**，`AgriGenome` 达到了 **0.82**。
   - 这表明变异点本身的表征变化是极其剧烈的，独立分箱能更真实地反映模型对点突变的即时感知能力。

2. **因果模型（rice_1B / final / AgriGenome）的单向感知特征**：
   - 在中心 bin `[-3, +2]` 处，感知强度显著上升（0.42 - 0.56）。
   - 在下游 bin `[+3, +8]` 和 `[+9, +14]` 处，感知强度依然维持在极高水平（0.49 - 0.72），甚至高于中心。
   - 在上游 bin `[-15, -10]` 处，感知强度极低（0.23 - 0.29）。
   - 这完美印证了 **Causal LM（因果语言模型）** 的单向注意力机制：变异的影响只能向右（下游）传播，无法向左（上游）传播。

2. **双向模型（Botanic0-L / agront_1b / PlantCAD2-L）的对称感知特征**：
   - 感知强度以中心 bin `[-3, +2]` 为核心，向两侧对称递减。
   - 例如 `PlantCAD2-L` 在 `cds` 区：上游 `[-15, -10]` 为 0.28，中心 `[-3, +2]` 达到峰值 0.55，下游 `[+9, +14]` 降回 0.28。
   - 这表明 **Masked LM（双向模型）** 和 **State Space Model（如 Caduceus/PlantCAD）** 能够将变异的影响双向对称地传播到上下文。

3. **不同功能区（CDS vs Intron vs Intergenic）的感知差异**：
   - **NTv3_650M** 表现出极强的区域特异性：其在 `intron`（0.31）和 `intergenic`（0.28）的中心感知显著强于 `cds`（0.17）。
   - **PlantCAD2-L** 在 `cds` 区的中心感知（0.55）强于 `intron` 区（0.50）。
   - **AgriGenome** 和 **agront_1b** 在三类功能区上的感知强度非常均衡，说明其预训练语料和表征空间对全基因组具有极好的泛化能力。
- **NTv3 增益异常高 (25×, CV=1.53)**: 卷积上下采样放大效应, 属架构伪影
- 结论: "感知能力"在不同架构有**不同机制** —— 因果模型靠上下文累积, 双向模型靠点位感知; 无 rand 时用上下文增益排序, 有 rand 时用 AUC 排序, 两者互补

**输出**: `heatmap_ccsn_pure.png` (模型×位点归一化热图), `ccsn_ranking.png` (增益排名),
`ccsn_Ncurve.png` (累积曲线), `ccsn_pure_summary.json`
