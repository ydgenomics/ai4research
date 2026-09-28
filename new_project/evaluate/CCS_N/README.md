# CCS-N: 变异感知能力评测

评测 DNA 基础模型对单核苷酸变异 (SNV) 的感知能力。

## 背景

CCS-N（Cumulative Context Score，累积上下文得分）通过对比参考序列与变异序列的模型表征差异，量化模型对变异的感知：

1. **输入**：对每个 SNV，构建参考序列 (ref) 和变异序列 (alt)
2. **逐 token 打分**：将两条序列输入同一 DNA 基础模型，在嵌入空间计算每个对应 token 的差异
3. **确定中心**：以变异所在 token 为中心，记为 $s_0$
4. **对称累加**：
   $$\text{CCS-N} = s_0 + \sum_{i=1}^{N}(s_i^+ + s_i^-)$$
   其中 $s_i^+$/$s_i^-$ 是中心右侧/左侧第 $i$ 个 token 的得分

真实 SNV 应显著高于随机碱基替换（对照），高 AUC 表示模型对变异敏感。

## 数据

- **VCF**: `DATA/shang_data/vcf/251.SNP.final.vcf.gz` (251 样本, 573 万变异, 含 13.6 万 multi-allelic)
- **参考基因组**: `rice_server/source/rice_mut/osa1_r7.asm.ch.fa`
- 过滤: 保留双等位 SNP (约 559 万) → `data/251.SNP.bsnp.vcf.gz`

## 方法

### 为什么用 GenVarLoader (GVL)

GVL 预处理 VCF + BED 窗口 → 可直接按 (region, sample) 取序列：
- `with_seqs('reference')` → 参考序列（无变异）
- `with_seqs('haplotypes')` → 个性化单倍体（应用了该样本的真实变异）

### 关键坑（已验证）

1. **GVL Filter 有索引错位 bug**：`gvl.write` 拷贝未过滤的 `.gvi` 索引到 `variants.arrow`，
   但 genotypes 的变异索引基于内存过滤后的 index → 错位 → haplotypes 不应用变异。
   **必须上游过滤 VCF**（cyvcf2 保留双等位 SNP），GVL 直接用过滤文件、不用 Filter。
2. **参考纯合位点**：该样本 GT=0/0 的位点，haplotypes == reference（GVL 不写空变异），
   无法构造 alt 序列，自动跳过。
3. **坐标**：VCF POS 1-based；BED chromStart 0-based；窗口内变异下标 = `pos - 1 - chromStart`。

## 使用

### 1. 数据准备

```bash
python scripts/prepare_data.py \
    --vcf ../../../../../DATA/shang_data/vcf/251.SNP.final.vcf.gz \
    --ref ../../../../rice_server/source/rice_mut/osa1_r7.asm.ch.fa \
    --out-dir data \
    --samples NH001 \
    --window-size 256 \
    --max-variants 5000   # 冒烟测试; 省略则全量
```

生成:
- `data/251.SNP.bsnp.vcf.gz`: 双等位 SNP VCF
- `data/windows.bed`: 每 SNP 256bp 窗口 (变异位于 idx=127)
- `data/gvl/`: GVL 数据集

### 2. 运行 CCS-N 评测

```bash
python scripts/run_ccs_n.py \
    --model-dir /path/rice_1B_stage2_8k_hf \
    --model-dir /path/variant_cpt_32k_10samp/final \
    --data-dir data \
    --out-dir output \
    --ref ../../../../rice_server/source/rice_mut/osa1_r7.asm.ch.fa \
    --sample NH001 \
    --batch-size 16 \
    --device cuda:0 \
    --max-regions 500
```

### 3. 结果

`output/ccs_n_summary.json`:
- 每个模型 × 每个距离度量 (euclidean/cosine) × 每个 N (0/1/2/4/8/16)
- `mean_real` / `mean_rand`: 真实 SNV vs 随机对照的平均 CCS-N
- `ratio`: 真实/随机 比值（>1 表示真实变异引起更大扰动）
- `auc`: 真实 vs 随机二分类 AUC（0.5 无区分度，1.0 完全区分）

## 代码结构

```
CCS_N/
├── scripts/
│   ├── prepare_data.py   # 过滤VCF + 窗口BED + GVL写入
│   ├── run_ccs_n.py      # 主脚本: GVL读序列 + 模型前向 + CCS-N
│   └── utils.py          # 模型加载 / 距离度量 / CCS-N公式 / 区分度
├── data/                 # 数据 (VCF, BED, GVL)
└── output/               # 结果
```

## 环境

- conda env: `vllm` (python 路径 `/root/miniconda3/envs/vllm/bin/python`)
- 依赖: genvarloader 0.42.1, genoray 4.0.2, cyvcf2, polars, transformers, torch
