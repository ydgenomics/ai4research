# 说明不同指标计算公式以及在不同分辨率、不同全局层面其代表的意义


**计算公式（数学定义）：**

### pcc
$$
\text{pcc} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\sqrt{\sum (y_i - \bar{y})^2}}
$$
- **bp**: 所有碱基（含零值）→ 零值天然相关，pcc 偏高
- **exon/gene**: 均值后跨特征计算 → 衡量模型对基因表达水平的排序能力

### log1p_pcc
$$
\text{log1p\_pcc} = \text{Pearson}\big(\log(1 + y_{\text{true}}),\; \log(1 + y_{\text{pred}})\big)
$$
- 压缩高表达极值，放大低表达区偏差，使指标不被高表达基因主导

### nozero_pcc
$$
\text{nozero\_pcc} = \text{Pearson}\big(y_{\text{true}}[y_{\text{true}}>0 \cap y_{\text{pred}}>0],\; y_{\text{pred}}[\cdots]\big)
$$
- 只算预测和真实都 > 0 的位点/基因
- **bp**: 排除零值干扰，反映模型对"有表达区域"的建模精度
- **exon/gene**: 均值 > 0 时 ≈ pcc

### zero_ratio
$$
\text{zero\_ratio} = \frac{\sum \mathbb{1}(y_{\text{true}} = 0)}{n} \times 100\%
$$
- 仅 bp 分辨率有意义（exon/gene 理论上 ≈ 0）
- 反映基因组未表达区域占比

### r2
$$
R^2 = 1 - \frac{\sum (y_{\text{true}} - y_{\text{pred}})^2}{\sum (y_{\text{true}} - \bar{y}_{\text{true}})^2}
$$
- **区别于 pcc**: pcc 只衡量趋势一致性，r2 还衡量**尺度准确性**
- **bp**: 零值主导，参考价值低
- **gene**: 模型能否准确预测不同基因的表达量级

### delta_pcc（feature_ref）
$$
\tilde{y} = y / \mu,\quad \Delta = \tilde{y} - \mu_{\text{ref}}(g),\quad \text{delta\_pcc} = \text{Pearson}(\Delta_{\text{pred}}, \Delta_{\text{true}})
$$
- $\mu_{\text{ref}}(g)$ = 该基因在训练集所有品种中的归一化真实表达均值
- 去除"基础表达水平"，只看品种特异的偏差模式
- **只算 gene 分辨率**

---

**各全局层面的含义：**

| 全局 | 粒度 | 输入 | 用途 |
|------|------|------|------|
| **sample** | 全基因组 | 多染色体拼接 | 模型整体能力评估 |
| **chromosome** | 单染色体 | 单染色体数据 | 染色体间一致性检查 |
| **window** | 单窗口 | 每窗口一行 | 局部表现探查，不含 delta_pcc |
| **gene** | 单基因 | 每基因/exon 一行 | 具体基因诊断，仅 bp+exon，不含 delta_pcc |

**各分辨率的适用场景：**

| 分辨率 | 核心价值 |
|--------|---------|
| **bp** | 全基因组碱基级覆盖，zero_ratio 反映数据稀疏性 |
| **exon** | 排除 intron 零值后，外显子表达预测精度 |
| **gene** | **论文核心指标**：跨基因表达排序能力 + 尺度准确性 |
| **gene-low/medium/high** | 分层展示模型在不同表达水平的性能差异 |