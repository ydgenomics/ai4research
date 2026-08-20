# delta_pcc 系列指标说明

> 本文档说明 `run_evaluation.py` 中所有 delta 相关指标的定义、动机、计算逻辑与适用场景。
> 相关代码：`pearson_delta_scale` / `compute_pearson_delta_metrics` / `build_per_gene_reference` / `compute_all_delta_pearson` / `build_ref_sample_from_train` / `compute_ref_sample_delta_metrics` / `compute_ref_sample_delta_from_df` / `compute_pairwise_delta_metrics`。

---

## 1. 背景与动机

普通 pcc 衡量模型对基因表达**水平**的预测能力，但跨基因相关性主要由"基础表达水平"主导（高表达基因天然易预测、零值天然相关），**品种特异的差异表达模式**被淹没。

delta 系列指标的目标：**去除"基础表达水平"，只看品种特异的偏差（deviation）模式预测能力**。

另一个动机（真实应用约束）：真实世界里用户拿到的是**未归一化的绝对预测值**，无法事后做样本级 scale（如除以样本均值）。因此指标应当：
- 不依赖任何**事后统计量**（样本均值等），训练完即可固定参考；
- 训练集参考在部署时可直接套用。

---

## 2. 指标定义

### 2.1 原版 `delta_pcc`（feature_ref 模式，样本内归一化）

出现在 `00_main_summary.csv` 的 `resolution=gene` 与 `gene-low/medium/high` 行。

$$\tilde{y} = \frac{y}{\mu},\quad \Delta = \tilde{y} - \mu_{ref}(g),\quad \text{delta\_pcc} = \text{Pearson}(\Delta_{pred}, \Delta_{true})$$

- $\mu_{ref}(g)$ = 该基因在训练集所有品种中的**归一化真实表达均值**（每个样本先除以自身均值，再跨品种平均）；
- pred 与 true 各自除以**自己的样本均值**做归一化（`global_mean` 方法）；
- 逐基因减参考值，只保留能匹配到参考值的基因（≥2 个，否则 NaN）。

**特性**：
- 对"只学整体缩放、无品种特异偏差"的模型**诚实**（模拟中 0.07）；
- 依赖样本内归一化（事后统计量），真实应用**不可复现**；
- 逐基因共享参考向量带来的伪相关风险，指标可能略偏松。

### 2.2 `delta_pcc_pred`（ref_pred 参考，宽松版）

$$\Delta_{pred} = pred - \bar{p}_{train}(g),\quad \Delta_{true} = true - \bar{t}_{train}(g)$$

- $\bar{p}_{train}(g)$ = 该基因在训练集所有品种中的**预测均值**（ref_pred_mean）；
- $\bar{t}_{train}(g)$ = 该基因在训练集所有品种中的**真实均值**（ref_true_mean）；
- 绝对尺度，无归一化。

**特性**：
- 与 `cross_variety` 的 pairwise 口径一致（把"品种 B"换成训练集预测均值参考）；
- **宽松**：模型在训练集上的系统性偏差被 $\bar{p}_{train}$ 抵消，测试时偏差被"隐藏"；
- 模拟验证（系统性偏差模型：学到 6 成偏差 + 高表达家族高估 30%）→ 0.999（虚高）。

### 2.3 `delta_pcc_true`（ref_true 参考，严格版）

$$\Delta_{pred} = pred - \bar{t}_{train}(g),\quad \Delta_{true} = true - \bar{t}_{train}(g)$$

- 预测值与真值**减去同一个训练集真实均值参考**；
- 绝对尺度，无归一化。

**特性**：
- **严格**：模型系统性偏差无法通过参考抵消，能力不足如实反映；
- 同场景模拟 → 0.649；
- **推荐作为论文主报告的品种特异偏差指标**。

> 注：`delta_pcc_pred` 与 `delta_pcc_true` 的差异源于 ref_pred 与 ref_true 之间的差距（约等于模型的系统性偏差）。二者对照使用：pred 版回答"按 cross_variety 口径的能力"，true 版回答"真实品种特异偏差能力"。

---

## 3. 参考样本构建（`build_ref_sample_from_train`）

仅使用 `split == "train"` 的任务：

1. 取每个训练样本 gene 级 `feature_df` 的 `pred_mean` / `true_mean`，按 `feature_id` 去重（重复时 groupby 均值）；
2. 跨品种取均值：
   - `ref_pred_mean(g)` = 训练集各品种预测均值；
   - `ref_true_mean(g)` = 训练集各品种真实均值。

训练完即可导出固定，部署时直接套用。

---

## 4. 输出位置

`00_main_summary.csv`（`global=sample` + `global=chromosome`，仅 `resolution=gene` 与 `gene-low/medium/high`）：

| 列 | 定义 |
|------|------|
| `delta_pcc` | 原版，样本内归一化 + 逐基因减 ref（见 metrices.md） |
| `delta_pcc_pred` / `delta_spearman_pred` / `delta_rmse_pred` / `sign_accuracy_pred` | ref_pred 参考，宽松版 |
| `delta_pcc_true` / `delta_spearman_true` / `delta_rmse_true` / `sign_accuracy_true` | ref_true 参考，严格版 |

不包含 delta 指标的输出：`00_window_level.csv`、`00_gene_level.csv`。

---

## 5. 与 cross_variety 的区别

`00_cross_variety_delta_summary.csv`（`compute_pairwise_delta_metrics`）：所有样本两两配对（train_train / test_test / train_test）。

| 维度 | cross_variety delta_pearson | delta_pcc_true（严格版） |
|------|------|------|
| Δ_pred | pred_A − pred_B（预测减预测） | pred_test − ref_true（预测减真值参考） |
| Δ_true | true_A − true_B | true_test − ref_true |
| 参考对象 | 另一个具体品种（每对不同） | 训练集均值"平均品种"（固定） |
| 共享偏差 | **抵消 → 宽松** | **不抵消 → 严格** |
| 样本覆盖 | 两两组合，train_test 混入训练拟合值，**不可比** | 仅测试样本 |
| 语义 | 两品种间 DE 预测能力 | 相对平均品种的偏差模式 |

**使用建议**：cross_variety 适合报告"两个具体品种差异预测得多准"（实用），delta_pcc_true 适合报告"模型对品种特异模式的预测能力"（严谨）。cross_variety 若用于论文，建议单独报告 test_test 配对或标注 train_test 不可比，并与 delta_pcc_true 对照。

---

## 6. 数值验证（模拟）

模拟设置：2000 基因，8 训练品种 + 1 测试品种，基础水平 log-normal，品种特异偏差稀疏（30% 基因）。

| 模型行为 | 原版 delta_pcc | delta_pcc_pred | delta_pcc_true |
|---------|:---:|:---:|:---:|
| 完美（预测=真实+5% 噪声） | 0.9527 | ≈0.96 | ≈0.96 |
| 只学整体缩放，无品种特异偏差 | **0.0686** ✅ | **0.4282** ⚠️ | **0.4282** ⚠️ |
| 平均模型（预测=训练平均谱） | 0.0686 ✅ | NaN（Δ_pred 恒 0） | NaN（Δ_pred 恒 0） |
| 学到 6 成偏差 + 高表达家族高估 30% | 0.7940 | **0.9990** ⚠️ 虚高 | **0.6486** ✅ 严格 |

**结论**：
- 归一化本身不虚高（Pearson 对全局线性变换不变），真正影响的是"参考向量"的选择；
- pred 参考会抵消模型系统偏差（虚高），true 参考更诚实；
- 绝对尺度（无归一化）在真实应用中可复现，代价是对"纯缩放模型"不够敏感——建议与原始 pcc/r2 配套解读。

---

## 7. 边界处理

| 条件 | 处理 |
|------|------|
| pred/true 为空、或匹配不到参考值 | 返回 NaN |
| 匹配基因数 < 2 | 返回 NaN |
| ref_pred_mean 含 NaN | 仅 pred 版返回 NaN，true 版正常 |
| 无训练任务 | ref_sample 为空，相关列全 NaN |
| Δ_pred 恒为 0（平均模型） | Pearson 方差为 0，返回 NaN |
