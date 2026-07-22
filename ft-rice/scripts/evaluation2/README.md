# 🧬 Gene Expression Prediction Evaluation v2 — 重构说明

> **完全重构的评估模块**，围绕 **Delta Pearson** 核心指标，提供统一的三层级指标体系。

## 📋 概述

本评估框架对基因表达预测结果进行 **3 层级、8 类 CSV 输出** 的完整评估：

```
输入：*.predictions.csv（逐窗口预测值）
  ↓
Layer 1: Track级（全染色体逐碱基聚合）
  → 01_track_level.csv
  
Layer 2: Segment级（逐窗口聚合）
  → 02_segment_level.csv
  → 03_segment_per_window.csv

Layer 3: Feature级（基因/外显子聚合）
  → 04_feature_level.csv
  → 05_feature_per_gene.csv
  
补充：表达分桶与 Delta Pearson
  → 06_feature_stratified.csv （按表达量分桶）
  → 07_delta_pearson.csv     （跨品种调控变化一致性 ⭐ 论文核心）
  → 08_run_manifest.csv      （运行清单）
```

---

## 🚀 快速开始

### 方式一：命令行（最简单）

```bash
cd /mnt/rice/default/Workspace/yangdong/gene_expression_prediction

# 用法：bash run_evaluate.sh <predict_dir> <ref_dir>
bash scripts/evaluation2/run_evaluate.sh \
  outputs/predict/202607151304 \
  /mnt/rice/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref
```

### 方式二：Python 脚本（更灵活）

```bash
cd /mnt/rice/default/Workspace/yangdong/gene_expression_prediction

python scripts/evaluation2/run_evaluation.py \
  outputs/predict/202607151304 \
  /mnt/rice/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref \
  --output-dir outputs/predict/202607151304/evaluation
```

### 方式三：YAML 配置（推荐用于生产环境）

编辑 `scripts/evaluation2/config.yaml`，然后：

```bash
bash scripts/evaluation2/run_evaluate.sh \
  outputs/predict/202607151304 \
  /mnt/rice/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref
```

脚本自动读取 `config.yaml`。

---

## 📊 输出文件详解

### 01_track_level.csv — 碱基级聚合（每行 = 一个样本 × 染色体 × 组织）

| 列 | 说明 | 公式 |
|---|---|---|
| split | train/valid/test | |
| sample_dir | 样本目录名 | |
| chromosome | 染色体 | |
| biosample | 组织/品种 | |
| modality | 模态（RNA-seq等） | |
| **pearson** | ⭐ 全碱基 Pearson 相关系数 | $\text{corr}(y, \hat{y})$ |
| **r2** | R² 得分 | $1 - SS_{res}/SS_{tot}$ |
| **rmse** | 根均方差 | $\sqrt{\text{MSE}}$ |
| spearman | 秩相关系数 | rank corr |
| log_pearson | Log-transformed Pearson | $\text{corr}(\log y+1, \log\hat{y}+1)$ |
| mae | 平均绝对误差 | |
| zero_auroc | 零/非零判别 AUC | |
| nonzero_pearson | 非零位点 Pearson | 仅 $y>0, \hat{y}>0$ |
| zero_ratio | 零值占比 (%) | |

**用途**：快速判断模型在全染色体上的整体表现。

---

### 02_segment_level.csv — 窗口级聚合

| 列 | 说明 |
|---|---|
| split, sample_dir, chromosome, ... | 元数据 |
| **n_windows** | 总窗口数 |
| **pearson_mean / pearson_median / pearson_std** | 所有窗口 Pearson 的分布 |
| **r2_global** | 全局 R²（所有位点打平） |
| mse_mean, mae_mean | 窗口平均误差 |

**用途**：了解窗口级性能的分布，发现异常窗口。

---

### 03_segment_per_window.csv — 逐窗口明细（保留原有）

每行一个窗口，包含：
- `chromosome, start, end, length` — 坐标
- `pearson_corr, spearman_corr, mse, mae` — 窗口内指标
- `pred_mean, true_mean` — 窗口内均值
- `non_zero_count` — 非零位点数

**用途**：调试特定窗口性能不佳的原因。

---

### 04_feature_level.csv — 特征级汇总（每行 = 样本 × feature_type）

| 列 | 说明 | 优先级 |
|---|---|---|
| split, sample_dir, feature_type | 元数据 | — |
| n_features | 总特征数 | — |
| **feature_mean_pearson** ⭐ | **跨基因均值 Pearson**：所有基因 (pred_mean, true_mean) 对的 Pearson | P0 |
| **feature_mean_spearman** | 同上，Spearman | P1 |
| **feature_mean_r2** | 同上，R² | P1 |
| pearson_mean / pearson_median | 基因内逐碱基 Pearson 的均值/中位数 | P2 |
| mse_mean, mae_mean | 基因内平均误差 | P2 |
| coverage_fraction_mean | 平均覆盖率 | — |

**核心指标** `feature_mean_pearson`：衡量模型能否正确排序不同基因的表达水平。这是论文最重要的指标。

**用途**：
- `feature_mean_pearson` 用于论文主图
- 对比 `pearson` 与 `feature_mean_pearson` 来判断模型是否过度依赖基因本身的表达水平

---

### 05_feature_per_gene.csv — 逐基因明细

每行一个基因/外显子，包含：

| 列 | 说明 |
|---|---|
| feature_id, feature_type, parent_id | 基因标识 |
| chromosome, start, end, strand | 坐标 |
| coverage_fraction, overlap_bp | 覆盖信息 |
| pred_mean, true_mean | 基因区间内平均表达 |
| pearson, spearman, r2, mse | 基因内逐碱基指标 |
| **expression_bucket** | 🆕 low / medium / high |

**表达分桶标准**（基于训练集 true_mean 分布）：
- `low`：true_mean ≤ 33% 分位数
- `medium`：33% < true_mean ≤ 67% 分位数
- `high`：true_mean > 67% 分位数

**用途**：
- 分析模型在高/低表达基因上的表现差异
- 发现特异性高表达/低表达基因

---

### 06_feature_stratified.csv — 分桶聚合（每行 = 样本 × feature_type × bucket）

| 列 | 说明 |
|---|---|
| split, sample_dir, feature_type, expression_bucket | 元数据 |
| n_genes | 该桶内基因数 |
| **feature_mean_pearson** | 该桶内跨基因 Pearson |
| **feature_mean_spearman** | 该桶内跨基因 Spearman |
| pred_mean_avg, true_mean_avg | 桶内平均表达 |
| bucket_low_threshold, bucket_high_threshold | 分桶边界 |

**用途**：
- 对比高/低表达基因上的模型性能
- 判断模型是否有表达量偏差（比如过度预测高表达基因）

---

### 07_delta_pearson.csv — 🆕 核心指标：跨品种调控变化一致性

**这是论文最重要的补充指标。**

| 列 | 说明 |
|---|---|
| split, sample_dir, feature_type | 元数据 |
| delta_normalize | 归一化方式（`global_mean` / `nonzero_mean`） |
| **delta_pcc_zero** | 参照 = 0（基因开关一致性） |
| **delta_pcc_feature_ref** ⭐ | 参照 = 训练集该基因平均值（**论文推荐**） |
| **delta_pcc_global_ref** | 参照 = 当前样本全局均值 |
| delta_rmse | Delta 空间的 RMSE |

#### Delta Pearson 的含义

**普通 Pearson**：基因 A 在品种 X 的表达高、预测也高 → 可能只是因为基因 A 本身在所有品种都表达高（housekeeping gene）。

**Delta Pearson（推荐）**：基因 A 在品种 X 相对于其训练集中的典型表达**上调**，预测也**上调** → 说明模型学到了品种特异的调控变化（enhancer 突变等）。

**公式**：

$$\text{DeltaPearson} = \text{Pearson}\left( \frac{\hat{y}_g}{\mu_{scale}} - \mu_{ref,g}, \frac{y_g}{\mu_{scale}} - \mu_{ref,g} \right)$$

其中：
- $\mu_{scale}$ = `global_mean` 或 `nonzero_mean`（当前 track 的归一化因子）
- $\mu_{ref,g}$ = 基因 g 在所有训练品种中的平均表达（feature_normalized_mean 模式）

**用途**：论文图表中凸显模型捕获品种间差异的能力。

---

### 08_run_manifest.csv — 运行清单

| 列 | 说明 |
|---|---|
| split, sample_dir, ... | 元数据 |
| status | ok / error |
| error | 错误信息（若有） |

---

## ⚙️ 配置文件详解

### config.yaml

```yaml
# 路径
predict_dir: /path/to/predict   # 命令行优先，此处为默认
ref_dir: /path/to/ref
output_dir: /path/to/output     # 默认: <predict_dir>/evaluation

# Feature级
feature_level:
  feature_types: [gene, exon]       # 要评估的特征类型
  min_overlap_bp: 1                 # 最小重叠碱基数
  min_nonzero_bp: 2                 # Pearson 所需最小非零位点

# Delta Pearson
delta:
  normalize: global_mean            # "none" / "global_mean" / "nonzero_mean"
  ref_modes:
    - zero                          # 参照 = 0（基因开关）
    - feature_normalized_mean       # 参照 = 训练集该基因均值（推荐）
    - true_global_mean              # 参照 = 全局均值

# 表达分桶
buckets:
  n_buckets: 3                      # low / medium / high
  # 或手动指定阈值（覆盖自动计算）
  # low_threshold: 0.1
  # high_threshold: 1.0
```

---

## 🔍 工作流程图

```
┌─────────────────────────────────────────┐
│ 扫描 predict_dir 下所有 *_predictions.csv │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
   ┌────────────┐     ┌───────────────┐
   │ Track级    │     │ Segment级     │
   ├────────────┤     ├───────────────┤
   │ 逐碱基拼接 │     │ 逐窗口统计    │
   │ 全指标计算 │     │ 聚合分布      │
   └────┬───────┘     └────────┬──────┘
        │                      │
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ Feature级（如有GFF） │
        ├──────────────────────┤
        │ 1. 特征聚合          │
        │ 2. 表达分桶          │
        │ 3. Delta Pearson    │
        └──────────┬───────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │ 写入 8 个 CSV 文件    │
        └──────────────────────┘
```

---

## 🎯 指标优先级与论文用途

| 优先级 | 指标 | 出现位置 | 用途 |
|:--|:--|:--|:--|
| **P0** | `feature_mean_pearson` | 04, 06 | **论文主图**：基因表达排序能力 |
| **P0** | `delta_pcc_feature_ref` | 07 | **论文主图**：品种间调控变化预测能力 |
| **P1** | `pearson` (track级) | 01 | 论文补充：全染色体评估 |
| **P1** | `r2` (各层级) | 01, 04 | 论文补充：方差解释 |
| **P2** | `zero_auroc` | 01 | 补充：基因开关判别能力 |
| **P3** | `mse, mae` | 各层级 | 诊断：误差分析 |
| **P3** | `coverage_fraction` | 05, 06 | 诊断：特征覆盖率 |

---

## 📝 常见用途

### 用途1：快速评估新模型

```bash
# 只评估 track 和 segment 层级（无需 GFF），更快
bash run_evaluate.sh predict_dir ref_dir --skip-features

# 输出时间：~2 分钟（100+ CSVs）
```

### 用途2：完整评估（用于论文）

```bash
bash run_evaluate.sh predict_dir ref_dir

# 包含基因级指标 + Delta Pearson
# 输出时间：~15 分钟（100+ CSVs）
```

### 用途3：调试特定样本

```bash
python run_evaluation.py predict_dir ref_dir --max-csv 1

# 仅处理 1 个 CSV，逐步检查

# 查看 03_segment_per_window.csv 找异常窗口
# 查看 05_feature_per_gene.csv 找异常基因
```

### 用途4：自定义分析

```python
import pandas as pd
from scripts.evaluation2.metrics_core import (
    compute_track_metrics,
    compute_feature_mean_correlation,
)
from scripts.evaluation2.utils import load_prediction_csv

# 读取单个 CSV
df = load_prediction_csv("path/to/pred.csv")

# 计算自定义指标
my_metrics = compute_track_metrics(df["parsed_pred"], df["parsed_true"])
```

---

## 🐛 故障排查

### Q: GFF 文件找不到？

**A**: 脚本自动根据 `sample_id` 搜索 GFF 文件。如果搜索失败：
1. 检查 GFF 路径 (`ref_dir`)
2. 检查命名是否匹配 （通常 `P{N}*.gff3` 或 `{sample_id}_*.gff3`）
3. 手动指定 GFF 路径（暂不支持，需修改 `find_annotation_file()` 函数）

```python
# 查看 find_annotation_file() 函数的搜索策略
less scripts/evaluation2/utils.py | grep -A 20 "def find_annotation_file"
```

### Q: 速度很慢？

**A**:
1. 用 `--max-csv 10` 测试（只处理 10 个 CSV）
2. 用 `--skip-features` 跳过基因级评估
3. 并行运行（按 sample_id 或 chromosome 分割输入）

### Q: 输出 CSV 为空？

**A**: 可能原因：
1. 输入 CSV 格式不对（检查 `parsed_pred` 和 `parsed_true` 列的 JSON 格式）
2. GFF 特征与预测区间无重叠（检查坐标系是否一致，0-based vs 1-based）
3. 所有特征都因为 `min_overlap_bp` 被过滤（降低阈值）

---

## 📚 文件结构

```
scripts/evaluation2/
├── metrics_core.py        # 核心指标计算（纯数学，无IO）
├── utils.py               # 工具函数（CSV/GFF解析、特征聚合）
├── run_evaluation.py      # 主程序（5步流程）
├── run_evaluate.sh        # bash 入口脚本
├── config.yaml            # 配置文件
└── README.md              # 本文件
```

### 代码分层设计

- **metrics_core.py**：通用指标函数，不依赖文件系统，可独立单元测试
- **utils.py**：CSV/GFF I/O 和数据处理，提供统一接口
- **run_evaluation.py**：5 步评估流程：扫描 → 单CSV评估 → 聚合 → 输出

---

## 🤝 集成建议

### 与现有 run_evaluate.sh 的关系

| 脚本 | 位置 | 用途 |
|:--|:--|:--|
| **旧** `run_evaluate.sh` | `scripts/evaluation/` | 原始评估脚本（保留作为备份） |
| **新** `run_evaluate.sh` | `scripts/evaluation2/` | 📍 推荐使用 |

逐步迁移计划：
1. v1 版保留，v2 新增
2. 在项目 README 中指引新项目使用 v2
3. 6 个月后可废弃 v1

### 与论文图表生成的集成

推荐创建 `scripts/paper_figures.py`：

```python
import pandas as pd

# 读取评估结果
df_delta = pd.read_csv("outputs/evaluation/07_delta_pearson.csv")
df_strat = pd.read_csv("outputs/evaluation/06_feature_stratified.csv")

# 图1：Delta Pearson 散点图（按 split 颜色）
# 图2：按表达水平分桶的 Pearson 对比
# 图3：高/低表达基因上的 RMSE 差异
```

---

## 📖 引用与相关工作

本评估框架的设计基于：
- **OneGenome-Rice 论文**的指标定义（Track/Segment/Feature 三层级）
- **Delta Pearson** 概念源自 DeepCIS 等品种比较项目
- **表达分桶** 策略用于识别表达量特异的预测偏差

---

## 📧 反馈与改进

如有问题或改进建议，请提 Issue 或 PR。

---

**版本**: v2.0（2026-07-16）  
**状态**: 生产就绪 ✅  
**维护者**: YangDong Team  
