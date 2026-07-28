# Changelog

## [1.1.1] - 2026-07-27

### 修复
- **跨品种差异表达崩溃**：`build_cross_variety_delta` 中 `gene_df` 因 `feature_id` 重复索引（同一基因在多行出现），导致 `df_a.loc[common_ids]` 与 `df_b.loc[common_ids]` 返回不同行数，`pred_a - pred_b` 广播失败（shape `(176815,)` vs `(178442,)`）。
  - 修复：构建 `species_dfs` 时检测重复索引，通过 `groupby(level=0).mean(numeric_only=True)` 聚合并去重，确保每个基因每品种唯一一行。

### 变更文件
- `run_evaluation.py`: 在 `build_cross_variety_delta` 中 duplicated index 检测 + groupby 去重逻辑。

## [1.1.0] - 2026-07-24

### 优化
- **大幅降低峰值内存**：在 `load_and_merge_csvs` 中解析 expression 列后立即释放原始字符串列 `sequence`、`true_expression`、`predicted_expression`。
  - 原问题：`pd.read_csv()` 一次性加载 ~23GB CSV，原始字符串列（每行 ~800KB）在解析后仍驻留内存，导致峰值达 ~60-90 GB。
  - 优化后：解析后立即 drop，峰值内存降至 ~10-15 GB。
  - 安全性：已确认这三列在解析后（`parsed_pred`/`parsed_true`）再无任何引用。

### 变更文件
- `run_evaluation.py`: 在 `load_and_merge_csvs` 函数中新增 `df.drop(columns=[...], inplace=True, errors="ignore")`。

## [1.0.0] - 初始版本
