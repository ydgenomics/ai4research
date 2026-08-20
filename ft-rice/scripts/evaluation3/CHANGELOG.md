# Changelog

## [1.5.0] - 2026-08-20

### 修复
- **基因水平指标（gene/exon/分桶）正负链不区分**：plus 和 minus 链的 feature 级指标完全相同，因为计算时合并了所有链的窗口再做聚合，然后只是机械地给结果打上不同链标签。影响范围包括主表 `00_main_summary.csv` 的 `gene`/`exon`/`gene-low/medium/high` 行，以及 `build_gene_table`、`build_cross_variety_delta`、`_build_ref_table` 等下游函数。
  - 根因：`_evaluate_task` 中对聚合后的全量 df（plus+minus 混合）只调用了一次 `aggregate_to_features`，导致基因表达量混入了反义链窗口的信号。
  - 修复 1（`_evaluate_task`）：按链分别过滤 df 后调用 `aggregate_to_features`，结果存入 `results["feature_df_per_strand"]`（字典，key=strand）。
  - 修复 2（`build_main_summary`）：global=sample 和 global=chromosome 的 feature 级行改为从 `feature_df_per_strand[strand_val]` 取数，而非从混合的 `feature_df` 取数。
  - 修复 3（`aggregate_to_features`）：在内层窗口循环中加入按链匹配过滤（`feat.strand != "total" and win_strands[i] != feat.strand` 时跳过），防御性编程确保即使调用方传入混合链数据也能正确按链聚合。
  - 兼容性：`results["feature_df"]` 保留第一条链的结果，确保 `build_gene_table`、`build_cross_variety_delta`、`_build_ref_table` 等下游函数不受影响（仍使用旧接口，但输出需注意已是单链结果）。

### 变更文件
- `run_evaluation.py`:
  - `_evaluate_task`: 新增 `feature_df_per_strand` 字典，按链分别聚合 feature。
  - `build_main_summary`: 改为从 `feature_df_per_strand` 取数，使 feature 级指标按链独立计算。
  - `aggregate_to_features`: 内层循环加入按链过滤（`total` 时不过滤），确保窗口与基因链匹配。

## [1.4.0] - 2026-08-18

### 新增
- **新增均值参考样本 delta 指标（ref_sample 模式）**：从训练集构建"平均品种"参考样本，评估测试样本相对参考的品种特异偏差模式预测能力。
  - 参考样本 = 训练集所有品种 gene 级 `pred_mean` / `true_mean` 的跨品种均值（`build_ref_sample_from_train`），训练完即可固定，部署时直接套用，**不依赖任何事后统计量**（无需样本均值 scale），更贴近真实应用。
  - 两种参考模式（替换原 `*_ref` 系列列）：
    - `delta_pcc_pred` 及 `delta_spearman_pred` / `delta_rmse_pred` / `sign_accuracy_pred`：Δ_pred = pred − ref_pred_mean，与 cross_variety pairwise 同口径，**宽松版**（模型系统偏差被参考抵消）。
    - `delta_pcc_true` 及 `delta_spearman_true` / `delta_rmse_true` / `sign_accuracy_true`：Δ_pred / Δ_true 都减去同一个 ref_true_mean，**严格版**（模型系统偏差如实反映）。
  - 输出至 `00_main_summary.csv` 的 `resolution=gene` 与 `gene-low/medium/high` 行（`global=sample` + `global=chromosome`）。
  - 保留原 `delta_pcc`（feature_ref 归一化版）不变，三套可对照。
- **新增文档 `delta_pcc.md`**：完整说明背景动机、三个指标定义、参考样本构建、与 cross_variety 的区别、数值模拟验证与边界处理。

### 变更文件
- `run_evaluation.py`:
  - 新增 `build_ref_sample_from_train`（从训练集构建均值参考样本，含 `ref_pred_mean` / `ref_true_mean`）。
  - 新增 `compute_ref_sample_delta_metrics`（两种参考模式，复用 pairwise 口径）与 `compute_ref_sample_delta_from_df`（按 feature_id 匹配参考样本）。
  - `compute_stratified_metrics` / `build_main_summary` 接入 ref_sample 参数，输出 pred/true 两组共 8 列，替换原 `*_ref` 4 列。
  - 修复：`compute_ref_sample_delta_metrics` 返回 dict 始终含全部键（ref_pred 含 NaN 时 pred 版为 NaN、true 版正常）。
- 新增 `delta_pcc.md`。

## [1.3.0] - 2026-07-31

### 新增
- **碱基分辨率新增 `bp_gene` 指标**：计算碱基分辨率指标时只保留基因区域内的碱基。
  - 基因区域定义为 GFF 中 `gene` 特征的整基因跨度 start-end（含内含子），重叠/相邻区间合并为并集（`load_gene_regions_from_gff`）。
  - 例如某染色体原本有 2 万个碱基元素，按基因注释过滤后仅保留约 5 千。
  - 现有 `bp` 指标**保留不变**，另新增 `resolution=bp_gene` 行（`global=sample` 和 `global=chromosome` 两个层面均输出）。
  - `n_positions` 反映过滤后的碱基数（主表最终只保留固定列，未直接输出，但通过 pcc/zero_ratio 等体现）。

### 变更文件
- `run_evaluation.py`:
  - 新增 `load_gene_regions_from_gff`（读取 gene 整基因跨度，合并重叠区间）。
  - 新增 `flatten_to_genome_gene`（逐碱基累加后只保留基因区域碱基，供 bp_gene 使用）。
  - 重构 `flatten_to_genome_array`，抽出公共累加逻辑 `_accumulate_flatten`。
  - `evaluate_one_task` 新增 `gene_regions_cache` 参数与 `pred_gene`/`true_gene` per-strand 缓存。
  - `build_main_summary` 新增 `bp_gene` 行（global=sample + global=chromosome），并新增拼接辅助 `_concat_gene_track`。

## [1.2.0] - 2026-07-28

### 修复
- **跨品种差异表达表只有少量行**：`build_cross_variety_delta` 按 `biosample` 分组后再配对，导致只产生组内品种对，缺失跨 biosample 的对比。
  - 例如 YP 配置（2 biosample × 2 sample）只输出 2 行；SAM2 配置（1 biosample × 5 task）只输出 3 行。
  - 修复：取消分组逻辑，将每个 (sample, biosample, split) 组合视为独立实体，用 `itertools.combinations` 生成所有两两对比。
  - 输出列 `biosample` 拆分为 `biosample_a` / `biosample_b`，以区分两个个体的 biosample。

### 变更文件
- `run_evaluation.py`: 重写 `build_cross_variety_delta` 函数，取消 biosample 分组。

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
