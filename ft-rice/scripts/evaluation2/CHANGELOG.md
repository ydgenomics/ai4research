# CHANGELOG
## 2026-07-21 (4)

### Fixed: 04_feature_level.csv 和 00_feature_summary.csv 缺少 delta_pcc

**问题**：之前 delta_pcc 只追加到了分箱 CSV（06/00_stratified），
但 04_feature_level.csv 和 00_feature_summary.csv 中的 gene 行没有 delta_pcc。

**修改文件**：

- **`run_evaluation.py` — `build_feature_summary_df`**
  - 新增参数 `ref_table`, `normalize`
  - 对 gene 类型调用 `compute_all_delta_pearson` 计算 delta_pcc
  - 追加 `delta_pcc_zero`, `delta_pcc_feature_ref`, `delta_pcc_global_ref`, `delta_rmse` 列

- **`run_evaluation.py` — `write_one_task`** 节 04
  - 同样对 gene 类型计算并追加 delta_pcc 列

- **`run_evaluation.py` — `write_top_level_summaries`**
  - 调用 `build_feature_summary_df` 时传入 `ref_table` 和 `normalize`

**效果**：`04_feature_level.csv` 和 `00_feature_summary.csv` 中 gene 行包含 delta_pcc 四列

## 2026-07-21 (3)

### Changed: 基因表达改为只计算各 exon 区域，排除 intron 零值稀释

**修改背景**：RNA-seq 基因表达应只反映外显子区域。之前 gene 特征取
`start0~end0` 全长区间（含 intron），大量 intron 零值拉低基因表达均值，
且不同基因 intron 比例不同造成稀释不均匀。

**修改文件**：

- **`utils.py` — `load_features_from_gff`**
  - 若 `feature_types` 同时包含 `"gene"` 和 `"exon"`，则在加载后对 gene 特征
    做 post-processing
  - 收集所有 exon 区间，按 `feature_id` 分组
  - 将 gene 全长区间替换为该基因所有 exon 区间的**并集**
  - 同一 gene_id 的多个 exon 片段在 `aggregate_predictions_to_features` 中
    自动合并为一条记录，均值只计算外显子碱基

**效果**：gene 水平的 `true_mean/pred_mean` 不再被 intron 零值稀释，
与 RNA-seq TPM/FPKM 定量方式一致，更准确反映真实表达水平。

## 2026-07-21 (2)

### Changed: delta_pcc 仅保留 gene 级别，并入分箱 CSV，取消独立 CSV

**修改背景**：
- exon 级别 delta_pcc 在 liftover GFF 中 ID 无法跨品种对齐（exon 无独立 ID，
  fallback 到 gene_id 后同基因多个外显子被合并），不具实际意义
- 分箱后的各表达水平基因（low/medium/high）需要计算 delta_pcc，以反映
  模型在不同表达水平下的品种偏差预测能力
- 简化输出结构：不再单独输出 07_delta_pearson.csv 和 00_delta_pearson_summary.csv，
  delta_pcc 直接追加到 06_feature_stratified.csv 和 00_stratified_summary.csv

**修改文件**：

1. **`metrics_core.py` — `compute_all_delta_pearson`**
   - 函数入口过滤 `valid = valid[valid["feature_type"] == "gene"]`，仅计算 gene 级别
   - 输出 dict 只含 `gene_*` 前缀的 key

2. **`metrics_core.py` — `compute_stratified_feature_metrics`**
   - 新增参数 `ref_table`, `biosample`, `normalize`
   - 对每个分桶（low/medium/high），若为 gene 类型则计算 delta_pcc
   - 在每行输出末尾追加 `delta_pcc_zero`, `delta_pcc_feature_ref`,
     `delta_pcc_global_ref`, `delta_rmse` 列

3. **`run_evaluation.py` — `build_delta_pearson_df`**
   - 函数已删除（不再需要独立的 delta 汇总表）

4. **`run_evaluation.py` — `write_one_task`**
   - 节 06 调用 `compute_stratified_feature_metrics` 时传入 `ref_table` 等参数
   - 节 07 已删除（不再输出 07_delta_pearson.csv）
   - 输出横幅改为 `(01~06)`

5. **`run_evaluation.py` — `build_feature_stratified_df`**
   - 新增参数 `ref_table`, `normalize`
   - 调用 `compute_stratified_feature_metrics` 时传入 ref_table

6. **`run_evaluation.py` — `write_top_level_summaries`**
   - 在 00d 之前构建 ref_table 并传入 `build_feature_stratified_df`
   - 删除原有的 00e (00_delta_pearson_summary.csv) 输出

**效果**：
- 输出目录不再包含 `07_delta_pearson.csv`
- `06_feature_stratified.csv` 中 gene 分桶行新增 delta_pcc 四列
- `00_stratified_summary.csv` 同样包含 delta_pcc 列

## 2026-07-21

### Fixed: biosample 混用导致 delta_pcc_feature_ref 在多组织场景下混乱

**问题**：CSQ-YG 双组织训练配置下，`build_delta_pearson_df` 和 `_build_ref_table` 用
`species/chrom_unit` 作为 key 收集训练集 feature_df。同一 species/chrom 有 CSQ 和 YG
两个 biosample，后到的覆盖前面的，导致 ref_table 只包含一种组织数据。
CSQ 样本计算 delta_pcc 时减的是 YG 的 ref，Δ 中混杂组织差异而非品种特异偏差。

**修改文件**：

1. **`run_evaluation.py` — `build_delta_pearson_df`**
   - key 从 `f"{species}/{chrom_unit}"` 改为 `f"{species}/{chrom_unit}/{biosample}"`
   - 调用 `compute_all_delta_pearson` 时传入 `biosample` 参数

2. **`run_evaluation.py` — `_build_ref_table`**
   - key 同样加入 biosample，与 build_delta_pearson_df 保持一致

3. **`run_evaluation.py` — `write_one_task`**
   - 调用 `compute_all_delta_pearson` 时传入 `ctx.get("biosample", "")`

4. **`metrics_core.py` — `build_per_gene_reference`**
   - 从 `sample_key`（格式: `species/chrom_unit/biosample`）中提取 biosample
   - groupby 增加 `biosample` 维度，ref_table 列变更为: `feature_id, feature_type, biosample, delta_ref_value`

5. **`metrics_core.py` — `compute_all_delta_pearson`**
   - 新增 `biosample` 参数
   - merge ref_table 前按 biosample 过滤，确保 CSQ 样本用 CSQ 的 ref，YG 样本用 YG 的 ref

**效果**：多组织场景下，delta_pcc_feature_ref 的 ref 按组织分别计算，Δ 衡量的才是
"同组织内品种特异的偏差模式"，不再受组织表达差异干扰。

## 2026-07-17

### Changed: 输出目录结构重构 — 按 task 独立子目录 + 顶层汇总

**问题**：所有 task 的 01~08 CSV 混写在一个目录下，多次运行互相覆盖，且无法区分数据来源。

**修改**：

- 每个 task 写入独立子目录，命名规则 `{split}-{species}-{tissue}-{chromosome}/`
  - 包含 01~07 全部指标（track / segment / per-window / feature / per-gene / stratified / delta）
- 顶层新增跨任务汇总表（`00_*_summary.csv`），一行一个 task，方便横向对比：
  - `00_track_summary.csv`
  - `00_segment_summary.csv`
  - `00_feature_summary.csv`
  - `00_stratified_summary.csv`
  - `00_delta_pearson_summary.csv`
- `08_run_manifest.csv` 保留在顶层
- 执行流程：全部评估 → 计算全局阈值/参考表 → 逐 task 写入子目录 → 写顶层汇总
- 分桶阈值（06）和 Delta Pearson 参考表（07）统一从训练集计算，各 task 结果独立写入各自子目录

### Fixed: `06_feature_stratified.csv` 按 feature_type 分箱，修复 `n_genes` 统计错误

**问题**：分层统计对所有特征行（exon + gene）统一分箱，导致 `n_genes` 字段实际统计的是特征总数（如 6233），而非基因数（如 1095）。exon 和 gene 的 `true_mean` 量级差异造成桶分布失真。

**修改文件**：

1. **`metrics_core.py` — `compute_stratified_feature_metrics`**
   - 新增 `feature_type` 参数，作为列写入输出
   - 列名 `n_genes` → `number`（该桶内的特征数量）

2. **`run_evaluation.py` — `build_feature_stratified_df`**（顶层 `00_stratified_summary.csv`）
   - 分桶阈值仅用 **gene** 行的 `true_mean` 计算（修复混入 exon 导致的阈值偏移）
   - 按 `feature_type`（exon / gene）分别分组，各自分箱后合并输出

3. **`run_evaluation.py` — 子目录 `06_feature_stratified.csv`**
   - 同样改为按 `feature_type` 分组后 `concat` 输出

### Fixed: `07_delta_pearson.csv` 列名映射错误 + `delta_pcc_feature_ref` 对 valid/test 为 NaN

**Bug 1 — 列名全部映射为 `delta_delta`**：
`k.split('_')[:2]` 对 `_delta_pcc_feature_ref` 得到 `['', 'delta']`，所有 key 都映射到同一列名，最终 CSV 只保存了最后一个值（`delta_rmse`）。
修复：`k.lstrip('_')`，如 `_delta_pcc_feature_ref` → `delta_pcc_feature_ref`。

**Bug 2 — `delta_pcc_feature_ref` 全 NaN**：
`compute_pearson_delta_metrics` 中若 ref 数组包含 NaN（因品种间 feature_id 不匹配），整列返回 NaN。修复：merge 后过滤 NaN 行再计算。

### Added: 跨品种差异表达指标 → `00_cross_variety_delta_summary.csv`

**背景**：审稿人要求评估"linking sequence variation to transcriptional output"，即模型对品种间差异表达的预测精度。

**实现**：

- `metrics_core.py` — 新增 `compute_pairwise_delta_metrics`：
  对两个品种匹配的同源基因，计算 $\Delta = \text{expr}_A - \text{expr}_B$，输出：
  - `delta_pearson` / `delta_spearman`：Δ 预测值的幅度相关性
  - `log2fc_pearson` / `log2fc_spearman`：log2 fold change 相关性（差异表达分析标准）
  - `sign_accuracy`：Δ 方向一致性比例
  - `up_precision` / `down_precision`：预测上调/下调的精确率
  - `delta_rmse`：Δ RMSE

- `run_evaluation.py` — 新增 `build_cross_variety_delta_df`：
  - 按 (tissue, chromosome) 分组
  - 对组内所有品种对（$\binom{5}{2}=10$ 对）计算指标
  - 分类 `pair_type`：`train_train`（训练品种间） vs `train_test`（训练 vs 测试品种）

- 顶层输出 `00_cross_variety_delta_summary.csv`
## 2026-07-16

### Added
- 品种配置下沉到每个 task：`species`, `tissue`, `gff` 字段直接在 tasks 列表中指定
- Segment/Feature 级添加 `log_pearson` (log1p PCC) 指标
- `CHANGELOG.md`

### Changed
- 配置驱动：`tasks` 列表显式指定每个评估任务的所有文件路径
- `run_evaluate.sh` 简化为 `--config` + `-o` 模式
- `EvalTask` 数据类替代 `VarietyRefConfig` / `parse_csv_context` / `discover_prediction_csvs`

### Removed
- 目录扫描逻辑 (`discover_prediction_csvs`, `parse_csv_context`)
- 正则解析品��名 (`sample_id_from_dir`, `variety_from_dir`)
- GFF 搜索逻辑 (`find_annotation_file`)
- `VarietyRefConfig` 数据类
- 冗余文档: CHECKLIST.md, DESIGN.md, INDEX.md, SUMMARY.txt, QUICKSTART.md
