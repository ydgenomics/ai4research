# README-YD（碱基分辨率只保留基因区域）

## 需求
计算碱基分辨率指标的时候，只保留基因区域的值。
例如：某染色体原本有 2 万个碱基元素，根据基因组注释中基因的 start-end，最终保留的元素只有 5 千。
> 注意：`2 万` / `5 千` 只是**举例说明**，不是写死的数字，也没有长度截断。
> 实际保留的元素数量完全取决于基因组注释中基因区域占覆盖范围的比例。

## 实现（已完成，2026-07-31）
- 保留原有 `bp` 指标不变，**新增 `resolution=bp_gene`** 行。
- 基因区域定义为 GFF 中 `gene` 特征的**整基因跨度 start-end（含内含子）**，重叠/相邻区间合并为并集。
- `bp_gene` 行在 `global=sample` 与 `global=chromosome` 两个层面均输出（主表 `00_main_summary.csv`）。
- 输出示例（`2 万 / 5 千` 仅为示意比例）：

| global | chromosome | resolution | 说明 |
|--------|-----------|------------|------|
| sample | all | bp | 原有：全基因组碱基（示意：2 万个元素） |
| sample | all | bp_gene | 新增：仅基因区域碱基（示意：约 5 千） |
| chromosome | Chr01 | bp | 原有：单染色体碱基 |
| chromosome | Chr01 | bp_gene | 新增：单染色体基因区域碱基 |

## 关键代码
- `load_gene_regions_from_gff`：读取 gene 整基因跨度，合并重叠区间。
- `flatten_to_genome_gene`：逐碱基累加后只保留基因区域碱基。
- `build_main_summary`：新增 `bp_gene` 行；辅助函数 `_concat_gene_track`。