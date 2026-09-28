# eutilies.py — 玉米公开组学数据收集脚本

使用 NCBI E-utilities（`Bio.Entrez`）自动检索并抓取玉米（*Zea mays*, txid4577）的所有公开组学数据元数据，产出**多张互补的表格**（Run 级 / 实验级 / 项目级 / 文献级），便于按课题、品种、组织、处理条件做二次筛选。

> 数据均来自 NCBI SRA / BioProject / PubMed 公开元数据，不下载原始测序文件。

---

## 1. 依赖与配置

```bash
pip install biopython pandas numpy
```

脚本顶部配置区（修改后直接运行）：

```python
Entrez.email = "你的NCBI注册邮箱"                       # NCBI 强制要求
Entrez.api_key = os.environ.get("NCBI_API_KEY") or "你的API Key"
# 留 None 则匿名限速（3次/秒）；带 Key 可提速到 10次/秒
```

- `NCBI_API_KEY` 环境变量可覆盖内置 Key；**不要设为空字符串**，否则 NCBI 返回 HTTP 400。
- 检索式在 `DB_QUERY` 字典中定义：`txid4577[Organism]`（可换成其他物种 TaxID）。

---

## 2. 命令用法

```bash
# 全量抓取（SRA Run级 + BioProject 项目级），无参数时默认执行这两步
python3 eutilies.py -o 输出目录

# 只跑 SRA Run 级表（含过滤表）
python3 eutilies.py --sra -o 输出目录

# 只跑 BioProject 项目级表
python3 eutilies.py --bioproject -o 输出目录

# ★ 实验级摘要表（esummary XML）：三层标题/提交机构/样品属性/实验级统计
python3 eutilies.py --experiment -o 输出目录
# 自定义检索式（复现官网"实验级汇总"导出，如 WGS 清单）：
python3 eutilies.py --experiment --experiment-query "txid4577[Organism] AND wgs[Strategy]" -o 输出目录

# 为 SRA 表里的 Study_Pubmed_id 抓取文献题录（需先跑过 --sra）
python3 eutilies.py --pubmed -o 输出目录
```

| 参数 | 作用 |
|---|---|
| `-o, --outdir` | 输出目录（默认当前目录） |
| `--sra` | 只跑 SRA Run 级步骤 |
| `--bioproject` | 只跑 BioProject 项目级步骤 |
| `--experiment` | ★ 抓取实验级摘要（独立运行，不连带其他步骤） |
| `--experiment-query` | 实验级摘要检索式；默认同 SRA 全量检索式 |
| `--pubmed` | 抓取关联文献题录 |

> 显式指定任一开关时只跑对应步骤；一个都不指定时默认跑 `--sra` + `--bioproject`。

### 断点续传

- 每张表**边拉边存**，中断后重跑同一命令会自动从断点继续（按主键去重）。
- 实验级表还会有 `.csv.done` 哨兵文件：存在即视为已拉完，直接读取。
- 旧格式（无 `Omics_Type`/`Project_Omics` 列）的表会被自动备份改名为 `*_old_时间戳.csv` 后全新抓取。

---

## 3. 产出表格一览

| 文件 | 粒度 | 说明 |
|---|---|---|
| `maize_sra_omics_dataset.csv` | **Run 级** | 全量 SRA 元数据（含分类/多组学/热词标注），每行一个 Run（SRR） |
| `maize_sra_omics_filtered.csv` | **Run 级** | 精简过滤版：Genome 只留 WGS、Transcriptome 只留 RNA-Seq，染色质可及性/表观组/其他整类保留 |
| `maize_sra_experiments.csv` | **实验级** | ★ esummary 实验摘要：三层标题、提交机构、样品属性、实验级统计（对应 NCBI "实验级汇总"导出，信息更全） |
| `maize_bioproject_summary.csv` | **项目级** | BioProject 项目描述 + 各项目组学类别聚合 |
| `maize_pubmed_literature.csv` | **文献级** | 与数据关联的 PubMed 题录（需 `--pubmed`） |

> 三级粒度关联键：**Run 级**（`Experiment`/`Sample` 列）↔ **实验级**（`Experiment`）↔ **项目级**（`BioProject`/`Study`）。
> 例如用 `Experiment` 将实验级的 `Study_Title`/`Submitter` join 回 Run 级表，即可得到带语义标注的增强表。

---

## 4. 各表列字段详解

### 4.1 `maize_sra_omics_dataset.csv` / `maize_sra_omics_filtered.csv`（Run 级）

| 列名 | 含义 | 备注 |
|---|---|---|
| `Run` | SRA Run 编号（主键，SRRxxxxxxxx） | 一行一个 Run |
| `BioProject` | 项目编号（PRJNAxxxxxx） | 关联项目级表 |
| `SRAStudy` | 研究编号（SRP/ERP） | 同一研究可含多项目 |
| `Experiment` | 实验编号（SRX） | 关联实验级表 |
| `Sample` | 样品编号（SRS） | |
| `BioSample` | BioSample 编号（SAMN/ERS） | |
| `SampleName` | 提交者给样品的名称 | 常含品种/处理信息 |
| `LibraryStrategy` | 测序策略（WGS/RNA-Seq/ChIP-Seq…） | 分类依据 |
| `LibrarySelection` | 文库筛选（RANDOM/PCR/…） | |
| `LibrarySource` | 文库来源（GENOMIC/TRANSCRIPTOMIC） | |
| `LibraryLayout` | 单/双端（SINGLE/PAIRED） | |
| `Platform` / `Model` | 测序平台与仪器型号 | |
| `TaxID` / `ScientificName` | 物种 | |
| `ReleaseDate` | 公开日期 | |
| `spots` / `bases` / `size_MB` | 数据量：reads 数 / 碱基数 / 文件大小 | 下载前评估规模 |
| `avgLength` | 平均读长 | |
| `download_path` | SRA 下载地址（sra 文件） | |
| `Study_Pubmed_id` | 关联 PMID | 供 `--pubmed` 使用 |
| `Consent` | 访问控制（public/controlled） | |
| `CenterName` | 测序中心 | |
| `Omics_Type` | ★ 主类别（脚本按策略打标） | `Genome`/`Transcriptome`/`Epigenome`/`Chromatin_Accessibility`/`Other`，可多值`;`分隔 |
| `Assay_Subtype` | 类内子类（scRNA/snRNA/WGBS/ChIP/Hi-C…） | 关键词刮取，只标记不筛选 |
| `Multiomics_Info` | 多组学标注 | `Project:类别集合` / `Sample:类别集合` / `Keyword:multiome` |
| `Omics_Category` | 旧启发式标签（兼容早期 EDA） | 新分析请用 `Omics_Type` |
| `Context_Tags` | 热词记录（处理/组织等） | 扫描 SampleName/LibraryName，如 `drought;leaf;salt` |

`maize_sra_omics_filtered.csv` 与 dataset 同构，仅行数被 `filter_core_omics()` 精简：
**Genome 只留 `WGS`、Transcriptome 只留 `RNA-Seq`，Chromatin_Accessibility / Epigenome / Other 整类保留**，其余策略剔除。

### 4.2 `maize_sra_experiments.csv`（实验级，一行一个实验 SRX）

| 列名 | 含义 | 相对官网导出的增强 |
|---|---|---|
| `Experiment` | 实验编号（主键，SRX） | |
| `Experiment_Title` | 实验标题 | 官网表有 |
| `Study` | 研究编号（SRP） | |
| `Study_Title` | 研究标题 | 官网表有 |
| `Sample` | 样品编号（SRS） | |
| `Sample_Title` | **样品标题** | 🆕 官网导出常为空，此表可补全 |
| `Organism` | 物种 | |
| `Instrument` | 测序仪器型号 | 官网表有 |
| `Submitter` | 提交机构 | 官网表有 |
| `LibraryName` | 文库名 | 常含品种/处理命名 |
| `LibraryStrategy/Source/Selection/Layout` | 文库四参数 | Layout 为新增 |
| `Total_RUNs` | 该实验含的 Run 数 | 官网表有 |
| `Total_Spots` / `Total_Bases` / `Total_Size_MB` | 实验级聚合数据量 | 官网表有 |
| `Study_Abstract` | 研究摘要 | 🆕 |
| `Attr_isolate` / `Attr_cultivar` / `Attr_tissue` / `Attr_genotype` / `Attr_dev_stage` / `Attr_ecotype` | BioSample 常见属性 | 🆕 已过滤 "not applicable" |
| `Sample_Attributes` | 其余样品属性（`tag=value; …`） | 🆕 |

> 使用技巧：把实验级 join 回 Run 级表（`Experiment` 列），即可让 Run 级表带上标题/机构/样品属性。

### 4.3 `maize_bioproject_summary.csv`（项目级，一行一个 BioProject）

| 列名 | 含义 |
|---|---|
| `BioProject` | 项目编号（PRJNA） |
| `Title` / `Description` | 项目标题 / 描述（可从标题快速识别课题） |
| `ReleaseDate` | 公开日期 |
| `DataType` | 数据类型（如 `raw sequence reads`） |
| `CenterName` | 提交机构 |
| `Relevance` | 是否标记为 Agricultural |
| `GEO` | 关联的 GEO 编号（GSE…） |
| `SubmissionID` | 提交批次号 |
| `Project_Omics` | ★ 项目内所有 `Omics_Type` 类别集合（`;`分隔） |
| `Multiomics` | 是否多组学项目（yes/no） |

### 4.4 `maize_pubmed_literature.csv`（文献级）

| 列名 | 含义 |
|---|---|
| `PMID` | 文献 ID |
| `BioProjects` | 关联的项目编号 |
| `PubTitle` / `Journal` / `PubDate` | 标题 / 期刊 / 发表日期 |

> 多数 SRA 提交未填 PMID，`Study_Pubmed_id` 可能大量为空，属正常现象。

---

## 5. 常见问题

**Q1：跑 `--experiment` 会连带下载全量 runinfo 吗？**
不会。显式指定任一开关后只运行对应步骤（`--experiment` 只抓实验级摘要）。

**Q2：为什么 `maize_sra_experiments.csv` 行数比 runinfo 表少？**
runinfo 是 Run 级（一个 Run 一行），实验级一个实验可能对应多个 Run（`Total_RUNs`>1），所以实验数 ≤ Run 数。抓取循环会安全处理"越过最后一个实验包"的情况。

**Q3：想看某类的数据，从哪个文件开始？**
- 想知道"有多少 WGS / RNA-Seq / ChIP-seq" → `maize_sra_omics_filtered.csv` 的 `Omics_Type`/`Assay_Subtype` 列
- 想按课题/品种/机构检索 → 实验级表的 `Study_Title`/`Experiment_Title`/`Submitter`，或项目级表 `Title`
- 想下载数据 → Run 级表 `download_path`（配合 `Consent=public` 过滤）

**Q4：为什么 `Omics_Category` 和 `Omics_Type` 都要？**
`Omics_Category` 是早期启发式标签（Single-Cell/Spatial、ATAC-seq、ChIP-seq、Bulk RNA-seq），为兼容旧 EDA 保留；新分析一律使用 `Omics_Type`（按官方 LibraryStrategy 映射，更严谨）。