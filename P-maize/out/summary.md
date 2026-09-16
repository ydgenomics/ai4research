我来基于过滤后的结果和 BioProject 表做一次完整的数据总结统计 

Ran terminal command:  cd /mnt/rice/default/Workspace/yangdong/ai4research/P-maize && python3 - <<'PY'
import pandas as pd
pd.set_option("display.width", 200)

f = pd.read_csv("out/maize_sra_omics_filtered.csv", low_memory=False, dtype={"TaxID": str, "BioProject": str, "SRAStudy": str, "Sample": str, "BioSample": str})
bp = pd.read_csv("out/maize_bioproject_summary.csv", low_memory=False, dtype={"BioProject": str})

print("=" * 60)
print("过滤后 SRA 表整体概况 (maize_sra_omics_filtered.csv)")
print("=" * 60)
print("总行数:", len(f))
print("唯一 Run:", f["Run"].nunique())

for c in ["BioProject", "SRAStudy", "Experiment", "Sample", "BioSample"]:
    print(f"唯一 {c}: {f[c].nunique()}")

b = pd.to_numeric(f["bases"], errors="coerce")
mb = pd.to_numeric(f["size_MB"], errors="coerce")
sp = pd.to_numeric(f["spots"], errors="coerce")
print("\n总碱基: %.1f Tb" % (b.sum()/1e12))
print("总 spots: %.2f B" % (sp.sum()/1e9))
print("总存储: %.1f TB" % (mb.sum()/1e6))

print("\n--- Omics_Type ---")
print(f["Omics_Type"].value_counts().to_string())

print("\n--- 平台 ---")
print(f["Platform"].value_counts().head(8).to_string())

print("\n--- 物种 ---")
print(f["ScientificName"].value_counts().head(5).to_string())

print("\n--- 年份(ReleaseDate) ---")
yr = f["ReleaseDate"].astype(str).str[:4]
yr = yr[yr != "nan"]
yr = yr[(yr >= "2008") & (yr <= "2026")]
print(yr.value_counts().sort_index().to_string())

print("\n--- Consent ---")
print(f["Consent"].value_counts(dropna=False).to_string())

print("\n--- Context_Tags 命中 ---")
ct = f["Context_Tags"].fillna("")
print(f"带标签行: {(ct!='').sum()} ({(ct!='').mean()*100:.1f}%)")
print(ct[ct!=""].value_counts().head(10).to_string())

print("\n--- Multiomics_Info 命中 ---")
m = f["Multiomics_Info"].fillna("")
print(f"多组学标注行: {(m!='').sum()} ({(m!='').mean()*100:.1f}%)")
print(m[m!=""].value_counts().head(8).to_string())
PY

继续统计 WGS 与 Transcriptome 各自的细分情况、BioProject 表层面，以及 LibrarySelection/Source 等特征分布。

 

Ran terminal command:  cd /mnt/rice/default/Workspace/yangdong/ai4research/P-maize && python3 - <<'PY'
import pandas as pd
pd.set_option("display.width", 200)

f = pd.read_csv("out/maize_sra_omics_filtered.csv", low_memory=False, dtype={"TaxID": str, "BioProject": str})
bp = pd.read_csv("out/maize_bioproject_summary.csv", low_memory=False, dtype={"BioProject": str})

g = f[f["Omics_Type"] == "Genome"]
t = f[f["Omics_Type"] == "Transcriptome"]

print("=" * 60)
print("Genome (WGS) 子集 —", len(g), "rows")
print("=" * 60)
b = pd.to_numeric(g["bases"], errors="coerce"); mb = pd.to_numeric(g["size_MB"], errors="coerce")
print(f"总碱基 {b.sum()/1e12:.0f} Tb | 存储 {mb.sum()/1e6:.0f} TB")
print("LibrarySource:", g["LibrarySource"].value_counts().head(4).to_dict())
print("LibrarySelection:", g["LibrarySelection"].value_counts().head(6).to_dict())
print("Model top:", g["Model"].value_counts().head(5).to_dict())
print("唯一 BioProject:", g["BioProject"].nunique())
print("唯一 Sample:", g["Sample"].nunique())

print("\n" + "=" * 60)
print("Transcriptome (RNA-Seq) 子集 —", len(t), "rows")
print("=" * 60)
b = pd.to_numeric(t["bases"], errors="coerce"); mb = pd.to_numeric(t["size_MB"], errors="coerce")
print(f"总碱基 {b.sum()/1e12:.0f} Tb | 存储 {mb.sum()/1e6:.0f} TB")
print("LibrarySelection:", t["LibrarySelection"].value_counts().head(6).to_dict())
print("LibraryLayout:", t["LibraryLayout"].value_counts().to_dict())
print("Model top:", t["Model"].value_counts().head(5).to_dict())
print("唯一 BioProject:", t["BioProject"].nunique())
print("唯一 Sample:", t["Sample"].nunique())

print("\n" + "=" * 60)
print("BioProject 汇总表概况 (maize_bioproject_summary.csv)")
print("=" * 60)
print("总项目数:", len(bp))
print("\nProject_Omics 标注:")
print(bp["Project_Omics"].fillna("(未打标)").value_counts().head(10).to_string())
print("\nMultiomics:", bp["Multiomics"].value_counts().to_dict())
print("\nRelevance:", bp["Relevance"].value_counts().to_dict())
print("\nCenterName top:", bp["CenterName"].value_counts().head(10).to_string())

# 过滤后表里的项目在两表交叉
f_proj = set(f["BioProject"].dropna())
bp_proj = set(bp["BioProject"].dropna())
print(f"\nfiltered 中唯一 BioProject: {len(f_proj)} | 在 BioProject 表中能找到: {len(f_proj & bp_proj)}")
PY

收集数据汇总如下（基于 `maize_sra_omics_filtered.csv` + `maize_bioproject_summary.csv`）：

---

## 玉米组学数据收集总结

### 一、总体规模（过滤后：WGS + RNA-Seq + Chromatin_Accessibility + Epigenome + Other）

| 指标 | 数量 |
|---|---|
| **Run 总数** | **109,969** |
| 唯一 Experiment | 107,821 |
| 唯一 Sample / BioSample | 89,563 |
| 唯一 SRA Study | 4,538 |
| 唯一 BioProject | 4,504 |
| 总碱基 | **774 Tb** |
| 总 reads | 3.06 T |
| 总存储 | **321 TB**（全部 public 可下载） |

### 二、五类组学构成

| 类别 | Run 数 | 占比 | 碱基 | 存储 | 唯一项目 | 唯一样本 |
|---|---|---|---|---|---|---|
| **Transcriptome (RNA-Seq)** | 60,131 | 54.7% | 300 Tb | 116 TB | 2,194 | 51,306 |
| **Genome (WGS)** | 28,203 | 25.6% | 360 Tb | 159 TB | 335 | 20,098 |
| **Other** | 18,098 | 16.5% | 52 Tb | 21 TB | 2,022 | 16,065 |
| **Epigenome** | 2,518 | 2.3% | 34 Tb | 14 TB | 194 | 2,098 |
| **Chromatin_Accessibility** | 1,019 | 0.9% | 28 Tb | 10 TB | 84 | 939 |

- **RNA-Seq**：以 PAIRED 39,972 / SINGLE 20,159 为主；建库选择 cDNA(26.7k)、RANDOM(8.7k)、PCR(7.9k)、Oligo-dT/PolyA(8.1k)
- **WGS**：GENOMIC 占 99.7%；RANDOM 19.9k、PCR 3.4k、size fractionation 877（HiFi）
- **Epigenome**：ChIP-Seq 1,449、Bisulfite-Seq(WGBS) 1,029、MeDIP-Seq 32、MRE-Seq 8
- **Chromatin_Accessibility**：ATAC-seq 595、MNase-Seq 281、Hi-C 130、DNase-Hypersensitivity 13
- **Other**：OTHER 16,156、WXS 1,306、GBS 224、Ribo-seq 18 等（未归类策略）

### 三、单细胞数据（含在 Transcriptome 内）

- 过滤表中 **LibrarySource 含 SINGLE CELL 共 2,649 行**，其中 **单细胞转录组（scRNA-seq/snRNA-seq）2,582 行**（`TRANSCRIPTOMIC SINGLE CELL`、含 10x 平台）、单细胞基因组 13 行
- 可通过 `LibrarySource == "TRANSCRIPTOMIC SINGLE CELL"` 或 `Assay_Subtype` 列（scRNA/snRNA/scATAC，共 80 行已标注）筛出；其余 57,549 行 RNA-Seq 为 Bulk 转录组

### 四、平台与测序技术

- **Illumina 主导**：102,196（93%），主力机型 NovaSeq 6000、HiSeq 2000、HiSeq 2500
- 长读长：PacBio 1,781、Oxford Nanopore 695
- 华大系：BGISEQ 2,475、DNBSEQ 1,036

### 五、时间趋势（按 ReleaseDate）

2020 年最高峰 **10,458** Run，近三年持续高位（2022: 8,854 / 2023: 7,844 / 2024: 8,743 / **2025: 11,526**），数据仍在高速增长。

### 六、物种构成

- `Zea mays` 80,491（91.1%）+ `subsp. mays` 7,189
- 含近缘种：teosinte `parviglumis` 358、`mexicana` 288（对群体进化研究有价值）

### 七、情境标签（Context_Tags，6.9% 命中，7,560 行）

root 1,235、leaf 997、seed 947、shoot 729、embryo 530、inbred_line 489、endosperm 486、ear 455

### 八、多组学（8.0% 命中，8,836 行）

同项目内多类测序：`Genome;Transcriptome`（项目级+样本级）、`Epigenome;Transcriptome`、`Chromatin_Accessibility;Genome` 等

### 九、BioProject 汇总表（6,483 个项目）

- **项目级分类**：Transcriptome 2,095、Genome 368、Epigenome 99、多类组合 176；**161 个项目为多组学（Multiomics=yes）**
- **相关性**：Relevance=yes 1,787（27.6%），可作为下游筛选优先级
- **机构分布**：DOE JGI 3,477（大量玉米基因组项目）、中国农大 217、河南农大 91、华中农大 82 等
- 过滤后 4,504 个项目中有 **4,341（96%）在 BioProject 表中有描述**，可关联项目标题/描述进一步筛选

---

## 数据总结（一段话）

本研究从 NCBI SRA 公开数据库中收集了玉米（*Zea mays*）全量测序数据约 **774 Tb**，包括 **20,098 个样品的 WGS 数据（28,203 个 Run，约 360 Tb）** 和 **51,306 个样品的 RNA-seq 数据（60,131 个 Run，约 300 Tb，其中单细胞转录组约 2,582 个 Run）**，另包含染色质可及性（ATAC-seq/Hi-C/MNase 等 1,019 个 Run，约 28 Tb）、表观基因组（ChIP-seq/WGBS 等 2,518 个 Run，约 34 Tb）及其他未归类测序（18,098 个 Run，约 52 Tb），合计 **109,969 个测序 Run**，覆盖 89,563 个唯一生物样品与 4,504 个 BioProject，全部数据公开可下载；测序平台以 Illumina 为主（约 93%），辅以华大 DNBSEQ/BGISEQ（约 3,500 个 Run）及 PacBio、Oxford Nanopore 等长读长平台（约 2,500 个 Run）；数据时间跨度 2009–2026 年，近五年每年新增 7,000–11,000 个 Run 且持续增长；同时配套 6,483 个 BioProject 的项目级描述表，可直接按组织、处理条件、品种、单细胞/多组学等维度进行定向筛选，可支撑玉米遗传变异解析、基因表达调控、染色质可及性及多组学关联分析等大规模研究。