import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Entrez

# ---------------------------------------------------------------------------
# 1. 配置参数（直接在下方填写默认值）
# ---------------------------------------------------------------------------
Entrez.email = "ydgenomics@gmail.com"  # 你的 NCBI 注册邮箱
# 默认内置 NCBI API Key（限速提升至 10 次/秒）
# 环境变量 NCBI_API_KEY 可覆盖；若不想用，可改为 None 使用匿名限速。
# 注意：勿设为 ""（空字符串），Biopython 会把空的 api_key 参数发给 NCBI，
#       导致 HTTP 400 "API key invalid"。
Entrez.api_key = os.environ.get("NCBI_API_KEY") or "83b22d1e0bd8988c025a3a2b563c29f03408"

# 检索式：全量收集玉米（Zea mays, txid4577）组学数据，不做关键词过滤
DB_QUERY = {
    "sra": "txid4577[Organism]",
    "bioproject": "txid4577[Organism]",
}

# ---------------------------------------------------------------------------
# 2. 组学分类映射与本地打标（向量化）
# ---------------------------------------------------------------------------
# LibraryStrategy -> 主类别（NCBI SRA 枚举；未覆盖值归 Other 兜底）
STRATEGY_TO_OMICS = {
    # Genome 基因组（全基因组/简化测序/靶向捕获/克隆）
    "WGS": "Genome", "WGA": "Genome", "WCS": "Genome", "RAD-Seq": "Genome",
    "Amplicon": "Genome", "Targeted-Capture": "Genome",
    "CLONE": "Genome", "POOLCLONE": "Genome", "CLONEEND": "Genome",
    "FINISHING": "Genome", "Synthetic-Long-Read": "Genome",
    # Transcriptome 转录组
    "RNA-Seq": "Transcriptome", "ssRNA-Seq": "Transcriptome",
    "miRNA-Seq": "Transcriptome", "ncRNA-Seq": "Transcriptome",
    "EST": "Transcriptome", "FL-cDNA": "Transcriptome",
    "CTS": "Transcriptome", "RIP-Seq": "Transcriptome", "SAGE": "Transcriptome",
    # Epigenome 表观组（新增：蛋白-DNA 结合 / DNA 甲基化）
    "ChIP-Seq": "Epigenome", "Bisulfite-Seq": "Epigenome",
    "MeDIP-Seq": "Epigenome", "MBD-Seq": "Epigenome", "MRE-Seq": "Epigenome",
    # Chromatin_Accessibility 染色质可及性
    "ATAC-seq": "Chromatin_Accessibility",
    "DNase-Hypersensitivity": "Chromatin_Accessibility",
    "MNase-Seq": "Chromatin_Accessibility", "NOMe-Seq": "Chromatin_Accessibility",
    "Tethered Chromatin Conformation Capture": "Chromatin_Accessibility",
    "Hi-C": "Chromatin_Accessibility",
}
_STRATEGY_LOOKUP = {k.lower(): v for k, v in STRATEGY_TO_OMICS.items()}

# 类内子类关键词（扫描 Strategy/Selection/LibraryName/SampleName，只标注不筛选）
SUBTYPE_KEYWORDS = [
    ("scRNA-seq", ["SINGLE CELL", "SCRNA", "SC-RNA", "SC-RNA-SEQ", "10X CHROMIUM", "10X GENOMICS", "10XGENOMICS"]),
    ("snRNA-seq", ["SNRNA", "SINGLE NUCLEUS"]),
    ("scATAC-seq", ["SC-ATAC", "SCATAC", "SINGLE CELL ATAC"]),
    ("spatial", ["SPATIAL", "VISIUM", "SLIDE-SEQ", "MERFISH", "STEREOSEQ"]),
    ("Multiome(RNA+ATAC)", ["MULTIOME", "RNA+ATAC", "RNA AND ATAC", "CO-ASSAY", "JOINT PROFILING", "EASY-MULTIOME", "DUAL-OMIC"]),
    ("WGBS", ["BISULFITE", "WGBS", "RRBS", "METHYL"]),
    ("ChIP", ["CHIP", "CUT&RUN", "CUT&TAG", "CUT-RUN"]),
    ("Hi-C", ["HI-C", "CHIA-PET", "HIC"]),
    ("WGS", ["WGS", "WHOLE GENOME", "RESEQUENCING", "NANOPORE", "PACBIO"]),
]

# 热词记录（Context_Tags）：扫描 SampleName/LibraryName，仅记录不筛选
CONTEXT_KEYWORDS = [
    ("DROUGHT", "drought"), ("DEHYDRATION", "drought"), ("WATER STRESS", "drought"),
    ("COLD", "cold"), ("FREEZING", "cold"), ("CHILLING", "cold"), ("4C", "cold"),
    ("HEAT", "heat"), ("HIGH TEMPERATURE", "heat"), ("HEAT SHOCK", "heat"),
    ("SALT", "salt"), ("SALINITY", "salt"), ("NACL", "salt"),
    ("DISEASE", "disease"), ("PATHOGEN", "disease"), ("FUNGAL", "disease"),
    ("INFECTION", "disease"), ("INOCULAT", "disease"),
    ("RESISTANCE", "resistance"), ("TOLERANCE", "tolerance"),
    ("DEVELOPMENT", "development"), ("DEVELOPING", "development"),
    ("TISSUE", "tissue"),
    ("INBRED", "inbred_line"),
    ("ROOT", "root"), ("LEAF", "leaf"), ("SEED", "seed"), ("KERNEL", "kernel"),
    ("POLLEN", "pollen"), ("TASSEL", "tassel"), ("EAR", "ear"),
    ("ENDOSPERM", "endosperm"), ("EMBRYO", "embryo"), ("SHOOT", "shoot"),
    ("SINGLE CELL", "single_cell"), ("SCRNA", "single_cell"), ("10X", "single_cell"),
    ("SPATIAL", "spatial"),
]


def rate_limit_sleep():
    """按是否带 API Key 使用不同的限速间隔"""
    time.sleep(0.3 if Entrez.api_key else 0.5)


# ---------------------------------------------------------------------------
# 2. 核心 API 辅助函数
# ---------------------------------------------------------------------------
def search_ncbi_history(db, query):
    """使用 History Server 检索数据库，返回 (Count, WebEnv, QueryKey)。

    两步法：先 retmax=1 拿总数，再以完整 Count 重建 history。
    否则服务器上只保存默认 20 条，后续 efetch 取不到剩余记录。
    """
    print(f"[{db.upper()}] 正在提交检索式: {query}")

    # 第一步：仅获取命中总数
    handle = Entrez.esearch(db=db, term=query, retmax=1)
    count = int(Entrez.read(handle)["Count"])
    handle.close()

    # 第二步：以完整 retmax 重新检索，确保全部结果保存到 History Server
    handle = Entrez.esearch(db=db, term=query, usehistory="y", retmax=count)
    results = Entrez.read(handle)
    handle.close()

    webenv = results["WebEnv"]
    query_key = results["QueryKey"]
    print(f"[{db.upper()}] 共命中 {count} 条记录。")
    return count, webenv, query_key


def fetch_with_retry(fetch_fn, *args, retries=3, **kwargs):
    """带线性退避重试的请求封装；全部失败返回 None"""
    for attempt in range(1, retries + 1):
        try:
            return fetch_fn(*args, **kwargs)
        except Exception as e:
            print(f"  ! 第 {attempt}/{retries} 次请求失败: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
    return None


# SRA runinfo CSV 标准列序（NCBI 文档序；注意：现在该接口不再返回表头，须自行赋列名）
SRA_RUNINFO_COLS = [
    "Run", "ReleaseDate", "LoadDate", "spots", "bases", "spots_with_mates",
    "avgLength", "size_MB", "assemblyName", "download_path", "Experiment",
    "LibraryName", "LibraryStrategy", "LibrarySelection", "LibrarySource",
    "LibraryLayout", "InsertSize", "InsertDev", "Platform", "Model",
    "SRAStudy", "BioProject", "Study_Pubmed_id", "ProjectID", "Sample",
    "BioSample", "SampleType", "TaxID", "ScientificName", "SampleName",
    "g1k_pop_code", "source", "g1k_analysis_group", "Subject_ID", "Sex",
    "Disease", "Tumor", "Affection_Status", "Analyte_Type",
    "Histological_Type", "Body_Site", "CenterName", "Submission",
    "dbGaP_Consent", "Consent", "md5_1", "md5_2",
]


def _apply_runinfo_header(df):
    """runinfo 响应无表头：按 NCBI 标准列序赋予列名（多余列用通用名填充）"""
    n = df.shape[1]
    cols = list(SRA_RUNINFO_COLS[:n])
    if n > len(SRA_RUNINFO_COLS):
        cols += [f"extra_col_{i}" for i in range(len(SRA_RUNINFO_COLS), n)]
    df.columns = cols
    return df


def _fetch_sra_batch(webenv, query_key, start, batch_size):
    handle = Entrez.efetch(
        db="sra", rettype="runinfo", retmode="csv",
        retstart=start, retmax=batch_size,
        webenv=webenv, query_key=query_key,
    )
    # 关键：SRA runinfo 现在不返回表头，必须 header=None 读取后手动赋列名
    df_batch = pd.read_csv(handle, header=None)
    handle.close()
    return _apply_runinfo_header(df_batch)


def _resume_rows(path, expect_col=None):
    """返回已有增量 CSV 的有效行数。

    文件不存在/损坏，或列结构不符（无表头/旧格式，缺 expect_col）时返回 0，
    触发从头重跑（旧的无表头废文件会被覆盖）。
    """
    if path is None or not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if expect_col and expect_col not in df.columns:
        return 0
    return len(df)


def fetch_sra_metadata(webenv, query_key, count, batch_size=500, out_path=None):
    """从 SRA 获取 RunInfo 元数据；边拉边存，支持断点续传"""
    resume = _resume_rows(out_path, expect_col="Run")
    if resume >= count:
        print(f"  - 已有完整数据（{resume} 行），跳过下载")
        return pd.read_csv(out_path)
    if resume > 0:
        print(f"  - 检测到已有 {resume} 行增量数据，从断点继续...")

    sra_records = []
    # 续传时先把已落盘的数据载入返回结果，避免后续打标只作用于新批次导致丢行
    if resume > 0 and out_path is not None:
        try:
            sra_records.append(pd.read_csv(out_path, low_memory=False))
        except Exception as e:
            print(f"  ! 载入已有增量表失败，将只基于新批处理: {e}")

    print(f"[SRA] 开始批量下载 {count} 条 SRA RunInfo 元数据...")

    for start in range(resume, count, batch_size):
        df_batch = fetch_with_retry(_fetch_sra_batch, webenv, query_key, start, batch_size)
        if df_batch is None:
            print(f"  ! 批次 {start} 重试后仍失败，跳过")
            continue
        sra_records.append(df_batch)
        if out_path is not None:
            df_batch.to_csv(out_path, index=False, header=(start == 0), mode="a")
        print(f"  - 已拉取 {start + len(df_batch)} / {count} 条")
        rate_limit_sleep()

    if sra_records:
        return pd.concat(sra_records, ignore_index=True)
    return pd.DataFrame()


def _fetch_bioproject_batch_raw(webenv, query_key, start, batch_size):
    handle = Entrez.efetch(
        db="bioproject", retmode="xml",
        retstart=start, retmax=batch_size,
        webenv=webenv, query_key=query_key,
    )
    xml_data = handle.read()
    handle.close()
    return xml_data


def _parse_bioproject_xml(xml_data):
    """解析 BioProject XML 批次，返回 dict 列表。

    NCBI efetch(db=bioproject, retmode=xml) 现返回 esummary 式结构：
        <RecordSet>
          <DocumentSummary uid="1524312">
            <Project>
              <ProjectID><ArchiveID accession="PRJNA..." archive="NCBI" id="..."/></ProjectID>
              <ProjectDescr><Title>..</Title><Description>..</Description>
                <Relevance><Agricultural>yes</Agricultural></Relevance></ProjectDescr>
              <ProjectType>...</ProjectType>
            </Project>
            <Submission submission_id=".." submitted="2026-.."><Description>..
               <Organization role="owner"><Name>..</Name>...</Organization></Description></Submission>
    （早期为 <PackageSet>/<Package>/<Project>，两种都兼容）

    注意：编号在 ArchiveID 的 accession *属性* 上，须用 .get("accession")。
    """
    root = ET.fromstring(xml_data)
    # 兼容两种顶层包裹结构
    records = root.findall(".//DocumentSummary") or root.findall(".//Package")
    batch = []
    for rec in records:
        proj = rec.find("./Project")
        if proj is None:          # 结构异常时退回以记录本身作为查找根
            proj = rec

        acc_el = proj.find("ProjectID/ArchiveID[@accession]") \
            or proj.find("ProjectID/ArchiveID")
        proj_acc = acc_el.get("accession") if acc_el is not None else ""

        title = proj.findtext("ProjectDescr/Title") or ""
        desc = proj.findtext("ProjectDescr/Description") or ""

        # 发布日期：旧结构有 ProjectReleaseDate；新结构退回 Submission 的 submitted
        release = (proj.findtext("ProjectDescr/ProjectReleaseDate") or "").strip()
        sub = rec.find(".//Submission")
        if not release and sub is not None:
            release = (sub.get("submitted") or sub.get("last_update") or "").strip()

        # 数据类型（如 raw sequence reads）
        datatypes = list(dict.fromkeys(
            n.text.strip() for n in rec.findall(".//DataType")
            if n.text and n.text.strip()
        ))
        if not datatypes:
            obj = rec.find(".//Objectives/Data")
            if obj is not None and obj.get("data_type"):
                datatypes = [obj.get("data_type")]
        datatype = ";".join(datatypes)

        # 提交机构（owner）
        org = rec.find(".//Organization[@role='owner']/Name")
        center = org.text.strip() if org is not None and org.text else ""

        # 农学相关性
        ag = (proj.findtext("ProjectDescr/Relevance/Agricultural") or "").strip().lower()
        relevance = "yes" if ag == "yes" else "no"

        # 关联 GEO 编号（如 GSE123；尽力提取）
        geo = sorted(set(re.findall(r"GSE\d+", ET.tostring(proj, encoding="unicode"))))
        sub_id = sub.get("submission_id") if sub is not None else ""

        batch.append({
            "BioProject": proj_acc,
            "Title": title,
            "Description": desc,
            "ReleaseDate": release,
            "DataType": datatype,
            "CenterName": center,
            "Relevance": relevance,
            "GEO": ";".join(geo),
            "SubmissionID": sub_id,
        })
    return batch


def _fetch_pubmed_batch(ids):
    handle = Entrez.esummary(db="pubmed", id=ids, retmode="xml")
    raw = handle.read()
    handle.close()
    return raw


def fetch_pubmed_summaries(pmids, batch_size=200):
    """按 PMID 列表批量抓取 PubMed 题录(esummary)，返回 {pmid: record} 字典"""
    pids = sorted({str(p).strip() for p in pmids if str(p).strip().isdigit()})
    out = {}
    print(f"[PubMed] 共有 {len(pids)} 个 PMID 待抓取...")
    for i in range(0, len(pids), batch_size):
        chunk = pids[i:i + batch_size]
        raw = fetch_with_retry(_fetch_pubmed_batch, ",".join(chunk))
        if raw is None:
            print(f"  ! PMID 批次 {i} 请求失败，跳过")
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            print(f"  ! PMID 批次 {i} 解析失败: {e}")
            continue
        for ds in root.findall(".//DocumentSummary"):
            uid = ds.get("uid")
            if uid is None:
                continue
            out[uid] = {
                "PMID": uid,
                "PubTitle": (ds.findtext("Title") or "").strip(),
                "Journal": (ds.findtext("FullJournalName") or ds.findtext("Source") or "").strip(),
                "PubDate": (ds.findtext("PubDate") or "").strip(),
            }
        rate_limit_sleep()
    return out


def fetch_bioproject_metadata(webenv, query_key, count, batch_size=200, out_path=None):
    """抓取 BioProject 项目信息；边拉边存，支持断点续传"""
    resume = _resume_rows(out_path, expect_col="BioProject")
    if resume >= count:
        print(f"  - 已有完整数据（{resume} 行），跳过下载")
        return pd.read_csv(out_path)
    if resume > 0:
        print(f"  - 检测到已有 {resume} 行增量数据，从断点继续...")

    projects = []
    print(f"[BioProject] 开始批量抓取 {count} 个 BioProject 信息...")
    done = resume

    for start in range(resume, count, batch_size):
        xml_data = fetch_with_retry(_fetch_bioproject_batch_raw, webenv, query_key, start, batch_size)
        if xml_data is None:
            print(f"  ! 批次 {start} 重试后仍失败，跳过")
            continue
        try:
            batch = _parse_bioproject_xml(xml_data)
        except ET.ParseError as e:
            print(f"  ! 批次 {start} XML 解析失败: {e}")
            continue
        projects.extend(batch)
        done += len(batch)
        if out_path is not None:
            pd.DataFrame(batch).to_csv(out_path, index=False, header=(start == 0), mode="a")
        print(f"  - 已解析 {done} / {count} 个项目")
        rate_limit_sleep()

    return pd.DataFrame(projects)


# ---------------------------------------------------------------------------
# 2.5 实验级摘要（esummary XML）——对应 NCBI 官网"实验级汇总"导出表
# ---------------------------------------------------------------------------
# 与 Run 级 runinfo 表互补：runinfo 重下载路径/访问控制/逐 RUN 数据量；
# esummary 实验级表重三层标题/研究摘要/提交机构/BioSample 属性/实验级统计，
# 信息量 > 手工导出的 sra_result_wgs_wei.csv（连 wgs_wei 缺失的 Sample Title 都能补全）。

# 实验级摘要表的目标列（对齐 wgs_wei 语义并增强）
EXP_EXPORT_COLS = [
    "Experiment", "Experiment_Title", "Study", "Study_Title",
    "Sample", "Sample_Title", "Organism", "Instrument", "Submitter",
    "LibraryName", "LibraryStrategy", "LibrarySource", "LibrarySelection", "LibraryLayout",
    "Total_RUNs", "Total_Spots", "Total_Bases", "Total_Size_MB",
    "Study_Abstract",
    "Attr_isolate", "Attr_cultivar", "Attr_tissue", "Attr_genotype",
    "Attr_dev_stage", "Attr_ecotype", "Sample_Attributes",
]


def _fetch_experiment_summary_batch(webenv, query_key, start, batch_size):
    """抓取一批 esummary XML（实验级摘要）"""
    handle = Entrez.efetch(
        db="sra", rettype="esummary", retmode="xml",
        retstart=start, retmax=batch_size,
        webenv=webenv, query_key=query_key,
    )
    xml_data = handle.read()
    handle.close()
    return xml_data


def _parse_experiment_package_xml(xml_data):
    """解析 esummary <EXPERIMENT_PACKAGE_SET>，返回实验级摘要 dict 列表。

    每个 <EXPERIMENT_PACKAGE> 聚合一个实验（含其全部 RUN）的元数据：
      - EXPERIMENT/TITLE                         实验标题
      - STUDY/(STUDY_TITLE|STUDY_ABSTRACT)       研究标题 / 摘要
      - SAMPLE/(TITLE|SAMPLE_NAME|ATTRIBUTES)    样品标题 / 物种 / BioSample 属性
      - SUBMISSION center_name + Organization    提交机构
      - LIBRARY_DESCRIPTOR                       文库参数
      - PLATFORM/INSTRUMENT_MODEL                测序平台
      - RUN_SET(runs/bases/spots/bytes)          实验级聚合统计
    """
    root = ET.fromstring(xml_data)
    pkgs = root.findall(".//EXPERIMENT_PACKAGE")
    rows = []
    for pkg in pkgs:
        exp = pkg.find("EXPERIMENT")
        if exp is None:
            continue
        rec = {"Experiment": exp.get("accession", "")}
        rec["Experiment_Title"] = exp.findtext("TITLE", "").strip()

        # 研究（标题 + 摘要）
        study = pkg.find("STUDY")
        sdesc = study.find("DESCRIPTOR") if study is not None else None
        rec["Study"] = study.get("accession", "") if study is not None else ""
        rec["Study_Title"] = sdesc.findtext("STUDY_TITLE", "").strip() if sdesc is not None else ""
        rec["Study_Abstract"] = sdesc.findtext("STUDY_ABSTRACT", "").strip() if sdesc is not None else ""

        # 文库参数（Name/Strategy/Source/Selection/Layout）
        lib = pkg.find(".//LIBRARY_DESCRIPTOR")
        if lib is not None:
            rec["LibraryName"] = lib.findtext("LIBRARY_NAME", "").strip()
            rec["LibraryStrategy"] = lib.findtext("LIBRARY_STRATEGY", "").strip()
            rec["LibrarySource"] = lib.findtext("LIBRARY_SOURCE", "").strip()
            rec["LibrarySelection"] = lib.findtext("LIBRARY_SELECTION", "").strip()
            layout = lib.find("LIBRARY_LAYOUT")
            rec["LibraryLayout"] = layout[0].tag if layout is not None and len(layout) else ""
        else:
            for k in ("LibraryName", "LibraryStrategy", "LibrarySource",
                      "LibrarySelection", "LibraryLayout"):
                rec[k] = ""

        # 平台仪器
        model = pkg.find(".//INSTRUMENT_MODEL")
        rec["Instrument"] = model.text.strip() if model is not None and model.text else ""

        # 提交机构（优先 SUBMISSION center_name，退回 Organization/Name）
        sub = pkg.find("SUBMISSION")
        rec["Submitter"] = sub.get("center_name", "").strip() if sub is not None else ""
        org = pkg.find("Organization/Name")
        if org is not None and org.text and not rec["Submitter"]:
            rec["Submitter"] = org.text.strip()

        # 样品（标题 + 物种 + BioSample 属性）
        samp = pkg.find("SAMPLE")
        rec["Sample"] = samp.get("accession", "") if samp is not None else ""
        rec["Sample_Title"] = samp.findtext("TITLE", "").strip() if samp is not None else ""
        sn = samp.find("SAMPLE_NAME") if samp is not None else None
        rec["Organism"] = sn.findtext("SCIENTIFIC_NAME", "").strip() if sn is not None else ""
        attrs = {}
        if samp is not None:
            for a in samp.findall("SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE"):
                tag = a.findtext("TAG", "").strip().lower()
                val = a.findtext("VALUE", "").strip()
                if tag and val and val.lower() not in ("not applicable", "n/a", ""):
                    attrs.setdefault(tag, []).append(val)
        common = {"isolate", "cultivar", "tissue", "genotype", "dev_stage",
                  "ecotype", "biomaterial_provider", "collection_date",
                  "geo_loc_name", "isolation_source", "cell_line", "age"}
        for k in common:
            vel = attrs.pop(k, None)
            rec[f"Attr_{k}"] = ";".join(vel) if vel else ""
        rec["Sample_Attributes"] = "; ".join(
            f"{k}={','.join(v)}" for k, v in sorted(attrs.items()))

        # RUN_SET 实验级聚合（runs/bases/spots/bytes）
        runs = pkg.find("RUN_SET")
        rec["Total_RUNs"] = int(runs.get("runs", 0)) if runs is not None else 0
        rec["Total_Spots"] = int(runs.get("spots", 0)) if runs is not None else 0
        rec["Total_Bases"] = int(runs.get("bases", 0)) if runs is not None else 0
        rec["Total_Size_MB"] = round(int(runs.get("bytes", 0)) / 1e6, 2) if runs is not None else 0.0

        rows.append(rec)
    return rows


def fetch_experiment_summary(webenv, query_key, count, batch_size=200, out_path=None):
    """实验级摘要（esummary XML 解析）：标题/机构/样品/实验级统计。

    注意：esearch 的 count 是 Run 级命中数，而 esummary 按实验包返回（包数 ≤ Run 数），
    因此遇到空批次会提前 break，并在完成后写 <out>.done 哨兵文件，避免下次重复拉取。
    """
    done_flag = None
    if out_path is not None:
        done_flag = Path(str(out_path) + ".done")
        if done_flag.exists():
            print(f"  - 实验摘要表已完成（{done_flag.name} 存在），直接读取")
            try:
                return pd.read_csv(out_path, low_memory=False)
            except Exception as e:
                print(f"  ! 读取完成态失败，重新拉取: {e}")

    resume = _resume_rows(out_path, expect_col="Experiment")
    seen = set()
    if resume > 0 and out_path is not None:
        try:
            old = pd.read_csv(out_path, low_memory=False)
            seen = set(old["Experiment"].dropna().astype(str))
            print(f"  - 检测到已有 {resume} 行，从断点继续（按 Experiment 去重）...")
        except Exception as e:
            print(f"  ! 载入已有增量表失败，将只基于新批处理: {e}")
            resume = 0

    recs = []
    print(f"[SRA-ESUMMARY] 开始批量抓取实验级摘要（命中 {count} 个 Run，实验包 ≤ 该数）...")
    for start in range(resume, count, batch_size):
        xml_data = fetch_with_retry(
            _fetch_experiment_summary_batch, webenv, query_key, start, batch_size)
        if xml_data is None:
            print(f"  ! 批次 {start} 重试后仍失败，跳过")
            continue
        try:
            batch = _parse_experiment_package_xml(xml_data)
        except ET.ParseError as e:
            print(f"  ! 批次 {start} XML 解析失败: {e}")
            continue
        if not batch:                       # 已越过最后一个实验包
            print(f"  - 批次 {start} 返回空，提前结束（实验包数 < Run 命中数）")
            break
        fresh = [r for r in batch if r["Experiment"] not in seen]
        if fresh:
            seen.update(r["Experiment"] for r in fresh)
            recs.append(pd.DataFrame(fresh))
            if out_path is not None:
                pd.DataFrame(fresh).to_csv(out_path, index=False,
                                           header=(start == 0), mode="a")
        print(f"  - 已解析 {start + len(batch)} 个实验包（新增 {len(fresh)}）")
        rate_limit_sleep()

    if recs:
        df = pd.concat(recs, ignore_index=True)\
            .drop_duplicates(subset="Experiment", keep="last")
    elif resume > 0 and out_path is not None:
        df = pd.read_csv(out_path, low_memory=False)   # 上次已拉完但未写哨兵
    else:
        df = pd.DataFrame()
    if done_flag is not None and (len(df) > 0 or resume > 0):
        done_flag.touch()
    return df


# ---------------------------------------------------------------------------
# 3. 组学分类 / 多组学 / 热词打标函数
# ---------------------------------------------------------------------------
def _strategy_to_omics(s):
    """单个 LibraryStrategy 值 -> 主类别（多值;分隔取并集，全失败归 Other）"""
    cats = sorted({
        _STRATEGY_LOOKUP[p.strip().lower()]
        for p in str(s).split(";") if p.strip()
        if p.strip().lower() in _STRATEGY_LOOKUP
    })
    return ";".join(cats) if cats else "Other"


def classify_omics(df):
    """按 LibraryStrategy 打四类主标签（Omics_Type）"""
    df = df.copy()
    df["Omics_Type"] = df["LibraryStrategy"].fillna("").astype(str).map(_strategy_to_omics)
    return df


def assign_subtype(df):
    """类内子类标签（Assay_Subtype）：按关键词刮取，只标记不筛选"""
    text = (
        df["LibraryStrategy"].fillna("").astype(str) + " " +
        df["LibrarySelection"].fillna("").astype(str) + " " +
        df["LibraryName"].fillna("").astype(str) + " " +
        df["SampleName"].fillna("").astype(str)
    ).str.upper()
    hits = np.column_stack([
        text.str.contains("|".join(re.escape(k) for k in kws), regex=True, na=False)
        for _, kws in SUBTYPE_KEYWORDS
    ])
    tags = [
        ";".join(SUBTYPE_KEYWORDS[i][0] for i, h in enumerate(row) if h)
        for row in hits
    ]
    df["Assay_Subtype"] = tags
    return df


def _cat_set(series):
    """把一组 Omics_Type 值摊平成去重且不含 Other 的类别集合"""
    seen = set()
    for v in series.fillna("").astype(str):
        for c in v.split(";"):
            c = c.strip()
            if c and c != "Other":
                seen.add(c)
    return seen


def assign_multiomics(df):
    """三粒度多组学标注（Multiomics_Info，;连接可叠加）：
    - Project:  同 BioProject 内 >=2 类主类别
    - Sample:   同 BioSample 内 >=2 类主类别
    - Keyword:  LibraryName/SampleName 含 Multiome 等联合测序关键词
    """
    df = df.copy()

    proj_set = df.groupby("BioProject")["Omics_Type"].apply(_cat_set)
    proj_map = proj_set.apply(
        lambda s: "Project:" + ";".join(sorted(s)) if len(s) >= 2 else "").to_dict()
    df["_proj"] = df["BioProject"].map(proj_map).fillna("")

    samp_set = df.groupby("BioSample")["Omics_Type"].apply(_cat_set)
    samp_map = samp_set.apply(
        lambda s: "Sample:" + ";".join(sorted(s)) if len(s) >= 2 else "").to_dict()
    df["_samp"] = df["BioSample"].map(samp_map).fillna("")

    kw = (df["LibraryName"].fillna("").astype(str) + " " +
          df["SampleName"].fillna("").astype(str)).str.upper()
    df["_kw"] = ""
    mask = kw.str.contains(
        "MULTIOME|CO-ASSAY|JOINT PROFILING|RNA\\+ATAC|EASY-MULTIOME|DUAL-OMIC|CITE-SEQ",
        regex=True, na=False)
    df.loc[mask, "_kw"] = "Keyword:multiome"

    df["Multiomics_Info"] = (
        (df["_proj"] + ";" + df["_samp"] + ";" + df["_kw"])
        .str.strip(";").str.replace(r";+", ";", regex=True)
    )
    return df.drop(columns=["_proj", "_samp", "_kw"])


def assign_context_tags(df):
    """热词记录（Context_Tags）：扫描 SampleName/LibraryName，只记录不筛选"""
    text = (df["SampleName"].fillna("").astype(str) + " " +
            df["LibraryName"].fillna("").astype(str)).str.upper()
    hits = np.column_stack([
        text.str.contains(kw, regex=False, na=False) for kw, _ in CONTEXT_KEYWORDS
    ])
    df["Context_Tags"] = [
        ";".join(CONTEXT_KEYWORDS[i][1] for i, h in enumerate(row) if h)
        for row in hits
    ]
    return df


def tag_omics_category(df):
    """旧启发式标签（Omics_Category）向量化版：Single-Cell/Spatial、ATAC-seq、
    ChIP-seq、Bulk RNA-seq。该列仅为兼容早期 EDA 保留，新分类请用 Omics_Type。
    """
    def _safe(col):
        return df[col].fillna("").astype(str) if col in df.columns else ""

    text = (
        _safe("LibraryStrategy") + " " +
        _safe("LibrarySelection") + " " +
        _safe("LibrarySource") + " " +
        _safe("LibraryName") + " " +
        _safe("SampleName") + " " +
        _safe("Sample") + " " +
        _safe("Model")
    ).str.upper()

    is_sc = text.str.contains(
        "10X CHROMIUM|10X GENOMICS|10XGENOMICS|SINGLE CELL|SCRNA|SC-RNA|VISIUM|SPATIAL",
        regex=True, na=False)
    is_atac = text.str.contains("ATAC", regex=False, na=False)
    is_chip = text.str.contains("CHIP", regex=False, na=False)
    is_bulk = text.str.contains(
        "RNA-SEQ|TRANSCRIPTOMIC|CDNA|POLYA|EXPRESSION PROFILING", regex=True, na=False) & ~is_sc

    tags = []
    for i in range(len(df)):
        parts = []
        if is_sc.iloc[i]:
            parts.append("Single-Cell/Spatial")
        if is_atac.iloc[i]:
            parts.append("ATAC-seq")
        if is_chip.iloc[i]:
            parts.append("ChIP-seq")
        if is_bulk.iloc[i]:
            parts.append("Bulk RNA-seq")
        tags.append(";".join(parts) if parts else "Other")
    df["Omics_Category"] = tags
    return df


def filter_core_omics(df):
    """策略级精简过滤：主类别内只保留最常见、最能代表该组学类的测序策略；
    附加类别整体保留，便于下游按需取用。
    - Genome（精简）：       只保留 LibraryStrategy == WGS
    - Transcriptome（精简）：只保留 LibraryStrategy == RNA-Seq
    - Chromatin_Accessibility / Epigenome / Other：类别整体保留
    其余策略（CLONE/FINISHING/RAD-Seq/Amplicon/…、miRNA-seq/ncRNA/…）一律剔除。
    LibraryStrategy 缺失或大小写变体以大小写不敏感比较兜底。
    """
    strat = df["LibraryStrategy"].fillna("").astype(str).str.upper()
    t = df["Omics_Type"]
    keep = ((t == "Genome") & (strat == "WGS")) | \
           ((t == "Transcriptome") & (strat == "RNA-SEQ")) | \
           t.isin(["Chromatin_Accessibility", "Epigenome", "Other"])
    return df[keep].copy()


# ---------------------------------------------------------------------------
# 4. 主流程控制
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="玉米公开组学数据集（NCBI E-utilities）收集")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."),
                        help="输出目录（默认当前目录）")
    parser.add_argument("--sra", action="store_true", help="只跑 SRA 步骤")
    parser.add_argument("--bioproject", action="store_true", help="只跑 BioProject 步骤")
    parser.add_argument("--experiment", action="store_true",
                        help="抓取实验级摘要(esummary XML)：三层标题/提交机构/样品属性/实验级统计")
    parser.add_argument("--experiment-query", type=str, default=None,
                        help="实验级摘要检索式（默认同 SRA 全量检索式；"
                             "如 'txid4577[Organism] AND wgs[Strategy]' 复现 WGS 实验表）")
    parser.add_argument("--pubmed", action="store_true",
                        help="为已有 SRA 表(Study_Pubmed_id)抓取文献题录(需先跑 SRA)")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sra_csv = outdir / "maize_sra_omics_dataset.csv"
    bp_csv = outdir / "maize_bioproject_summary.csv"
    pubmed_csv = outdir / "maize_pubmed_literature.csv"

    # 显式指定任一开关时只跑指定步骤；一个都不指定时默认全跑
    selective = args.sra or args.bioproject or args.experiment or args.pubmed

    print("=== 玉米公开组学数据集（NCBI E-utilities）收集任务开始 ===")

    # 步骤 1：检索 SRA 数据库（Run 级全量元数据）
    if args.sra or not selective:
        # 旧表为关键词限定版子集（含 Omics_Category 但无 Omics_Type），需移存备份后全新抓取；
        # 新格式增量表（含 Omics_Type 或仅有 runinfo 原始列）则保留，由断点续传继续。
        if sra_csv.exists():
            try:
                sra_head_cols = pd.read_csv(sra_csv, nrows=0).columns.tolist()
            except Exception:
                sra_head_cols = []
            is_old = (
                "Omics_Category" in sra_head_cols
                and "Omics_Type" not in sra_head_cols
            )
            if is_old:
                backup = outdir / "maize_sra_omics_dataset_old.csv"
                if backup.exists():
                    backup = outdir / f"maize_sra_omics_dataset_old_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                os.replace(str(sra_csv), str(backup))
                print(f"--> [备份] 原表已移存至: {backup}（将全新抓取全量数据）")
            else:
                print(f"--> [续传] 检测到新格式增量表，将从断点继续: {sra_csv}")

        sra_count, sra_webenv, sra_query_key = search_ncbi_history("sra", DB_QUERY["sra"])
        if sra_count > 0:
            print(f"  ! 本次命中 {sra_count} 条 Run，数据量较大，请耐心等待...")
            df_sra = fetch_sra_metadata(sra_webenv, sra_query_key, sra_count, out_path=sra_csv)

            if not df_sra.empty:
                # 四类主类别 / 类内子类 / 三粒度多组学 / 热词记录 / 旧启发式标签（兼容）
                df_sra = classify_omics(df_sra)
                df_sra = assign_subtype(df_sra)
                df_sra = assign_multiomics(df_sra)
                df_sra = assign_context_tags(df_sra)
                df_sra = tag_omics_category(df_sra)

                # 导出关键列字段（均为 runinfo 真实存在的列）
                sra_export_cols = [
                    'Run', 'BioProject', 'SRAStudy', 'Experiment',
                    'Sample', 'BioSample', 'SampleName',
                    'LibraryStrategy', 'LibrarySelection', 'LibrarySource',
                    'LibraryLayout', 'Platform', 'Model', 'TaxID',
                    'ScientificName', 'ReleaseDate',
                    # 数据量 / 下载 / 访问控制 / 文献关联
                    'spots', 'bases', 'size_MB', 'avgLength',
                    'download_path', 'Study_Pubmed_id', 'Consent',
                    'CenterName',
                    # 分类与多组学标注
                    'Omics_Type', 'Assay_Subtype', 'Multiomics_Info',
                    'Omics_Category', 'Context_Tags',
                ]
                valid_cols = [c for c in sra_export_cols if c in df_sra.columns]
                df_sra[valid_cols].to_csv(sra_csv, index=False)
                print(f"--> [成功] 已保存 SRA 元数据表: {sra_csv} ({len(df_sra)} 行)")

                # 精简过滤：Genome 只留 WGS、Transcriptome 只留 RNA-Seq
                if "Omics_Type" in df_sra.columns and "LibraryStrategy" in df_sra.columns:
                    df_f = filter_core_omics(df_sra)
                    filtered_csv = outdir / "maize_sra_omics_filtered.csv"
                    df_f[valid_cols].to_csv(filtered_csv, index=False)
                    print(f"--> [过滤] 仅保留 WGS(Genome)+RNA-Seq(Transcriptome)，"
                          f"已保存: {filtered_csv} ({len(df_f)} 行)")

    # 步骤 1.5（可选）：实验级摘要——实验/研究/样品标题、提交机构、实验级统计
    # （对应 NCBI "实验级汇总"导出表如 out/sra_result_wgs_wei.csv，且信息更全）
    if args.experiment:
        exp_query = args.experiment_query or DB_QUERY["sra"]
        exp_csv = outdir / "maize_sra_experiments.csv"
        exp_count, exp_webenv, exp_qk = search_ncbi_history("sra", exp_query)
        if exp_count > 0:
            df_exp = fetch_experiment_summary(exp_webenv, exp_qk, exp_count, out_path=exp_csv)
            if not df_exp.empty:
                exp_cols = [c for c in EXP_EXPORT_COLS if c in df_exp.columns]
                df_exp[exp_cols].to_csv(exp_csv, index=False)
                print(f"--> [成功] 已保存实验级摘要表: {exp_csv} ({len(df_exp)} 行)")

    # 步骤 2：检索 BioProject 数据库（项目级全量描述，方便按课题筛选）
    if args.bioproject or not selective:
        # 旧表同为关键词限定子集，同样仅在新格式时保留续传、旧格式才备份重抓
        if bp_csv.exists():
            try:
                bp_head_cols = pd.read_csv(bp_csv, nrows=0).columns.tolist()
            except Exception:
                bp_head_cols = []
            is_old = "Project_Omics" not in bp_head_cols
            if is_old:
                backup = outdir / "maize_bioproject_summary_old.csv"
                if backup.exists():
                    backup = outdir / f"maize_bioproject_summary_old_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                os.replace(str(bp_csv), str(backup))
                print(f"--> [备份] 原 BioProject 表已移存至: {backup}（将全新抓取全量数据）")
            else:
                print(f"--> [续传] 检测到新格式 BioProject 增量表，将从断点继续: {bp_csv}")

        bp_count, bp_webenv, bp_query_key = search_ncbi_history("bioproject", DB_QUERY["bioproject"])
        if bp_count > 0:
            print(f"  ! 本次命中 {bp_count} 个项目，数据量较大，请耐心等待...")
            df_bp = fetch_bioproject_metadata(bp_webenv, bp_query_key, bp_count, out_path=bp_csv)
            if not df_bp.empty:
                # 从 Run 级表聚合各项目的主类别集合与是否多组学
                proj_omics, proj_multi = {}, {}
                if sra_csv.exists():
                    try:
                        agg = pd.read_csv(
                            sra_csv, usecols=["BioProject", "Omics_Type"],
                            low_memory=False).groupby("BioProject")["Omics_Type"].apply(_cat_set)
                        proj_omics = agg.apply(lambda s: ";".join(sorted(s))).to_dict()
                        proj_multi = agg.apply(lambda s: "yes" if len(s) >= 2 else "no").to_dict()
                    except Exception as e:
                        print(f"  ! 聚合 SRA 表项目类别失败(不影响导出): {e}")
                df_bp["Project_Omics"] = df_bp["BioProject"].map(proj_omics).fillna("")
                df_bp["Multiomics"] = df_bp["BioProject"].map(proj_multi).fillna("no")
                df_bp.to_csv(bp_csv, index=False)
                print(f"--> [成功] 已保存 BioProject 项目汇总表: {bp_csv} ({len(df_bp)} 行)")

    # 步骤 3（可选）：基于 SRA 表的 Study_Pubmed_id 抓取文献题录
    if args.pubmed:
        if not sra_csv.exists():
            print("[--pubmed] 未找到 SRA 表，请先运行 SRA 步骤。")
        else:
            df_sra = pd.read_csv(sra_csv, usecols=lambda c: c in ("Study_Pubmed_id", "BioProject"))
            if "Study_Pubmed_id" not in df_sra.columns:
                print("[--pubmed] SRA 表缺少 Study_Pubmed_id 列，无法关联文献。")
            else:
                pm_to_bp = {}
                for raw_pid, bp in zip(df_sra["Study_Pubmed_id"].fillna(""), df_sra["BioProject"].fillna("")):
                    for pid in str(raw_pid).replace(";", ",").split(","):
                        pid = pid.strip()
                        if pid.isdigit():
                            pm_to_bp.setdefault(pid, set()).add(str(bp))
                if not pm_to_bp:
                    print("[--pubmed] SRA 表中没有任何关联 PMID（多数提交未填，属正常）。")
                else:
                    recs = fetch_pubmed_summaries(pm_to_bp.keys())
                    rows = []
                    for pid, bps in pm_to_bp.items():
                        r = recs.get(pid, {})
                        rows.append({
                            "PMID": pid,
                            "BioProjects": ";".join(sorted(bps)),
                            "PubTitle": r.get("PubTitle", ""),
                            "Journal": r.get("Journal", ""),
                            "PubDate": r.get("PubDate", ""),
                        })
                    df_pm = pd.DataFrame(rows)
                    df_pm.to_csv(pubmed_csv, index=False)
                    print(f"--> [成功] 已保存文献题录表: {pubmed_csv} ({len(df_pm)} 篇)")

    print("\n=== 数据收集完成！下一步可在 Pandas 中根据亲本名称、组织、处理条件进行二次过滤 ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，增量 CSV 已保留，可重新运行以断点续传。")
        sys.exit(130)