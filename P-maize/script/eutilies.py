import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from Bio import Entrez

# ---------------------------------------------------------------------------
# 1. 配置参数（直接在下方填写默认值）
# ---------------------------------------------------------------------------
Entrez.email = "ydgenomics@gmail.com"  # 你的 NCBI 注册邮箱
# 你的 NCBI API Key（NCBI 账号 Settings 页面生成，可选）
# 填写后速率限制从 3 次/秒 提升至 10 次/秒。留 None 使用匿名限速。
# 注意：勿设为 ""（空字符串），Biopython 会把空的 api_key 参数发给 NCBI，
#       导致 HTTP 400 "API key invalid"。
Entrez.api_key = os.environ.get("NCBI_API_KEY") or None

# 检索词逻辑构建：限定玉米物种 (txid4577 = Zea mays) + 针对性关键词组合
KEYWORDS_BUILDER = {
    "SingleCell_Spatial": '("single cell" OR "single-cell" OR "scRNA-seq" OR "spatial transcriptomics" OR "10x Genomics")',
    "Epigenomics_ATAC": '("ATAC-seq" OR "chromatin accessibility")',
    "Bulk_RNA": '("RNA-seq" OR "transcriptome" OR "expression profiling")',
    "Stress_Tissue": '(drought OR cold OR heat OR salt OR disease OR resistance OR development OR tissue OR "inbred line")',
}

_OMICS_CLAUSE = (
    f'({KEYWORDS_BUILDER["SingleCell_Spatial"]} OR {KEYWORDS_BUILDER["Epigenomics_ATAC"]} '
    f'OR ({KEYWORDS_BUILDER["Bulk_RNA"]} AND {KEYWORDS_BUILDER["Stress_Tissue"]}))'
)

# 按数据库构建检索式：taxid 形式在 BioProject / SRA 中最稳定
DB_QUERY = {
    "sra": f"txid4577[Organism] AND {_OMICS_CLAUSE}",
    "bioproject": f"txid4577[Organism] AND {_OMICS_CLAUSE}",
}


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
# 3. 主流程控制与关键词标签分类
# ---------------------------------------------------------------------------
def tag_experiment_type(row):
    """基于 SRA runinfo 实际字段（无 ExperimentTitle 列）标定组学类型标签"""
    def _u(key):
        return str(row.get(key, '')).upper()

    text = " ".join([
        _u('LibraryStrategy'), _u('LibrarySelection'), _u('LibrarySource'),
        _u('LibraryLayout'), _u('LibraryName'), _u('SampleName'),
        _u('Sample'), _u('Model'),
    ])

    tags = []
    # 单细胞 / 空间转录组
    if any(k in text for k in (
        '10X CHROMIUM', '10X GENOMICS', '10XGENOMICS', 'SINGLE CELL',
        'SCRNA', 'SC-RNA', 'VISIUM', 'SPATIAL',
    )):
        tags.append('Single-Cell/Spatial')
    # 染色质可及性
    if 'ATAC' in text:
        tags.append('ATAC-seq')
    # 蛋白-DNA 结合 (ChIP-seq)
    if 'CHIP' in text:
        tags.append('ChIP-seq')
    # Bulk RNA-seq（转录组类且非单细胞）
    if any(k in text for k in (
        'RNA-SEQ', 'TRANSCRIPTOMIC', 'CDNA', 'POLYA', 'EXPRESSION PROFILING',
    )) and 'Single-Cell/Spatial' not in tags:
        tags.append('Bulk RNA-seq')

    return ";".join(tags) if tags else "Other"

def main():
    parser = argparse.ArgumentParser(description="玉米公开组学数据集（NCBI E-utilities）收集")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."),
                        help="输出目录（默认当前目录）")
    parser.add_argument("--sra", action="store_true", help="只跑 SRA 步骤")
    parser.add_argument("--bioproject", action="store_true", help="只跑 BioProject 步骤")
    parser.add_argument("--pubmed", action="store_true",
                        help="为已有 SRA 表(Study_Pubmed_id)抓取文献题录(需先跑 SRA)")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sra_csv = outdir / "maize_sra_omics_dataset.csv"
    bp_csv = outdir / "maize_bioproject_summary.csv"
    pubmed_csv = outdir / "maize_pubmed_literature.csv"

    print("=== 玉米公开组学数据集（NCBI E-utilities）收集任务开始 ===")

    # 步骤 1：检索 SRA 数据库（获得具体 Samples / Runs 级别元数据）
    if not args.bioproject:
        sra_count, sra_webenv, sra_query_key = search_ncbi_history("sra", DB_QUERY["sra"])
        if sra_count > 0:
            df_sra = fetch_sra_metadata(sra_webenv, sra_query_key, sra_count, out_path=sra_csv)

            # 标签打标与分类过滤
            if not df_sra.empty:
                df_sra['Omics_Category'] = df_sra.apply(tag_experiment_type, axis=1)
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
                    'Omics_Category',
                ]
                valid_cols = [c for c in sra_export_cols if c in df_sra.columns]
                df_sra[valid_cols].to_csv(sra_csv, index=False)
                print(f"--> [成功] 已保存 SRA 元数据表: {sra_csv} ({len(df_sra)} 行)")

    # 步骤 2：检索 BioProject 数据库（获得整体项目级描述，方便按课题筛选）
    if not args.sra:
        bp_count, bp_webenv, bp_query_key = search_ncbi_history("bioproject", DB_QUERY["bioproject"])
        if bp_count > 0:
            df_bp = fetch_bioproject_metadata(bp_webenv, bp_query_key, bp_count, out_path=bp_csv)
            if not df_bp.empty:
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