# -*- coding: utf-8 -*-
"""GFF 基因解析 + pyfaidx 参考序列提取（负链反互补、flank、max_len 截断）。

支持两种基因来源:
- GFF3 注释文件 (`load_genes`): 按特征类别取全部基因;
- 面板 CSV (`load_genes_from_csv`): 按 csv 行的 MSU 列(缺省回退 RAPdb)
  定位基因坐标, 只提取面板里列出的基因, 输出顺序 = CSV 行序。
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

logger = logging.getLogger(__name__)

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

_ID_PAT = re.compile(r"(?:^|[;\s])[Ii][Dd]=([^;]+)")


def load_genes(gff: str, feature: str | list[str] = "gene") -> pd.DataFrame:
    """读 GFF3 指定特征类别; attributes 提取 ID。列: id chr start end strand。

    feature: 第 3 列特征类别, 单个字符串或列表。默认 "gene"(=原行为);
             也可用 "CDS" / "5_UTR" / "Intron" / "Intergenic" 等, 或
             ["CDS", "5_UTR"] 合并多类。
    """
    if isinstance(feature, str):
        feature = [feature]
    df = pd.read_csv(
        gff, sep="\t", comment="#", header=None,
        usecols=[0, 2, 3, 4, 6, 8],
        names=["chr", "feature", "start", "end", "strand", "attrs"],
    )
    genes = df[df.feature.isin(feature)].copy()
    # 注意: 列名是 attrs, 但 genes.attrs 是 DataFrame 的元数据属性(返回 dict),
    # 必须用下标 genes["attrs"] 访问 GFF 第 9 列。
    # ID= / Id= 两种写法都兼容 (integrated_features.gff 用 Id=)。
    genes["id"] = genes["attrs"].str.extract(_ID_PAT, expand=False)
    genes = genes.dropna(subset=["id"])
    genes = genes.drop(columns=["feature", "attrs"])[["id", "chr", "start", "end", "strand"]]
    logger.info(f"GFF 特征类别 {feature} 行数: {len(genes)}")
    return genes


def load_genes_from_csv(csv_path: str, id_col: str = "MSU",
                        coord_gff: str | None = None,
                        source_gff: str | None = None) -> pd.DataFrame:
    """从面板 CSV 读基因列表, 按 id_col(默认 MSU) 定位坐标。列: id chr start end strand。

    - 每个基因一条记录, 输出顺序 = CSV 行序;
    - id 优先取 id_col(MSU, 如 LOC_Os01g06280); 该列为空/None 时回退到
      gene_id(RAPdb, 如 Os01g0155500), 再回退到坐标本身的 chr:start-end;
    - 坐标来源: 参数 coord_gff(默认 IRGSP-1.0 注释 gff, 按 ID 匹配);
      找不到 ID 时用 CSV 自带的 chr/start/end/strand 列(格式需与 FASTA 一致);
    - source_gff: 可选; 提供 MSU 版本 GFF(如 osa1_r7.all_models.gff3, ID=LOC_..)
      时, 坐标以它为准(与 MSU 坐标体系一致)。
    """
    # 注意: 面板 CSV 的 Description 列含逗号且部分行引号不配对, 不能用
    # csv.DictReader(表头会出现 None); 改用 csv.reader 按列名定位所需列。
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    cols = [c.strip() for c in header]
    if not rows:
        raise ValueError(f"CSV 为空: {csv_path}")
    need = [id_col, "gene_id", "chr", "start", "end", "strand"]
    idx = {c: cols.index(c) for c in need if c in cols}
    missing = [c for c in (id_col, "chr", "start", "end", "strand") if c not in idx]
    if missing:
        raise ValueError(f"CSV {csv_path} 缺少列: {missing} (现有: {cols})")

    def _cell(r: list, c: str) -> str:
        i = idx.get(c)
        return r[i].strip() if i is not None and i < len(r) else ""

    # 坐标候选 1: coord_gff 按 ID 匹配(IRGSP-1.0 注释, ID=Os01g..)
    coord_by_id: dict[str, tuple] = {}
    if coord_gff:
        try:
            for g in load_genes(coord_gff, "gene").itertuples(index=False):
                coord_by_id.setdefault(g.id, (g.chr, int(g.start), int(g.end), g.strand))
        except Exception as e:
            logger.warning(f"解析 coord_gff {coord_gff} 失败, 仅用 CSV 自带坐标: {e}")

    # 坐标候选 2: source_gff 按 MSU ID 匹配(rice_mut 的 osa1_r7, ID=LOC_..)
    msu_coord: dict[str, tuple] = {}
    if source_gff:
        try:
            for g in load_genes(source_gff, "gene").itertuples(index=False):
                msu_coord.setdefault(g.id, (g.chr, int(g.start), int(g.end), g.strand))
        except Exception as e:
            logger.warning(f"解析 source_gff {source_gff} 失败: {e}")

    # CSV 自带坐标(1-based inclusive): 需要和 FASTA 染色体名一致, 由调用方负责归一化
    csv_coord = {}
    for i, r in enumerate(rows):
        try:
            csv_coord[i] = (str(_cell(r, "chr")), int(float(_cell(r, "start"))),
                            int(float(_cell(r, "end"))), _cell(r, "strand") or "+")
        except (ValueError, TypeError):
            csv_coord[i] = None

    genes, dropped = [], []
    for i, r in enumerate(rows):
        rap = _cell(r, "gene_id")
        msu = _cell(r, id_col)
        if msu in ("", "None"):
            msu = ""
        # 坐标优先级: source_gff(MSU) > coord_gff(RAP) > CSV 自带坐标
        coord = None
        if msu and msu in msu_coord:
            coord = msu_coord[msu]
        elif rap and rap in coord_by_id:
            coord = coord_by_id[rap]
        elif csv_coord.get(i):
            coord = csv_coord[i]
        if coord is None:
            dropped.append(rap or msu or f"row{i}")
            continue
        gid = msu if msu else (rap if rap else f"{coord[0]}:{coord[1]}-{coord[2]}")
        genes.append((gid, *coord))

    if dropped:
        logger.warning(f"CSV 中 {len(dropped)} 个基因找不到坐标, 已跳过: {dropped[:20]}")
    if not genes:
        raise ValueError(f"CSV {csv_path} 中没有可用的基因坐标")

    df = pd.DataFrame(genes, columns=["id", "chr", "start", "end", "strand"])
    logger.info(f"CSV {csv_path}: {len(rows)} 行 -> {len(df)} 个基因 (id 列: {id_col})")
    return df


def normalize_chrom(genes: pd.DataFrame, fasta: str) -> pd.DataFrame:
    """把基因坐标里的染色体名归一化到 FASTA 的命名 (Chr1 <-> chr01 <-> 1 <-> chromosome01)。"""
    fa = Fasta(fasta)
    fasta_names = set(fa.keys())
    g = genes.copy()
    if set(g["chr"].astype(str)) <= fasta_names:
        return g
    for i, c in enumerate(g["chr"].astype(str)):
        if c in fasta_names:
            continue
        cand = [c]
        m = re.match(r"^(?:[Cc]hr|chromosome|CHR)?0*(\d+|[A-Za-z]+)$", c)
        if m:
            n = m.group(1)
            cand += [n, f"chr{n}", f"Chr{n}", f"CHR{n}", f"chromosome{n}"]
            if n.isdigit():
                cand += [f"chr{n:0>2}", f"Chr{n:0>2}", f"CHR{n:0>2}", f"chromosome{n:0>2}"]
        hit = next((x for x in cand if x in fasta_names), None)
        if hit is None:
            raise KeyError(f"FASTA 中找不到染色体 {c!r} (可选: {sorted(fasta_names)[:6]}...)")
        g.loc[g.index[i], "chr"] = hit
    return g


def fetch_seq(fa: Fasta, chrom: str, start: int, end: int, strand: str, flank: int) -> str:
    """取含侧翼的基因序列（pyfaidx 切片 0-based 半开），负链反互补。"""
    lo, hi = max(start - flank, 1), end + flank  # 侧翼自然跟随正/负链
    s = str(fa[chrom][lo - 1:hi])
    return s if strand == "+" else s.translate(_COMP)[::-1]


def extract_sequences(genes: pd.DataFrame, fasta: str, flank: int, max_len: int) -> list[str]:
    """一次性提取所有基因序列（所有模型共用同一份）。max_len=0 不截断。"""
    fa = Fasta(fasta)
    seqs: list[str] = []
    for _, r in genes.iterrows():
        s = fetch_seq(fa, r.chr, r.start, r.end, r.strand, flank)
        if max_len and len(s) > max_len:
            s = s[:max_len]
        seqs.append(s)
    return seqs