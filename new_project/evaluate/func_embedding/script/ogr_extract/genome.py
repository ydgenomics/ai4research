# -*- coding: utf-8 -*-
"""GFF 基因解析 + pyfaidx 参考序列提取（负链反互补、flank、max_len 截断）。"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

logger = logging.getLogger(__name__)

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


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
    # 必须用下标 genes["attrs"] 访问 GFF 第 9 列
    genes["id"] = genes["attrs"].str.extract(r"ID=([^;]+)", expand=False)
    genes = genes.dropna(subset=["id"])
    genes = genes.drop(columns=["feature", "attrs"])[["id", "chr", "start", "end", "strand"]]
    logger.info(f"GFF 特征类别 {feature} 行数: {len(genes)}")
    return genes


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