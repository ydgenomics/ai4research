"""对 1000 个变异位点进行 GFF 注释分类 (intergenic, cds, intron)。

数据来源:
- BED: data/carrier_hom.bed
- GFF: /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.all_models.gff3
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_bed(bed_path: str):
    regions = []
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom, start, end, name = parts[0], int(parts[1]), int(parts[2]), parts[3]
            p = name.split(":")
            ref_alt = next(x for x in p if ">" in x)
            pos = int(p[p.index(ref_alt) - 1])
            chrom_n = p[p.index(ref_alt) - 2]
            ref, alt = ref_alt.split(">")[:2]
            regions.append((chrom, pos, ref, alt))
    return regions


def classify_sites(bed_path: str, gff_path: str):
    print("[classify] 加载 BED 位点...")
    sites = parse_bed(bed_path)  # [(chrom, pos, ref, alt)]
    
    print("[classify] 加载 GFF3 注释...")
    # 为了快速检索，我们将 GFF 中的区间按染色体存入 interval trees 或 list
    # 格式: Chr1, MSU_osa1r7, gene, start, end, ...
    from collections import defaultdict
    genes = defaultdict(list)
    exons = defaultdict(list)
    cds = defaultdict(list)
    
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, source, ftype, start, end = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
            if ftype == "gene":
                genes[chrom].append((start, end))
            elif ftype == "exon":
                exons[chrom].append((start, end))
            elif ftype == "CDS":
                cds[chrom].append((start, end))

    print("[classify] 开始分类位点...")
    categories = []
    for chrom, pos, ref, alt in sites:
        # 检查是否在 CDS 内
        in_cds = False
        for s, e in cds[chrom]:
            if s <= pos <= e:
                in_cds = True
                break
        
        if in_cds:
            categories.append("cds")
            continue
            
        # 检查是否在 gene 内
        in_gene = False
        for s, e in genes[chrom]:
            if s <= pos <= e:
                in_gene = True
                break
                
        if in_gene:
            # 在 gene 内但不在 CDS 内，归为 intron (或 UTR，这里统一按用户要求归为 intron)
            categories.append("intron")
        else:
            categories.append("intergenic")
            
    # 统计
    from collections import Counter
    counts = Counter(categories)
    print(f"[classify] 分类结果: {dict(counts)}")
    return sites, categories


if __name__ == "__main__":
    classify_sites(
        "data/carrier_hom.bed",
        "/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.all_models.gff3"
    )
