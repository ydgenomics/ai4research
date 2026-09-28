"""变异位点统计分析: 对评测位点 (carrier_hom.bed) 做群体/序列水平统计。

统计项:
1. 染色体分布 (每个染色体位点数)
2. SNP 替换类型分布 (A>C, A>G, A>T, C>A, ... 12 类) + Ti/Tv 比值
3. 等位基因频率分布 (INFO/AF), 分组: 低频(<0.05) / 中频(0.05-0.5) / 高频(>=0.5)
4. 变异在窗口内的相对位置分布 (BED 内偏移)
5. 窗口序列 GC 含量分布 (参考 FASTA 256bp 窗口)
6. 变异位点上下文保守性 (参考序列中变异碱基出现频率, 作为背景参考)

用法:
    python scripts/variant_stats.py \
        --bed data/carrier_hom.bed \
        --vcf data/251.SNP.bsnp.vcf.gz \
        --ref <osa1_r7.asm.ch.fa> \
        --out output/hom_1000_multi7/variant_stats.json \
        --out-png output/hom_1000_multi7/variant_stats.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TITV = {
    ("A", "G"): "ts", ("G", "A"): "ts", ("C", "T"): "ts", ("T", "C"): "ts",
    ("A", "C"): "tv", ("A", "T"): "tv", ("C", "A"): "tv", ("C", "G"): "tv",
    ("G", "C"): "tv", ("G", "T"): "tv", ("T", "A"): "tv", ("T", "G"): "tv",
}
BASES = "ACGT"


def load_vcf_info(vcf_path: str, regions: list = None):
    """收集 (chrom, pos, ref, alt, AF)。

    若提供 regions (BED 位点), 用 tabix 索引按位点精确查询, 避免全量扫描。
    返回 {(chrom,pos): (ref, alt, AF)}。
    """
    import cyvcf2

    vcf = cyvcf2.VCF(str(vcf_path))
    info = {}
    if regions is not None:
        # 每个位点一次 tabix 精确区间查询: chrom:pos-pos
        n_hit = 0
        for r in regions:
            chrom, pos = r[0], r[4]  # (chrom, start, end, chrom_n, pos, ref, alt)
            key = (chrom, pos)
            if key in info:
                continue
            for rec in vcf(f"{chrom}:{pos}-{pos}"):
                info[(rec.CHROM, rec.POS)] = (rec.REF, rec.ALT[0], rec.INFO.get("AF"))
                n_hit += 1
        print(f"[stats] VCF 命中 {n_hit} 个位点 (查询 {len({(r[0], r[4]) for r in regions})} 个去重位点)")
        return info

    for rec in vcf:
        info[(rec.CHROM, rec.POS)] = (rec.REF, rec.ALT[0], rec.INFO.get("AF"))
    return info


def parse_bed(bed_path: str):
    """解析 BED, 返回 regions: [(chrom, start, end, chrom, pos, ref, alt)]。"""
    regions = []
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom, start, end, name = parts[0], int(parts[1]), int(parts[2]), parts[3]
            # name: {chrom}:{pos}:{ref}>{alt}  (可能有前缀)
            p = name.split(":")
            ref_alt = next(x for x in p if ">" in x)
            pos = int(p[p.index(ref_alt) - 1])
            chrom_n = p[p.index(ref_alt) - 2]
            ref, alt = ref_alt.split(">")[:2]
            regions.append((chrom, start, end, chrom_n, pos, ref, alt))
    return regions


def gc_content_of_windows(regions, ref_fasta: str):
    """从参考 FASTA 提取窗口序列, 计算 GC 含量。返回 [B] float (缺失=nan)。"""
    from pyfaidx import Fasta

    fa = Fasta(ref_fasta)
    gcs = []
    for chrom, start, end, *_ in regions:
        try:
            seq = str(fa[chrom][start:end])
            n = len(seq)
            gc = (seq.count("G") + seq.count("C")) / n if n else float("nan")
        except Exception:
            gc = float("nan")
        gcs.append(gc)
    return np.array(gcs, dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bed", required=True)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=None, help="统计 JSON 输出路径")
    ap.add_argument("--out-png", default=None, help="统计图输出路径")
    args = ap.parse_args()

    out_json = Path(args.out) if args.out else Path(args.bed).parent / "variant_stats.json"
    out_png = Path(args.out_png) if args.out_png else out_json.with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1. 读 BED + VCF
    print("[stats] 解析 BED ...")
    regions = parse_bed(args.bed)
    print(f"[stats] BED 位点数: {len(regions)}")
    print("[stats] 扫描 VCF (按位点索引查询) ...")
    vcf_info = load_vcf_info(args.vcf, regions=regions)
    print(f"[stats] VCF 记录数: {len(vcf_info)}")

    # 2. 组装每位点特征
    chroms, poss, refs, alts, afs = [], [], [], [], []
    for r in regions:
        chrom, start, end, chrom_n, pos, ref, alt = r
        af = vcf_info.get((chrom_n, pos), (None, None, None))[2]
        chroms.append(chrom_n)
        poss.append(pos)
        refs.append(ref)
        alts.append(alt)
        afs.append(af)
    refs = np.array(refs)
    alts = np.array(alts)
    afs = np.array([x if x is not None else np.nan for x in afs])

    # 3. 各统计量
    # 3.1 染色体分布
    uniq_chroms, chrom_counts = np.unique(chroms, return_counts=True)
    order = np.argsort([int(c.replace("Chr", "")) if c.replace("Chr", "").isdigit() else 99 for c in uniq_chroms])
    uniq_chroms, chrom_counts = uniq_chroms[order], chrom_counts[order]

    # 3.2 SNP 类型 (12 类) + Ti/Tv
    n_ts = n_tv = 0
    types = {f"{a}>{b}": 0 for a in BASES for b in BASES if a != b}
    for a, b in zip(refs, alts):
        key = f"{a}>{b}"
        if key in types:
            types[key] += 1
            n_ts += 1 if TITV[(a, b)] == "ts" else 0
            n_tv += 1 if TITV[(a, b)] == "tv" else 0
    titv = n_ts / n_tv if n_tv else float("inf")

    # 3.3 AF 分布
    af_valid = afs[~np.isnan(afs)]
    af_bins = {
        "low (<0.05)": int(((af_valid < 0.05)).sum()),
        "mid (0.05-0.5)": int(((af_valid >= 0.05) & (af_valid < 0.5)).sum()),
        "high (>=0.5)": int((af_valid >= 0.5).sum()),
    }
    af_mean = float(af_valid.mean()) if len(af_valid) else float("nan")
    af_median = float(np.median(af_valid)) if len(af_valid) else float("nan")

    # 3.4 窗口内相对位置
    rel_pos = np.array([pos - start for _, start, _, _, pos, _, _ in regions], dtype=float) / (
        np.array([end - start for _, start, end, *_ in regions])
    )

    # 3.5 GC 含量
    print("[stats] 计算窗口 GC 含量 ...")
    gcs = gc_content_of_windows(regions, args.ref)
    gc_valid = gcs[~np.isnan(gcs)]

    stats = {
        "n_variants": len(regions),
        "chrom_distribution": {c: int(n) for c, n in zip(uniq_chroms, chrom_counts)},
        "snp_types": types,
        "ti": n_ts, "tv": n_tv, "titv": round(titv, 3),
        "af": {"bins": af_bins, "mean": round(af_mean, 4), "median": round(af_median, 4),
               "missing": int(np.isnan(afs).sum())},
        "rel_pos": {"mean": round(float(rel_pos.mean()), 4), "std": round(float(rel_pos.std()), 4),
                    "min": round(float(rel_pos.min()), 4), "max": round(float(rel_pos.max()), 4)},
        "gc": {"mean": round(float(gc_valid.mean()), 4), "std": round(float(gc_valid.std()), 4),
               "median": round(float(np.median(gc_valid)), 4)},
    }

    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[stats] JSON 已写入 {out_json}")

    # 4. 绘图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # 4.1 染色体分布
    ax = axes[0][0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(uniq_chroms)))
    bars = ax.bar(uniq_chroms, chrom_counts, color=colors)
    ax.set_title(f"Chromosome distribution (n={len(regions)})")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Variants")
    ax.tick_params(axis="x", rotation=60)
    for b, v in zip(bars, chrom_counts):
        ax.text(b.get_x() + b.get_width() / 2, v + max(chrom_counts) * 0.01, str(v),
                ha="center", va="bottom", fontsize=8)

    # 4.2 SNP 类型
    ax = axes[0][1]
    tkeys = [k for k in types if types[k] > 0]
    tvals = [types[k] for k in tkeys]
    colors2 = ["#d62728" if TITV[(k[0], k[2])] == "tv" else "#1f77b4" for k in tkeys]
    ax.bar(tkeys, tvals, color=colors2)
    ax.set_title(f"SNP substitution types (Ti/Tv={titv:.2f})")
    ax.set_xlabel("Ref>Alt")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=60)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="Transition (Ti)"),
                       Patch(color="#d62728", label="Transversion (Tv)")], fontsize=8)

    # 4.3 AF 直方图
    ax = axes[0][2]
    ax.hist(af_valid, bins=30, color="#2ca02c", alpha=0.8)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_title(f"Allele frequency (mean={af_mean:.3f}, med={af_median:.3f})")
    ax.set_xlabel("AF (INFO)")
    ax.set_ylabel("Variants")

    # 4.4 相对位置
    ax = axes[1][0]
    ax.hist(rel_pos, bins=40, color="#9467bd", alpha=0.8)
    ax.axvline(0.5, color="red", ls="--", lw=0.8, label="window center")
    ax.set_title("Relative position in window (0=start, 1=end)")
    ax.set_xlabel("(pos - start) / window_size")
    ax.set_ylabel("Variants")
    ax.legend(fontsize=8)

    # 4.5 GC 含量
    ax = axes[1][1]
    ax.hist(gc_valid, bins=40, color="#ff7f0e", alpha=0.8)
    ax.set_title(f"Window GC content (mean={gc_valid.mean():.3f}, std={gc_valid.std():.3f})")
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Windows")

    # 4.6 AF 分组条形
    ax = axes[1][2]
    bnames = list(af_bins)
    bvals = list(af_bins.values())
    ax.bar(bnames, bvals, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_title(f"AF groups (missing={stats['af']['missing']})")
    ax.set_ylabel("Variants")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle(f"Variant site statistics — {Path(args.bed).parent.name} ({len(regions)} sites)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    print(f"[stats] 图已保存 {out_png}")


if __name__ == "__main__":
    main()
