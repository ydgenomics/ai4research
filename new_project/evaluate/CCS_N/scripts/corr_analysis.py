"""CCS-N 得分 × 位点属性关联分析。

输入:
- output/<dir>/per_site/<model>/ccs_n_per_site.npz (per-site CCS-N)
- data/carrier_hom.bed (位点信息)
- data/251.SNP.bsnp.vcf.gz (AF)

输出:
1. 各模型 per-site CCS-N 与 AF/TiTv/GC 的 Spearman 相关
2. hardest/easiest 位点的属性对比
3. 各模型 top-10 差异位点 (某模型认为最难但其他模型认为容易 → 模型特异性)

用法:
    python scripts/corr_analysis.py output/hom_1000_multi7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ROOT = out_dir.parent.parent
    from heatmap_analysis import load_per_site
    from variant_stats import load_vcf_info, parse_bed

    # 1. 加载数据
    print("[corr] 加载 per-site 分数 ...")
    models, real, rand = load_per_site(out_dir, (0, 1, 2, 4, 8, 16), "euclidean", 16)
    M, B = real.shape

    bed_path = ROOT / "data" / "carrier_hom.bed"
    vcf_path = ROOT / "data" / "251.SNP.bsnp.vcf.gz"
    regions = parse_bed(str(bed_path))
    info = load_vcf_info(str(vcf_path), regions=regions)

    # 2. 位点属性
    refs, alts, afs, gcs = [], [], [], []
    for r in regions:
        chrom, start, end, chrom_n, pos, ref, alt = r
        af = info.get((chrom, pos), (None, None, None))[2]
        afs.append(af if af is not None else np.nan)
        refs.append(ref)
        alts.append(alt)
    refs, alts = np.array(refs), np.array(alts)
    afs = np.array(afs, dtype=float)

    # Ti/Tv
    titv = np.array([1 if {r, a} in ({"A", "G"}, {"C", "T"}) else 0 for r, a in zip(refs, alts)])
    # 替换类型 → 碱基类别 (嘌呤/嘧啶)
    purine = {"A", "G"}
    is_tv = 1 - titv

    # 3. 相关分析
    from scipy.stats import spearmanr
    print("\n===== per-site CCS-N (euclidean N=16) 与位点属性的 Spearman 相关 =====")
    print(f"{'模型':<42s} {'AF':>8s} {'Ti/Tv(1=Ti)':>12s}")
    corr_table = {}
    for i, name in enumerate(models):
        c_af = spearmanr(real[i][~np.isnan(afs)], afs[~np.isnan(afs)]).statistic
        c_tv = spearmanr(real[i], titv).statistic
        corr_table[name] = {"AF": c_af, "Ti/Tv": c_tv}
        print(f"{name[:40]:<42s} {c_af:>8.3f} {c_tv:>12.3f}")

    # 4. 模型间 per-site 一致性 (哪些位点模型间分歧大)
    std_across = real.std(axis=0)
    rank_std = np.argsort(std_across)[::-1]  # 分歧从大到小
    print("\n===== 模型间分歧最大的 10 个位点 (各模型对该位点 CCS-N) =====")
    print(f"{'位点':<24s} {'std':>6s} " + "".join(f"{m[:8]:>10s}" for m in models))
    for idx in rank_std[:10]:
        r = regions[idx]
        line = f"{r[0]}:{r[4]}:{r[5]}>{r[6]:<8s} {std_across[idx]:>6.1f} "
        line += "".join(f"{real[i, idx]:>10.1f}" for i in range(M))
        print(line)

    # 5. 各模型独有"难点"位点: 该模型得分排名前 10% 中, 其他模型得分排名后 50% 的位点
    print("\n===== 模型特异性难点位点 (该模型 rank top-5%, 其他模型 median 以下) =====")
    for i, name in enumerate(models):
        my_rank = np.argsort(np.argsort(real[i]))  # 升序排名
        others_median = np.median(np.delete(real, i, axis=0), axis=0)
        cand = np.where((my_rank >= B * 0.95) & (others_median < np.median(others_median)))[0]
        if len(cand):
            idx = cand[np.argmax(real[i, cand])]
            r = regions[idx]
            print(f"  {name[:32]:<34s} {r[0]}:{r[4]}:{r[5]}>{r[6]}  "
                  f"AF={afs[idx]:.3f}  my_ccsn={real[i, idx]:.1f}  others_med={others_median[idx]:.1f}")


if __name__ == "__main__":
    main()
