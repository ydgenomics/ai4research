"""从过滤 VCF 中采样目标样本携带变异的位点, 生成窗口 BED。

与 build_windows_bed 同格式: BED 0-based [pos-1-ws//2, pos-1+ws//2), 变异在 idx=ws//2-1。
name 列: {chrom}:{pos}:{ref}>{alt}。

用法:
    python sample_carriers.py --vcf data/251.SNP.bsnp.vcf.gz \
        --sample NH001 --genotype hom \
        --out data/carrier_hom.bed --max-variants 100000 --seed 42 --window-size 256
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cyvcf2
import numpy as np


def build_carrier_bed(
    vcf_path: Path,
    bed_path: Path,
    sample: str,
    genotype: str = "hom",
    window_size: int = 256,
    max_variants: int | None = None,
    seed: int = 42,
) -> int:
    """扫描 VCF, 收集目标样本携带变异的位点, 随机采样后写窗口 BED。

    genotype:
      - hom: 纯合 1/1 (两个拷贝都含 ALT, 无歧义)
      - het: 杂合 0/1 (至少一个拷贝含 ALT)
      - any: 携带任何 ALT
    返回写出的区间数。
    """
    ws = window_size
    half = ws // 2
    vcf = cyvcf2.VCF(str(vcf_path))
    si = vcf.samples.index(sample)

    rows: list[tuple[str, int, str, str]] = []  # (chrom, pos, ref, alt)
    n_total = 0
    t0 = time.time()
    for rec in vcf:
        n_total += 1
        a, b = rec.genotypes[si][0], rec.genotypes[si][1]
        if genotype == "hom":
            ok = a == 1 and b == 1
        elif genotype == "het":
            ok = (a == 1 and b == 0) or (a == 0 and b == 1)
        else:  # any
            ok = a > 0 or b > 0
        if ok:
            rows.append((rec.CHROM, rec.POS, rec.REF, rec.ALT[0]))
        if n_total % 1_000_000 == 0:
            print(f"  [scan] {n_total/1e6:.0f}M 行, 收集 {len(rows)}", flush=True)
    print(f"[scan] 完成: 总 {n_total}, {sample} 携带({genotype})位点 {len(rows)}, 耗时 {time.time()-t0:.0f}s")

    # 随机采样 (保持染色体/位置排序)
    idx = np.arange(len(rows))
    if max_variants is not None and max_variants < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_variants, replace=False)
        idx.sort()
    print(f"[scan] 采样 {len(idx)} 个位点 (seed={seed})")

    with open(bed_path, "w") as f:
        for i in idx:
            chrom, pos, ref, alt = rows[i]
            start = max(0, pos - half)
            end = start + ws
            name = f"{chrom}:{pos}:{ref}>{alt}"
            f.write(f"{chrom}\t{start}\t{end}\t{name}\n")
    print(f"[scan] BED写入: {bed_path} ({len(idx)} 区间)")
    return len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, help="过滤后双等位SNP VCF")
    ap.add_argument("--sample", required=True, help="目标样本名")
    ap.add_argument("--out", required=True, help="输出 BED 路径")
    ap.add_argument("--genotype", choices=["hom", "het", "any"], default="hom", help="位点类型 (默认 hom)")
    ap.add_argument("--window-size", type=int, default=256, help="窗口大小")
    ap.add_argument("--max-variants", type=int, default=None, help="采样上限 (省略则全量)")
    ap.add_argument("--seed", type=int, default=42, help="随机采样 seed")
    args = ap.parse_args()

    build_carrier_bed(
        Path(args.vcf), Path(args.out), args.sample,
        genotype=args.genotype, window_size=args.window_size,
        max_variants=args.max_variants, seed=args.seed,
    )


if __name__ == "__main__":
    main()
