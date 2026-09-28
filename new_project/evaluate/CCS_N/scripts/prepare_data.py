"""CCS-N 数据准备: 过滤 VCF + 生成窗口 BED + GVL 数据集写入。

流程:
1. 用 cyvcf2 全量扫描原 VCF, 保留双等位 SNP (rec.is_snp and len(rec.ALT)==1)
   输出到 data/251.SNP.bsnp.vcf.gz
2. 从过滤后的 VCF 读取全部 SNP 位点, 以每个 SNP 为中心生成 256bp 窗口 BED
   (窗口默认 256bp, 中心 SNP 位于窗口内 idx=127 处, 可通过 --window-size 修改)
3. 调用 gvl.write 将 BED + 过滤 VCF 预处理成 GVL 数据集
   (只含目标样本, 可通过 --samples 指定)

用法:
    python prepare_data.py \
        --vcf /path/251.SNP.final.vcf.gz \
        --ref /path/osa1_r7.asm.ch.fa \
        --out-dir ./data \
        --samples NH001 \
        --window-size 256 \
        --max-variants 5000      # 冒烟测试用; 省略则全量
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import time
from pathlib import Path

import cyvcf2


def filter_biallelic_snp(vcf_path: Path, out_path: Path) -> int:
    """过滤: 保留双等位 SNP, 返回保留数。"""
    if out_path.exists():
        print(f"[prepare] 过滤VCF已存在: {out_path}")
        _build_tabix_index(out_path)
        return _count_variants(out_path)
    t0 = time.time()
    reader = cyvcf2.VCF(str(vcf_path))
    writer = cyvcf2.Writer(str(out_path), reader)
    n_keep = n_total = 0
    for rec in reader:
        n_total += 1
        if rec.is_snp and len(rec.ALT) == 1:
            writer.write_record(rec)
            n_keep += 1
        if n_total % 1_000_000 == 0:
            print(f"  [prepare] {n_total} 行, 保留 {n_keep}, {time.time()-t0:.0f}s", flush=True)
    writer.close()
    print(f"[prepare] VCF过滤完成: 总 {n_total}, 保留双等位SNP {n_keep}, 耗时 {time.time()-t0:.0f}s")
    _build_tabix_index(out_path)
    return n_keep


def _build_tabix_index(vcf_path: Path) -> None:
    """为过滤后的 bgzip VCF 建 tabix 索引 (GVL/genoray 依赖 .tbi)。"""
    import pysam

    tbi = Path(str(vcf_path) + ".tbi")
    if tbi.exists():
        print(f"[prepare] tabix索引已存在: {tbi}")
        return
    t0 = time.time()
    pysam.tabix_index(str(vcf_path), preset="vcf", force=True)
    print(f"[prepare] tabix索引完成: {tbi} ({time.time()-t0:.0f}s)")


def _count_variants(vcf_path: Path) -> int:
    """统计 VCF 变异数 (快路径: 用 gzip 读非注释行计数, 仅作日志用)。"""
    n = 0
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                n += 1
    return n


def build_windows_bed(vcf_path: Path, bed_path: Path, window_size: int, max_variants: int | None) -> int:
    """以每个双等位 SNP 为中心生成窗口 BED。

    BED 0-based: [pos-1-ws//2, pos-1+ws//2) 保证 SNP 位于 idx = ws//2 - 1 处
    (窗口长度 ws, 变异在索引 ws//2 - 1, 即左右各 ws//2-1 bp + 变异1bp)。
    若窗口越界则平移贴边。
    name 列: {chrom}:{pos}:{ref}>{alt} 供后续对齐变异位置。
    返回窗口数。
    """
    ws = window_size
    half = ws // 2
    n = 0
    with gzip.open(vcf_path, "rt") as fin, open(bed_path, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = parts[0], int(parts[1]), parts[3], parts[4]
            # pos 1-based -> 窗口 0-based, 保证变异位于 idx = half-1 (窗口中心偏左1)
            # start = pos - 1 - (half-1) = pos - half
            start = max(0, pos - half)
            end = start + ws
            name = f"{chrom}:{pos}:{ref}>{alt}"
            fout.write(f"{chrom}\t{start}\t{end}\t{name}\n")
            n += 1
            if max_variants is not None and n >= max_variants:
                break
    print(f"[prepare] 窗口BED: {n} 个区间 -> {bed_path}")
    return n


def write_gvl_dataset(
    bed_path: Path,
    vcf_path: Path,
    ref_path: Path,
    out_dir: Path,
    samples: list[str] | None,
) -> None:
    """gvl.write 预处理。"""
    import genvarloader as gvl

    t0 = time.time()
    gvl.write(
        str(out_dir),
        bed=str(bed_path),
        variants=str(vcf_path),
        samples=samples,
        overwrite=True,
    )
    print(f"[prepare] GVL写入完成: {out_dir} ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf", required=True, help="原始 VCF (可含 multi-allelic)")
    ap.add_argument("--ref", required=True, help="参考基因组 FASTA")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "data"), help="输出目录")
    ap.add_argument("--samples", nargs="*", default=None, help="样本列表, 默认全部")
    ap.add_argument("--window-size", type=int, default=256, help="窗口大小 (默认256)")
    ap.add_argument("--max-variants", type=int, default=None, help="最多处理变异数 (冒烟测试用)")
    ap.add_argument("--skip-gvl", action="store_true", help="只生成过滤VCF和BED, 不写GVL")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = Path(args.vcf)

    # 1. 过滤 VCF
    filtered_vcf = out_dir / "251.SNP.bsnp.vcf.gz"
    filter_biallelic_snp(vcf_path, filtered_vcf)

    # 2. 窗口 BED
    bed_path = out_dir / "windows.bed"
    build_windows_bed(filtered_vcf, bed_path, args.window_size, args.max_variants)

    # 3. GVL 写入
    if not args.skip_gvl:
        gvl_dir = out_dir / "gvl"
        if gvl_dir.exists():
            shutil.rmtree(gvl_dir)
        write_gvl_dataset(bed_path, filtered_vcf, Path(args.ref), gvl_dir, args.samples)

    print("[prepare] 完成。")


if __name__ == "__main__":
    main()
