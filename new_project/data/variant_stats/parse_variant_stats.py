#!/usr/bin/env python3
"""
解析 bcftools stats 输出 + 补充实测统计 → 生成结构化的描述性统计表格。

用法（在 new_project/data/variant_stats 下执行）:
    python3 parse_variant_stats.py

输入:
    bcftools_stats.txt  （bcftools stats 原始输出）
    samples.txt         （样本列表）
輸出:
    variant_summary.csv
    variant_by_chrom.csv
    allele_frequency.csv
    quality_dist.csv
    depth_dist.csv
    ts_tv_by_af.csv
    indel_length_dist.csv
    sample_counts.csv
    missing_by_sample.csv  （若提供 GT QC）
"""
import re
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = HERE / "bcftools_stats.txt"
SAMPLES = HERE / "samples.txt"

# ---------------------------------------------------------------------------
# 1. 概要（SN 块）
# ---------------------------------------------------------------------------
summary = {}
with open(STATS) as f:
    for line in f:
        if line.startswith("SN\t"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                summary[parts[2].rstrip(":")] = parts[3]  # 键如 "number of samples:" → 去冒号

def s(key):
    return int(summary.get(key, 0))

n_samples = s("number of samples")
n_records = s("number of records")
n_snps = s("number of SNPs")
n_indels = s("number of indels")
n_mnp = s("number of MNPs")
n_multi = s("number of multiallelic sites")

# ---------------------------------------------------------------------------
# 2. 染色体分布（每染色体的记录数来自 bcftools query 实测，见 run_all.sh）
#    文件 chrom_counts.csv: 两列 chrom,count（制表符分隔）
# ---------------------------------------------------------------------------
chrom_counts = {}
chrom_counts_file = HERE / "chrom_counts.tsv"
if chrom_counts_file.exists():
    with open(chrom_counts_file) as f:
        header = f.readline()  # 跳过表头
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                continue
            chrom_counts[parts[0]] = int(parts[1])

# ---------------------------------------------------------------------------
# 3. 等位基因频率（AF 块）
# ---------------------------------------------------------------------------
af_rows = []  # (af, n_snps, n_ts, n_tv, n_indels)
with open(STATS) as f:
    for line in f:
        if line.startswith("AF\t"):
            _, _, af, s_, ts, tv, ind_ = line.rstrip("\n").split("\t")[:7]
            af_rows.append((float(af), int(s_), int(ts), int(tv), int(ind_)))

# ---------------------------------------------------------------------------
# 4. QUAL 分布
# ---------------------------------------------------------------------------
qual_rows = []
with open(STATS) as f:
    for line in f:
        if line.startswith("QUAL\t"):
            parts = line.rstrip("\n").split("\t")
            qual_rows.append((float(parts[2]), int(parts[3]), int(parts[4]),
                              int(parts[5]), int(parts[6])))

# ---------------------------------------------------------------------------
# 5. DP 分布（每样本深度）
# ---------------------------------------------------------------------------
dp_rows = []
with open(STATS) as f:
    for line in f:
        if line.startswith("DP\t"):
            # DP [2]id [3]bin [4]n_genotypes [5]frac_genotypes [6]n_sites [7]frac_sites
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7:
                try:
                    bin_val = int(parts[2])
                except ValueError:
                    bin_val = -1  # '>500' 等末位 bin，统一记为 -1
                dp_rows.append((bin_val, int(parts[3]), float(parts[4]),
                                int(parts[5]), float(parts[6])))

# ---------------------------------------------------------------------------
# 6. 转变/颠换（TSTV）
# ---------------------------------------------------------------------------
tstv = None
with open(STATS) as f:
    for line in f:
        if line.startswith("TSTV\t"):
            parts = line.rstrip("\n").split("\t")
            tstv = {"ts": int(parts[2]), "tv": int(parts[3]), "ratio": float(parts[4])}
            break

# ---------------------------------------------------------------------------
# 7. indel 长度分布（IDD）
# ---------------------------------------------------------------------------
indel_rows = []
with open(STATS) as f:
    for line in f:
        if line.startswith("IDD\t"):
            parts = line.rstrip("\n").split("\t")
            indel_rows.append((int(parts[2]), int(parts[3]), int(parts[4]),
                               parts[5]))  # length, n_sites, n_genotypes, mean_vaf

# ---------------------------------------------------------------------------
# 8. 输出 CSV
# ---------------------------------------------------------------------------
def write_csv(name, header, rows):
    with open(HERE / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# 8.1 variant_summary.csv
write_csv("variant_summary.csv",
          ["metric", "value"],
          [
              ["number_of_samples", n_samples],
              ["number_of_records_total", n_records],
              ["number_of_SNP_sites", n_snps],
              ["number_of_indel_sites", n_indels],
              ["number_of_MNP_sites", n_mnp],
              ["number_of_multiallelic_sites", n_multi],
              ["Ts", tstv["ts"] if tstv else ""],
              ["Tv", tstv["tv"] if tstv else ""],
              ["Ts/Tv", f'{tstv["ratio"]:.4f}' if tstv else ""],
              ["avg_SNP_per_sample", f"{n_snps / n_samples:.1f}" if n_samples else ""],
              ["avg_indel_per_sample", f"{n_indels / n_samples:.1f}" if n_samples else ""],
          ])

# 8.2 allele_frequency.csv（AF 稀疏表 → 汇总 bins）
af_bins = defaultdict(lambda: [0, 0, 0, 0])  # bin -> [snps, ts, tv, indels]
for af, s_, ts, tv, ind_ in af_rows:
    if af < 0.005:
        bin_name = "0-0.005"
    elif af < 0.01:
        bin_name = "0.005-0.01"
    elif af < 0.05:
        bin_name = "0.01-0.05"
    elif af < 0.1:
        bin_name = "0.05-0.1"
    elif af < 0.5:
        bin_name = "0.1-0.5"
    else:
        bin_name = "0.5-1.0"
    af_bins[bin_name][0] += s_
    af_bins[bin_name][1] += ts
    af_bins[bin_name][2] += tv
    af_bins[bin_name][3] += ind_

order = ["0-0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.5", "0.5-1.0"]
write_csv("allele_frequency.csv",
          ["af_bin", "n_SNPs", "n_transitions", "n_transversions", "n_indels"],
          [[b] + af_bins[b] for b in order])

# ---------------------------------------------------------------------------
# 8b. QUAL 分布分箱（避免 24 万行裸输出）
# ---------------------------------------------------------------------------
qual_summary = []
if qual_rows:
    qvals = [q for q, *_ in qual_rows]
    qmin, qmax = min(qvals), max(qvals)
    # 按 [0,30,50,100,200,500,1000,∞) 分箱
    q_edges = [0, 30, 50, 100, 200, 500, 1000]
    q_bins = {}
    for q, s_, ts, tv, ind_ in qual_rows:
        b = None
        for i, e in enumerate(q_edges):
            if q < e:
                b = f"{q_edges[i-1] if i else 0}-{e}"
                break
        if b is None:
            b = ">1000"
        q_bins.setdefault(b, [0, 0, 0, 0])
        q_bins[b][0] += s_; q_bins[b][1] += ts; q_bins[b][2] += tv; q_bins[b][3] += ind_
    q_order = ["0-30", "30-50", "50-100", "100-200", "200-500", "500-1000", ">1000"]
    q_rows = []
    for b in q_order:
        if b in q_bins:
            q_rows.append([b] + q_bins[b])
    write_csv("quality_binned.csv",
              ["qual_bin", "n_SNPs", "n_ts", "n_tv", "n_indels"], q_rows)
    total_q = sum(v[0] for v in q_bins.values()) or 1
    qual_summary = [(f"min_qual", qmin), (f"max_qual", qmax),
                    (f"snp_density_below30", f'{100*q_bins.get("0-30",[0])[0]/total_q:.3f}'),
                    (f"snp_density_below50", f'{100*(q_bins.get("0-30",[0])[0]+q_bins.get("30-50",[0])[0])/total_q:.3f}')]
# QUAL 分位数（5/10/25/50/75/90/95/99 百分位）
qual_sorted = sorted(qual_rows)
nq = len(qual_sorted)
q_at = {}
if nq:
    for i in (5, 10, 25, 50, 75, 90, 95, 99):
        idx = i * nq // 100
        q_at[str(i)] = qual_sorted[idx][0]
write_csv("quality_quantiles.csv",
          ["percentile", "value"],
          [[k, v] for k, v in q_at.items()])

# 8.4 depth_dist.csv（去掉无意义的零堆积，输出深度>=1）
# 列: bin, n_genotypes, frac_genotypes(%), n_sites, frac_sites(%)
write_csv("depth_dist.csv",
          ["depth", "n_genotypes", "frac_genotypes_pct", "n_sites", "frac_sites_pct"],
          [[d, g, f"{fg:.6f}", s_, f"{fs:.6f}"]
           for d, g, fg, s_, fs in dp_rows if d >= 1][:500])  # 上限防爆

# 8.5 ts_tv_by_af.csv
write_csv("ts_tv_by_af.csv",
          ["af_bin", "ts", "tv", "ts_tv"],
          [[b, af_bins[b][1], af_bins[b][2],
            f"{(af_bins[b][1] / af_bins[b][2]):.4f}" if af_bins[b][2] else "NA"]
           for b in order])

# 8.6 indel_length_dist.csv
write_csv("indel_length_dist.csv",
          ["indel_length", "n_sites", "n_genotypes", "mean_vaf"],
          [[l, s_, g_, v] for l, s_, g_, v in indel_rows])

# 8.7 SNV 替换类型（ST）
sub_rows = []
with open(STATS) as f:
    for line in f:
        if line.startswith("ST\t"):
            parts = line.rstrip("\n").split("\t")
            sub_rows.append((parts[2], int(parts[3])))
write_csv("substitution_types.csv", ["type", "count"], sub_rows)

# 8.8 singleton (SiS)
sis_rows = []
sis_total_snp = 0
sis_total_indel = 0
with open(STATS) as f:
    for line in f:
        if line.startswith("SiS\t"):
            parts = line.rstrip("\n").split("\t")
            sis_rows.append((int(parts[2]), int(parts[3]), int(parts[4]),
                             int(parts[5]), int(parts[6])))
            sis_total_snp += int(parts[3])
            sis_total_indel += int(parts[6])
write_csv("singleton_stats.csv",
          ["allele_count", "n_SNPs", "n_ts", "n_tv", "n_indels"], sis_rows)

# 8.9 染色体分布
if chrom_counts:
    total_chrom = sum(chrom_counts.values()) or 1
    write_csv("variant_by_chrom.csv",
              ["chrom", "n_records", "pct"],
              [[c, n, f"{100*n/total_chrom:.3f}"]
               for c, n in sorted(chrom_counts.items(), key=lambda x: -x[1])])
print("\n[OK] 全部 CSV 已生成")
print(f"  记录数     : {n_records:,}")
print(f"  SNP 位点   : {n_snps:,}")
print(f"  Indel 位点 : {n_indels:,}")
print(f"  MNP 位点   : {n_mnp:,}")
if n_records:
    print(f"  多等位位点 : {n_multi:,} ({100*n_multi/n_records:.2f}%)")
print(f"  Ts/Tv      : {tstv['ratio']:.4f}" if tstv else "  Ts/Tv : NA")
print(f"  AF 分箱行  : {len(order)} 类, 累计 SNP = {sum(af_bins[b][0] for b in order):,}")
print(f"  Singleton SNP/Indel : {sis_total_snp:,} / {sis_total_indel:,}")
if chrom_counts:
    print(f"  染色体数   : {len(chrom_counts)}")