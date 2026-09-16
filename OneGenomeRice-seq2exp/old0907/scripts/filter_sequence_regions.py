#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_sequence_regions.py
==========================
按染色体区域(百分比)过滤序列窗口，并同步更新索引文件。

背景
----
data_prepare.sh 生成的每个 multitrack 目录包含:
  - sequence_split_train.csv : 全基因组滑动窗口列表 (chromosome, start, end)
  - bigWig_labels_meta.csv   : track 级元数据 (target_file_name, nonzero_mean, ...)
  - index_stat.json          : 数据集统计 (counts.num_samples 等)

本脚本按染色体长度百分比过滤窗口(例如只保留后半段 50%~100%)，
同步更新 index_stat.json 的样本计数; 可选用保留区域重算 track 级
nonzero_mean 并回写 bigWig_labels_meta.csv 与 index_stat.json。

用法
----
python filter_sequence_regions.py --indices-dir DIR [DIR...] [选项]

示例
----
# 1) 只保留每条染色体后半段窗口, 原地覆盖
python filter_sequence_regions.py --indices-dir data/indices --region 50 100

# 2) 先预览 10%~40% 区域过滤结果, 不写盘
python filter_sequence_regions.py --indices-dir data/indices --region 10 40 --dry-run

# 3) 只过滤训练目录并保留 .bak 备份
python filter_sequence_regions.py \
    --indices-dir "data/indices/train_*_multitrack" \
    --region 50 100 --backup-suffix .bak

# 4) 同时用保留区域重算 nonzero_mean (需 pyBigWig)
python filter_sequence_regions.py --indices-dir data/indices \
    --region 50 100 --recompute-track-stats
"""

import argparse
import glob
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

SEQ_CSV = "sequence_split_train.csv"
META_CSV = "bigWig_labels_meta.csv"
STAT_JSON = "index_stat.json"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def is_index_dir(d):
    return (os.path.isfile(os.path.join(d, SEQ_CSV))
            and os.path.isfile(os.path.join(d, STAT_JSON)))


def find_index_dirs(patterns, max_depth=2):
    """展开 glob 模式, 返回包含 sequence_split_train.csv 的目录列表。

    规则:
      1. glob 展开 / 目录路径匹配;
      2. 匹配到的目录若直接包含索引文件 -> 视为 multitrack 目录, 直接收录;
      3. 否则视为父目录, 向下递归(最多 max_depth 层)查找 multitrack 子目录。
    """
    dirs = []

    def walk(d, depth):
        if depth < 0:
            return
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            return
        for name in entries:
            if name.startswith("."):
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                if is_index_dir(full):
                    dirs.append(full)
                else:
                    walk(full, depth - 1)

    for pat in patterns:
        pat = os.path.expanduser(os.path.expandvars(pat))
        matched = glob.glob(pat)
        if not matched and os.path.isdir(pat):
            matched = [pat]
        for m in matched:
            if not os.path.isdir(m):
                continue
            if is_index_dir(m):
                dirs.append(m)
            else:
                walk(m, max_depth - 1)

    if not dirs:
        raise SystemExit(f"[error] 未找到包含 {SEQ_CSV} 与 {STAT_JSON} 的目录: {patterns}")
    return sorted(set(dirs))


def chrom_lengths(df):
    """窗口坐标 0-based 半开区间, 最后一个窗口 end 即染色体长度。"""
    return df.groupby("chromosome")["end"].max().to_dict()


def filter_windows(df, lo_pct, hi_pct, mode="center"):
    """按染色体长度百分比过滤窗口。lo_pct/hi_pct 为 0~100。"""
    chrom_len = chrom_lengths(df)
    keep = pd.Series(False, index=df.index)
    for chrom, clen in chrom_len.items():
        lo_bound = clen * lo_pct / 100.0
        hi_bound = clen * hi_pct / 100.0
        mask = df["chromosome"] == chrom
        starts = df.loc[mask, "start"].to_numpy()
        ends = df.loc[mask, "end"].to_numpy()
        if mode == "center":
            mid = (starts + ends) / 2.0
            sel = (mid >= lo_bound) & (mid <= hi_bound)
        elif mode == "overlap":
            # 窗口与目标区间有交集
            sel = (starts < hi_bound) & (ends > lo_bound)
        else:
            raise ValueError(f"未知 keep-mode: {mode}")
        keep.loc[mask] = sel
    return df[keep].copy()


def backup_files(index_dir, suffix):
    """覆盖前将三个索引文件复制为 *.suffix。"""
    if not suffix:
        return
    for name in (SEQ_CSV, META_CSV, STAT_JSON):
        src = os.path.join(index_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, src + suffix)
    print(f"  [backup] 已备份到 *{suffix}")


def load_stat(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_stat(path, stat):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stat, f, indent=2, ensure_ascii=False)


def recompute_nonzero_means(df, stat, meta_df):
    """用保留窗口区域重新计算每条 track 的 nonzero_mean (value>0 的均值)。"""
    try:
        import pyBigWig
    except ImportError:
        raise SystemExit("[error] 重新计算 nonzero_mean 需要 pyBigWig")

    bw_dir = stat["inputs"]["processed_bw_dir"]
    target_files = list(stat["counts"].get("target_file_name", []))
    means = []
    for tf in target_files:
        path = os.path.join(bw_dir, tf)
        bw = pyBigWig.open(path)
        total, cnt = 0.0, 0
        for row in df.to_dict("records"):
            vals = np.array(
                bw.values(str(row["chromosome"]), int(row["start"]), int(row["end"])),
                dtype=np.float64,
            )
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            nz = vals[vals > 0]
            if nz.size:
                total += float(nz.sum())
                cnt += int(nz.size)
        bw.close()
        means.append(total / cnt if cnt else 0.0)

    # meta 中 target_file_name 顺序与 index_stat 一致(同一生成流程)
    meta_col = "nonzero_mean"
    if meta_col in meta_df.columns:
        name_to_mean = dict(zip(target_files, means))
        meta_df[meta_col] = meta_df["target_file_name"].map(name_to_mean).fillna(0.0)
    return means


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_dir(index_dir, lo_pct, hi_pct, mode, dry_run, recompute):
    print(f"\n=== {index_dir} ===")
    seq_path = os.path.join(index_dir, SEQ_CSV)
    stat_path = os.path.join(index_dir, STAT_JSON)
    meta_path = os.path.join(index_dir, META_CSV)

    df = pd.read_csv(seq_path)
    before = int(len(df))
    if before == 0:
        print("  [skip] 空窗口列表")
        return None

    n_chrom_before = df["chromosome"].nunique()
    kept = filter_windows(df, lo_pct, hi_pct, mode)
    after = int(len(kept))
    pct_keep = 100.0 * after / before if before else 0.0
    print(f"  [region] {lo_pct}%~{hi_pct}% (mode={mode})  "
          f"{before} -> {after} 窗口 (保留 {pct_keep:.1f}%)")

    if kept.empty:
        print("  [error] 过滤后为空, 终止处理该目录(不写文件)")
        return None

    if dry_run:
        print("  [dry-run] 不写盘")
        return {"before": before, "after": after}

    # 1) 覆盖 sequence_split_train.csv
    kept.to_csv(seq_path, index=False)

    # 2) 同步更新 index_stat.json 计数
    stat = load_stat(stat_path)
    by_chrom = kept["chromosome"].value_counts().to_dict()
    by_chrom = {str(k): int(v) for k, v in by_chrom.items()}
    stat["counts"]["num_samples"] = after
    stat["counts"]["num_samples_by_chromosome"] = by_chrom
    stat["filtered"] = {
        "region_percent": [lo_pct, hi_pct],
        "keep_mode": mode,
        "num_windows_before": before,
        "num_windows_after": after,
        "filtered_at": datetime.now().isoformat(timespec="seconds"),
    }
    if n_chrom_before != len(by_chrom):
        print(f"  [warn] 染色体数变化: {n_chrom_before} -> {len(by_chrom)}")
    save_stat(stat_path, stat)
    print(f"  已更新 {STAT_JSON}: num_samples={after}, "
          f"num_samples_by_chromosome={by_chrom}")

    # 3) 可选: 重算 track 级 nonzero_mean
    if recompute:
        if not os.path.isfile(meta_path):
            print("  [skip] 未找到 bigWig_labels_meta.csv, 跳过 nonzero_mean 重算")
        else:
            meta_df = pd.read_csv(meta_path)
            means = recompute_nonzero_means(kept, stat, meta_df)
            meta_df.to_csv(meta_path, index=False)
            stat["counts"]["nonzero_mean"] = [float(m) for m in means]
            save_stat(stat_path, stat)
            print(f"  已重算并回写 {META_CSV} / {STAT_JSON}: nonzero_mean={means}")

    return {"before": before, "after": after}


def main():
    parser = argparse.ArgumentParser(
        description="按染色体区域(百分比)过滤窗口, 同步更新 index_stat.json / bigWig_labels_meta.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--indices-dir", nargs="+", default=["data/indices"],
                        help="multitrack 目录路径或 glob; 也可传父目录(自动递归发现子目录), 可传多个")
    parser.add_argument("--region", nargs=2, type=float, metavar=("LO", "HI"),
                        default=[50.0, 100.0],
                        help="保留区域占染色体长度的百分比区间 [LO, HI] (0~100)")
    parser.add_argument("--keep-mode", choices=["center", "overlap"], default="center",
                        help="center: 窗口中心落在区域内; overlap: 窗口与区域有交集")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="传入父目录时, 向下递归查找 multitrack 子目录的最大深度")
    parser.add_argument("--dry-run", action="store_true", help="只打印不写盘")
    parser.add_argument("--backup-suffix", default="",
                        help="覆盖前备份文件后缀, 例如 .bak (留空则不备份)")
    parser.add_argument("--recompute-track-stats", action="store_true",
                        help="用保留区域重算 nonzero_mean 并回写 meta/json")
    args = parser.parse_args()

    lo, hi = args.region
    if not (0.0 <= lo <= hi <= 100.0):
        raise SystemExit(f"[error] --region 需满足 0 <= LO <= HI <= 100, 当前 {lo}~{hi}")

    index_dirs = find_index_dirs(args.indices_dir, max_depth=args.max_depth)
    print(f"共找到 {len(index_dirs)} 个 multitrack 目录")

    summary = []
    for d in index_dirs:
        if not args.dry_run and args.backup_suffix:
            backup_files(d, args.backup_suffix)
        res = process_dir(d, lo, hi, args.keep_mode, args.dry_run,
                          args.recompute_track_stats)
        if res:
            summary.append((d, res["before"], res["after"]))

    print("\n==================== 汇总 ====================")
    print(f"{'目录':<60} {'过滤前':>10} {'过滤后':>10} {'保留%':>8}")
    for d, b, a in summary:
        print(f"{d:<60} {b:>10} {a:>10} {100.0*a/b if b else 0.0:>7.1f}%")
    if not args.dry_run:
        print("完成。注意: 若之后训练需要 chromosome_features, "
              "建议用过滤后的 sequence_split 重建(见 build_chromosome_features.py)。")


if __name__ == "__main__":
    main()