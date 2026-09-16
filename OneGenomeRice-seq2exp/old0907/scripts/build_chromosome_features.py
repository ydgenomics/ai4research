#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build chromosome-level feature JSON for chromosome feature embedding.

Use only training splits/index_stats to avoid validation/test leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 chromosome_features.json")
    parser.add_argument("--sequence_splits", nargs="+", required=True,
                        help="训练集 sequence_split_train.csv，可传多个时期")
    parser.add_argument("--index_stats", nargs="+", required=True,
                        help="训练集 index_stat.json，数量需与 sequence_splits 一致")
    parser.add_argument("--chromosomes", nargs="+", required=True,
                        help="要统计的染色体，如 Chr01 Chr03 Chr05 Chr07 Chr09 Chr11")
    parser.add_argument("--output", required=True,
                        help="输出 chromosome_features.json")
    parser.add_argument("--max_windows_per_chrom", type=int, default=128,
                        help="每个时期、每条染色体最多抽样窗口数，默认 128")
    parser.add_argument("--max_values_per_track_chrom", type=int, default=200000,
                        help="每个 track/染色体最多保留多少非零值用于近似分位数，默认 200000")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_chrom(chrom: str) -> str:
    match = re.search(r"(?:Chr|chr)0?(1[0-2]|[1-9])(?!\d)", str(chrom))
    if match:
        return f"Chr{int(match.group(1)):02d}"
    return str(chrom)


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def safe_log1p(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.log1p(max(float(x), 0.0)))


class RunningStats:
    def __init__(self):
        self.total = 0
        self.zero = 0
        self.nonzero = 0
        self.nonzero_sum = 0.0
        self.sum = 0.0
        self.sumsq = 0.0
        self.max_value = 0.0
        self.active_windows = 0
        self.windows = 0
        self.values = []

    def update(self, arr: np.ndarray, max_values: int, rng: np.random.Generator):
        arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        self.windows += 1
        self.total += int(arr.size)
        self.zero += int(np.count_nonzero(arr <= 0))
        self.sum += float(arr.sum())
        self.sumsq += float(np.square(arr, dtype=np.float64).sum())
        if arr.size:
            self.max_value = max(self.max_value, float(arr.max()))

        nz = arr[arr > 0]
        self.nonzero += int(nz.size)
        if nz.size:
            self.active_windows += 1
            self.nonzero_sum += float(nz.sum())
            capacity = max_values - len(self.values)
            if capacity > 0:
                take = min(capacity, nz.size)
                if take < nz.size:
                    nz = rng.choice(nz, size=take, replace=False)
                self.values.extend(float(x) for x in nz)

    def summary(self) -> Dict[str, float]:
        zero_ratio = self.zero / self.total if self.total else 0.0
        mean = self.sum / self.total if self.total else 0.0
        variance = max(self.sumsq / self.total - mean * mean, 0.0) if self.total else 0.0
        nonzero_mean = self.nonzero_sum / self.nonzero if self.nonzero else 0.0
        active_window_rate = self.active_windows / self.windows if self.windows else 0.0
        values = np.asarray(self.values, dtype=np.float32)
        if values.size:
            median = float(np.quantile(values, 0.50))
            q95 = float(np.quantile(values, 0.95))
            q99 = float(np.quantile(values, 0.99))
        else:
            median = q95 = q99 = 0.0
        return {
            "zero_ratio": float(zero_ratio),
            "mean_log1p": safe_log1p(mean),
            "std_log1p": safe_log1p(math.sqrt(variance)),
            "nonzero_mean_log1p": safe_log1p(nonzero_mean),
            "nonzero_median_log1p": safe_log1p(median),
            "q95_log1p": safe_log1p(q95),
            "q99_log1p": safe_log1p(q99),
            "max_log1p": safe_log1p(self.max_value),
            "active_window_rate": float(active_window_rate),
        }


def choose_windows(df: pd.DataFrame, chrom_key: str, max_windows: int, seed: int) -> pd.DataFrame:
    sub = df[df["chromosome"].map(canonical_chrom) == chrom_key].copy()
    if max_windows > 0 and len(sub) > max_windows:
        sub = sub.sample(n=max_windows, random_state=seed)
    return sub.sort_values(["chromosome", "start", "end"])


def build_features(args: argparse.Namespace) -> dict:
    if len(args.sequence_splits) != len(args.index_stats):
        raise ValueError("--sequence_splits 和 --index_stats 数量必须一致")

    rng = np.random.default_rng(args.seed)
    chrom_keys = [canonical_chrom(c) for c in args.chromosomes]
    chrom_features = {chrom: defaultdict(float) for chrom in chrom_keys}
    signal_stats = {chrom: {} for chrom in chrom_keys}
    dataset_counts = defaultdict(int)

    for split_path, stat_path in zip(args.sequence_splits, args.index_stats):
        stat = read_json(stat_path)
        df = pd.read_csv(split_path)
        fasta = pyfaidx.Fasta(stat["inputs"]["genome_fasta"])
        bw_dir = stat["inputs"]["processed_bw_dir"]
        target_files = list(stat["counts"]["target_file_name"])
        biosamples = list(stat["counts"].get("biosample_order", []))
        if len(biosamples) != len(target_files):
            biosamples = [Path(name).stem for name in target_files]

        bw_handles = {}
        try:
            for target_file in target_files:
                bw_handles[target_file] = pyBigWig.open(os.path.join(bw_dir, target_file))

            for chrom_key in chrom_keys:
                windows = choose_windows(df, chrom_key, args.max_windows_per_chrom, args.seed)
                if windows.empty:
                    continue
                dataset_counts[chrom_key] += 1

                gc_count = 0
                n_count = 0
                seq_total = 0
                chrom_length = 0
                per_track_stats = {biosample: RunningStats() for biosample in biosamples}

                for _, row in windows.iterrows():
                    chrom = str(row["chromosome"])
                    start = int(row["start"])
                    end = int(row["end"])
                    chrom_length = max(chrom_length, end)

                    seq = str(fasta[chrom][start:end]).upper()
                    seq_total += len(seq)
                    gc_count += seq.count("G") + seq.count("C")
                    n_count += seq.count("N")

                    for biosample, target_file in zip(biosamples, target_files):
                        vals = np.asarray(bw_handles[target_file].values(chrom, start, end), dtype=np.float32)
                        per_track_stats[biosample].update(vals, args.max_values_per_track_chrom, rng)

                chrom_features[chrom_key]["chrom_length_log1p"] += safe_log1p(chrom_length)
                chrom_features[chrom_key]["gc_ratio"] += gc_count / seq_total if seq_total else 0.0
                chrom_features[chrom_key]["n_ratio"] += n_count / seq_total if seq_total else 0.0
                chrom_features[chrom_key]["sampled_windows_log1p"] += safe_log1p(len(windows))

                for biosample, stats in per_track_stats.items():
                    signal_stats[chrom_key].setdefault(biosample, []).append(stats.summary())
        finally:
            try:
                fasta.close()
            except Exception:
                pass
            for bw in bw_handles.values():
                try:
                    bw.close()
                except Exception:
                    pass

    output_features = {}
    feature_names = [
        "chrom_length_log1p",
        "gc_ratio",
        "n_ratio",
        "sampled_windows_log1p",
    ]
    stat_names = [
        "zero_ratio",
        "mean_log1p",
        "std_log1p",
        "nonzero_mean_log1p",
        "nonzero_median_log1p",
        "q95_log1p",
        "q99_log1p",
        "max_log1p",
        "active_window_rate",
    ]

    biosample_names = sorted({b for chrom in signal_stats.values() for b in chrom.keys()})
    for biosample in biosample_names:
        for name in stat_names:
            feature_names.append(f"{biosample}_{name}")
    for name in stat_names:
        feature_names.append(f"all_tracks_{name}")

    for chrom in chrom_keys:
        denom = max(dataset_counts[chrom], 1)
        base = {
            "chrom_length_log1p": chrom_features[chrom]["chrom_length_log1p"] / denom,
            "gc_ratio": chrom_features[chrom]["gc_ratio"] / denom,
            "n_ratio": chrom_features[chrom]["n_ratio"] / denom,
            "sampled_windows_log1p": chrom_features[chrom]["sampled_windows_log1p"] / denom,
        }

        values_by_feature = dict(base)
        all_track_summaries = []
        for biosample in biosample_names:
            summaries = signal_stats[chrom].get(biosample, [])
            for name in stat_names:
                if summaries:
                    value = float(np.mean([s[name] for s in summaries]))
                else:
                    value = 0.0
                values_by_feature[f"{biosample}_{name}"] = value
            all_track_summaries.extend(summaries)

        for name in stat_names:
            if all_track_summaries:
                value = float(np.mean([s[name] for s in all_track_summaries]))
            else:
                value = 0.0
            values_by_feature[f"all_tracks_{name}"] = value

        output_features[chrom] = [float(values_by_feature.get(name, 0.0)) for name in feature_names]

    return {
        "feature_names": feature_names,
        "features": output_features,
        "metadata": {
            "sequence_splits": args.sequence_splits,
            "index_stats": args.index_stats,
            "chromosomes": chrom_keys,
            "max_windows_per_chrom": args.max_windows_per_chrom,
            "max_values_per_track_chrom": args.max_values_per_track_chrom,
            "note": "Use training data only. train_zeroloss.py will z-score these features by default.",
        },
    }


def main() -> None:
    args = parse_args()
    features = build_features(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(features, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved chromosome features: {out}")
    print(f"Feature dim: {len(features['feature_names'])}")
    print(f"Chromosomes: {', '.join(features['features'].keys())}")


if __name__ == "__main__":
    main()
