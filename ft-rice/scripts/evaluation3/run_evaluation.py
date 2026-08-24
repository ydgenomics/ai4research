#!/usr/bin/env python3
"""
run_evaluation.py — 基因表达预测评估 v3（单脚本版）

用法:
  python run_evaluation.py --config config.yaml

输出:
  00_main_summary.csv             主表 (sample + chromosome 全局)
  00_window_level.csv             窗口全局表 (每窗口一行)
  00_gene_level.csv               基因全局表 (每基因/exon 一行, 仅 bp+exon)
  00_cross_variety_delta_summary.csv  跨品种差异表达
  08_run_manifest.csv             运行清单
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from tqdm import tqdm


# =============================================================================
# 0. 性能工具
# =============================================================================

def _parse_json_array(series: pd.Series) -> pd.Series:
    """快速批量解析 JSON 数组列（list comprehension 代替 .apply()，速度提升 3-5x）。"""
    def _parse(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.array([], dtype=np.float32)
        text = str(val).strip()
        if not text:
            return np.array([], dtype=np.float32)
        try:
            return np.asarray(json.loads(text), dtype=np.float32)
        except (json.JSONDecodeError, TypeError):
            try:
                return np.asarray(ast.literal_eval(text), dtype=np.float32)
            except (ValueError, SyntaxError):
                return np.array([], dtype=np.float32)
    return pd.Series([_parse(v) for v in series], index=series.index, dtype=object)


# =============================================================================
# 1. 数据类
# =============================================================================

@dataclass(frozen=True)
class CsvTriplet:
    """单个 CSV 的加载参数。"""
    csv_path: Path
    chromosome: str          # 重命名后 chrom 列的值；"all" 表示不重写
    strand: str              # "total" / "plus" / "minus"


@dataclass
class EvalTask:
    """一个评估任务。"""
    sample: str
    biosample: str
    split: str
    modality: str
    gff: Optional[Path]
    triplets: list[CsvTriplet]    # (csv, chrom, strand) 三元组列表


@dataclass(frozen=True)
class EvalConfig:
    """评估配置。"""
    output_dir: Path
    tasks: list[EvalTask] = field(default_factory=list)
    feature_types: tuple[str, ...] = ("gene", "exon")
    feature_flank_bp: int = 0
    min_overlap_bp: int = 1
    n_expression_buckets: int = 3


@dataclass(frozen=True)
class Feature:
    """GFF 中的基因/外显子特征。"""
    chrom: str
    start0: int
    end0: int
    feature_type: str
    feature_id: str
    parent_id: str
    strand: str


# =============================================================================
# 2. 工具函数
# =============================================================================


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """安全计算 Pearson 相关系数。"""
    if a.size < 2:
        return float("nan")
    if float(np.std(a)) <= 1e-8 or float(np.std(b)) <= 1e-8:
        return float("nan")
    return float(stats.pearsonr(a, b).statistic)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """安全计算 Spearman 秩相关系数。"""
    if a.size < 2:
        return float("nan")
    if len(np.unique(a)) <= 1 or len(np.unique(b)) <= 1:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


# =============================================================================
# 3. CSV 加载与解析
# =============================================================================

def load_and_merge_csvs(triplets: list[CsvTriplet]) -> pd.DataFrame:
    """加载多个 CSV，重命名 chromosome，添加 strand 列，按行合并。

    优化：指定列类型减少内存，批量解析 JSON 数组代替逐行 parse_expression_column。
    """
    all_dfs = []
    for t in triplets:
        # 只读需要的列，并指定 dtypes 减少内存
        usecols = {"chromosome", "start", "end", "predicted_expression", "true_expression"}
        # 如果存在 sequence 列，读进来以便之后 drop（避免 warning）
        # 先用较少的列读取
        df = pd.read_csv(
            t.csv_path,
            usecols=lambda c: c in usecols or c == "sequence",
            dtype={"start": "int32", "end": "int32"},
            low_memory=False,
        )
        required = {"chromosome", "start", "end", "predicted_expression", "true_expression"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{t.csv_path} missing columns: {missing}")

        # 批量解析 expression 列（比逐行 apply parse_expression_column 更快）
        df["parsed_pred"] = _parse_json_array(df["predicted_expression"])
        df["parsed_true"] = _parse_json_array(df["true_expression"])

        # 释放原始字符串列
        df.drop(columns=["sequence", "true_expression", "predicted_expression"], inplace=True, errors="ignore")

        # 验证长度一致性
        calc_len = df["end"] - df["start"]
        parsed_len = df["parsed_pred"].apply(len)
        mismatch = calc_len != parsed_len
        if mismatch.any():
            print(f"  ⚠️  {t.csv_path.name}: {mismatch.sum()} rows with length mismatch, filtered")
            df = df[~mismatch].copy()

        # 重命名 chromosome 列
        if t.chromosome != "all":
            df["chromosome"] = t.chromosome

        # 添加 strand 列
        df["strand"] = t.strand

        # 自动转换染色体命名: chr+数字(无前导零) → chr+两位数字, 例如 chr1→chr01, chr9→chr09
        # chr10/chr11/chr12 保持不变, chrPltd 等非标准命名保持不变
        def _pad_chrom(chrom: str) -> str:
            m = re.match(r'^chr(\d)$', str(chrom))
            return f"chr0{m.group(1)}" if m else chrom
        orig_chroms = df["chromosome"].unique().tolist()
        df["chromosome"] = df["chromosome"].apply(_pad_chrom)
        new_chroms = df["chromosome"].unique().tolist()
        if orig_chroms != new_chroms:
            print(f"  🔄 Chromosome renamed: {sorted(orig_chroms)} → {sorted(new_chroms)}")

        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"  📄 Loaded {len(triplets)} CSV(s) → {len(merged)} total rows")
    return merged


def _accumulate_flatten(
    group: pd.DataFrame, value_col: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """对单条染色体的窗口组做逐碱基累加。

    Returns: (sum_arr, count_arr, origin, span)
        sum_arr/count_arr 为相对 origin 的累加和/覆盖计数（float64 / int32）
    """
    starts = group["start"].to_numpy(dtype=np.int64)
    values_list = group[value_col].to_numpy()
    lengths = np.array([len(v) for v in values_list], dtype=np.int64)
    offsets = starts - int(starts.min())
    span = int((starts + lengths).max() - starts.min())
    origin = int(starts.min())

    sum_arr = np.zeros(span, dtype=np.float64)
    count_arr = np.zeros(span, dtype=np.int32)

    for i in range(len(starts)):
        o = int(offsets[i])
        n = int(lengths[i])
        sum_arr[o:o + n] += values_list[i]
        count_arr[o:o + n] += 1

    return sum_arr, count_arr, origin, span


def flatten_to_genome_array(
    df: pd.DataFrame, value_col: str = "parsed_pred"
) -> dict[str, np.ndarray]:
    """将逐窗口的值按染色体位置平均后拼接。

    优化：按染色体分组后，使用 numpy 数组累积，避免嵌套 dict 的 Python 逐碱基循环。
    使用 float32 累加以减少内存带宽，offsets 预计算而非逐次减法。

    Returns: dict[chromosome] → np.ndarray (逐碱基平均值)
    """
    result = {}
    for chrom, group in df.groupby("chromosome", sort=True):
        sum_arr, count_arr, _origin, _span = _accumulate_flatten(group, value_col)
        mask = count_arr > 0
        result[str(chrom)] = np.divide(sum_arr, count_arr, where=mask, dtype=np.float32)[mask]
    return result


def flatten_to_genome_gene(
    df: pd.DataFrame, value_col: str,
    gene_intervals_by_chrom: Optional[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """将逐窗口的值按染色体位置平均后，只保留基因区域内的碱基。

    与 flatten_to_genome_array 相同的累加逻辑，但在输出前用基因区间掩码过滤，
    只返回落在任一基因区间（整基因跨度 start-end，含内含子）内的逐碱基平均值。
    用于 bp_gene 碱基分辨率指标。

    Returns: dict[chromosome] → np.ndarray (仅基因区域内的逐碱基平均值)
    """
    result = {}
    for chrom, group in df.groupby("chromosome", sort=True):
        intervals = (gene_intervals_by_chrom or {}).get(str(chrom))
        if intervals is None or intervals.size == 0:
            continue

        sum_arr, count_arr, origin, span = _accumulate_flatten(group, value_col)

        # 基因区域掩码：直接用绝对坐标映射到 span 内，避免物化 int64 坐标数组
        gene_mask = np.zeros(span, dtype=bool)
        for s, e in intervals:
            lo = max(int(s) - origin, 0)
            hi = min(int(e) - origin, span)
            if hi > lo:
                gene_mask[lo:hi] = True

        mask = gene_mask & (count_arr > 0)
        if not mask.any():
            continue
        result[str(chrom)] = np.divide(sum_arr, count_arr, where=mask, dtype=np.float32)[mask]

    return result


# =============================================================================
# 4. 核心指标计算
# =============================================================================

def compute_track_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Track 级指标：pcc, log1p_pcc, nozero_pcc, zero_ratio, r2（输入已为 1D float32）。"""
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    n = len(y_pred)

    if n == 0:
        return {"pcc": np.nan, "log1p_pcc": np.nan, "nozero_pcc": np.nan,
                "zero_ratio": np.nan, "r2": np.nan, "n_positions": 0}

    pcc = safe_pearson(y_true, y_pred)
    log1p_pcc = safe_pearson(np.log1p(y_true), np.log1p(y_pred))
    r2 = float(r2_score(y_true, y_pred))
    zero_ratio = float(np.mean(y_true == 0) * 100)

    nonzero_mask = (y_true > 0) & (y_pred > 0)
    y_true_nz = y_true[nonzero_mask]
    y_pred_nz = y_pred[nonzero_mask]
    nozero_pcc = safe_pearson(y_true_nz, y_pred_nz) if len(y_true_nz) >= 2 else np.nan

    spearman = safe_spearman(y_true, y_pred)

    return {
        "pcc": round(pcc, 6) if not np.isnan(pcc) else np.nan,
        "spearman": round(spearman, 6) if not np.isnan(spearman) else np.nan,
        "log1p_pcc": round(log1p_pcc, 6) if not np.isnan(log1p_pcc) else np.nan,
        "nozero_pcc": round(nozero_pcc, 6) if not np.isnan(nozero_pcc) else np.nan,
        "zero_ratio": round(zero_ratio, 4),
        "r2": round(r2, 6),
        "n_positions": n,
    }


def compute_window_metrics(
    pred_arrays: list[np.ndarray],
    true_arrays: list[np.ndarray],
    chromosomes: list[str],
    starts: list[int],
    ends: list[int],
    strands: list[str],
) -> pd.DataFrame:
    """对每个预测窗口独立计算指标（pred/true 已为 ndarray，不再重复转换）。"""
    rows: list[dict] = []
    for pred, true, chrom, start, end, strand in zip(
        pred_arrays, true_arrays, chromosomes, starts, ends, strands
    ):
        length = len(pred)
        if length != len(true) or length == 0:
            continue

        nonzero_mask = (true > 0) & (pred > 0)
        pred_nz = pred[nonzero_mask]
        true_nz = true[nonzero_mask]
        has_nz = len(pred_nz) >= 2

        pcc_val = safe_pearson(true_nz, pred_nz) if has_nz else np.nan
        log1p_val = safe_pearson(np.log1p(true_nz), np.log1p(pred_nz)) if has_nz else np.nan
        spearman_val = safe_spearman(true_nz, pred_nz) if has_nz else np.nan
        zero_ratio = float(np.mean(true == 0) * 100)

        diff = true - pred
        ss_res = float(diff @ diff)
        true_mean = float(np.mean(true))
        ss_tot = float(np.sum((true - true_mean) ** 2))
        r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

        rows.append({
            "chromosome": str(chrom),
            "start": int(start),
            "end": int(end),
            "strand": str(strand),
            "length": length,
            "pcc": round(pcc_val, 6) if not np.isnan(pcc_val) else np.nan,
            "spearman": round(spearman_val, 6) if not np.isnan(spearman_val) else np.nan,
            "log1p_pcc": round(log1p_val, 6) if not np.isnan(log1p_val) else np.nan,
            "nozero_pcc": round(pcc_val, 6) if not np.isnan(pcc_val) else np.nan,
            "zero_ratio": round(zero_ratio, 4),
            "r2": round(r2_val, 6) if not np.isnan(r2_val) else np.nan,
            "pred_mean": round(float(np.mean(pred)), 6),
            "true_mean": round(float(np.mean(true)), 6),
        })
    return pd.DataFrame(rows)


def compute_feature_basic_metrics(
    pred: np.ndarray, true: np.ndarray
) -> dict[str, float]:
    """单个基因/外显子区间内的基础指标（pred/true 已为 ndarray）。"""
    if len(pred) == 0:
        return {"pcc": np.nan, "spearman": np.nan, "log1p_pcc": np.nan, "r2": np.nan}

    nonzero_mask = (pred > 0) & (true > 0)
    pred_nz = pred[nonzero_mask]
    true_nz = true[nonzero_mask]

    if len(pred_nz) >= 2 and len(np.unique(pred_nz)) > 1 and len(np.unique(true_nz)) > 1:
        pcc_val = float(stats.pearsonr(pred_nz, true_nz).statistic)
        log1p_val = safe_pearson(np.log1p(true_nz), np.log1p(pred_nz))
        spearman_val = safe_spearman(pred_nz, true_nz)
    else:
        pcc_val = np.nan
        log1p_val = np.nan
        spearman_val = np.nan

    diff = true - pred
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

    return {
        "pcc": round(pcc_val, 6) if not np.isnan(pcc_val) else np.nan,
        "spearman": round(spearman_val, 6) if not np.isnan(spearman_val) else np.nan,
        "log1p_pcc": round(log1p_val, 6) if not np.isnan(log1p_val) else np.nan,
        "r2": round(r2_val, 6) if not np.isnan(r2_val) else np.nan,
    }


def compute_feature_mean_correlation(
    df: pd.DataFrame, min_features: int = 3
) -> dict[str, float]:
    """跨基因的均值相关性。"""
    valid = df.dropna(subset=["pred_mean", "true_mean"])
    if len(valid) < min_features:
        return {"pcc": np.nan, "spearman": np.nan, "log1p_pcc": np.nan, "nozero_pcc": np.nan, "r2": np.nan}

    pred = valid["pred_mean"].to_numpy(dtype=float)
    true = valid["true_mean"].to_numpy(dtype=float)

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

    nz_mask = (pred > 0) & (true > 0)
    nozero_pcc = safe_pearson(pred[nz_mask], true[nz_mask]) if np.sum(nz_mask) >= min_features else np.nan

    return {
        "pcc": round(safe_pearson(pred, true), 6),
        "spearman": round(safe_spearman(pred, true), 6),
        "log1p_pcc": round(safe_pearson(np.log1p(pred), np.log1p(true)), 6),
        "nozero_pcc": round(nozero_pcc, 6) if not np.isnan(nozero_pcc) else np.nan,
        "r2": round(r2_val, 6),
    }


# =============================================================================
# 5. 表达分箱 + Delta Pearson
# =============================================================================

def assign_expression_buckets(
    df: pd.DataFrame,
    true_mean_col: str = "true_mean",
    n_buckets: int = 3,
    thresholds: Optional[tuple[float, float]] = None,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """按真实表达值分桶 (low / medium / high)。"""
    df = df.copy()
    true_means = df[true_mean_col].dropna().to_numpy()
    if len(true_means) == 0:
        df["expression_bucket"] = "unknown"
        return df, (0.0, 0.0)

    if thresholds is None:
        perc = 100.0 / n_buckets
        low_thresh = np.percentile(true_means, perc)
        high_thresh = np.percentile(true_means, 100 - perc)
    else:
        low_thresh, high_thresh = thresholds

    def _bucket(v):
        if pd.isna(v):
            return "unknown"
        if v <= low_thresh:
            return "low"
        elif v <= high_thresh:
            return "medium"
        else:
            return "high"

    df["expression_bucket"] = df[true_mean_col].apply(_bucket)
    return df, (low_thresh, high_thresh)


# ---- Delta Pearson (仅 feature_ref 模式) ----

def pearson_delta_scale(arr: np.ndarray, method: str = "global_mean") -> float:
    """归一化因子。"""
    values = np.asarray(arr, dtype=np.float64)
    mean_value = float(np.mean(values))
    if mean_value <= 0.0 or not np.isfinite(mean_value):
        return float("nan")
    return mean_value


def compute_pearson_delta_metrics(
    pred: np.ndarray, true: np.ndarray,
    normalize: str = "global_mean",
    ref_value: float | np.ndarray = 0.0,
) -> dict[str, float]:
    """Delta Pearson 计算。"""
    if len(pred) == 0 or len(true) == 0:
        return {"delta_pcc": np.nan, "delta_rmse": np.nan}

    scale_pred = pearson_delta_scale(pred, normalize)
    scale_true = pearson_delta_scale(true, normalize)
    pred_norm = np.asarray(pred, dtype=np.float64) / scale_pred if np.isfinite(scale_pred) else np.full_like(pred, np.nan)
    true_norm = np.asarray(true, dtype=np.float64) / scale_true if np.isfinite(scale_true) else np.full_like(true, np.nan)

    if np.isscalar(ref_value):
        ref = np.full_like(true_norm, float(ref_value))
    else:
        ref = np.asarray(ref_value, dtype=np.float64)
        if ref.shape != true_norm.shape:
            return {"delta_pcc": np.nan, "delta_rmse": np.nan}

    if np.isnan(pred_norm).any() or np.isnan(true_norm).any() or np.isnan(ref).any():
        return {"delta_pcc": np.nan, "delta_rmse": np.nan}

    delta_pred = pred_norm - ref
    delta_true = true_norm - ref
    diff = pred_norm - true_norm

    return {
        "delta_pcc": round(safe_pearson(delta_pred, delta_true), 6),
        "delta_rmse": round(float(np.sqrt(np.mean(diff ** 2))), 6),
    }


def build_per_gene_reference(
    feature_dfs: dict[str, pd.DataFrame], normalize: str = "global_mean"
) -> pd.DataFrame:
    """构建 per_gene_reference 表 (feature_ref 模式)。

    key 格式: sample/chromosome/biosample
    """
    all_rows = []
    for sample_key, df in feature_dfs.items():
        if df.empty:
            continue
        df = df.copy()
        biosample = sample_key.rsplit("/", 1)[-1] if "/" in sample_key else ""
        for ftype in df["feature_type"].unique():
            mask = df["feature_type"] == ftype
            scale = pearson_delta_scale(df.loc[mask, "true_mean"].to_numpy(), normalize)
            df.loc[mask, "true_mean_norm"] = df.loc[mask, "true_mean"] / scale if np.isfinite(scale) else np.nan
        df["biosample"] = biosample
        all_rows.append(df[["feature_id", "feature_type", "true_mean_norm", "biosample"]])

    if not all_rows:
        return pd.DataFrame(columns=["feature_id", "feature_type", "biosample", "delta_ref_value"])

    combined = pd.concat(all_rows, ignore_index=True)
    ref = combined.groupby(["feature_id", "feature_type", "biosample"], dropna=False)["true_mean_norm"].mean()
    ref = ref.reset_index()
    ref.rename(columns={"true_mean_norm": "delta_ref_value"}, inplace=True)
    return ref


def compute_all_delta_pearson(
    feature_df: pd.DataFrame,
    ref_table: pd.DataFrame,
    normalize: str = "global_mean",
    biosample: str = "",
) -> dict[str, float]:
    """对单个样本计算 delta_pcc (feature_ref 模式, 仅 gene 级别)。"""
    valid = feature_df.dropna(subset=["pred_mean", "true_mean"])
    valid = valid[valid["feature_type"] == "gene"]
    if valid.empty:
        return {"delta_pcc": np.nan, "delta_rmse": np.nan}

    if biosample and "biosample" in ref_table.columns:
        ref_filtered = ref_table[ref_table["biosample"] == biosample]
    else:
        ref_filtered = ref_table

    pred = valid["pred_mean"].to_numpy(dtype=float)
    true = valid["true_mean"].to_numpy(dtype=float)

    # feature_ref 模式: 按基因匹配 ref_table 中的 delta_ref_value
    merged = valid[["feature_id", "feature_type"]].merge(
        ref_filtered, on=["feature_id", "feature_type"], how="left"
    )
    valid_mask = merged["delta_ref_value"].notna().to_numpy()
    if valid_mask.sum() < 2:
        return {"delta_pcc": np.nan, "delta_rmse": np.nan}

    ref_vals = merged.loc[valid_mask, "delta_ref_value"].to_numpy(dtype=float)
    return compute_pearson_delta_metrics(pred[valid_mask], true[valid_mask], normalize, ref_vals)


# ---- Delta vs 均值参考样本 (ref_sample 模式, 绝对尺度, 无归一化) ----
# 参考样本 = 训练集所有品种 gene 级 true/pred 的跨品种均值。
# 两种参考模式：
#   delta_pcc_pred  (ref_pred 参考): Δ_pred = pred − ref_pred_mean, Δ_true = true − ref_true_mean
#       与 cross_variety 的 pairwise 口径一致 (把品种 B 换成训练集预测均值参考)，宽松版
#   delta_pcc_true  (ref_true 参考): Δ_pred = pred − ref_true_mean, Δ_true = true − ref_true_mean
#       预测值与真值减去同一个训练集真实均值参考，模型系统性偏差无法被抵消，严格版
# 不依赖任何事后统计量（无需样本均值 scale），训练完即可固定，
# 测试/部署时直接套用，更贴近真实应用。

def build_ref_sample_from_train(all_results: list[dict]) -> pd.DataFrame:
    """从所有训练集样本构建均值参考样本 (gene 级)。

    Returns: DataFrame with columns [feature_id, ref_pred_mean, ref_true_mean]。
    delta_pcc_pred 用 ref_pred_mean，delta_pcc_true 用 ref_true_mean。
    """
    pred_cols: list[pd.Series] = []
    true_cols: list[pd.Series] = []
    for r in all_results:
        if r.get("error"):
            continue
        if r["context"].get("split") != "train":
            continue
        fdf = r.get("feature_df")
        if fdf is None or fdf.empty:
            continue
        gene_df = fdf[fdf["feature_type"] == "gene"].set_index("feature_id")
        if gene_df.empty:
            continue
        if gene_df.index.duplicated().any():
            gene_df = gene_df.groupby(level=0).mean(numeric_only=True)
        pred_cols.append(gene_df["pred_mean"])
        true_cols.append(gene_df["true_mean"])
    if not pred_cols:
        return pd.DataFrame(columns=["feature_id", "ref_pred_mean", "ref_true_mean"])
    ref_pred = pd.concat(pred_cols, axis=1).mean(axis=1)
    ref_true = pd.concat(true_cols, axis=1).mean(axis=1)
    return pd.DataFrame({
        "feature_id": ref_pred.index.to_numpy(),
        "ref_pred_mean": ref_pred.to_numpy(dtype=float),
        "ref_true_mean": ref_true.to_numpy(dtype=float),
    })


def compute_ref_sample_delta_metrics(
    pred: np.ndarray, true: np.ndarray,
    ref_true: np.ndarray, ref_pred: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """与均值参考样本的 delta 指标 (两种参考模式, 绝对尺度, 无归一化)。

    - ref_true 模式: Δ_pred = pred − ref_true, Δ_true = true − ref_true (严格)
    - ref_pred 模式: Δ_pred = pred − ref_pred, Δ_true = true − ref_true (cross_variety 同口径)
    复用 pairwise 口径 (与 compute_pairwise_delta_metrics 一致)。
    """
    empty = {
        "delta_pcc_pred": np.nan, "delta_spearman_pred": np.nan,
        "delta_rmse_pred": np.nan, "sign_accuracy_pred": np.nan,
        "delta_pcc_true": np.nan, "delta_spearman_true": np.nan,
        "delta_rmse_true": np.nan, "sign_accuracy_true": np.nan,
    }
    if len(pred) < 2 or len(ref_true) != len(pred):
        return empty
    # ref_true 模式: 两边减同一个训练集真值均值参考
    mt = compute_pairwise_delta_metrics(pred, true, ref_true, ref_true)
    out = dict(empty)
    out.update({
        "delta_pcc_true": mt.get("delta_pearson", np.nan),
        "delta_spearman_true": mt.get("delta_spearman", np.nan),
        "delta_rmse_true": mt.get("delta_rmse", np.nan),
        "sign_accuracy_true": mt.get("sign_accuracy", np.nan),
    })
    # ref_pred 模式: 预测减训练集预测均值参考 (cross_variety 同口径)
    if ref_pred is not None and len(ref_pred) == len(pred) and not np.isnan(ref_pred).any():
        mp = compute_pairwise_delta_metrics(pred, true, ref_pred, ref_true)
        out.update({
            "delta_pcc_pred": mp.get("delta_pearson", np.nan),
            "delta_spearman_pred": mp.get("delta_spearman", np.nan),
            "delta_rmse_pred": mp.get("delta_rmse", np.nan),
            "sign_accuracy_pred": mp.get("sign_accuracy", np.nan),
        })
    return out


def compute_ref_sample_delta_from_df(
    feature_df: pd.DataFrame, ref_sample: pd.DataFrame,
) -> dict[str, float]:
    """对单个样本的 gene 级 feature_df 计算与参考样本的 delta 指标 (pred + true 两种参考)。"""
    empty = {
        "delta_pcc_pred": np.nan, "delta_spearman_pred": np.nan,
        "delta_rmse_pred": np.nan, "sign_accuracy_pred": np.nan,
        "delta_pcc_true": np.nan, "delta_spearman_true": np.nan,
        "delta_rmse_true": np.nan, "sign_accuracy_true": np.nan,
    }
    valid = feature_df.dropna(subset=["pred_mean", "true_mean"])
    if valid.empty:
        return empty
    merged = valid[["feature_id"]].merge(ref_sample, on="feature_id", how="left")
    mask = merged["ref_true_mean"].notna().to_numpy()
    if mask.sum() < 2:
        return empty
    ref_true = merged.loc[mask, "ref_true_mean"].to_numpy(dtype=float)
    ref_pred = merged.loc[mask, "ref_pred_mean"].to_numpy(dtype=float)
    return compute_ref_sample_delta_metrics(
        valid["pred_mean"].to_numpy(dtype=float)[mask],
        valid["true_mean"].to_numpy(dtype=float)[mask],
        ref_true, ref_pred,
    )


def compute_stratified_metrics(
    df: pd.DataFrame, feature_type: str = "",
    ref_table: Optional[pd.DataFrame] = None,
    biosample: str = "", normalize: str = "global_mean",
    ref_sample: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """按 expression_bucket 分组计算指标 (gene 级别含 delta_pcc 与 ref_sample delta)。"""
    rows = []
    for bucket in ["low", "medium", "high"]:
        part = df[df["expression_bucket"] == bucket]
        if len(part) == 0:
            continue
        metrics = compute_feature_mean_correlation(part)
        row = {
            "feature_type": feature_type,
            "expression_bucket": bucket,
            "number": len(part),
            "pred_mean_avg": round(float(part["pred_mean"].mean()), 6),
            "true_mean_avg": round(float(part["true_mean"].mean()), 6),
            **metrics,
        }
        if ref_table is not None and not ref_table.empty and feature_type == "gene":
            delta = compute_all_delta_pearson(part, ref_table, normalize, biosample)
            row["delta_pcc"] = delta.get("delta_pcc", np.nan)
            row["delta_rmse"] = delta.get("delta_rmse", np.nan)
        if ref_sample is not None and not ref_sample.empty and feature_type == "gene":
            ds = compute_ref_sample_delta_from_df(part, ref_sample)
            row["delta_pcc_pred"] = ds.get("delta_pcc_pred", np.nan)
            row["delta_spearman_pred"] = ds.get("delta_spearman_pred", np.nan)
            row["delta_rmse_pred"] = ds.get("delta_rmse_pred", np.nan)
            row["sign_accuracy_pred"] = ds.get("sign_accuracy_pred", np.nan)
            row["delta_pcc_true"] = ds.get("delta_pcc_true", np.nan)
            row["delta_spearman_true"] = ds.get("delta_spearman_true", np.nan)
            row["delta_rmse_true"] = ds.get("delta_rmse_true", np.nan)
            row["sign_accuracy_true"] = ds.get("sign_accuracy_true", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6. 跨品种差异表达
# =============================================================================

def compute_pairwise_delta_metrics(
    pred_a: np.ndarray, true_a: np.ndarray,
    pred_b: np.ndarray, true_b: np.ndarray,
) -> dict[str, float]:
    """计算两个品种之间的差异表达预测精度。"""
    delta_pred = pred_a - pred_b
    delta_true = true_a - true_b
    log2fc_pred = np.log2(pred_a + 1) - np.log2(pred_b + 1)
    log2fc_true = np.log2(true_a + 1) - np.log2(true_b + 1)

    sign_match = np.sign(delta_pred) == np.sign(delta_true)
    nonzero_mask = (delta_pred != 0) & (delta_true != 0)

    results = {
        "n_genes": float(len(delta_pred)),
        "delta_pearson": safe_pearson(delta_pred, delta_true),
        "delta_spearman": safe_spearman(delta_pred, delta_true),
        "log2fc_pearson": safe_pearson(log2fc_pred, log2fc_true),
        "log2fc_spearman": safe_spearman(log2fc_pred, log2fc_true),
        "delta_rmse": float(np.sqrt(np.mean((delta_pred - delta_true) ** 2))),
    }
    if nonzero_mask.sum() > 0:
        results["sign_accuracy"] = float(sign_match[nonzero_mask].mean())
    else:
        results["sign_accuracy"] = float("nan")

    pred_up = delta_pred > 0
    true_up = delta_true > 0
    n_pred_up = pred_up.sum()
    n_pred_down = (~pred_up).sum()
    results["up_precision"] = float((pred_up & true_up).sum() / n_pred_up) if n_pred_up > 0 else float("nan")
    results["down_precision"] = float((~pred_up & ~true_up).sum() / n_pred_down) if n_pred_down > 0 else float("nan")

    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in results.items()}


# =============================================================================
# 7. GFF 加载与特征聚合
# =============================================================================

def parse_gff_attr(attr_text: str) -> dict[str, str]:
    """解析 GFF 属性列。"""
    attrs: dict[str, str] = {}
    for item in attr_text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            key, value = item.split(" ", 1)
            value = value.strip().strip('"')
        else:
            continue
        from urllib.parse import unquote
        attrs[key.strip()] = unquote(value.strip())
    return attrs


def pick_feature_id(attrs: dict[str, str], feature_type: str) -> str:
    for key in ("ID", "gene_id", "transcript_id", "Name"):
        value = attrs.get(key)
        if value:
            return value
    return f"{feature_type}:unknown"


def pick_parent(attrs: dict[str, str]) -> str:
    for key in ("Parent", "gene_id", "transcript_id"):
        value = attrs.get(key)
        if value:
            return value.split(",")[0]
    return ""


def load_features_from_gff(
    gff_path: Path, feature_types: set[str], flank_bp: int = 0
) -> dict[str, list[Feature]]:
    """从 GFF 加载特征，gene 区间自动替换为 exon 并集。"""
    features_by_chrom: dict[str, list[Feature]] = defaultdict(list)
    transcript_to_gene: dict[str, str] = {}

    with gff_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, ftype, start, end, _, strand, _, attr_text = parts[:9]
            attrs = parse_gff_attr(attr_text)

            if ftype in ("mRNA", "transcript"):
                parent = pick_parent(attrs)
                tid = pick_feature_id(attrs, ftype)
                if parent and tid and tid != parent:
                    transcript_to_gene[tid] = parent
                continue

            if ftype not in feature_types:
                continue

            start0 = int(start) - 1
            end0 = int(end)
            if start0 < 0 or end0 <= start0:
                continue

            features_by_chrom[chrom].append(Feature(
                chrom=chrom, start0=start0, end0=end0,
                feature_type=ftype,
                feature_id=pick_feature_id(attrs, ftype),
                parent_id=pick_parent(attrs),
                strand=strand,
            ))

    # 排序
    result: dict[str, list[Feature]] = {}
    for chrom in features_by_chrom:
        features_by_chrom[chrom].sort(key=lambda x: (x.start0, x.end0))
        result[chrom] = features_by_chrom[chrom]

    # gene 区间替换为 exon 并集
    if "gene" in feature_types and "exon" in feature_types:
        def _resolve_gene_id(exon: Feature) -> str:
            if exon.parent_id and exon.parent_id in transcript_to_gene:
                return transcript_to_gene[exon.parent_id]
            return exon.parent_id or exon.feature_id

        exon_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chrom, features in result.items():
            for f in features:
                if f.feature_type == "exon":
                    gene_id = _resolve_gene_id(f)
                    exon_intervals[gene_id].append((f.start0, f.end0))

        new_result: dict[str, list[Feature]] = {}
        for chrom, features in result.items():
            new_features = []
            for f in features:
                if f.feature_type == "gene" and f.feature_id in exon_intervals:
                    intervals = sorted(exon_intervals[f.feature_id], key=lambda x: (x[0], x[1]))
                    for ex_start, ex_end in intervals:
                        new_features.append(Feature(
                            chrom=f.chrom, start0=ex_start, end0=ex_end,
                            feature_type="gene",
                            feature_id=f.feature_id,
                            parent_id=f.parent_id,
                            strand=f.strand,
                        ))
                else:
                    new_features.append(f)
            new_result[chrom] = new_features
        result = new_result

    return result


def load_gene_regions_from_gff(gff_path: Path) -> dict[str, np.ndarray]:
    """从 GFF 加载每个染色体的基因区间（整基因跨度 start-end，含内含子）。

    用于 bp_gene 碱基分辨率指标：只统计基因区域内的碱基。
    与 load_features_from_gff 不同，这里保留 gene 特征的原始跨度（不替换为 exon 并集）。

    Returns: dict[chromosome] → np.ndarray shape (N, 2) 的 [start0, end0) 半开区间，
             已合并重叠/相邻区间为并集，按 start 升序。
    """
    intervals_by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with gff_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, _source, ftype, start, end = parts[:5]
            if ftype != "gene":
                continue
            start0 = int(start) - 1
            end0 = int(end)
            if start0 < 0 or end0 <= start0:
                continue
            intervals_by_chrom[chrom].append((start0, end0))

    result: dict[str, np.ndarray] = {}
    for chrom, intervals in intervals_by_chrom.items():
        arr = np.asarray(sorted(intervals), dtype=np.int64)
        if arr.size == 0:
            continue
        # 合并重叠/相邻区间为并集
        merged: list[list[int]] = [[int(arr[0][0]), int(arr[0][1])]]
        for s, e in arr[1:]:
            if int(s) <= merged[-1][1]:
                if int(e) > merged[-1][1]:
                    merged[-1][1] = int(e)
            else:
                merged.append([int(s), int(e)])
        result[chrom] = np.asarray(merged, dtype=np.int64)
    return result


def aggregate_to_features(
    df: pd.DataFrame, features_by_chrom: dict[str, list[Feature]],
    min_overlap_bp: int = 1,
) -> pd.DataFrame:
    """将逐窗口预测值与 GFF 特征做 overlap 聚合。

    优化策略：
    1. 特征为中心的外循环 + 二分查找 + 游标推进（O(F·log W)）
    2. numpy 切片累加代替 Python dict 逐碱基循环（C 级速度）
    3. 每特征临时分配数组，用完即释放（低内存）
    """
    rows = []
    csv_chroms = set(df["chromosome"].unique())
    gff_chroms = set(features_by_chrom.keys())
    missing_chroms = gff_chroms - csv_chroms
    if missing_chroms:
        print(f"  ⚠️  GFF chromosomes with NO matching CSV data: {sorted(missing_chroms)}")
        print(f"     CSV has chromosomes: {sorted(csv_chroms)}")

    for chrom, feats in features_by_chrom.items():
        if not feats:
            continue

        chrom_df = df[df["chromosome"] == str(chrom)]
        if chrom_df.empty:
            continue

        # 按 start 排序窗口，建立二分查找索引
        # 注意：调用方已按 strand 过滤 df（_evaluate_task 中 per-strand 聚合），
        # 但为防御性编程，内层仍按特征链匹配窗口（strand="total" 时不过滤）
        chrom_df = chrom_df.sort_values(["start", "end"])
        win_starts = chrom_df["start"].to_numpy(dtype=np.int64)
        win_ends = chrom_df["end"].to_numpy(dtype=np.int64)
        win_preds = chrom_df["parsed_pred"].to_numpy()
        win_trues = chrom_df["parsed_true"].to_numpy()
        win_strands = chrom_df["strand"].to_numpy()
        n_wins = len(win_starts)

        # 游标：特征按染色体顺序排列，左边界只增不减
        left_cursor = 0

        for feat in tqdm(feats, desc=f"  Features on {chrom}", leave=False):
            f_start0 = feat.start0
            f_end0 = feat.end0
            f_len = f_end0 - f_start0
            feat_strand = feat.strand

            # 二分查找上界：第一个 start >= f_end0 的窗口
            right = int(np.searchsorted(win_starts, f_end0, side="left"))

            # 推进游标跳过 end <= f_start0 的窗口
            while left_cursor < right and win_ends[left_cursor] <= f_start0:
                left_cursor += 1
            left = left_cursor

            if left >= right:
                continue

            # 临时数组：按相对位置（距 feat.start0）累积
            pred_sum = np.zeros(f_len, dtype=np.float64)
            true_sum = np.zeros(f_len, dtype=np.float64)
            counts = np.zeros(f_len, dtype=np.int32)

            # 将 GFF 链符号（+/-）归一化为 CSV 格式（plus/minus/total），用于窗口匹配
            if feat_strand == "+":
                _feat_strand_norm = "plus"
            elif feat_strand == "-":
                _feat_strand_norm = "minus"
            else:
                _feat_strand_norm = feat_strand

            for i in range(left, right):
                # 按链匹配：跳过链不匹配的窗口（strand="total" 时匹配所有链）
                if _feat_strand_norm != "total" and win_strands[i] != _feat_strand_norm:
                    continue

                w_start = int(win_starts[i])
                w_end = int(win_ends[i])

                o_start = w_start if w_start > f_start0 else f_start0
                o_end = w_end if w_end < f_end0 else f_end0
                if o_end <= o_start:
                    continue

                src_start = o_start - w_start
                src_end = o_end - w_start
                dst_start = o_start - f_start0
                dst_end = o_end - f_start0

                pred_arr = win_preds[i]   # 已是 float32 ndarray
                true_arr = win_trues[i]   # 已是 float32 ndarray
                pred_sum[dst_start:dst_end] += pred_arr[src_start:src_end]
                true_sum[dst_start:dst_end] += true_arr[src_start:src_end]
                counts[dst_start:dst_end] += 1

            covered_mask = counts > 0
            overlap_bp = int(covered_mask.sum())
            if overlap_bp < min_overlap_bp:
                continue

            pred_vals = np.divide(pred_sum, counts, where=covered_mask, dtype=np.float32)[covered_mask]
            true_vals = np.divide(true_sum, counts, where=covered_mask, dtype=np.float32)[covered_mask]
            metrics = compute_feature_basic_metrics(pred_vals, true_vals)

            nonzero_mask = (pred_vals > 0) & (true_vals > 0)
            first_covered = int(np.argmax(covered_mask)) + f_start0
            last_covered = int(f_len - np.argmax(covered_mask[::-1]) - 1) + f_start0
            eval_length = last_covered - first_covered + 1
            feature_length = f_len

            rows.append({
                "feature_type": feat.feature_type,
                "feature_id": feat.feature_id,
                "parent_id": feat.parent_id,
                "chromosome": feat.chrom,
                "start": feat.start0,
                "end": feat.end0,
                "strand": feat.strand,
                "feature_length": feature_length,
                "eval_length": eval_length,
                "overlap_bp": overlap_bp,
                "coverage_fraction": round(overlap_bp / eval_length, 4) if eval_length else np.nan,
                "pred_mean": float(np.mean(pred_vals)),
                "true_mean": float(np.mean(true_vals)),
                "pred_zero_ratio": round(float(np.mean(pred_vals == 0)), 4),
                "true_zero_ratio": round(float(np.mean(true_vals == 0)), 4),
                "nonzero_bp": int(np.sum(nonzero_mask)),
                **metrics,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 8. 评估主流程
# =============================================================================

def evaluate_one_task(
    task: EvalTask, config: EvalConfig,
    features_cache: Optional[dict] = None,
    gene_regions_cache: Optional[dict] = None,
) -> dict:
    """对单个评估任务执行全部计算。"""
    print(f"\n{'='*60}")
    strands_display = ",".join(set(t.strand for t in task.triplets))
    print(f"📊 Evaluating: [{task.split}] {task.sample}/{task.biosample}/{strands_display}")

    # 加载并合并 CSV
    df = load_and_merge_csvs(task.triplets)
    if len(df) == 0:
        return {"task": task, "error": "empty dataframe"}

    context = {
        "sample": task.sample,
        "biosample": task.biosample,
        "split": task.split,
        "modality": task.modality,
    }

    results = {"task": task, "context": context, "df": df}

    # ---- 基因区域（整基因跨度 start-end）加载 —— 用于 bp_gene 碱基分辨率指标 ----
    gene_regions: Optional[dict[str, np.ndarray]] = None
    if task.gff is not None and task.gff.is_file():
        if gene_regions_cache is not None:
            gene_regions = gene_regions_cache.get(str(task.gff))
        if gene_regions is None:
            gene_regions = load_gene_regions_from_gff(task.gff)
            if gene_regions_cache is not None:
                gene_regions_cache[str(task.gff)] = gene_regions
        n_interval = sum(len(v) for v in gene_regions.values())
        print(f"  🧬 Gene regions loaded: {n_interval} intervals across {len(gene_regions)} chroms")
    results["gene_regions"] = gene_regions

    # ---- Track 级 ----
    print("  📐 Track-level...")
    pred_dict = flatten_to_genome_array(df, "parsed_pred")
    true_dict = flatten_to_genome_array(df, "parsed_true")

    all_preds = []
    all_trues = []
    for chrom in sorted(set(list(pred_dict.keys()) + list(true_dict.keys()))):
        if chrom in pred_dict and chrom in true_dict:
            all_preds.append(pred_dict[chrom])
            all_trues.append(true_dict[chrom])

    global_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float32)
    global_true = np.concatenate(all_trues) if all_trues else np.array([], dtype=np.float32)
    results["global_pred"] = global_pred
    results["global_true"] = global_true
    results["track_metrics"] = compute_track_metrics(global_pred, global_true)

    # ---- 逐染色体 track 级（直接复用 pred_dict/true_dict）----
    chrom_track = {}
    for chrom in sorted(set(df["chromosome"])):
        sp = pred_dict.get(chrom, np.array([], dtype=np.float32))
        st = true_dict.get(chrom, np.array([], dtype=np.float32))
        if len(sp) > 0 and len(st) > 0:
            chrom_track[chrom] = compute_track_metrics(sp, st)
    results["chrom_track"] = chrom_track
    # 缓存 chromosomes 的 flatten 结果供 build_main_summary 复用
    results["flatten_pred"] = pred_dict
    results["flatten_true"] = true_dict

    # ---- 窗口级 ----
    print("  📐 Window-level...")
    # parsed_pred/parsed_true 已是 float32 ndarray，无需重复 np.asarray
    pred_arrays = df["parsed_pred"].tolist()
    true_arrays = df["parsed_true"].tolist()
    chromosomes = df["chromosome"].tolist()
    starts = df["start"].tolist()
    ends = df["end"].tolist()
    strands = df["strand"].tolist()
    results["window_df"] = compute_window_metrics(
        pred_arrays, true_arrays, chromosomes, starts, ends, strands
    )
    if not results["window_df"].empty:
        for key, value in context.items():
            results["window_df"].insert(0, key, value)

    # ---- Feature 级 ----
    results["feature_df"] = pd.DataFrame()
    results["feature_summary"] = {}
    results["feature_df_per_strand"] = {}

    if task.gff is not None and task.gff.is_file() and features_cache is not None:
        try:
            print("  📐 Feature-level (gene/exon)...")
            gff_path = task.gff
            features_by_chrom = features_cache.get(str(gff_path))
            if features_by_chrom is None:
                features_by_chrom = load_features_from_gff(
                    gff_path, set(config.feature_types), config.feature_flank_bp
                )
                features_cache[str(gff_path)] = features_by_chrom

            # 按链分别聚合，确保基因水平指标区分正负链
            fdf_per_strand: dict[str, pd.DataFrame] = {}
            for strand_val in df["strand"].unique():
                df_s = df[df["strand"] == strand_val]
                fdf_s = aggregate_to_features(df_s, features_by_chrom, config.min_overlap_bp)
                if not fdf_s.empty:
                    for key, value in context.items():
                        fdf_s.insert(0, key, value)
                    fdf_per_strand[strand_val] = fdf_s
            results["feature_df_per_strand"] = fdf_per_strand

            # 兼容旧接口：取第一条链的结果作为 feature_df（用于 build_gene_table 等下游）
            if fdf_per_strand:
                first_strand_fdf = next(iter(fdf_per_strand.values()))
                results["feature_df"] = first_strand_fdf

                for ftype in first_strand_fdf["feature_type"].unique():
                    part = first_strand_fdf[first_strand_fdf["feature_type"] == ftype]
                    corr = compute_feature_mean_correlation(part)
                    results["feature_summary"][ftype] = {
                        "n_features": len(part),
                        "n_valid": len(part.dropna(subset=["pred_mean", "true_mean"])),
                        **corr,
                    }
        except FileNotFoundError as e:
            print(f"     ⚠️ Skipping feature-level: {e}")

    # 缓存各 strand 的 flatten 结果供 build_main_summary 复用，避免重复计算
    # pred_gene / true_gene 为仅基因区域（整基因跨度）内的碱基，用于 bp_gene 分辨率
    per_strand_flatten: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for strand_val in df["strand"].unique():
        df_s = df[df["strand"] == strand_val]
        per_strand_flatten[strand_val] = {
            "pred": flatten_to_genome_array(df_s, "parsed_pred"),
            "true": flatten_to_genome_array(df_s, "parsed_true"),
            "pred_gene": flatten_to_genome_gene(df_s, "parsed_pred", gene_regions),
            "true_gene": flatten_to_genome_gene(df_s, "parsed_true", gene_regions),
        }
    results["per_strand_flatten"] = per_strand_flatten

    return results


def _concat_gene_track(
    pred_gene: Optional[dict[str, np.ndarray]],
    true_gene: Optional[dict[str, np.ndarray]],
    chromosomes: list[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """把逐染色体的基因区域过滤结果拼接，供 bp_gene 指标使用。

    Returns: (pred_gene_concat, true_gene_concat)；无基因区域或无可保留碱基时返回 (None, None)。
    """
    if pred_gene is None or true_gene is None:
        return None, None
    all_p, all_t = [], []
    for c in chromosomes:
        if c in pred_gene and c in true_gene and len(pred_gene[c]) > 0 and len(true_gene[c]) > 0:
            all_p.append(pred_gene[c])
            all_t.append(true_gene[c])
    if not all_p:
        return None, None
    return np.concatenate(all_p), np.concatenate(all_t)


def build_main_summary(
    all_results: list[dict], config: EvalConfig,
    bucket_thresholds=None, ref_table: Optional[pd.DataFrame] = None,
    ref_sample: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """构建主表: global=sample + global=chromosome。"""
    rows = []

    for r in all_results:
        if r.get("error"):
            continue
        ctx = r["context"]
        task = r["task"]

        # 汇总该 task 中出现的 strand
        strands_in_task = sorted(set(t.strand for t in task.triplets))

        for strand_val in strands_in_task:
            # 使用预缓存的 per-strand flatten 结果，避免重复计算
            flatten_cache = r.get("per_strand_flatten", {}).get(strand_val)
            if flatten_cache is None:
                continue
            sp_cache = flatten_cache["pred"]
            st_cache = flatten_cache["true"]
            chroms_in_strand = sorted(sp_cache.keys())

            # ---- global=sample: 跨染色体聚合 ----
            all_p = [sp_cache[c] for c in chroms_in_strand if c in st_cache]
            all_t = [st_cache[c] for c in chroms_in_strand if c in st_cache]
            if all_p:
                gp = np.concatenate(all_p)
                gt = np.concatenate(all_t)
                track = compute_track_metrics(gp, gt)
                rows.append({
                    "sample": ctx["sample"], "biosample": ctx["biosample"],
                    "split": ctx["split"], "global": "sample",
                    "chromosome": "all", "resolution": "bp",
                    "strand": strand_val, **track,
                })

                # ---- global=sample: bp_gene（只保留基因区域内碱基）----
                gp_g, gt_g = _concat_gene_track(
                    flatten_cache.get("pred_gene"), flatten_cache.get("true_gene"),
                    chroms_in_strand,
                )
                if gp_g is not None:
                    track_g = compute_track_metrics(gp_g, gt_g)
                    rows.append({
                        "sample": ctx["sample"], "biosample": ctx["biosample"],
                        "split": ctx["split"], "global": "sample",
                        "chromosome": "all", "resolution": "bp_gene",
                        "strand": strand_val, **track_g,
                    })

            # Feature 级 (exon/gene) — 使用 per-strand feature_df
            fdf_per_strand = r.get("feature_df_per_strand", {})
            fdf = fdf_per_strand.get(strand_val)
            if fdf is not None and not fdf.empty:
                for ftype in config.feature_types:
                    part = fdf[fdf["feature_type"] == ftype]
                    if part.empty:
                        continue
                    corr = compute_feature_mean_correlation(part)
                    row = {
                        "sample": ctx["sample"], "biosample": ctx["biosample"],
                        "split": ctx["split"], "global": "sample",
                        "chromosome": "all", "resolution": ftype,
                        "strand": strand_val, **corr,
                    }
                    if ref_table is not None and not ref_table.empty and ftype == "gene":
                        delta = compute_all_delta_pearson(
                            part, ref_table, "global_mean", ctx.get("biosample", "")
                        )
                        row["delta_pcc"] = delta.get("delta_pcc", np.nan)
                    if ref_sample is not None and not ref_sample.empty and ftype == "gene":
                        ds = compute_ref_sample_delta_from_df(part, ref_sample)
                        row["delta_pcc_pred"] = ds.get("delta_pcc_pred", np.nan)
                        row["delta_spearman_pred"] = ds.get("delta_spearman_pred", np.nan)
                        row["delta_rmse_pred"] = ds.get("delta_rmse_pred", np.nan)
                        row["sign_accuracy_pred"] = ds.get("sign_accuracy_pred", np.nan)
                        row["delta_pcc_true"] = ds.get("delta_pcc_true", np.nan)
                        row["delta_spearman_true"] = ds.get("delta_spearman_true", np.nan)
                        row["delta_rmse_true"] = ds.get("delta_rmse_true", np.nan)
                        row["sign_accuracy_true"] = ds.get("sign_accuracy_true", np.nan)
                    rows.append(row)

                # 分桶 (gene-low/medium/high)
                if bucket_thresholds is not None:
                    for ftype in config.feature_types:
                        part = fdf[fdf["feature_type"] == ftype]
                        if part.empty:
                            continue
                        bucketed, _ = assign_expression_buckets(
                            part, "true_mean", n_buckets=config.n_expression_buckets,
                            thresholds=bucket_thresholds,
                        )
                        stratified = compute_stratified_metrics(
                            bucketed, feature_type=ftype,
                            ref_table=ref_table, biosample=ctx.get("biosample", ""),
                            normalize="global_mean", ref_sample=ref_sample,
                        )
                        for _, srow in stratified.iterrows():
                            bucket_label = srow.get("expression_bucket", "unknown")
                            rows.append({
                                "sample": ctx["sample"], "biosample": ctx["biosample"],
                                "split": ctx["split"], "global": "sample",
                                "chromosome": "all",
                                "resolution": f"{ftype}-{bucket_label}",
                                "strand": strand_val,
                                "pcc": srow.get("pcc", np.nan),
                                "spearman": srow.get("spearman", np.nan),
                                "log1p_pcc": srow.get("log1p_pcc", np.nan),
                                "nozero_pcc": srow.get("nozero_pcc", np.nan),
                                "zero_ratio": np.nan,
                                "r2": srow.get("r2", np.nan),
                                "delta_pcc": srow.get("delta_pcc", np.nan),
                                "delta_pcc_pred": srow.get("delta_pcc_pred", np.nan),
                                "delta_spearman_pred": srow.get("delta_spearman_pred", np.nan),
                                "delta_rmse_pred": srow.get("delta_rmse_pred", np.nan),
                                "sign_accuracy_pred": srow.get("sign_accuracy_pred", np.nan),
                                "delta_pcc_true": srow.get("delta_pcc_true", np.nan),
                                "delta_spearman_true": srow.get("delta_spearman_true", np.nan),
                                "delta_rmse_true": srow.get("delta_rmse_true", np.nan),
                                "sign_accuracy_true": srow.get("sign_accuracy_true", np.nan),
                            })

            # ---- global=chromosome: 逐染色体（使用缓存结果）----
            for chrom in chroms_in_strand:
                sp = sp_cache.get(chrom, np.array([], dtype=np.float32))
                st = st_cache.get(chrom, np.array([], dtype=np.float32))

                if len(sp) > 0 and len(st) > 0:
                    track = compute_track_metrics(sp, st)
                    rows.append({
                        "sample": ctx["sample"], "biosample": ctx["biosample"],
                        "split": ctx["split"], "global": "chromosome",
                        "chromosome": chrom, "resolution": "bp",
                        "strand": strand_val, **track,
                    })

                    # ---- 逐染色体 bp_gene（只保留基因区域内碱基）----
                    pg_cache = flatten_cache.get("pred_gene")
                    tg_cache = flatten_cache.get("true_gene")
                    sp_g = pg_cache.get(chrom) if pg_cache else None
                    st_g = tg_cache.get(chrom) if tg_cache else None
                    if sp_g is not None and st_g is not None and len(sp_g) > 0 and len(st_g) > 0:
                        track_g = compute_track_metrics(sp_g, st_g)
                        rows.append({
                            "sample": ctx["sample"], "biosample": ctx["biosample"],
                            "split": ctx["split"], "global": "chromosome",
                            "chromosome": chrom, "resolution": "bp_gene",
                            "strand": strand_val, **track_g,
                        })

                # 逐染色体的 feature 级 — 使用 per-strand feature_df
                if fdf is not None and not fdf.empty:
                    fdf_chrom = fdf[fdf["chromosome"] == chrom]
                    if not fdf_chrom.empty:
                        for ftype in config.feature_types:
                            part = fdf_chrom[fdf_chrom["feature_type"] == ftype]
                            if part.empty:
                                continue
                            corr = compute_feature_mean_correlation(part)
                            row = {
                                "sample": ctx["sample"], "biosample": ctx["biosample"],
                                "split": ctx["split"], "global": "chromosome",
                                "chromosome": chrom, "resolution": ftype,
                                "strand": strand_val, **corr,
                            }
                            if ref_table is not None and not ref_table.empty and ftype == "gene":
                                delta = compute_all_delta_pearson(
                                    part, ref_table, "global_mean", ctx.get("biosample", "")
                                )
                                row["delta_pcc"] = delta.get("delta_pcc", np.nan)
                            if ref_sample is not None and not ref_sample.empty and ftype == "gene":
                                ds = compute_ref_sample_delta_from_df(part, ref_sample)
                                row["delta_pcc_pred"] = ds.get("delta_pcc_pred", np.nan)
                                row["delta_spearman_pred"] = ds.get("delta_spearman_pred", np.nan)
                                row["delta_rmse_pred"] = ds.get("delta_rmse_pred", np.nan)
                                row["sign_accuracy_pred"] = ds.get("sign_accuracy_pred", np.nan)
                                row["delta_pcc_true"] = ds.get("delta_pcc_true", np.nan)
                                row["delta_spearman_true"] = ds.get("delta_spearman_true", np.nan)
                                row["delta_rmse_true"] = ds.get("delta_rmse_true", np.nan)
                                row["sign_accuracy_true"] = ds.get("sign_accuracy_true", np.nan)
                            rows.append(row)

                        # 分桶
                        if bucket_thresholds is not None:
                            for ftype in config.feature_types:
                                part = fdf_chrom[fdf_chrom["feature_type"] == ftype]
                                if part.empty:
                                    continue
                                bucketed, _ = assign_expression_buckets(
                                    part, "true_mean", n_buckets=config.n_expression_buckets,
                                    thresholds=bucket_thresholds,
                                )
                                stratified = compute_stratified_metrics(
                                    bucketed, feature_type=ftype,
                                    ref_table=ref_table, biosample=ctx.get("biosample", ""),
                                    normalize="global_mean", ref_sample=ref_sample,
                                )
                                for _, srow in stratified.iterrows():
                                    bucket_label = srow.get("expression_bucket", "unknown")
                                    rows.append({
                                        "sample": ctx["sample"], "biosample": ctx["biosample"],
                                        "split": ctx["split"], "global": "chromosome",
                                        "chromosome": chrom,
                                        "resolution": f"{ftype}-{bucket_label}",
                                        "strand": strand_val,
                                        "pcc": srow.get("pcc", np.nan),
                                        "spearman": srow.get("spearman", np.nan),
                                        "log1p_pcc": srow.get("log1p_pcc", np.nan),
                                        "nozero_pcc": srow.get("nozero_pcc", np.nan),
                                        "zero_ratio": np.nan,
                                        "r2": srow.get("r2", np.nan),
                                        "delta_pcc": srow.get("delta_pcc", np.nan),
                                        "delta_pcc_pred": srow.get("delta_pcc_pred", np.nan),
                                        "delta_spearman_pred": srow.get("delta_spearman_pred", np.nan),
                                        "delta_rmse_pred": srow.get("delta_rmse_pred", np.nan),
                                        "sign_accuracy_pred": srow.get("sign_accuracy_pred", np.nan),
                                        "delta_pcc_true": srow.get("delta_pcc_true", np.nan),
                                        "delta_spearman_true": srow.get("delta_spearman_true", np.nan),
                                        "delta_rmse_true": srow.get("delta_rmse_true", np.nan),
                                        "sign_accuracy_true": srow.get("sign_accuracy_true", np.nan),
                                    })

    df_out = pd.DataFrame(rows)
    # 确保列顺序
    cols = ["sample", "biosample", "split", "global", "chromosome",
            "resolution", "strand", "pcc", "spearman", "log1p_pcc", "nozero_pcc",
            "zero_ratio", "r2", "delta_pcc",
            "delta_pcc_pred", "delta_spearman_pred", "delta_rmse_pred", "sign_accuracy_pred",
            "delta_pcc_true", "delta_spearman_true", "delta_rmse_true", "sign_accuracy_true"]
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    return df_out[cols]


def build_window_table(all_results: list[dict]) -> pd.DataFrame:
    """构建窗口全局表 (每窗口一行)。"""
    all_windows = []
    for r in all_results:
        if r.get("error"):
            continue
        wdf = r.get("window_df")
        if wdf is not None and not wdf.empty:
            all_windows.append(wdf)
    return pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()


def build_gene_table(all_results: list[dict]) -> pd.DataFrame:
    """构建基因全局表 (每基因/exon 一行, 仅 bp+exon, 无 delta_pcc)。"""
    all_genes = []
    for r in all_results:
        if r.get("error"):
            continue
        fdf = r.get("feature_df")
        if fdf is not None and not fdf.empty:
            # 只保留 bp 和 exon 分辨率，去掉 delta_pcc
            keep_cols = [c for c in fdf.columns if c not in ("delta_pcc", "delta_rmse")]
            all_genes.append(fdf[keep_cols])
    return pd.concat(all_genes, ignore_index=True) if all_genes else pd.DataFrame()


def build_cross_variety_delta(all_results: list[dict]) -> pd.DataFrame:
    """构建跨品种差异表达表。

    将每个 (sample, biosample) 组合视为独立品种，生成所有两两对比，
    包括同 biosample 内不同 sample、不同 biosample 间同 sample 和跨 sample 的组合。
    """
    # 收集所有结果中的 gene 级 feature_df
    entries: list[dict] = []
    for r in all_results:
        if r.get("error"):
            continue
        fdf = r.get("feature_df")
        if fdf is None or fdf.empty:
            continue
        ctx = r["context"]
        gene_df = fdf[fdf["feature_type"] == "gene"].set_index("feature_id")
        if gene_df.empty:
            continue
        if gene_df.index.duplicated().any():
            gene_df = gene_df.groupby(level=0).mean(numeric_only=True)
        entries.append({
            "sample": ctx["sample"],
            "biosample": ctx["biosample"],
            "split": ctx["split"],
            "gene_df": gene_df,
        })

    if len(entries) < 2:
        return pd.DataFrame()

    rows = []
    for entry_a, entry_b in combinations(entries, 2):
        df_a = entry_a["gene_df"]
        df_b = entry_b["gene_df"]
        common_ids = df_a.index.intersection(df_b.index)
        if len(common_ids) < 10:
            continue

        split_a = entry_a["split"]
        split_b = entry_b["split"]
        if split_a == "train" and split_b == "train":
            pair_type = "train_train"
        elif split_a == "test" and split_b == "test":
            pair_type = "test_test"
        else:
            pair_type = "train_test"

        metrics = compute_pairwise_delta_metrics(
            df_a.loc[common_ids, "pred_mean"].to_numpy(dtype=float),
            df_a.loc[common_ids, "true_mean"].to_numpy(dtype=float),
            df_b.loc[common_ids, "pred_mean"].to_numpy(dtype=float),
            df_b.loc[common_ids, "true_mean"].to_numpy(dtype=float),
        )
        rows.append({
            "biosample_a": entry_a["biosample"],
            "biosample_b": entry_b["biosample"],
            "cultivar_a": entry_a["sample"],
            "cultivar_b": entry_b["sample"],
            "pair_type": pair_type,
            **metrics,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 9. Config 加载
# =============================================================================

def load_config(config_path: Path) -> EvalConfig:
    """加载 YAML 配置文件。"""
    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    output_dir = Path(data.get("output_dir", "./evaluation_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(output_dir=output_dir)

    if "feature_level" in data:
        fl = data["feature_level"]
        config.feature_types = tuple(fl.get("feature_types", ["gene", "exon"]))
        config.min_overlap_bp = fl.get("min_overlap_bp", 1)
        config.feature_flank_bp = fl.get("flank_bp", 0)

    if "buckets" in data:
        config.n_expression_buckets = data["buckets"].get("n_buckets", 3)

    # 解析 tasks
    for td in data.get("tasks", []):
        predict_csvs = td["predict_csv"]
        if isinstance(predict_csvs, str):
            predict_csvs = [predict_csvs]
        chromosomes = td["chromosome"]
        if isinstance(chromosomes, str):
            chromosomes = [chromosomes]
        strands = td["strand"]
        if isinstance(strands, str):
            strands = [strands]

        n = len(predict_csvs)
        if len(chromosomes) != n or len(strands) != n:
            raise ValueError(
                f"predict_csv ({n}), chromosome ({len(chromosomes)}), "
                f"strand ({len(strands)}) counts must match"
            )

        triplets = [
            CsvTriplet(csv_path=Path(predict_csvs[i]),
                       chromosome=chromosomes[i],
                       strand=strands[i])
            for i in range(n)
        ]

        config.tasks.append(EvalTask(
            sample=td["sample"],
            biosample=td["biosample"],
            split=td["split"],
            modality=td.get("modality", "RNA-seq"),
            gff=Path(td["gff"]) if td.get("gff") else None,
            triplets=triplets,
        ))

    return config


def _compute_global_thresholds(
    all_results: list[dict], n_buckets: int = 3
) -> Optional[tuple[float, float]]:
    """从所有训练集的 true_mean 计算全局分桶阈值。"""
    all_train_true = []
    for r in all_results:
        if r.get("error"):
            continue
        if r["context"].get("split") == "train":
            fdf = r.get("feature_df")
            if fdf is not None and not fdf.empty:
                gene_part = fdf[fdf["feature_type"] == "gene"]
                if not gene_part.empty:
                    all_train_true.append(gene_part["true_mean"].dropna())
    if not all_train_true:
        return None
    combined = pd.concat([pd.Series(a) for a in all_train_true])
    perc = 100.0 / n_buckets
    return (float(np.percentile(combined, perc)),
            float(np.percentile(combined, 100 - perc)))


def _build_ref_table(
    all_results: list[dict], normalize: str = "global_mean"
) -> pd.DataFrame:
    """从所有训练集的 feature_df 构建基因参考表。"""
    train_features: dict[str, pd.DataFrame] = {}
    for r in all_results:
        if r.get("error"):
            continue
        if r["context"].get("split") != "train":
            continue
        fdf = r.get("feature_df")
        if fdf is None or fdf.empty:
            continue
        # key: sample/chromosome/biosample （兼容多组织）
        ctx = r["context"]
        # 取该 task 的染色体集合
        for chrom in fdf["chromosome"].unique():
            key = f"{ctx['sample']}/{chrom}/{ctx['biosample']}"
            sub = fdf[fdf["chromosome"] == chrom]
            if not sub.empty:
                train_features[key] = sub

    if not train_features:
        return pd.DataFrame()
    return build_per_gene_reference(train_features, normalize)


# =============================================================================
# 10. Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Gene Expression Prediction Evaluation (v3)")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("-o", "--output_dir", type=Path, default=None, help="Override output dir")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature-level eval")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧬 Gene Expression Prediction Evaluation (v3)")
    print("=" * 60)
    print(f"  Output: {config.output_dir}")
    print(f"  Tasks:  {len(config.tasks)}")

    # 逐任务评估
    features_cache = {} if not args.skip_features else None
    gene_regions_cache: dict[str, dict[str, np.ndarray]] = {}
    all_results = []
    for task in tqdm(config.tasks, desc="Evaluating", unit="task"):
        result = evaluate_one_task(task, config, features_cache, gene_regions_cache)
        all_results.append(result)

    # 全局阈值 + 参考表 + 均值参考样本
    bucket_thresholds = _compute_global_thresholds(all_results, config.n_expression_buckets)
    ref_table = _build_ref_table(all_results, "global_mean")
    if not ref_table.empty:
        print(f"  Built reference table: {len(ref_table)} genes")
    ref_sample = build_ref_sample_from_train(all_results)
    if not ref_sample.empty:
        print(f"  Built mean ref-sample (pred+true): {len(ref_sample)} genes")

    # 写输出
    print(f"\n{'='*60}")
    print("📝 Writing outputs...")

    outdir = config.output_dir

    # 00_main_summary.csv
    df = build_main_summary(all_results, config, bucket_thresholds, ref_table, ref_sample)
    if not df.empty:
        df.to_csv(outdir / "00_main_summary.csv", index=False)
        print(f"  ✅ 00_main_summary.csv ({len(df)} rows)")

    # 00_window_level.csv
    df = build_window_table(all_results)
    if not df.empty:
        df.to_csv(outdir / "00_window_level.csv", index=False)
        print(f"  ✅ 00_window_level.csv ({len(df)} rows)")

    # 00_gene_level.csv
    df = build_gene_table(all_results)
    if not df.empty:
        df.to_csv(outdir / "00_gene_level.csv", index=False)
        print(f"  ✅ 00_gene_level.csv ({len(df)} rows)")

    # 00_cross_variety_delta_summary.csv
    df = build_cross_variety_delta(all_results)
    if not df.empty:
        df.to_csv(outdir / "00_cross_variety_delta_summary.csv", index=False)
        print(f"  ✅ 00_cross_variety_delta_summary.csv ({len(df)} rows)")

    # 08_run_manifest.csv
    manifest_rows = []
    for r in all_results:
        ctx = r.get("context", {})
        if r.get("error"):
            manifest_rows.append({**ctx, "status": "error", "error": r["error"]})
        else:
            manifest_rows.append({**ctx, "status": "ok"})
    pd.DataFrame(manifest_rows).to_csv(outdir / "08_run_manifest.csv", index=False)
    print(f"  ✅ 08_run_manifest.csv ({len(manifest_rows)} rows)")

    print(f"\n🎉 Done → {outdir}/")


if __name__ == "__main__":
    main()
