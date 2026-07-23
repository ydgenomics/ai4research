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

def parse_expression_column(value: object) -> np.ndarray:
    """将 JSON 字符串解析为 float32 numpy 数组。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=np.float32)
    text = str(value).strip()
    if not text:
        return np.array([], dtype=np.float32)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return np.asarray(parsed, dtype=np.float32)


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
    """加载多个 CSV，重命名 chromosome，添加 strand 列，按行合并。"""
    all_dfs = []
    for t in triplets:
        df = pd.read_csv(t.csv_path)
        required = {"chromosome", "start", "end", "predicted_expression", "true_expression"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{t.csv_path} missing columns: {missing}")

        # 解析 expression 列
        df["parsed_pred"] = df["predicted_expression"].apply(parse_expression_column)
        df["parsed_true"] = df["true_expression"].apply(parse_expression_column)

        # 验证长度一致性
        df["calc_length"] = df["end"] - df["start"]
        df["parsed_length"] = df["parsed_pred"].apply(len)
        mismatch = df["calc_length"] != df["parsed_length"]
        if mismatch.any():
            print(f"  ⚠️  {t.csv_path.name}: {mismatch.sum()} rows with length mismatch, filtered")
            df = df[~mismatch].copy()

        # 重命名 chromosome 列
        if t.chromosome != "all":
            df["chromosome"] = t.chromosome

        # 添加 strand 列
        df["strand"] = t.strand

        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"  📄 Loaded {len(triplets)} CSV(s) → {len(merged)} total rows")
    return merged


def flatten_to_genome_array(
    df: pd.DataFrame, value_col: str = "parsed_pred"
) -> dict[str, np.ndarray]:
    """将逐窗口的值按染色体位置平均后拼接。

    Returns: dict[chromosome] → np.ndarray (逐碱基平均值)
    """
    pos_data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for _, row in df.iterrows():
        chrom = str(row["chromosome"])
        start = int(row["start"])
        values = np.asarray(row[value_col], dtype=float)
        for j, val in enumerate(values):
            p = start + j
            pos_data[chrom][p][0] += val
            pos_data[chrom][p][1] += 1

    result = {}
    for chrom in sorted(pos_data.keys()):
        positions = sorted(pos_data[chrom].keys())
        result[chrom] = np.array(
            [pos_data[chrom][p][0] / pos_data[chrom][p][1] for p in positions],
            dtype=np.float32,
        )
    return result


# =============================================================================
# 4. 核心指标计算
# =============================================================================

def compute_track_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Track 级指标：pcc, log1p_pcc, nozero_pcc, zero_ratio, r2。"""
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    y_true = np.asarray(y_true, dtype=np.float32).flatten()

    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    n = len(y_pred)

    if n == 0:
        return {"pcc": np.nan, "log1p_pcc": np.nan, "nozero_pcc": np.nan,
                "zero_ratio": np.nan, "r2": np.nan, "n_positions": 0}

    pcc = safe_pearson(y_true, y_pred)
    log1p_pcc = safe_pearson(np.log(y_true + 1), np.log(y_pred + 1))
    r2 = float(r2_score(y_true, y_pred))
    zero_ratio = float(np.mean(y_true == 0) * 100)

    nonzero_mask = (y_true > 0) & (y_pred > 0)
    y_true_nz = y_true[nonzero_mask]
    y_pred_nz = y_pred[nonzero_mask]
    nozero_pcc = safe_pearson(y_true_nz, y_pred_nz) if len(y_true_nz) >= 2 else np.nan

    return {
        "pcc": round(pcc, 6) if not np.isnan(pcc) else np.nan,
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
    """对每个预测窗口独立计算指标。"""
    rows = []
    for pred, true, chrom, start, end, strand in zip(
        pred_arrays, true_arrays, chromosomes, starts, ends, strands
    ):
        pred = np.asarray(pred, dtype=float)
        true = np.asarray(true, dtype=float)
        length = len(pred)
        if length != len(true) or length == 0:
            continue

        nonzero_mask = (true > 0) & (pred > 0)
        pred_nz = pred[nonzero_mask]
        true_nz = true[nonzero_mask]

        pcc_val = safe_pearson(true_nz, pred_nz) if len(pred_nz) >= 2 else np.nan
        log1p_val = safe_pearson(np.log(true_nz + 1), np.log(pred_nz + 1)) if len(pred_nz) >= 2 else np.nan
        zero_ratio = float(np.mean(true == 0) * 100)

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

        rows.append({
            "chromosome": str(chrom),
            "start": int(start),
            "end": int(end),
            "strand": str(strand),
            "length": length,
            "pcc": round(pcc_val, 6) if not np.isnan(pcc_val) else np.nan,
            "log1p_pcc": round(log1p_val, 6) if not np.isnan(log1p_val) else np.nan,
            "nozero_pcc": round(pcc_val, 6) if not np.isnan(pcc_val) else np.nan,
            "zero_ratio": round(zero_ratio, 4),
            "r2": round(r2_val, 6) if not np.isnan(r2_val) else np.nan,
            "pred_mean": round(float(np.mean(pred)), 6),
            "true_mean": round(float(np.mean(true)), 6),
        })
    return pd.DataFrame(rows)


def compute_feature_basic_metrics(
    pred_values: np.ndarray, true_values: np.ndarray
) -> dict[str, float]:
    """单个基因/外显子区间内的基础指标。"""
    pred = np.asarray(pred_values, dtype=np.float32)
    true = np.asarray(true_values, dtype=np.float32)

    if len(pred) == 0:
        return {"pcc": np.nan, "log1p_pcc": np.nan, "r2": np.nan}

    nonzero_mask = (pred > 0) & (true > 0)
    pred_nz = pred[nonzero_mask]
    true_nz = true[nonzero_mask]

    if len(pred_nz) >= 2 and len(np.unique(pred_nz)) > 1 and len(np.unique(true_nz)) > 1:
        pcc_val = float(stats.pearsonr(pred_nz, true_nz).statistic)
        log1p_val = safe_pearson(np.log(true_nz + 1), np.log(pred_nz + 1))
    else:
        pcc_val = np.nan
        log1p_val = np.nan

    diff = true - pred
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

    return {
        "pcc": round(pcc_val, 6) if not np.isnan(pcc_val) else np.nan,
        "log1p_pcc": round(log1p_val, 6) if not np.isnan(log1p_val) else np.nan,
        "r2": round(r2_val, 6) if not np.isnan(r2_val) else np.nan,
    }


def compute_feature_mean_correlation(
    df: pd.DataFrame, min_features: int = 3
) -> dict[str, float]:
    """跨基因的均值相关性。"""
    valid = df.dropna(subset=["pred_mean", "true_mean"])
    if len(valid) < min_features:
        return {"pcc": np.nan, "log1p_pcc": np.nan, "nozero_pcc": np.nan, "r2": np.nan}

    pred = valid["pred_mean"].to_numpy(dtype=float)
    true = valid["true_mean"].to_numpy(dtype=float)

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

    nz_mask = (pred > 0) & (true > 0)
    nozero_pcc = safe_pearson(pred[nz_mask], true[nz_mask]) if np.sum(nz_mask) >= min_features else np.nan

    return {
        "pcc": round(safe_pearson(pred, true), 6),
        "log1p_pcc": round(safe_pearson(np.log(pred + 1), np.log(true + 1)), 6),
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


def compute_stratified_metrics(
    df: pd.DataFrame, feature_type: str = "",
    ref_table: Optional[pd.DataFrame] = None,
    biosample: str = "", normalize: str = "global_mean",
) -> pd.DataFrame:
    """按 expression_bucket 分组计算指标 (gene 级别含 delta_pcc)。"""
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


def aggregate_to_features(
    df: pd.DataFrame, features_by_chrom: dict[str, list[Feature]],
    min_overlap_bp: int = 1,
) -> pd.DataFrame:
    """将逐窗口预测值与 GFF 特征做 overlap 聚合。"""
    feature_data: dict[str, list[Feature]] = {
        chrom: feats for chrom, feats in features_by_chrom.items()
    }
    feature_values: dict[tuple, dict] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Aggregating to features", leave=False):
        chrom = str(row["chromosome"])
        window_start = int(row["start"])
        window_end = int(row["end"])
        pred = np.asarray(row["parsed_pred"], dtype=float)
        true = np.asarray(row["parsed_true"], dtype=float)

        feats = feature_data.get(chrom, [])
        if not feats:
            continue

        for feat in feats:
            if feat.end0 <= window_start:
                continue
            if feat.start0 >= window_end:
                break

            overlap_start = max(window_start, feat.start0)
            overlap_end = min(window_end, feat.end0)
            if overlap_end <= overlap_start:
                continue

            src_start = overlap_start - window_start
            src_end = overlap_end - window_start

            key = (feat.feature_type, feat.feature_id, feat.chrom)
            entry = feature_values.setdefault(key, {
                "feature": feat,
                "pred_sum": defaultdict(float),
                "true_sum": defaultdict(float),
                "counts": defaultdict(int),
            })
            for offset in range(overlap_start, overlap_end):
                entry["pred_sum"][offset] += float(pred[src_start + (offset - overlap_start)])
                entry["true_sum"][offset] += float(true[src_start + (offset - overlap_start)])
                entry["counts"][offset] += 1

    rows = []
    for entry in feature_values.values():
        feat = entry["feature"]
        covered = sorted(entry["counts"].keys())
        overlap_bp = len(covered)
        if overlap_bp < min_overlap_bp:
            continue

        pred_vals = np.array([entry["pred_sum"][p] / entry["counts"][p] for p in covered], dtype=np.float32)
        true_vals = np.array([entry["true_sum"][p] / entry["counts"][p] for p in covered], dtype=np.float32)
        metrics = compute_feature_basic_metrics(pred_vals, true_vals)

        nonzero_mask = (pred_vals > 0) & (true_vals > 0)
        eval_length = covered[-1] - covered[0] + 1 if covered else 0
        feature_length = feat.end0 - feat.start0

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

    # ---- 逐染色体 track 级 ----
    chrom_track = {}
    for chrom in sorted(set(df["chromosome"])):
        sub = df[df["chromosome"] == chrom]
        sp = flatten_to_genome_array(sub, "parsed_pred").get(chrom, np.array([], dtype=np.float32))
        st = flatten_to_genome_array(sub, "parsed_true").get(chrom, np.array([], dtype=np.float32))
        if len(sp) > 0 and len(st) > 0:
            chrom_track[chrom] = compute_track_metrics(sp, st)
    results["chrom_track"] = chrom_track

    # ---- 窗口级 ----
    print("  📐 Window-level...")
    pred_arrays = [np.asarray(v, dtype=float) for v in df["parsed_pred"]]
    true_arrays = [np.asarray(v, dtype=float) for v in df["parsed_true"]]
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

            fdf = aggregate_to_features(df, features_by_chrom, config.min_overlap_bp)
            if not fdf.empty:
                for key, value in context.items():
                    fdf.insert(0, key, value)
                results["feature_df"] = fdf

                for ftype in fdf["feature_type"].unique():
                    part = fdf[fdf["feature_type"] == ftype]
                    corr = compute_feature_mean_correlation(part)
                    results["feature_summary"][ftype] = {
                        "n_features": len(part),
                        "n_valid": len(part.dropna(subset=["pred_mean", "true_mean"])),
                        **corr,
                    }
        except FileNotFoundError as e:
            print(f"     ⚠️ Skipping feature-level: {e}")

    return results


def build_main_summary(
    all_results: list[dict], config: EvalConfig,
    bucket_thresholds=None, ref_table: Optional[pd.DataFrame] = None,
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
            # 筛选该 strand 的数据
            df_sub = r["df"]
            if "strand" in df_sub.columns:
                df_strand = df_sub[df_sub["strand"] == strand_val]
            else:
                df_strand = df_sub

            if df_strand.empty:
                continue

            # ---- global=sample: 跨染色体聚合 ----
            sp = flatten_to_genome_array(df_strand, "parsed_pred")
            st = flatten_to_genome_array(df_strand, "parsed_true")
            all_p = []
            all_t = []
            for chrom in sorted(set(list(sp.keys()) + list(st.keys()))):
                if chrom in sp and chrom in st:
                    all_p.append(sp[chrom])
                    all_t.append(st[chrom])
            gp = np.concatenate(all_p) if all_p else np.array([], dtype=np.float32)
            gt = np.concatenate(all_t) if all_t else np.array([], dtype=np.float32)

            if len(gp) > 0:
                track = compute_track_metrics(gp, gt)
                rows.append({
                    "sample": ctx["sample"], "biosample": ctx["biosample"],
                    "split": ctx["split"], "global": "sample",
                    "chromosome": "all", "resolution": "bp",
                    "strand": strand_val, **track,
                })

            # Feature 级 (exon/gene)
            fdf = r.get("feature_df")
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
                            normalize="global_mean",
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
                                "log1p_pcc": srow.get("log1p_pcc", np.nan),
                                "nozero_pcc": srow.get("nozero_pcc", np.nan),
                                "zero_ratio": np.nan,
                                "r2": srow.get("r2", np.nan),
                                "delta_pcc": srow.get("delta_pcc", np.nan),
                            })

            # ---- global=chromosome: 逐染色体 ----
            for chrom in sorted(df_strand["chromosome"].unique()):
                sub = df_strand[df_strand["chromosome"] == chrom]
                sp = flatten_to_genome_array(sub, "parsed_pred").get(chrom, np.array([], dtype=np.float32))
                st = flatten_to_genome_array(sub, "parsed_true").get(chrom, np.array([], dtype=np.float32))

                if len(sp) > 0 and len(st) > 0:
                    track = compute_track_metrics(sp, st)
                    rows.append({
                        "sample": ctx["sample"], "biosample": ctx["biosample"],
                        "split": ctx["split"], "global": "chromosome",
                        "chromosome": chrom, "resolution": "bp",
                        "strand": strand_val, **track,
                    })

                # 逐染色体的 feature 级
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
                                    normalize="global_mean",
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
                                        "log1p_pcc": srow.get("log1p_pcc", np.nan),
                                        "nozero_pcc": srow.get("nozero_pcc", np.nan),
                                        "zero_ratio": np.nan,
                                        "r2": srow.get("r2", np.nan),
                                        "delta_pcc": srow.get("delta_pcc", np.nan),
                                    })

    df_out = pd.DataFrame(rows)
    # 确保列顺序
    cols = ["sample", "biosample", "split", "global", "chromosome",
            "resolution", "strand", "pcc", "log1p_pcc", "nozero_pcc",
            "zero_ratio", "r2", "delta_pcc"]
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
    """构建跨品种差异表达表。"""
    # 按 (biosample, chromosome) 分组
    groups: dict[tuple, list[dict]] = {}
    for r in all_results:
        if r.get("error"):
            continue
        fdf = r.get("feature_df")
        if fdf is None or fdf.empty:
            continue
        ctx = r["context"]
        # 按 (biosample) 分组即可 — chromosome 已在 feature_df 中有
        key = ctx["biosample"]
        groups.setdefault(key, []).append(r)

    rows = []
    for biosample, group_results in groups.items():
        species_dfs: dict[str, pd.DataFrame] = {}
        species_splits: dict[str, str] = {}
        for r in group_results:
            fdf = r.get("feature_df")
            if fdf is None or fdf.empty:
                continue
            sp = r["context"]["sample"]
            gene_df = fdf[fdf["feature_type"] == "gene"].set_index("feature_id")
            if not gene_df.empty:
                species_dfs[sp] = gene_df
                species_splits[sp] = r["context"]["split"]

        species_list = sorted(species_dfs.keys())
        if len(species_list) < 2:
            continue

        for sp_a, sp_b in combinations(species_list, 2):
            df_a = species_dfs[sp_a]
            df_b = species_dfs[sp_b]
            common_ids = df_a.index.intersection(df_b.index)
            if len(common_ids) < 10:
                continue

            split_a = species_splits.get(sp_a, "unknown")
            split_b = species_splits.get(sp_b, "unknown")
            pair_type = "train_train" if split_a == "train" and split_b == "train" else "train_test"

            metrics = compute_pairwise_delta_metrics(
                df_a.loc[common_ids, "pred_mean"].to_numpy(dtype=float),
                df_a.loc[common_ids, "true_mean"].to_numpy(dtype=float),
                df_b.loc[common_ids, "pred_mean"].to_numpy(dtype=float),
                df_b.loc[common_ids, "true_mean"].to_numpy(dtype=float),
            )
            rows.append({
                "biosample": biosample,
                "cultivar_a": sp_a,
                "cultivar_b": sp_b,
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
    all_results = []
    for task in tqdm(config.tasks, desc="Evaluating", unit="task"):
        result = evaluate_one_task(task, config, features_cache)
        all_results.append(result)

    # 全局阈值 + 参考表
    bucket_thresholds = _compute_global_thresholds(all_results, config.n_expression_buckets)
    ref_table = _build_ref_table(all_results, "global_mean")
    if not ref_table.empty:
        print(f"  Built reference table: {len(ref_table)} genes")

    # 写输出
    print(f"\n{'='*60}")
    print("📝 Writing outputs...")

    outdir = config.output_dir

    # 00_main_summary.csv
    df = build_main_summary(all_results, config, bucket_thresholds, ref_table)
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
