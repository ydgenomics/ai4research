#!/usr/bin/env python3
"""
metrics_core.py — 核心指标计算模块（无IO依赖，纯数学计算）

提供三个层级的所有指标函数：
  - Track级（碱基级）：全染色体逐碱基拼接后的指标
  - Segment级（窗口级）：逐窗口指标 + 聚合统计
  - Feature级（基因/外显子级）：基因区间聚合 + 表达分桶 + Delta Pearson

设计原则：
  - 纯函数，输入 numpy 数组，输出 dict/DataFrame
  - 所有函数独立可测，不依赖文件系统
  - 与 OneGenome-Rice 论文的指标定义保持一致
"""

from __future__ import annotations

import json
import ast
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from math import sqrt


# =============================================================================
# 0. 工具函数
# =============================================================================

def parse_expression_column(value: object) -> np.ndarray:
    """将 JSON 字符串（如 "[1.2, 0.5, 0.0, ...]"）解析为 float32 numpy 数组。

    支持 json.loads 和 ast.literal_eval 两种解析方式。
    """
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
    """安全计算 Pearson 相关系数，处理退化情况返回 NaN。"""
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
# 1. Track级（碱基级）指标
# =============================================================================

def compute_track_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """计算全染色体逐碱基拼接后的 12 项核心指标。

    Parameters
    ----------
    y_pred : np.ndarray, shape (n_positions,)
        逐碱基预测值（多窗口按位置平均后拼接）。
    y_true : np.ndarray, shape (n_positions,)
        逐碱基真实值。

    Returns
    -------
    dict[str, float]
        包含 12 项指标和 n_positions 的字典。
    """
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    y_true = np.asarray(y_true, dtype=np.float32).flatten()

    # 过滤 NaN/Inf
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    n = len(y_pred)

    if n == 0:
        return {k: np.nan for k in _track_metric_keys()}

    # ---- 全位点指标 ----
    pearson = safe_pearson(y_true, y_pred)
    spearman = safe_spearman(y_true, y_pred)
    log_pearson = safe_pearson(np.log(y_true + 1), np.log(y_pred + 1))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = sqrt(mse)

    # ---- 零膨胀指标 ----
    true_zero = (y_true == 0)
    zero_ratio = float(np.mean(true_zero) * 100)

    try:
        zero_auroc = float(roc_auc_score(true_zero, -y_pred))
        zero_auprc = float(average_precision_score(true_zero, -y_pred))
    except ValueError:
        zero_auroc = float("nan")
        zero_auprc = float("nan")

    # ---- 非零位点指标 ----
    nonzero_mask = (y_true > 0) & (y_pred > 0)
    y_true_nz = y_true[nonzero_mask]
    y_pred_nz = y_pred[nonzero_mask]

    if len(y_true_nz) >= 2:
        nonzero_pearson = safe_pearson(y_true_nz, y_pred_nz)
        nonzero_spearman = safe_spearman(y_true_nz, y_pred_nz)
    else:
        nonzero_pearson = float("nan")
        nonzero_spearman = float("nan")

    return {
        "n_positions": n,
        "pearson": round(pearson, 6),
        "spearman": round(spearman, 6),
        "log_pearson": round(log_pearson, 6),
        "r2": round(r2, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "zero_auroc": round(zero_auroc, 6),
        "zero_auprc": round(zero_auprc, 6),
        "nonzero_pearson": round(nonzero_pearson, 6),
        "nonzero_spearman": round(nonzero_spearman, 6),
        "zero_ratio": round(zero_ratio, 4),
    }


def _track_metric_keys() -> list[str]:
    return [
        "n_positions", "pearson", "spearman", "log_pearson", "r2",
        "mae", "rmse", "zero_auroc", "zero_auprc",
        "nonzero_pearson", "nonzero_spearman", "zero_ratio",
    ]


# =============================================================================
# 2. Segment级（窗口级）指标
# =============================================================================

def compute_segment_metrics_per_window(
    pred_arrays: list[np.ndarray],
    true_arrays: list[np.ndarray],
    chromosomes: list[str],
    starts: list[int],
    ends: list[int],
) -> pd.DataFrame:
    """对每个预测窗口独立计算指标。

    Parameters
    ----------
    pred_arrays : list of np.ndarray
        每个窗口的预测值数组。
    true_arrays : list of np.ndarray
        每个窗口的真实值数组。
    chromosomes, starts, ends : 窗口坐标信息。

    Returns
    -------
    pd.DataFrame, 每行一个窗口，列包含 pearson_corr, spearman_corr, mse, mae 等。
    """
    rows = []
    for pred, true, chrom, start, end in zip(pred_arrays, true_arrays, chromosomes, starts, ends):
        pred = np.asarray(pred, dtype=float)
        true = np.asarray(true, dtype=float)
        length = len(pred)

        if length != len(true) or length == 0:
            continue

        nonzero_mask = (true > 0) & (pred > 0)
        pred_nz = pred[nonzero_mask]
        true_nz = true[nonzero_mask]

        pearson_corr = safe_pearson(true_nz, pred_nz) if len(pred_nz) >= 2 else np.nan
        spearman_corr = safe_spearman(true_nz, pred_nz) if len(pred_nz) >= 2 else np.nan
        log_pearson_corr = safe_pearson(np.log(true_nz + 1), np.log(pred_nz + 1)) if len(pred_nz) >= 2 else np.nan
        mse_val = float(np.mean((pred - true) ** 2))
        mae_val = float(np.mean(np.abs(pred - true)))

        rows.append({
            "chromosome": str(chrom),
            "start": int(start),
            "end": int(end),
            "length": length,
            "pearson_corr": round(pearson_corr, 6) if not np.isnan(pearson_corr) else np.nan,
            "spearman_corr": round(spearman_corr, 6) if not np.isnan(spearman_corr) else np.nan,
            "log_pearson_corr": round(log_pearson_corr, 6) if not np.isnan(log_pearson_corr) else np.nan,
            "mse": round(mse_val, 6),
            "mae": round(mae_val, 6),
            "pred_mean": round(float(np.mean(pred)), 6),
            "true_mean": round(float(np.mean(true)), 6),
            "non_zero_count": int(np.sum(nonzero_mask)),
        })
    return pd.DataFrame(rows)


def compute_segment_level_summary(
    per_window_df: pd.DataFrame,
    global_pred: np.ndarray,
    global_true: np.ndarray,
) -> dict[str, float]:
    """从逐窗口指标计算聚合统计。

    Parameters
    ----------
    per_window_df : 逐窗口 DataFrame（compute_segment_metrics_per_window 的输出）。
    global_pred, global_true : 全局打平的一维数组，用于计算全局 R²。

    Returns
    -------
    dict[str, float]
    """
    if per_window_df.empty:
        return {"n_windows": 0, "n_valid_windows": 0, "nonzero_pearson_global": np.nan}

    valid = per_window_df.dropna(subset=["pearson_corr"])

    ss_res = np.sum((global_true - global_pred) ** 2)
    ss_tot = np.sum((global_true - np.mean(global_true)) ** 2)
    r2_global = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

    # ---- 全局非零位点 Pearson ----
    nz_mask = (global_true > 0) & (global_pred > 0)
    if nz_mask.sum() >= 2:
        nonzero_pearson_global = safe_pearson(global_true[nz_mask], global_pred[nz_mask])
    else:
        nonzero_pearson_global = np.nan

    return {
        "n_windows": int(len(per_window_df)),
        "n_valid_windows": int(len(valid)),
        "pearson_mean": round(float(valid["pearson_corr"].mean()), 6),
        "pearson_median": round(float(valid["pearson_corr"].median()), 6),
        "pearson_std": round(float(valid["pearson_corr"].std()), 6),
        "spearman_mean": round(float(valid["spearman_corr"].mean()), 6),
        "spearman_median": round(float(valid["spearman_corr"].median()), 6),
        "spearman_std": round(float(valid["spearman_corr"].std()), 6),
        "log_pearson_mean": round(float(valid["log_pearson_corr"].mean()), 6),
        "log_pearson_median": round(float(valid["log_pearson_corr"].median()), 6),
        "mse_mean": round(float(per_window_df["mse"].mean()), 6),
        "mae_mean": round(float(per_window_df["mae"].mean()), 6),
        "r2_global": round(r2_global, 6),
        "nonzero_pearson_global": round(nonzero_pearson_global, 6) if not np.isnan(nonzero_pearson_global) else np.nan,
        "pred_mean_global": round(float(np.mean(global_pred)), 6),
        "true_mean_global": round(float(np.mean(global_true)), 6),
    }


# =============================================================================
# 3. Feature级（基因/外显子级）指标
# =============================================================================

def compute_feature_basic_metrics(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    min_nonzero_bp: int = 2,
    min_r2_variance: float = 1e-8,
) -> dict[str, float]:
    """计算单个基因/外显子区间内的基础指标。

    Parameters
    ----------
    pred_values, true_values : 该 feature 区间内逐碱基的值（已按位置平均）。
    min_nonzero_bp : 计算 Pearson/Spearman 所需的最小非零位点数。
    min_r2_variance : R² 分母的最小方差阈值。

    Returns
    -------
    dict[str, float]
    """
    pred = np.asarray(pred_values, dtype=np.float32)
    true = np.asarray(true_values, dtype=np.float32)

    if len(pred) == 0:
        return {"pearson": np.nan, "spearman": np.nan, "log_pearson": np.nan, "r2": np.nan, "mse": np.nan, "mae": np.nan}

    nonzero_mask = (pred > 0) & (true > 0)
    pred_nz = pred[nonzero_mask]
    true_nz = true[nonzero_mask]

    if len(pred_nz) >= min_nonzero_bp and len(np.unique(pred_nz)) > 1 and len(np.unique(true_nz)) > 1:
        pearson = float(stats.pearsonr(pred_nz, true_nz).statistic)
        spearman = float(stats.spearmanr(pred_nz, true_nz).statistic)
        log_pearson = safe_pearson(np.log(true_nz + 1), np.log(pred_nz + 1))
    else:
        pearson = np.nan
        spearman = np.nan
        log_pearson = np.nan

    diff = true - pred
    mse_val = float(np.mean(diff ** 2))
    mae_val = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > min_r2_variance else np.nan

    nonzero_pcc_val = round(pearson, 6) if not np.isnan(pearson) else np.nan

    return {
        "pearson": nonzero_pcc_val,
        "nonzero_pcc": nonzero_pcc_val,
        "spearman": round(spearman, 6) if not np.isnan(spearman) else np.nan,
        "log_pearson": round(log_pearson, 6) if not np.isnan(log_pearson) else np.nan,
        "r2": round(r2_val, 6) if not np.isnan(r2_val) else np.nan,
        "mse": round(mse_val, 6),
        "mae": round(mae_val, 6),
    }


def compute_feature_mean_correlation(
    df: pd.DataFrame,
    min_features: int = 3,
) -> dict[str, float]:
    """跨基因的均值相关性：将所有基因的 (pred_mean, true_mean) 对做 Pearson/Spearman/R²。

    这是衡量"模型能否正确排序不同基因表达水平"的核心指标。

    Parameters
    ----------
    df : DataFrame，必须包含 pred_mean 和 true_mean 列。
    min_features : 最少需要多少个基因才计算。

    Returns
    -------
    dict[str, float]
    """
    valid = df.dropna(subset=["pred_mean", "true_mean"])
    if len(valid) < min_features:
        return {
            "feature_mean_pearson": np.nan, "feature_mean_spearman": np.nan,
            "feature_mean_log_pearson": np.nan, "feature_mean_r2": np.nan,
            "feature_mean_nonzero_pearson": np.nan,
        }

    pred = valid["pred_mean"].to_numpy(dtype=float)
    true = valid["true_mean"].to_numpy(dtype=float)

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan

    # nonzero: 只保留 pred_mean > 0 且 true_mean > 0 的基因
    nz_mask = (pred > 0) & (true > 0)
    if np.sum(nz_mask) >= min_features:
        nonzero_pearson = safe_pearson(pred[nz_mask], true[nz_mask])
    else:
        nonzero_pearson = np.nan

    return {
        "feature_mean_pearson": round(safe_pearson(pred, true), 6),
        "feature_mean_spearman": round(safe_spearman(pred, true), 6),
        "feature_mean_log_pearson": round(safe_pearson(np.log(pred + 1), np.log(true + 1)), 6),
        "feature_mean_r2": round(r2_val, 6),
        "feature_mean_nonzero_pearson": round(nonzero_pearson, 6) if not np.isnan(nonzero_pearson) else np.nan,
    }


# =============================================================================
# 4. 表达分桶
# =============================================================================

def assign_expression_buckets(
    df: pd.DataFrame,
    true_mean_col: str = "true_mean",
    n_buckets: int = 3,
    thresholds: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """按真实表达值分桶（low / medium / high）。

    Parameters
    ----------
    df : 逐基因 DataFrame。
    true_mean_col : 真实表达均值列名。
    n_buckets : 桶数量（默认3，即 low/medium/high）。
    thresholds : (low_thresh, high_thresh) 预计算的分位数阈值。
                 如果为 None，则从 df 自动计算。

    Returns
    -------
    df : 添加了 expression_bucket 列的 DataFrame。
    thresholds : 实际使用的阈值元组。
    """
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


def compute_stratified_feature_metrics(
    df: pd.DataFrame,
    feature_type: str = "",
    ref_table: pd.DataFrame | None = None,
    biosample: str = "",
    normalize: str = "global_mean",
) -> pd.DataFrame:
    """按 expression_bucket 分组计算特征级指标（含 delta_pcc）。

    Parameters
    ----------
    df : 逐特征 DataFrame，必须包含 expression_bucket, pred_mean, true_mean 列。
    feature_type : 该分组对应的特征类型（如 "exon" 或 "gene"），会作为列写入输出。
    ref_table : per_gene_reference 表，用于计算 delta_pcc；None 则跳过。
    biosample : 用于按组织匹配 ref_table。
    normalize : 归一化方式。

    Returns
    -------
    pd.DataFrame，每行一个分桶，含 delta_pcc 相关列。
    """
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
        # 仅对 gene 级别计算 delta_pcc
        if ref_table is not None and not ref_table.empty and feature_type == "gene":
            delta = compute_all_delta_pearson(part, ref_table, normalize, biosample)
            row["delta_pcc_zero"] = delta.get("gene_delta_pcc_zero", np.nan)
            row["delta_pcc_feature_ref"] = delta.get("gene_delta_pcc_feature_ref", np.nan)
            row["delta_pcc_global_ref"] = delta.get("gene_delta_pcc_global_ref", np.nan)
            row["delta_rmse"] = delta.get("gene_delta_rmse", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 5. Delta Pearson
# =============================================================================

def pearson_delta_scale(arr: np.ndarray, method: str) -> float:
    """计算 Delta Pearson 的归一化因子 μ_scale。

    Parameters
    ----------
    arr : 当前 track 的值数组。
    method : "none" | "global_mean" | "nonzero_mean"

    Returns
    -------
    float : 归一化因子。
    """
    values = np.asarray(arr, dtype=np.float64)
    if method == "none":
        return 1.0
    if method == "global_mean":
        mean_value = float(np.mean(values))
    elif method == "nonzero_mean":
        nonzero = values[values != 0.0]
        if nonzero.size == 0:
            return float("nan")
        mean_value = float(np.mean(nonzero))
    else:
        raise ValueError(f"Unknown pearson_delta normalization: {method}")
    if mean_value <= 0.0 or not np.isfinite(mean_value):
        return float("nan")
    return mean_value


def normalize_for_pearson_delta(arr: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    """归一化：arr / μ_scale。

    Returns
    -------
    (normalized_array, scale_factor)
    """
    scale = pearson_delta_scale(arr, method)
    values = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(scale):
        return np.full_like(values, np.nan, dtype=np.float64), scale
    return values / scale, scale


def resolve_pearson_delta_ref_value(
    pred: np.ndarray,
    true: np.ndarray,
    ref_mode: str,
) -> float:
    """根据 ref_mode 计算标量参照值。

    Parameters
    ----------
    ref_mode : "zero" | "true_global_mean" | "pred_true_pooled_mean"
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if ref_mode == "zero":
        return 0.0
    if ref_mode == "true_global_mean":
        return float(np.mean(true)) if true.size else float("nan")
    if ref_mode == "pred_true_pooled_mean":
        pooled = np.concatenate([pred, true]) if pred.size or true.size else np.array([], dtype=np.float64)
        return float(np.mean(pooled)) if pooled.size else float("nan")
    raise ValueError(f"Unknown pearson_delta ref mode: {ref_mode}")


def compute_pearson_delta_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    normalize: str = "global_mean",
    ref_value: float | np.ndarray = 0.0,
) -> dict[str, float]:
    """计算 Delta Pearson 及相关指标。

    公式: Pearson( (pred_norm - ref), (true_norm - ref) )

    Parameters
    ----------
    pred : np.ndarray, shape (n_features,)
        各基因的预测均值。
    true : np.ndarray, shape (n_features,)
        各基因的真实均值。
    normalize : str, "none" | "global_mean" | "nonzero_mean"
        归一化方式。
    ref_value : float or np.ndarray
        参照值。如果是标量，所有基因用同一个参照；
        如果是数组，每个基因用其专属参照（feature_normalized_mean 模式）。

    Returns
    -------
    dict[str, float]
    """
    if len(pred) == 0 or len(true) == 0:
        return {
            "pearson_delta": np.nan, "delta_rmse": np.nan,
            "pearson_delta_scale_pred": np.nan, "pearson_delta_scale_true": np.nan,
        }

    pred_norm, scale_pred = normalize_for_pearson_delta(pred, normalize)
    true_norm, scale_true = normalize_for_pearson_delta(true, normalize)

    if np.isscalar(ref_value):
        ref = np.full_like(np.asarray(true, dtype=np.float64), float(ref_value), dtype=np.float64)
    else:
        ref = np.asarray(ref_value, dtype=np.float64)
        if ref.shape != np.asarray(true).shape:
            raise ValueError(f"ref shape mismatch: ref={ref.shape}, true={np.asarray(true).shape}")

    if np.isnan(pred_norm).any() or np.isnan(true_norm).any() or np.isnan(ref).any():
        return {
            "pearson_delta": np.nan, "delta_rmse": np.nan,
            "pearson_delta_scale_pred": scale_pred, "pearson_delta_scale_true": scale_true,
        }

    delta_pred = pred_norm - ref
    delta_true = true_norm - ref
    diff = pred_norm - true_norm

    return {
        "pearson_delta": round(safe_pearson(delta_pred, delta_true), 6),
        "delta_rmse": round(float(np.sqrt(np.mean(diff ** 2))), 6),
        "pearson_delta_scale_pred": scale_pred,
        "pearson_delta_scale_true": scale_true,
    }


def build_per_gene_reference(
    feature_dfs: dict[str, pd.DataFrame],
    normalize: str = "global_mean",
) -> pd.DataFrame:
    """构建 per_gene_reference 表（feature_normalized_mean 模式所需）。

    对每个基因，按 biosample 分组，计算所有训练样本中 true_mean_norm 的均值作为 ref。

    Parameters
    ----------
    feature_dfs : dict[str, pd.DataFrame]
        key = sample_key (格式: species/chrom_unit/biosample),
        value = 该样本的逐基因 DataFrame。
    normalize : 归一化方式。

    Returns
    -------
    pd.DataFrame，列: feature_id, feature_type, biosample, delta_ref_value
    """
    all_rows = []
    for sample_key, df in feature_dfs.items():
        if df.empty:
            continue
        df = df.copy()
        # 从 sample_key 中提取 biosample（格式: species/chrom_unit/biosample）
        biosample = sample_key.rsplit("/", 1)[-1] if "/" in sample_key else ""
        # 按 feature_type 组内归一化
        for ftype in df["feature_type"].unique():
            mask = df["feature_type"] == ftype
            scale = pearson_delta_scale(df.loc[mask, "true_mean"].to_numpy(), normalize)
            df.loc[mask, "true_mean_norm"] = df.loc[mask, "true_mean"] / scale if np.isfinite(scale) else np.nan
        df["sample"] = sample_key
        df["biosample"] = biosample
        all_rows.append(df[["feature_id", "feature_type", "true_mean_norm", "sample", "biosample"]])

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
    """对单个样本计算所有 ref_mode 下的 Delta Pearson（仅 gene 级别）。

    Parameters
    ----------
    feature_df : 该样本的逐基因 DataFrame（含 pred_mean, true_mean）。
    ref_table : per_gene_reference 表（build_per_gene_reference 的输出）。
    normalize : 归一化方式。
    biosample : 该样本的 biosample 名称，用于按组织匹配 ref_table。

    Returns
    -------
    dict[str, float]，包含各个 ref_mode 的 pearson_delta。
    """
    valid = feature_df.dropna(subset=["pred_mean", "true_mean"])
    if valid.empty:
        return {}

    # 仅计算 gene 级别 delta_pcc
    valid = valid[valid["feature_type"] == "gene"]
    if valid.empty:
        return {}

    # 按 biosample 过滤 ref_table
    if biosample and "biosample" in ref_table.columns:
        ref_filtered = ref_table[ref_table["biosample"] == biosample]
    else:
        ref_filtered = ref_table

    results = {}

    for ftype in valid["feature_type"].unique():
        part = valid[valid["feature_type"] == ftype]
        pred = part["pred_mean"].to_numpy(dtype=float)
        true = part["true_mean"].to_numpy(dtype=float)

        # ---- 模式1: zero ----
        m = compute_pearson_delta_metrics(pred, true, normalize, ref_value=0.0)
        results[f"{ftype}_delta_pcc_zero"] = m["pearson_delta"]
        results[f"{ftype}_delta_rmse"] = m["delta_rmse"]

        # ---- 模式2: feature_normalized_mean ----
        merged = part[["feature_id", "feature_type"]].merge(
            ref_filtered, on=["feature_id", "feature_type"], how="left"
        )
        valid_mask = merged["delta_ref_value"].notna().to_numpy()
        if valid_mask.sum() < 2:
            results[f"{ftype}_delta_pcc_feature_ref"] = float("nan")
        else:
            ref_vals = merged.loc[valid_mask, "delta_ref_value"].to_numpy(dtype=float)
            m2 = compute_pearson_delta_metrics(
                pred[valid_mask], true[valid_mask], normalize, ref_value=ref_vals
            )
            results[f"{ftype}_delta_pcc_feature_ref"] = m2["pearson_delta"]

        # ---- 模式3: true_global_mean ----
        ref_scalar = resolve_pearson_delta_ref_value(pred, true, "true_global_mean")
        m3 = compute_pearson_delta_metrics(pred, true, normalize, ref_value=ref_scalar)
        results[f"{ftype}_delta_pcc_global_ref"] = m3["pearson_delta"]

    return results


# =============================================================================
# 6. Cross-Variety Pairwise Differential Expression
# =============================================================================

def compute_pairwise_delta_metrics(
    pred_a: np.ndarray,
    true_a: np.ndarray,
    pred_b: np.ndarray,
    true_b: np.ndarray,
) -> dict[str, float]:
    """计算两个品种之间的差异表达预测精度。

    对每个匹配的同源基因，计算品种间表达差异：
      Δ_pred = pred_A - pred_B
      Δ_true = true_A - true_B

    然后评估模型对差异幅度和方向的预测能力。

    Parameters
    ----------
    pred_a, true_a : 品种 A 的预测和真实基因均值。
    pred_b, true_b : 品种 B 的预测和真实基因均值。

    Returns
    -------
    dict[str, float]
    """
    delta_pred = pred_a - pred_b
    delta_true = true_a - true_b

    # log2 fold change (pseudocount=1, standard in differential expression)
    log2fc_pred = np.log2(pred_a + 1) - np.log2(pred_b + 1)
    log2fc_true = np.log2(true_a + 1) - np.log2(true_b + 1)

    results: dict[str, float] = {}
    results["n_genes"] = float(len(delta_pred))

    # Δ 幅度相关性
    results["delta_pearson"] = safe_pearson(delta_pred, delta_true)
    results["delta_spearman"] = safe_spearman(delta_pred, delta_true)

    # log2FC 相关性（差异表达分析的标准视角）
    results["log2fc_pearson"] = safe_pearson(log2fc_pred, log2fc_true)
    results["log2fc_spearman"] = safe_spearman(log2fc_pred, log2fc_true)

    # 方向一致性：Δ 符号匹配的比例（排除 ties）
    sign_match = np.sign(delta_pred) == np.sign(delta_true)
    nonzero_mask = (delta_pred != 0) & (delta_true != 0)
    if nonzero_mask.sum() > 0:
        results["sign_accuracy"] = float(sign_match[nonzero_mask].mean())
    else:
        results["sign_accuracy"] = float("nan")

    # Up/Down 精确率
    pred_up = delta_pred > 0
    true_up = delta_true > 0
    n_pred_up = pred_up.sum()
    n_pred_down = (~pred_up).sum()
    results["up_precision"] = float((pred_up & true_up).sum() / n_pred_up) if n_pred_up > 0 else float("nan")
    results["down_precision"] = float((~pred_up & ~true_up).sum() / n_pred_down) if n_pred_down > 0 else float("nan")

    # Δ RMSE
    results["delta_rmse"] = float(np.sqrt(np.mean((delta_pred - delta_true) ** 2)))

    return results
