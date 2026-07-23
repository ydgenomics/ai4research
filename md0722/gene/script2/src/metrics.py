"""Evaluation metrics: safe Pearson/R², per-gene, macro aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation with NaN-safe handling."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score with NaN-safe handling."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    variance = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if len(y_true) == 0 or variance == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / variance)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, Pearson, R² for a set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "pearson": safe_pearson(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "target_std": float(np.std(y_true)),
        "prediction_std": float(np.std(y_pred)),
    }


def prefixed_metrics(
    prefix: str, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Return metrics dict with a prefix on each key."""
    return {
        f"{prefix}_{k}": v for k, v in regression_metrics(y_true, y_pred).items()
    }


def per_gene_metrics(
    frame: pd.DataFrame, variable_by_gene: np.ndarray
) -> pd.DataFrame:
    """Compute per-gene evaluation metrics from a prediction DataFrame.

    Expected columns: gene_id, embedding_name, target_diff, prediction_diff

    Args:
        frame: predictions with per-sample rows
        variable_by_gene: [n_genes] bool, whether gene has train variance > eps

    Returns:
        DataFrame with one row per gene
    """
    variable = np.asarray(variable_by_gene, dtype=bool)
    rows = []
    for (gene_id, embedding_name), group in frame.groupby(
        ["gene_id", "embedding_name"], sort=True
    ):
        y_true = group["target_diff"].to_numpy(dtype=float)
        y_pred = group["prediction_diff"].to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        target_std = float(np.std(y_true))
        prediction_std = float(np.std(y_pred))
        eligible = bool(variable[int(gene_id)] and target_std > 0)

        rows.append(
            {
                "gene_id": int(gene_id),
                "embedding_name": embedding_name,
                "n": int(len(y_true)),
                "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                "mae": float(np.mean(np.abs(y_pred - y_true))),
                "pearson": safe_pearson(y_true, y_pred) if eligible else float("nan"),
                "r2": safe_r2(y_true, y_pred) if eligible else float("nan"),
                "target_std": target_std,
                "prediction_std": prediction_std,
                "prediction_target_std_ratio": (
                    prediction_std / target_std if target_std > 0 else float("nan")
                ),
                "eligible_correlation": eligible,
            }
        )
    return pd.DataFrame(rows)


def macro_gene_metrics(per_gene_frame: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate per-gene metrics into macro averages.

    Only includes genes with eligible_correlation=True.
    """
    eligible = per_gene_frame["eligible_correlation"].astype(bool)
    pearson = pd.to_numeric(
        per_gene_frame.loc[eligible, "pearson"], errors="coerce"
    ).dropna()
    r2 = pd.to_numeric(
        per_gene_frame.loc[eligible, "r2"], errors="coerce"
    ).dropna()
    return {
        "macro_rmse": float(per_gene_frame["rmse"].mean()),
        "macro_mae": float(per_gene_frame["mae"].mean()),
        "macro_pearson": float(pearson.mean()) if len(pearson) else float("nan"),
        "macro_r2": float(r2.mean()) if len(r2) else float("nan"),
        "n_genes_total": int(len(per_gene_frame)),
        "n_genes_variable": int(eligible.sum()),
        "n_pearson_valid": int(len(pearson)),
        "n_r2_valid": int(len(r2)),
    }
