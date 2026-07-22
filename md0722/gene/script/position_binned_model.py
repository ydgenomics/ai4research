from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


def bin_snp_hidden_states(
    hidden_states,
    positions,
    window_start: int,
    window_end: int,
    n_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(hidden_states, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    if states.ndim != 2 or pos.ndim != 1 or len(states) != len(pos):
        raise ValueError("hidden_states and positions must align")
    if window_end <= window_start or n_bins <= 0:
        raise ValueError("window and n_bins must be positive")
    if np.any(pos < window_start) or np.any(pos >= window_end):
        raise ValueError("SNP position outside the half-open gene window")

    bin_ids = np.minimum(
        (pos - int(window_start)) * int(n_bins) // (int(window_end) - int(window_start)),
        int(n_bins) - 1,
    )
    embedding_dim = states.shape[1]
    features = np.zeros((n_bins, embedding_dim * 2 + 1), dtype=np.float32)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.int64)
    mask = counts > 0
    for bin_id in np.flatnonzero(mask):
        selected = states[bin_ids == bin_id]
        features[bin_id, :embedding_dim] = selected.mean(axis=0)
        features[bin_id, embedding_dim : 2 * embedding_dim] = selected.max(axis=0)
        features[bin_id, -1] = np.log1p(len(selected))
    return features, mask, counts


def fit_gene_target_scalers(
    y,
    gene_ids,
    train_mask,
    n_genes: int,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    values = np.asarray(y, dtype=np.float32)
    ids = np.asarray(gene_ids, dtype=np.int64)
    mask = np.asarray(train_mask, dtype=bool)
    if values.ndim != 1 or ids.shape != values.shape or mask.shape != values.shape:
        raise ValueError("y, gene_ids, and train_mask must be aligned 1D arrays")

    means = np.empty(n_genes, dtype=np.float32)
    stds = np.empty(n_genes, dtype=np.float32)
    variable = np.empty(n_genes, dtype=bool)
    for gene_id in range(n_genes):
        selected = values[mask & (ids == gene_id)]
        if selected.size == 0:
            raise ValueError(f"gene_id={gene_id} has no training targets")
        means[gene_id] = selected.mean()
        observed_std = float(selected.std())
        variable[gene_id] = observed_std >= eps
        stds[gene_id] = observed_std if variable[gene_id] else 1.0
    return {"mean": means, "std": stds, "variable": variable}


def scale_gene_targets(y, gene_ids, scalers: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32)
    ids = np.asarray(gene_ids, dtype=np.int64)
    return ((values - scalers["mean"][ids]) / scalers["std"][ids]).astype(np.float32)


def inverse_scale_gene_targets(y_scaled, gene_ids, scalers: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(y_scaled, dtype=np.float32)
    ids = np.asarray(gene_ids, dtype=np.int64)
    return (values * scalers["std"][ids] + scalers["mean"][ids]).astype(np.float32)


class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PositionBinnedRegressor(nn.Module):
    def __init__(
        self,
        bin_feature_dim: int,
        n_genes: int,
        projection_dim: int = 128,
        gene_embedding_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bin_projection = nn.Sequential(
            nn.LayerNorm(bin_feature_dim),
            nn.Linear(bin_feature_dim, projection_dim),
            nn.GELU(),
        )
        self.conv_blocks = nn.Sequential(
            ResidualConvBlock(projection_dim, dropout),
            ResidualConvBlock(projection_dim, dropout),
        )
        self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(projection_dim * 2 + gene_embedding_dim),
            nn.Linear(projection_dim * 2 + gene_embedding_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        bin_mask: torch.Tensor,
        gene_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or bin_mask.shape != x.shape[:2]:
            raise ValueError("x and bin_mask must have shapes [batch, bins, features] and [batch, bins]")
        if torch.any(~bin_mask.any(dim=1)):
            raise ValueError("every sample must have at least one nonempty bin")
        hidden = self.bin_projection(x)
        hidden = self.conv_blocks(hidden.transpose(1, 2)).transpose(1, 2)
        mask = bin_mask.unsqueeze(-1)
        masked_sum = (hidden * mask).sum(dim=1)
        masked_mean = masked_sum / mask.sum(dim=1).clamp_min(1)
        masked_max = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min).max(dim=1).values
        gene_vector = self.gene_embedding(gene_ids)
        return self.head(torch.cat([masked_mean, masked_max, gene_vector], dim=1))


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    variance = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if len(y_true) == 0 or variance == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / variance)


def per_gene_metrics(frame: pd.DataFrame, variable_by_gene) -> pd.DataFrame:
    variable = np.asarray(variable_by_gene, dtype=bool)
    rows = []
    for (gene_id, embedding_name), group in frame.groupby(["gene_id", "embedding_name"], sort=True):
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
                "pearson": _safe_pearson(y_true, y_pred) if eligible else float("nan"),
                "r2": _safe_r2(y_true, y_pred) if eligible else float("nan"),
                "target_std": target_std,
                "prediction_std": prediction_std,
                "prediction_target_std_ratio": prediction_std / target_std if target_std > 0 else float("nan"),
                "eligible_correlation": eligible,
            }
        )
    return pd.DataFrame(rows)


def macro_gene_metrics(per_gene_frame: pd.DataFrame) -> dict[str, float | int]:
    eligible = per_gene_frame["eligible_correlation"].astype(bool)
    pearson = pd.to_numeric(per_gene_frame.loc[eligible, "pearson"], errors="coerce").dropna()
    r2 = pd.to_numeric(per_gene_frame.loc[eligible, "r2"], errors="coerce").dropna()
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


def build_checkpoint(
    *,
    model: nn.Module,
    model_config: dict,
    ordered_genes: list[dict],
    gene_to_id: dict[str, int],
    target_scalers: dict[str, np.ndarray],
    n_bins: int,
    splits: dict[str, list[str]],
    training_config: dict,
    best_epochs: dict,
) -> dict:
    return {
        "format_version": 1,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "model_config": dict(model_config),
        "ordered_genes": list(ordered_genes),
        "gene_to_id": dict(gene_to_id),
        "target_scalers": {
            key: torch.as_tensor(value).detach().cpu().clone()
            for key, value in target_scalers.items()
        },
        "binning": {"n_bins": int(n_bins), "window_semantics": "half_open"},
        "splits": {key: list(value) for key, value in splits.items()},
        "training_config": dict(training_config),
        "best_epochs": dict(best_epochs),
    }
