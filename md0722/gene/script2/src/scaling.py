"""Per-gene target scaling: Z-score normalize within each gene."""

from __future__ import annotations

import numpy as np


def fit_gene_target_scalers(
    y: np.ndarray,
    gene_ids: np.ndarray,
    train_mask: np.ndarray,
    n_genes: int,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute per-gene mean and std from training targets.

    Genes with near-zero variance (std < eps) get std=1 and variable=False.

    Returns:
        {"mean": [n_genes], "std": [n_genes], "variable": [n_genes] bool}
    """
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


def scale_gene_targets(
    y: np.ndarray, gene_ids: np.ndarray, scalers: dict[str, np.ndarray]
) -> np.ndarray:
    """Z-score normalize: (y - mean) / std."""
    values = np.asarray(y, dtype=np.float32)
    ids = np.asarray(gene_ids, dtype=np.int64)
    return ((values - scalers["mean"][ids]) / scalers["std"][ids]).astype(np.float32)


def inverse_scale_gene_targets(
    y_scaled: np.ndarray, gene_ids: np.ndarray, scalers: dict[str, np.ndarray]
) -> np.ndarray:
    """Inverse Z-score: y_scaled * std + mean."""
    values = np.asarray(y_scaled, dtype=np.float32)
    ids = np.asarray(gene_ids, dtype=np.int64)
    return (values * scalers["std"][ids] + scalers["mean"][ids]).astype(np.float32)
