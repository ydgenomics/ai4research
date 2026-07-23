"""Loss functions and batch samplers for paired/pairwise training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# Pairwise difference losses
# ---------------------------------------------------------------------------

def pairwise_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_ids: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss on within-gene pairwise expression differences.

    For each gene, enumerate all pairs (i,j) and minimize:
        Huber((pred_i - pred_j), (target_i - target_j))

    This forces the model to learn relative ordering between individuals
    rather than absolute expression values.

    Args:
        prediction: [B, 1] or [B] predicted values
        target:     [B, 1] or [B] ground truth
        gene_ids:   [B] gene index per sample
        delta:      Huber loss threshold

    Returns:
        scalar loss (mean over all valid pairs)
    """
    pred = prediction.reshape(-1)
    truth = target.reshape(-1)
    ids = gene_ids.reshape(-1)
    if not (len(pred) == len(truth) == len(ids)):
        raise ValueError("prediction, target, and gene_ids must align")

    losses = []
    for gene_id in torch.unique(ids):
        selected = torch.nonzero(ids == gene_id, as_tuple=False).reshape(-1)
        if len(selected) < 2:
            continue
        pairs = torch.combinations(selected, r=2)
        pred_diff = pred[pairs[:, 0]] - pred[pairs[:, 1]]
        truth_diff = truth[pairs[:, 0]] - truth[pairs[:, 1]]
        losses.append(
            F.huber_loss(pred_diff, truth_diff, delta=float(delta), reduction="mean")
        )
    if not losses:
        raise ValueError("batch must contain at least one same-gene pair")
    return torch.stack(losses).mean()


def pairwise_difference_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_ids: torch.Tensor,
) -> torch.Tensor:
    """MSE version of pairwise_difference_loss (no Huber clipping)."""
    pred = prediction.reshape(-1)
    truth = target.reshape(-1)
    ids = gene_ids.reshape(-1)
    if not (len(pred) == len(truth) == len(ids)):
        raise ValueError("prediction, target, and gene_ids must align")

    losses = []
    for gene_id in torch.unique(ids):
        selected = torch.nonzero(ids == gene_id, as_tuple=False).reshape(-1)
        if len(selected) < 2:
            continue
        pairs = torch.combinations(selected, r=2)
        pred_diff = pred[pairs[:, 0]] - pred[pairs[:, 1]]
        truth_diff = truth[pairs[:, 0]] - truth[pairs[:, 1]]
        losses.append(F.mse_loss(pred_diff, truth_diff, reduction="mean"))
    if not losses:
        raise ValueError("batch must contain at least one same-gene pair")
    return torch.stack(losses).mean()


def mixed_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_ids: torch.Tensor,
    pairwise_weight: float = 0.5,
    delta: float = 1.0,
) -> torch.Tensor:
    """Mixed loss: pointwise Huber + pairwise difference.

    Args:
        pairwise_weight: weight for pairwise term (0 = pure pointwise, 1 = pure pairwise)
    """
    loss_pointwise = F.huber_loss(
        prediction.reshape(-1), target.reshape(-1), delta=delta
    )
    loss_pairwise = pairwise_difference_loss(prediction, target, gene_ids, delta)
    return loss_pointwise + pairwise_weight * loss_pairwise


# ---------------------------------------------------------------------------
# Batch sampler for pairwise training
# ---------------------------------------------------------------------------

class SameGenePairBatchSampler(Sampler[list[int]]):
    """Sampler that ensures each batch contains ≥2 samples from the same gene.

    This is required for pairwise_difference_loss to work, since it needs
    at least one pair per batch.
    """

    def __init__(
        self, gene_ids: np.ndarray, selected_indices: np.ndarray, seed: int = 0
    ) -> None:
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if self.gene_ids.ndim != 1 or self.selected_indices.ndim != 1:
            raise ValueError("gene_ids and selected_indices must be 1D")
        if self.selected_indices.size == 0:
            raise ValueError("selected_indices must not be empty")
        if (
            self.selected_indices.min() < 0
            or self.selected_indices.max() >= len(self.gene_ids)
        ):
            raise IndexError("selected sample index is out of range")
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        selected_gene_ids = self.gene_ids[self.selected_indices]
        for gene_id in np.unique(selected_gene_ids):
            indices = self.selected_indices[selected_gene_ids == gene_id].copy()
            rng.shuffle(indices)
            batches.extend(
                indices[offset : offset + 2].tolist()
                for offset in range(0, len(indices) - 1, 2)
            )
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        selected_gene_ids = self.gene_ids[self.selected_indices]
        return int(
            sum(
                np.sum(selected_gene_ids == gene_id) // 2
                for gene_id in np.unique(selected_gene_ids)
            )
        )
