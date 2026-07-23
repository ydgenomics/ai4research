"""Checkpoint save / load utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def build_checkpoint(
    *,
    model: nn.Module,
    model_config: dict,
    ordered_genes: list[dict],
    gene_to_id: dict[str, int],
    target_scalers: dict[str, np.ndarray],
    snp_centers: Sequence[np.ndarray],
    positions: Sequence[np.ndarray],
    variant_keys: Sequence[Sequence[str]],
    max_snps: int,
    splits: dict[str, list[str]],
    training_config: dict,
    best_validation: dict,
    best_training: dict,
) -> dict:
    """Build a self-contained checkpoint dict.

    Contains everything needed to reload the model and reproduce predictions:
    weights, gene metadata, preprocessing parameters, and train/val/test splits.
    """
    source = str(training_config.get("embedding_source", "hap1"))
    if source != "hap1":
        raise ValueError("this checkpoint format expects hap1 hidden states")
    model_variant = str(
        training_config.get("model_variant", "specific_snp_delta_hap1")
    )
    return {
        "format_version": 2,
        "model_variant": model_variant,
        "model_config": dict(model_config),
        "model_state_dict": _cpu_state_dict(model.state_dict()),
        "best_validation_state_dict": _cpu_state_dict(best_validation["state"]),
        "best_training_state_dict": _cpu_state_dict(best_training["state"]),
        "ordered_genes": list(ordered_genes),
        "gene_to_id": dict(gene_to_id),
        "target_scalers": {
            k: torch.as_tensor(v).detach().cpu().clone()
            for k, v in target_scalers.items()
        },
        "snp_centers": [
            torch.as_tensor(v).detach().cpu().clone() for v in snp_centers
        ],
        "positions": [
            torch.as_tensor(v, dtype=torch.long).cpu() for v in positions
        ],
        "variant_keys": [list(values) for values in variant_keys],
        "preprocessing": {
            "embedding_source": source,
            "embedding_frozen": bool(
                training_config.get("embedding_frozen", True)
            ),
            "centering": "per_gene_per_snp_train_individual_mean",
            "max_snps": int(max_snps),
            "padding_value": 0.0,
        },
        "splits": {k: list(v) for k, v in splits.items()},
        "training_config": dict(training_config),
        "best_epochs": {
            "validation": {
                "epoch": int(best_validation["epoch"]),
                "loss": float(best_validation["loss"]),
            },
            "training": {
                "epoch": int(best_training["epoch"]),
                "loss": float(best_training["loss"]),
            },
        },
    }


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint dict from disk."""
    return torch.load(str(path), map_location="cpu", weights_only=False)
