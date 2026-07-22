from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Sampler

from position_binned_model import (
    fit_gene_target_scalers,
    inverse_scale_gene_targets,
    macro_gene_metrics,
    per_gene_metrics,
    scale_gene_targets,
)


def pairwise_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_ids: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
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
        losses.append(F.huber_loss(pred_diff, truth_diff, delta=float(delta), reduction="mean"))
    if not losses:
        raise ValueError("batch must contain at least one same-gene pair")
    return torch.stack(losses).mean()


def pairwise_difference_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_ids: torch.Tensor,
) -> torch.Tensor:
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


class SameGenePairBatchSampler(Sampler[list[int]]):
    def __init__(self, gene_ids, selected_indices, seed: int = 0) -> None:
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if self.gene_ids.ndim != 1 or self.selected_indices.ndim != 1:
            raise ValueError("gene_ids and selected_indices must be 1D")
        if self.selected_indices.size == 0:
            raise ValueError("selected_indices must not be empty")
        if self.selected_indices.min() < 0 or self.selected_indices.max() >= len(self.gene_ids):
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
            batches.extend(indices[offset : offset + 2].tolist() for offset in range(0, len(indices) - 1, 2))
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        selected_gene_ids = self.gene_ids[self.selected_indices]
        return int(sum(np.sum(selected_gene_ids == gene_id) // 2 for gene_id in np.unique(selected_gene_ids)))


def _as_hidden_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected [samples, snps, hidden_dim], got {array.shape}")
    if array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError("every gene must contain at least one SNP and hidden channel")
    if not np.isfinite(array).all():
        raise ValueError("hidden states must be finite")
    return array


def fit_snp_centers(
    hidden_by_gene: Sequence[np.ndarray],
    train_sample_indices,
) -> list[np.ndarray]:
    indices = np.asarray(train_sample_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("train_sample_indices must be a nonempty 1D array")
    centers = []
    for hidden in hidden_by_gene:
        array = _as_hidden_array(hidden)
        if indices.min() < 0 or indices.max() >= array.shape[0]:
            raise IndexError("training sample index is out of range")
        centers.append(array[indices].mean(axis=0, dtype=np.float32))
    return centers


def center_and_pad_snp_hidden(
    hidden_by_gene: Sequence[np.ndarray],
    centers: Sequence[np.ndarray],
    max_snps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not hidden_by_gene or len(hidden_by_gene) != len(centers):
        raise ValueError("hidden_by_gene and centers must be nonempty and aligned")
    arrays = [_as_hidden_array(value) for value in hidden_by_gene]
    n_samples = arrays[0].shape[0]
    hidden_dim = arrays[0].shape[2]
    if any(array.shape[0] != n_samples or array.shape[2] != hidden_dim for array in arrays):
        raise ValueError("all genes must have the same sample count and hidden dimension")
    observed_max = max(array.shape[1] for array in arrays)
    max_snps = observed_max if max_snps is None else int(max_snps)
    if max_snps < observed_max:
        raise ValueError("max_snps is smaller than an observed SNP count")
    output = np.zeros((len(arrays), n_samples, max_snps, hidden_dim), dtype=np.float32)
    masks = np.zeros((len(arrays), max_snps), dtype=bool)
    for gene_id, (array, center) in enumerate(zip(arrays, centers)):
        center_array = np.asarray(center, dtype=np.float32)
        if center_array.shape != array.shape[1:]:
            raise ValueError(f"center shape {center_array.shape} does not match {array.shape[1:]}")
        n_snps = array.shape[1]
        output[gene_id, :, :n_snps] = array - center_array[None, :, :]
        masks[gene_id, :n_snps] = True
    return output, masks


def normalize_snp_positions(
    positions_by_gene: Sequence[np.ndarray],
    starts: Sequence[int],
    ends: Sequence[int],
    max_snps: int,
) -> np.ndarray:
    if not (len(positions_by_gene) == len(starts) == len(ends)):
        raise ValueError("positions, starts, and ends must align")
    if max_snps <= 0:
        raise ValueError("max_snps must be positive")
    output = np.zeros((len(positions_by_gene), max_snps), dtype=np.float32)
    for gene_id, (positions, start, end) in enumerate(zip(positions_by_gene, starts, ends)):
        pos = np.asarray(positions, dtype=np.int64)
        if pos.ndim != 1 or len(pos) > max_snps:
            raise ValueError("positions must be 1D and fit max_snps")
        zero_based_pos = pos - 1
        if (
            int(end) <= int(start)
            or np.any(zero_based_pos < int(start))
            or np.any(zero_based_pos >= int(end))
        ):
            raise ValueError("1-based SNP position outside the 0-based half-open gene window")
        output[gene_id, : len(pos)] = (zero_based_pos - int(start)) / float(int(end) - int(start))
    return output


def validate_aligned_snp_payloads(payloads: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not payloads:
        raise ValueError("at least one payload is required")
    reference_positions = None
    reference_keys = None
    hidden_rows = []
    for payload in payloads:
        for key in ("hidden_states", "positions", "variant_keys"):
            if key not in payload:
                raise KeyError(f"payload is missing {key!r}")
        hidden = np.asarray(payload["hidden_states"], dtype=np.float32)
        positions = np.asarray(payload["positions"], dtype=np.int64).reshape(-1)
        variant_keys = tuple(str(value) for value in payload["variant_keys"])
        if hidden.ndim != 2 or len(hidden) != len(positions) or len(hidden) != len(variant_keys):
            raise ValueError("hidden_states, positions, and variant_keys must align")
        if not np.isfinite(hidden).all():
            raise ValueError("hidden states must be finite")
        if reference_positions is None:
            reference_positions = positions.copy()
            reference_keys = variant_keys
        else:
            if not np.array_equal(positions, reference_positions):
                raise ValueError("positions are not aligned across individuals")
            if variant_keys != reference_keys:
                raise ValueError("variant_keys are not aligned across individuals")
            if hidden.shape != hidden_rows[0].shape:
                raise ValueError("hidden_states shapes are not aligned across individuals")
        hidden_rows.append(hidden)
    return np.stack(hidden_rows), reference_positions, reference_keys


class SpecificSNPRegressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        projection_dim: int = 64,
        gene_embedding_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.snp_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(1, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(projection_dim + gene_embedding_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(projection_dim + gene_embedding_dim),
            nn.Linear(projection_dim + gene_embedding_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        snp_mask: torch.Tensor,
        relative_positions: torch.Tensor,
        gene_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or snp_mask.shape != x.shape[:2] or relative_positions.shape != x.shape[:2]:
            raise ValueError("x, snp_mask, and relative_positions must align")
        if gene_ids.ndim != 1 or len(gene_ids) != len(x):
            raise ValueError("gene_ids must have one value per sample")
        if torch.any(~snp_mask.any(dim=1)):
            raise ValueError("every sample must contain at least one real SNP")
        hidden = self.snp_projection(x) + self.position_projection(relative_positions.unsqueeze(-1))
        gene_vector = self.gene_embedding(gene_ids)
        gene_per_snp = gene_vector.unsqueeze(1).expand(-1, x.shape[1], -1)
        logits = self.attention(torch.cat([hidden, gene_per_snp], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~snp_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(torch.cat([pooled, gene_vector], dim=-1))


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        if dim <= 0 or max_len <= 0:
            raise ValueError("dim and max_len must be positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] > self.encoding.shape[1]:
            raise ValueError("input must be [batch, positions, channels] and fit max_len")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class MultiScaleConvBlock(nn.Module):
    def __init__(self, dim: int, kernels: Sequence[int], dropout: float) -> None:
        super().__init__()
        kernel_values = tuple(int(kernel) for kernel in kernels)
        if not kernel_values or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_values):
            raise ValueError("cnn kernels must be positive odd integers")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel in kernel_values
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(dim * len(kernel_values), dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        branches = [branch(channels_first).transpose(1, 2) for branch in self.branches]
        return self.projection(torch.cat(branches, dim=-1))


class SpecificSNPTransformerCNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        max_snps: int,
        model_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        gene_embedding_dim: int = 32,
        dropout: float = 0.2,
        cnn_kernels: Sequence[int] = (3, 5, 9, 15),
    ) -> None:
        super().__init__()
        if model_dim <= 0 or n_heads <= 0 or model_dim % n_heads != 0:
            raise ValueError("model_dim must be positive and divisible by n_heads")
        if n_layers <= 0 or max_snps <= 0:
            raise ValueError("n_layers and max_snps must be positive")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_encoding = SinusoidalPositionEncoding(model_dim, max_len=max_snps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.cnn = MultiScaleConvBlock(model_dim, cnn_kernels, dropout)
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim * 2 + gene_embedding_dim),
            nn.Linear(model_dim * 2 + gene_embedding_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        snp_mask: torch.Tensor,
        relative_positions: torch.Tensor,
        gene_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or snp_mask.shape != x.shape[:2] or relative_positions.shape != x.shape[:2]:
            raise ValueError("x, snp_mask, and relative_positions must align")
        if gene_ids.ndim != 1 or len(gene_ids) != len(x):
            raise ValueError("gene_ids must have one value per sample")
        if torch.any(~snp_mask.any(dim=1)):
            raise ValueError("every sample must contain at least one real SNP")
        token_mask = snp_mask.unsqueeze(-1)
        hidden = self.position_encoding(self.input_projection(x))
        hidden = hidden.masked_fill(~token_mask, 0.0)
        attention_output = self.transformer(hidden, src_key_padding_mask=~snp_mask)
        cnn_output = self.cnn(hidden)
        fused = self.fusion_norm(hidden + attention_output + cnn_output)
        fused = fused.masked_fill(~token_mask, 0.0)
        masked_mean = fused.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        masked_max = fused.masked_fill(~token_mask, torch.finfo(fused.dtype).min).max(dim=1).values
        gene_vector = self.gene_embedding(gene_ids)
        return self.head(torch.cat([masked_mean, masked_max, gene_vector], dim=1))


def _cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def build_specific_snp_checkpoint(
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
    source = str(training_config.get("embedding_source", "hap1"))
    if source != "hap1":
        raise ValueError("this model expects the existing hap1 hidden states")
    model_variant = str(training_config.get("model_variant", "specific_snp_delta_hap1"))
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
            key: torch.as_tensor(value).detach().cpu().clone()
            for key, value in target_scalers.items()
        },
        "snp_centers": [torch.as_tensor(value).detach().cpu().clone() for value in snp_centers],
        "positions": [torch.as_tensor(value, dtype=torch.long).cpu() for value in positions],
        "variant_keys": [list(values) for values in variant_keys],
        "preprocessing": {
            "embedding_source": source,
            "embedding_frozen": bool(training_config.get("embedding_frozen", True)),
            "centering": "per_gene_per_snp_train_individual_mean",
            "max_snps": int(max_snps),
            "padding_value": 0.0,
        },
        "splits": {key: list(values) for key, values in splits.items()},
        "training_config": dict(training_config),
        "best_epochs": {
            "validation": {"epoch": int(best_validation["epoch"]), "loss": float(best_validation["loss"])},
            "training": {"epoch": int(best_training["epoch"]), "loss": float(best_training["loss"])},
        },
    }
