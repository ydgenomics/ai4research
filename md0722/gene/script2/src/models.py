"""Model architectures for SNP-embedding-based gene expression prediction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


# ---------------------------------------------------------------------------
# SpecificSNPRegressor — 轻量级 Attention Pooling 模型
# ---------------------------------------------------------------------------

class SpecificSNPRegressor(nn.Module):
    """Attention-pooling model for per-gene expression prediction from SNP embeddings.

    Input:
        x:                 [B, max_snps, hidden_dim]   SNP delta (after centering)
        snp_mask:          [B, max_snps]               True = real SNP, False = padding
        relative_positions: [B, max_snps]               [0,1] normalized genomic position
        gene_ids:          [B]                          gene index

    Output:
        [B, 1]  predicted expression difference
    """

    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        projection_dim: int = 64,
        gene_embedding_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Compress 1024-dim Genos hidden → 64-dim
        self.snp_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
        )
        # Encode scalar position → 64-dim space
        self.position_projection = nn.Sequential(
            nn.Linear(1, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        # Per-gene conditioning vector
        self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)
        # SNP-level attention (conditioned on gene identity)
        self.attention = nn.Sequential(
            nn.Linear(projection_dim + gene_embedding_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, 1),
        )
        # Output head
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
        _validate_inputs(x, snp_mask, relative_positions, gene_ids)

        # Project SNP embedding + inject position signal
        hidden = self.snp_projection(x) + self.position_projection(
            relative_positions.unsqueeze(-1)
        )  # [B, max_snps, proj_dim]

        # Expand gene embedding to per-SNP
        gene_vector = self.gene_embedding(gene_ids)  # [B, gene_dim]
        gene_per_snp = gene_vector.unsqueeze(1).expand(-1, x.shape[1], -1)

        # Attention over SNPs (masked)
        logits = self.attention(
            torch.cat([hidden, gene_per_snp], dim=-1)
        ).squeeze(-1)  # [B, max_snps]
        logits = logits.masked_fill(~snp_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)  # [B, max_snps]

        # Weighted sum → gene-level representation
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)  # [B, proj_dim]

        return self.head(torch.cat([pooled, gene_vector], dim=-1))


# ---------------------------------------------------------------------------
# SpecificSNPTransformerCNN — 更强的 Transformer+CNN 混合模型
# ---------------------------------------------------------------------------

class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding for SNP sequence positions."""

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
        encoding[:, 1::2] = torch.cos(
            position * div_term[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                "input must be [batch, positions, channels] and fit max_len"
            )
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class MultiScaleConvBlock(nn.Module):
    """Multi-kernel 1D convolution block for capturing LD blocks at different scales."""

    def __init__(self, dim: int, kernels: Sequence[int], dropout: float) -> None:
        super().__init__()
        kernel_values = tuple(int(k) for k in kernels)
        if not kernel_values or any(k <= 0 or k % 2 == 0 for k in kernel_values):
            raise ValueError("cnn kernels must be positive odd integers")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in kernel_values
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(dim * len(kernel_values), dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] → [B, C, L] → conv → [B, L, C]
        channels_first = x.transpose(1, 2)
        branches = [
            branch(channels_first).transpose(1, 2) for branch in self.branches
        ]
        return self.projection(torch.cat(branches, dim=-1))


class SpecificSNPTransformerCNN(nn.Module):
    """Transformer + Multi-scale CNN hybrid model for SNP embedding → expression.

    Three parallel paths:
      - Transformer: captures long-range SNP interactions (epistasis)
      - MultiScale CNN: captures local LD blocks at multiple scales
      - Residual: preserves original signal
    Fused via add + LayerNorm, then mean+max pooling with gene conditioning.
    """

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
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.cnn = MultiScaleConvBlock(model_dim, cnn_kernels, dropout)
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)

        # Head: concat(mean_pool, max_pool, gene_vector)
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
        _validate_inputs(x, snp_mask, relative_positions, gene_ids)
        token_mask = snp_mask.unsqueeze(-1)

        # Project + position encoding
        hidden = self.position_encoding(self.input_projection(x))
        hidden = hidden.masked_fill(~token_mask, 0.0)

        # Three parallel branches
        attention_output = self.transformer(hidden, src_key_padding_mask=~snp_mask)
        cnn_output = self.cnn(hidden)
        fused = self.fusion_norm(hidden + attention_output + cnn_output)
        fused = fused.masked_fill(~token_mask, 0.0)

        # Pooling: mean + max
        masked_mean = fused.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        masked_max = (
            fused.masked_fill(~token_mask, torch.finfo(fused.dtype).min)
            .max(dim=1)
            .values
        )

        gene_vector = self.gene_embedding(gene_ids)
        return self.head(torch.cat([masked_mean, masked_max, gene_vector], dim=1))


# ---------------------------------------------------------------------------
# PositionBinnedRegressor — 位置分箱模型 (备用架构)
# ---------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """Residual 1D convolutional block with GELU activation."""

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
    """Position-binned regressor: SNPs aggregated into fixed genomic bins then CNN + MLP.

    This is an alternative to SpecificSNPRegressor that first bins SNPs by
    genomic position (e.g. 32 bins across the gene window) rather than using
    per-SNP attention.
    """

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
        self, x: torch.Tensor, bin_mask: torch.Tensor, gene_ids: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 3 or bin_mask.shape != x.shape[:2]:
            raise ValueError("x/bim_mask shape mismatch")
        if torch.any(~bin_mask.any(dim=1)):
            raise ValueError("every sample needs ≥1 nonempty bin")

        hidden = self.bin_projection(x)
        hidden = self.conv_blocks(hidden.transpose(1, 2)).transpose(1, 2)

        mask = bin_mask.unsqueeze(-1)
        masked_sum = (hidden * mask).sum(dim=1)
        masked_mean = masked_sum / mask.sum(dim=1).clamp_min(1)
        masked_max = (
            hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
            .max(dim=1)
            .values
        )

        gene_vector = self.gene_embedding(gene_ids)
        return self.head(torch.cat([masked_mean, masked_max, gene_vector], dim=1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    x: torch.Tensor,
    snp_mask: torch.Tensor,
    relative_positions: torch.Tensor,
    gene_ids: torch.Tensor,
) -> None:
    if x.ndim != 3 or snp_mask.shape != x.shape[:2] or relative_positions.shape != x.shape[:2]:
        raise ValueError("x, snp_mask, and relative_positions must align")
    if gene_ids.ndim != 1 or len(gene_ids) != len(x):
        raise ValueError("gene_ids must have one value per sample")
    if torch.any(~snp_mask.any(dim=1)):
        raise ValueError("every sample must contain at least one real SNP")
