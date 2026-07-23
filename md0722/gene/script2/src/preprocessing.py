"""SNP embedding preprocessing: centering, padding, position normalization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# SNP Embedding centering & padding
# ---------------------------------------------------------------------------

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
    train_sample_indices: np.ndarray,
) -> list[np.ndarray]:
    """Compute per-SNP mean hidden state across training individuals.

    For each gene, compute the mean hidden state at each SNP position using
    only training individuals. This "center" captures the population-average
    sequence context, so subtracting it isolates individual-specific signal.

    Args:
        hidden_by_gene: list of [n_samples, n_snps, hidden_dim] arrays
        train_sample_indices: indices of training samples

    Returns:
        list of [n_snps, hidden_dim] center arrays (one per gene)
    """
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
    """Center SNP hidden states and pad to uniform max_snps.

    For each gene × individual, compute delta = hidden - center, then
    zero-pad all genes to the same max_snps. Generates a boolean mask
    indicating real vs padded positions.

    Args:
        hidden_by_gene: list of [n_samples, n_snps, hidden_dim] arrays
        centers: list of [n_snps, hidden_dim] per-gene centers
        max_snps: target SNP count (default: max observed across genes)

    Returns:
        output: [n_genes, n_samples, max_snps, hidden_dim] delta array
        masks:  [n_genes, max_snps] boolean mask
    """
    if not hidden_by_gene or len(hidden_by_gene) != len(centers):
        raise ValueError("hidden_by_gene and centers must be nonempty and aligned")
    arrays = [_as_hidden_array(v) for v in hidden_by_gene]
    n_samples = arrays[0].shape[0]
    hidden_dim = arrays[0].shape[2]
    if any(a.shape[0] != n_samples or a.shape[2] != hidden_dim for a in arrays):
        raise ValueError("all genes must have same sample count and hidden dim")

    observed_max = max(a.shape[1] for a in arrays)
    max_snps = observed_max if max_snps is None else int(max_snps)
    if max_snps < observed_max:
        raise ValueError("max_snps is smaller than an observed SNP count")

    output = np.zeros(
        (len(arrays), n_samples, max_snps, hidden_dim), dtype=np.float32
    )
    masks = np.zeros((len(arrays), max_snps), dtype=bool)

    for gene_id, (array, center) in enumerate(zip(arrays, centers)):
        center_arr = np.asarray(center, dtype=np.float32)
        if center_arr.shape != array.shape[1:]:
            raise ValueError(
                f"center shape {center_arr.shape} != {array.shape[1:]}"
            )
        n_snps = array.shape[1]
        output[gene_id, :, :n_snps] = array - center_arr[None, :, :]
        masks[gene_id, :n_snps] = True
    return output, masks


# ---------------------------------------------------------------------------
# Position normalization
# ---------------------------------------------------------------------------

def normalize_snp_positions(
    positions_by_gene: Sequence[np.ndarray],
    starts: Sequence[int],
    ends: Sequence[int],
    max_snps: int,
) -> np.ndarray:
    """Normalize SNP genomic positions to [0, 1] within each gene's window.

    Args:
        positions_by_gene: list of [n_snps] 1-based genomic positions
        starts: gene window starts (0-based)
        ends: gene window ends (0-based, half-open)
        max_snps: pad to this length

    Returns:
        [n_genes, max_snps] normalized positions
    """
    if not (len(positions_by_gene) == len(starts) == len(ends)):
        raise ValueError("positions, starts, and ends must align")
    if max_snps <= 0:
        raise ValueError("max_snps must be positive")

    output = np.zeros((len(positions_by_gene), max_snps), dtype=np.float32)
    for gene_id, (positions, start, end) in enumerate(
        zip(positions_by_gene, starts, ends)
    ):
        pos = np.asarray(positions, dtype=np.int64)
        if pos.ndim != 1 or len(pos) > max_snps:
            raise ValueError("positions must be 1D and fit max_snps")
        zero_based = pos - 1
        if (
            int(end) <= int(start)
            or np.any(zero_based < int(start))
            or np.any(zero_based >= int(end))
        ):
            raise ValueError(
                "1-based SNP position outside the 0-based half-open gene window"
            )
        output[gene_id, : len(pos)] = (zero_based - int(start)) / float(
            int(end) - int(start)
        )
    return output


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_aligned_snp_payloads(
    payloads: Sequence[dict],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate that SNP payloads are aligned across individuals.

    All individuals for the same gene must have identical positions and
    variant_keys. Returns stacked hidden states and reference metadata.
    """
    if not payloads:
        raise ValueError("at least one payload is required")
    ref_positions = None
    ref_keys = None
    hidden_rows = []

    for payload in payloads:
        for key in ("hidden_states", "positions", "variant_keys"):
            if key not in payload:
                raise KeyError(f"payload is missing {key!r}")
        hidden = np.asarray(payload["hidden_states"], dtype=np.float32)
        positions = np.asarray(payload["positions"], dtype=np.int64).reshape(-1)
        variant_keys = tuple(str(v) for v in payload["variant_keys"])

        if hidden.ndim != 2 or len(hidden) != len(positions) or len(hidden) != len(variant_keys):
            raise ValueError("hidden_states, positions, variant_keys must align")
        if not np.isfinite(hidden).all():
            raise ValueError("hidden states must be finite")

        if ref_positions is None:
            ref_positions = positions.copy()
            ref_keys = variant_keys
        else:
            if not np.array_equal(positions, ref_positions):
                raise ValueError("positions not aligned across individuals")
            if variant_keys != ref_keys:
                raise ValueError("variant_keys not aligned across individuals")
            if hidden.shape != hidden_rows[0].shape:
                raise ValueError("hidden_states shapes not aligned")
        hidden_rows.append(hidden)

    return np.stack(hidden_rows), ref_positions, ref_keys


# ---------------------------------------------------------------------------
# Position binning (for PositionBinnedRegressor)
# ---------------------------------------------------------------------------

def bin_snp_hidden_states(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    window_start: int,
    window_end: int,
    n_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-SNP hidden states into fixed genomic bins.

    Each bin stores: mean embedding, max embedding, and log(1 + snp_count).
    """
    states = np.asarray(hidden_states, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    if states.ndim != 2 or pos.ndim != 1 or len(states) != len(pos):
        raise ValueError("hidden_states and positions must align")
    if window_end <= window_start or n_bins <= 0:
        raise ValueError("window and n_bins must be positive")

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
