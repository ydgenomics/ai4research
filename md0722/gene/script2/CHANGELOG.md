# Changelog

All notable changes to the `script2` SNP embedding gene expression modeling pipeline.

---

## [0.3.0] - 2026-07-20

### Changed (Notebook)

- **Notebook restructured**: clear sections with input/output tables, data flow diagram, consolidated config
- **Removed BED/GTF dependencies**: genes discovered from `meta.json` only; sorted by chrom/start; `N_GENES` controls count
- **Simplified gene selection**: single `N_GENES` parameter replaces multi-size gene sets (`genes_by_size`)
- **Merged hyperparameters**: all training params (LR, dropout, etc.) in one cell; removed `GENE_SET_SIZE` duplication
- **Updated descriptions**: title/model now described as predicting absolute log2(TPM), not "expression difference"
- **Fixed `head(sample_table)` → `display(sample_table.head())`**
- **Removed debug cells** (bare `print(len(...))` and `selected['pearson'].sum()/10`)

### Fixed

- **Gene filtering on NaN targets**: `build_multi_gene_data_from_matrix` now detects genes with zero training samples (all NaN in expression matrix), removes them, and remaps gene IDs to stay 0-contiguous. Prevents `gene_id=X has no training targets` crash.
- **`table["gene_id"]` sync**: updated alongside `gene_ids_arr` after gene filtering
- **Cached delta alignment**: new `precompute_snp_deltas_from_table()` builds cached array aligned to final `table` rows (after NaN/gene filtering), fixing `IndexError: index 95 is out of bounds for axis 0 with size 87`
- **`build_multi_gene_data_from_matrix`** now returns filtered `genes` list as 11th tuple element
- **`run_experiment`** auto-updates `model_config["n_genes"]` from filtered gene count
- **`build_multi_gene_data` (BigWig)** also returns `genes` for interface parity

### Hyperparameter Changes

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `DROPOUT` | 0.0 | 0.3 | Prevent overfitting (train >> test gap) |
| `WEIGHT_DECAY` | 0.0 | 1e-3 | L2 regularization |

---

## [0.2.0] - 2026-07-15

### Added
- `CachedSpecificSNPDataset`: memory-cached dataset that precomputes all SNP deltas once before training, eliminating per-batch disk I/O
- `precompute_snp_deltas()`: helper to load all `.vcf.pt` files, subtract gene centers, and pack into a single `[total_samples, max_snps, hidden_dim]` float16 buffer
- `build_multi_gene_data_from_matrix` now accepts `cache_deltas=True` (default) and returns `(cached_delta, cached_mask)` as extra tuple elements
- `build_multi_gene_data` (BigWig) returns `(None, None)` as compatible placeholders

### Changed
- `BATCH_SIZE` default increased from 2 → 64 (A40 GPU utilization was <10%)
- `run_experiment()` auto-selects `CachedSpecificSNPDataset` when cache is available
- Training Cell in `main.ipynb` updated with cache-mode comments

### Resource Impact
| Mode | 10 genes | 50 genes | 97 genes |
|------|----------|----------|----------|
| Cache (float16) | ~5 GB | ~25 GB | ~47 GB |
| GPU training | <6 GB @ BS=64 | <10 GB | <18 GB |

---

## [0.1.0] - 2026-07-15

### Added
- Initial refactored pipeline: `src/` + `main.ipynb`
- `SpecificSNPRegressor` model (~85K params): projection + position + gene_embedding + masked attention + head
- `SpecificSNPTransformerCNN` model: Transformer + multi-scale CNN hybrid
- `PositionBinnedRegressor`: alternative binned architecture
- Per-gene Z-score target scaling (`fit_gene_target_scalers`, `scale`, `inverse`)
- Pairwise difference loss (`pairwise_difference_loss`, `pairwise_difference_mse_loss`, `mixed_loss`)
- `SameGenePairBatchSampler` for pairwise training
- Expression matrix target loading (`load_expression_matrix`, `build_multi_gene_data_from_matrix`)
- ID harmonization: embedding short IDs (H005) ↔ expression matrix IDs (CIMA-H005)
- Full train→eval→checkpoint pipeline (`Trainer`, `run_experiment`)
- Evaluation: per-gene metrics, macro aggregation, overall metrics

### Ported from
- xxl's `SNPembedding/` project (original `specific_snp_model.py`, `position_binned_model.py`, `run_multi_gene_models.ipynb`)
