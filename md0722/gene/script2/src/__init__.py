# SNP Embedding Gene Expression Modeling
# Refactored from xxl's SNPembedding project
from .models import SpecificSNPRegressor, SpecificSNPTransformerCNN
from .preprocessing import (
    center_and_pad_snp_hidden,
    fit_snp_centers,
    normalize_snp_positions,
)
from .losses import pairwise_difference_loss, pairwise_difference_mse_loss
from .metrics import per_gene_metrics, macro_gene_metrics, regression_metrics
from .scaling import fit_gene_target_scalers, scale_gene_targets, inverse_scale_gene_targets
from .data import (
    GeneRecord,
    SpecificSNPDataset,
    CachedSpecificSNPDataset,
    precompute_snp_deltas,
    build_multi_gene_data,
    build_multi_gene_data_from_matrix,
    load_expression_matrix,
    extract_gene_symbol,
    read_genes_from_embedding_dir,
    read_genes_from_bed,
    order_genes_by_bed,
    read_sample_table,
)
from .training import Trainer, run_experiment
from .checkpoint import build_checkpoint, load_checkpoint
