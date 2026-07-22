#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SINGLE_NOTEBOOK = PROJECT_DIR / "run_single_gene_models.ipynb"
MULTI_NOTEBOOK = PROJECT_DIR / "run_multi_gene_models.ipynb"
TRANSFORMER_CNN_NOTEBOOK = PROJECT_DIR / "run_multi_gene_transformer_cnn_models.ipynb"


def notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_single_gene_notebook() -> None:
    require(SINGLE_NOTEBOOK.exists(), f"missing notebook: {SINGLE_NOTEBOOK}")
    source = notebook_source(SINGLE_NOTEBOOK)
    require("ElasticNet" not in source, "single-gene model notebook must not compute ElasticNet baseline")
    require("SingleGeneRegressor" in source, "single-gene model class is missing")
    require("run_single_gene_model" in source, "single-gene training function is missing")
    require("test_pearson" in source, "single-gene test Pearson metric is missing")
    require("test_r2" in source, "single-gene test R2 metric is missing")
    require("model_single_gene_metrics.csv" in source, "single-gene metrics output is missing")
    require("model_single_gene_predictions.csv" in source, "single-gene predictions output is missing")


def check_multi_gene_notebook() -> None:
    require(MULTI_NOTEBOOK.exists(), f"missing notebook: {MULTI_NOTEBOOK}")
    source = notebook_source(MULTI_NOTEBOOK)
    require("ElasticNet" not in source, "multi-gene model notebook must not compute ElasticNet baseline")
    require("SpecificSNPRegressor" in source, "specific-SNP model class is missing")
    require("SpecificSNPTransformerCNN" not in source, "Transformer+CNN leaked into the original notebook")
    require("GENE_SET_SIZES = [" in source, "gene-set size configuration is missing")
    require("fit_gene_snp_preprocessing" in source, "train-only per-SNP centering is missing")
    require("center_sum += states" in source, "train hidden states are not accumulated into the SNP center")
    require("center_and_pad_snp_hidden" in source, "specific-SNP delta padding is missing")
    require("snp_mask" in source, "SNP padding mask is missing")
    require('payload["positions"]' in source, "stored SNP positions are not loaded")
    require('payload["variant_keys"]' in source, "stored SNP identities are not loaded")
    require("fit_gene_target_scalers" in source, "per-gene target scaling is missing")
    require("nn.HuberLoss" in source, "Huber loss is missing")
    require("macro_gene_metrics" in source, "macro within-gene metrics are missing")
    require("prediction_target_std_ratio" in source, "prediction collapse diagnostics are missing")
    require("best_validation_state_dict" in source, "best-validation state is missing")
    require("best_training_state_dict" in source, "best-training state is missing")
    require("model_multi_gene_specific_snp_delta_10.pt" in source, "10-gene checkpoint is missing")
    require('embedding_source": "hap1"' in source, "hap1 embedding source is not explicit")
    require("alt_probability" not in source, "alt_probability must not be used")
    require("PositionBinnedRegressor" not in source, "old position-binned model remains")
    require("bin_snp_hidden_states" not in source, "old SNP position binning remains")
    require("pooled_snp_features(" not in source, "old pooled SNP features remain in multi-gene notebook")
    require('FEATURE_MODE = "pooled"' not in source, "old pooled feature mode remains")
    require("model_multi_gene_metrics_by_gene.csv" in source, "multi-gene by-gene metrics output is missing")
    require("model_multi_gene_metrics_macro.csv" in source, "multi-gene macro metrics output is missing")
    require("model_multi_gene_predictions.csv" in source, "multi-gene predictions output is missing")


def check_transformer_cnn_notebook() -> None:
    require(TRANSFORMER_CNN_NOTEBOOK.exists(), f"missing notebook: {TRANSFORMER_CNN_NOTEBOOK}")
    source = notebook_source(TRANSFORMER_CNN_NOTEBOOK)
    required = {
        "SpecificSNPTransformerCNN": "Transformer+CNN model class is missing",
        "MODEL_DIM = 64": "model dimension is missing",
        "N_HEADS = 4": "attention head count is missing",
        "N_LAYERS = 2": "Transformer layer count is missing",
        "CNN_KERNELS = (3, 5, 9, 15)": "multi-scale CNN kernels are missing",
        "DROPOUT = 0.2": "dropout regularization is missing",
        "WEIGHT_DECAY = 1e-4": "weight decay is missing",
        "PATIENCE = 15": "early-stopping patience is missing",
        "PAIRWISE_LOSS_WEIGHT = 0.25": "pairwise MSE weight is missing",
        "criterion = nn.MSELoss()": "absolute MSE loss is missing",
        "specific_snp_delta_hap1_transformer_cnn": "model variant is missing",
        "multi_gene_models_specific_snp_transformer_cnn": "dedicated output directory is missing",
        "model_multi_gene_specific_snp_transformer_cnn_delta_97.pt": "97-gene checkpoint is missing",
        "SameGenePairBatchSampler": "same-gene pair sampler is missing",
        "pairwise_difference_mse_loss": "pairwise MSE loss is missing",
        "fit_gene_snp_preprocessing": "train-only SNP centering is missing",
    }
    for text, message in required.items():
        require(text in source, message)
    require("SpecificSNPRegressor(**model_config)" not in source, "old attention model is instantiated")
    require("pairwise_difference_loss(" not in source, "Huber pairwise loss remains")
    require("nn.HuberLoss" not in source, "Huber absolute loss remains")
    require("bin_snp_hidden_states" not in source, "SNP binning must not be used")


def main() -> int:
    check_single_gene_notebook()
    check_multi_gene_notebook()
    check_transformer_cnn_notebook()
    print("model notebook checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
