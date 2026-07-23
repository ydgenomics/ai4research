"""Training loop, prediction, and experiment runner."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .metrics import macro_gene_metrics, per_gene_metrics, regression_metrics
from .scaling import inverse_scale_gene_targets


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        Subset(dataset, np.flatnonzero(indices).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dataset.collate,
    )


def predict_scaled(model, loader, device):
    """Run inference and return scaled predictions + row indices."""
    model.eval()
    predictions, row_indices = [], []
    with torch.no_grad():
        for xb, mb, pb, gb, _, ib in loader:
            pred = model(
                xb.to(device, dtype=torch.float32),
                mb.to(device),
                pb.to(device),
                gb.to(device),
            )
            predictions.append(pred.detach().cpu().numpy().reshape(-1))
            row_indices.append(ib.numpy())
    return np.concatenate(predictions), np.concatenate(row_indices)


class Trainer:
    """Trainer for SpecificSNPRegressor / SpecificSNPTransformerCNN."""

    def __init__(
        self,
        model: nn.Module,
        dataset,
        table: pd.DataFrame,
        y: np.ndarray,
        gene_ids: np.ndarray,
        scalers: dict,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 2,
        epochs: int = 300,
        patience: int = 40,
        loss_type: str = "huber",       # "huber" | "pairwise" | "mixed"
        pairwise_weight: float = 0.5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.table = table
        self.y = y
        self.gene_ids = gene_ids
        self.scalers = scalers
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_type = loss_type
        self.pairwise_weight = pairwise_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.HuberLoss(delta=1.0)

        self.train_loader = make_loader(dataset, train_mask, batch_size, True)
        self.train_eval_loader = make_loader(dataset, train_mask, batch_size, False)
        self.val_loader = make_loader(dataset, val_mask, batch_size, False)

        self.best_val = {"loss": math.inf, "epoch": 0, "state": None}
        self.best_train = {"loss": math.inf, "epoch": 0, "state": None}
        self.history: list[dict] = []

    def _compute_loss(self, pred, target, gene_ids):
        if self.loss_type == "pairwise":
            from .losses import pairwise_difference_loss
            return pairwise_difference_loss(pred, target, gene_ids)
        elif self.loss_type == "mixed":
            from .losses import mixed_loss
            return mixed_loss(pred, target, gene_ids, self.pairwise_weight)
        else:
            return self.criterion(pred, target)

    def _huber_numpy(self, pred, target):
        """Compute Huber loss in numpy for validation metric."""
        diff = np.abs(pred - target)
        return float(
            np.mean(np.where(diff < 1, 0.5 * diff ** 2, diff - 0.5))
        )

    def run(self):
        bad_epochs = 0
        for epoch in range(1, self.epochs + 1):
            # --- Train ---
            self.model.train()
            batch_losses = []
            for xb, mb, pb, gb, yb, _ in self.train_loader:
                pred = self.model(
                    xb.to(self.device, dtype=torch.float32),
                    mb.to(self.device),
                    pb.to(self.device),
                    gb.to(self.device),
                )
                loss = self._compute_loss(pred, yb.to(self.device), gb.to(self.device))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(batch_losses))

            # --- Validation ---
            val_scaled, val_rows = predict_scaled(self.model, self.val_loader, self.device)
            val_loss = self._huber_numpy(val_scaled, self.dataset.y_scaled[val_rows])

            # --- Metrics ---
            train_scaled, train_rows = predict_scaled(
                self.model, self.train_eval_loader, self.device
            )
            train_pred = inverse_scale_gene_targets(
                train_scaled, self.gene_ids[train_rows], self.scalers
            )
            val_pred = inverse_scale_gene_targets(
                val_scaled, self.gene_ids[val_rows], self.scalers
            )
            train_macro = _macro_for_rows(
                self.table, train_rows, self.y[train_rows], train_pred, self.scalers["variable"]
            )
            val_macro = _macro_for_rows(
                self.table, val_rows, self.y[val_rows], val_pred, self.scalers["variable"]
            )

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_macro_pearson": train_macro["macro_pearson"],
                    "train_macro_r2": train_macro["macro_r2"],
                    "val_macro_pearson": val_macro["macro_pearson"],
                    "val_macro_r2": val_macro["macro_r2"],
                }
            )

            # --- Checkpointing ---
            if train_loss < self.best_train["loss"]:
                self.best_train = {
                    "loss": train_loss, "epoch": epoch,
                    "state": _cpu_state_dict(self.model),
                }
            if val_loss < self.best_val["loss"]:
                self.best_val = {
                    "loss": val_loss, "epoch": epoch,
                    "state": _cpu_state_dict(self.model),
                }
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch == 1 or epoch % 10 == 0:
                print(self.history[-1], flush=True)
            if bad_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if self.best_val["state"] is None or self.best_train["state"] is None:
            raise RuntimeError("Training did not produce restorable states")
        return pd.DataFrame(self.history), self.best_val, self.best_train


def predict_all(
    model, state_dict, dataset, table, y, gene_ids, scalers, device, state_name
):
    """Return full prediction DataFrame for a given model state."""
    model.load_state_dict(state_dict)
    loader = make_loader(dataset, np.ones(len(table), dtype=bool), batch_size=2, shuffle=False)
    pred_scaled, row_indices = predict_scaled(model, loader, device)
    prediction = inverse_scale_gene_targets(pred_scaled, gene_ids[row_indices], scalers)
    frame = table.iloc[row_indices].copy()
    frame["model_state"] = state_name
    frame["target_scaled"] = dataset.y_scaled[row_indices]
    frame["prediction_scaled"] = pred_scaled
    frame["target_diff"] = y[row_indices]
    frame["prediction_diff"] = prediction
    return frame


def summarize_predictions(predictions, variable, model_variant, gene_set_size):
    """Aggregate predictions into overall, per-gene, and macro metrics."""
    overall_rows, by_gene_frames, macro_rows = [], [], []
    for state_name in predictions["model_state"].unique():
        state_frame = predictions[predictions["model_state"].eq(state_name)]
        allowed = ["train"] if state_name == "best_training" else ["train", "val", "test"]
        for split in allowed:
            sf = state_frame[state_frame["split"].eq(split)]
            overall_rows.append(
                {
                    "gene_set_size": gene_set_size,
                    "model_variant": model_variant,
                    "model_state": state_name,
                    "split": split,
                    "n": len(sf),
                    **regression_metrics(
                        sf["target_diff"].to_numpy(), sf["prediction_diff"].to_numpy()
                    ),
                }
            )
            bg = per_gene_metrics(sf, variable)
            bg.insert(0, "model_state", state_name)
            bg.insert(0, "split", split)
            bg.insert(0, "model_variant", model_variant)
            bg.insert(0, "gene_set_size", gene_set_size)
            by_gene_frames.append(bg)
            macro_rows.append(
                {
                    "gene_set_size": gene_set_size,
                    "model_variant": model_variant,
                    "model_state": state_name,
                    "split": split,
                    **macro_gene_metrics(bg),
                }
            )
    return (
        pd.DataFrame(overall_rows),
        pd.concat(by_gene_frames, ignore_index=True),
        pd.DataFrame(macro_rows),
    )


def _macro_for_rows(table, row_indices, target, prediction, variable):
    frame = table.iloc[row_indices].copy()
    frame["target_diff"] = target
    frame["prediction_diff"] = prediction
    return macro_gene_metrics(per_gene_metrics(frame, variable))


# ---------------------------------------------------------------------------
# High-level experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    genes: list,
    all_individuals: list[str],
    train_individuals: list[str],
    val_individuals: set[str],
    test_individuals: set[str],
    embedding_root: Path,
    output_dir: Path,
    *,
    # Target source: BigWig or expression matrix (mutually exclusive)
    bigwig_dir: Path | None = None,
    reference_bw_path: Path | None = None,
    sample_to_accession: dict[str, str] | None = None,
    expression_matrix_path: str | Path | pd.DataFrame | None = None,
    # Model & training
    model_class,
    model_config: dict,
    gene_set_size: int,
    model_variant: str = "specific_snp_delta_hap1",
    batch_size: int = 2,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    patience: int = 40,
    loss_type: str = "huber",
    target_std_eps: float = 1e-6,
    device: str = "cuda",
    random_state: int = 20260712,
) -> dict:
    """Run a complete train→predict→evaluate pipeline.

    Target source: either BigWig (bigwig_dir + reference_bw_path) or
    expression matrix (expression_matrix_path).

    When expression_matrix_path is used, SNP deltas are precomputed into memory
    by default (CachedSpecificSNPDataset) to eliminate per-batch disk I/O.

    Returns dict of all results DataFrames.
    """
    from .data import (
        build_multi_gene_data, build_multi_gene_data_from_matrix,
        SpecificSNPDataset, CachedSpecificSNPDataset,
    )
    from .checkpoint import build_checkpoint

    # --- Data assembly ---
    print(f"Building data for {gene_set_size} genes...")

    if expression_matrix_path is not None:
        # ── Expression matrix mode ──
        if isinstance(expression_matrix_path, pd.DataFrame):
            print(f"  Target source: pre-loaded expression matrix ({expression_matrix_path.shape})")
        else:
            print(f"  Target source: expression matrix ({expression_matrix_path})")
        (
            table, y, gene_ids_arr, gene_to_id, centers,
            positions, variant_keys, normalized_positions, max_snps,
            cached_delta, cached_mask, genes,
        ) = build_multi_gene_data_from_matrix(
            genes, all_individuals, train_individuals,
            val_individuals, test_individuals,
            embedding_root, expression_matrix_path,
        )
    elif bigwig_dir is not None and reference_bw_path is not None:
        # ── BigWig mode ──
        print(f"  Target source: BigWig ({bigwig_dir})")
        (
            table, y, gene_ids_arr, gene_to_id, centers,
            positions, variant_keys, normalized_positions, max_snps,
            cached_delta, cached_mask, genes,
        ) = build_multi_gene_data(
            genes, all_individuals, train_individuals,
            val_individuals, test_individuals,
            embedding_root, bigwig_dir, reference_bw_path, sample_to_accession,
        )
    else:
        raise ValueError(
            "Must provide either expression_matrix_path or "
            "(bigwig_dir + reference_bw_path + sample_to_accession)"
        )

    train_mask = table["split"].eq("train").to_numpy()
    val_mask = table["split"].eq("val").to_numpy()

    # --- Scaling ---
    from .scaling import fit_gene_target_scalers, scale_gene_targets
    scalers = fit_gene_target_scalers(
        y, gene_ids_arr, train_mask, len(genes), target_std_eps
    )
    y_scaled = scale_gene_targets(y, gene_ids_arr, scalers)

    # --- Dataset ---
    if cached_delta is not None:
        print(f"  Using CachedSpecificSNPDataset (precomputed deltas)")
        dataset = CachedSpecificSNPDataset(
            table, cached_delta, cached_mask, normalized_positions, y_scaled,
        )
    else:
        dataset = SpecificSNPDataset(
            table, genes, centers, normalized_positions, y_scaled, max_snps, embedding_root
        )

    # --- Model ---
    # Update n_genes in case genes were filtered by data builder
    model_config["n_genes"] = len(genes)
    model = model_class(**model_config)

    # --- Train ---
    print(f"Training {model_variant}: {gene_set_size} genes...")
    trainer = Trainer(
        model, dataset, table, y, gene_ids_arr, scalers,
        train_mask, val_mask,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        epochs=epochs, patience=patience, loss_type=loss_type,
        device=device,
    )
    history, best_val, best_train = trainer.run()

    # --- Predict ---
    best_val_preds = predict_all(
        model, best_val["state"], dataset, table, y,
        gene_ids_arr, scalers, device, "best_validation",
    )
    best_train_preds = predict_all(
        model, best_train["state"], dataset, table, y,
        gene_ids_arr, scalers, device, "best_training",
    )
    predictions = pd.concat([best_val_preds, best_train_preds], ignore_index=True)

    # --- Evaluate ---
    overall, by_gene, macro = summarize_predictions(
        predictions, scalers["variable"], model_variant, gene_set_size
    )

    # --- Checkpoint ---
    model.load_state_dict(best_val["state"])
    checkpoint = build_checkpoint(
        model=model,
        model_config=model_config,
        ordered_genes=[
            {
                "name": g.name, "embedding_name": g.embedding_name,
                "chrom": g.chrom, "start": g.start, "end": g.end,
                "strand": g.strand, "transcript_id": g.transcript_id,
            }
            for g in genes
        ],
        gene_to_id=gene_to_id,
        target_scalers=scalers,
        snp_centers=centers,
        positions=positions,
        variant_keys=variant_keys,
        max_snps=max_snps,
        splits={
            "train": train_individuals,
            "val": sorted(val_individuals),
            "test": sorted(test_individuals),
        },
        training_config={
            "batch_size": batch_size, "epochs": epochs, "lr": lr,
            "weight_decay": weight_decay, "dropout": dropout,
            "patience": patience, "loss_type": loss_type,
            "random_state": random_state,
            "embedding_source": "hap1", "embedding_frozen": True,
        },
        best_validation=best_val,
        best_training=best_train,
    )
    ckpt_path = output_dir / f"model_{model_variant}_{gene_set_size}.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # --- Save CSVs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / "metrics_overall.csv", index=False)
    by_gene.to_csv(output_dir / "metrics_by_gene.csv", index=False)
    macro.to_csv(output_dir / "metrics_macro.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    history["gene_set_size"] = gene_set_size
    history["model_variant"] = model_variant
    history.to_csv(output_dir / "history.csv", index=False)

    print(macro.to_dict("records"))
    return {
        "overall": overall, "by_gene": by_gene, "macro": macro,
        "predictions": predictions, "history": history,
        "checkpoint_path": ckpt_path,
    }
