"""Trainer for predictor.type=block (region-masked loss)."""

from __future__ import annotations

import torch

from model.trainer import CustomTrainer


class CustomTrainerBlock(CustomTrainer):
    """Like CustomTrainer, but forwards region tensors to the model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            atac_signal=inputs["atac_signal"],
            region_signal=inputs["region_signal"],
            region_mask=inputs["region_mask"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _collate_fn(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "atac_signal": torch.stack([b["atac_signal"] for b in batch]),
            "region_signal": torch.stack([b["region_signal"] for b in batch]),
            "region_mask": torch.stack([b["region_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

