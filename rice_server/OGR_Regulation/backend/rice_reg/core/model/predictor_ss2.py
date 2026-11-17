"""Strand-split 1-channel head: same architecture as `predictor_ss`, downstream-only MSE loss."""

import torch.nn.functional as F

from model.predictor_ss import MultiModalPredictorSS


class MultiModalPredictorSS2(MultiModalPredictorSS):
    """Same stack as `MultiModalPredictorSS`; training loss is MSE on the downstream half only.

    For strand-split data, the downstream half matches ``calc_metrics`` slicing with
    ``dataset_type: strand`` (second half of the window, ``[:, :, L//2:]``).

    Evaluation configs should set ``metric_region`` to ``downstream`` or include downstream
    (e.g. ``both``); ``metric_region: full`` alone is inconsistent with this objective.
    """

    REQUIRED_DATASET_TYPE = "strand"

    def forward(self, input_ids, atac_signal, labels=None):
        # Full-window logits; loss only on downstream positions (same rule as metric downstream slice).
        out = super().forward(input_ids, atac_signal, labels=None)
        logits = out["logits"]
        loss = None
        if labels is not None:
            L = logits.size(-1)
            h = L // 2
            loss = F.mse_loss(
                logits[:, :, h:],
                labels[:, :, h:].to(logits.dtype),
            )
        return {"loss": loss, "logits": logits}
