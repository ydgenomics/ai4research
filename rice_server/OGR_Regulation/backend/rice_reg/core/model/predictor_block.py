"""Multi-modal predictor: region-masked MSE head (predictor.type=block).

Consumes an additional `region_signal` track (0/1 mask BigWig) and restricts loss
to positions where `region_mask == 1`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.distributed import dist_print


def _masked_mse_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error only where mask==1.

    Shapes:
    - logits: [B, C, L]
    - labels: [B, C, L]
    - mask:   [B, L] or [B, 1, L]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=logits.dtype)
    se = (logits - labels.to(logits.dtype)) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(1.0)
    return se.sum() / denom


class MultiModalPredictorBlock(nn.Module):
    """Architecture is based on predictor_v0, with an extra region fusion term."""

    REQUIRED_DATASET_TYPE = "v0"

    def __init__(self, base_model, atac_encoder, *, region_scale_init: float = 0.0):
        super().__init__()
        self.base = base_model
        self.atac_encoder = atac_encoder

        # Lightweight region encoder: map [B,L] -> [B,1024,L] like ATAC encoder output.
        self.region_proj = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 1024, kernel_size=1),
        )
        self.region_scale = nn.Parameter(torch.tensor(float(region_scale_init)))

        for param in self.base.parameters():
            param.requires_grad = False

        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            self.layers = base_model.model.layers
        elif hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
            self.layers = base_model.encoder.layer
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            self.layers = base_model.transformer.h
        elif hasattr(base_model, "layers"):
            self.layers = base_model.layers
        else:
            raise RuntimeError("Cannot identify transformer layer structure in base model")

        if not self.layers:
            raise RuntimeError("No transformer layers found")

        self.num_layers = len(self.layers)
        dist_print(f"Identified {self.num_layers} transformer layers")

        for param in self.layers[-1].parameters():
            param.requires_grad = True

        self.fusion = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc1 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.up1 = nn.ConvTranspose1d(1024, 1024, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.up2 = nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.final = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 2, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        self._ensure_float32()

    def _ensure_float32(self):
        for module in [
            self.region_proj,
            self.fusion,
            self.enc1,
            self.enc2,
            self.enc3,
            self.bottleneck,
            self.up1,
            self.dec1,
            self.up2,
            self.dec2,
            self.final,
        ]:
            for param in module.parameters():
                param.data = param.data.float()
        self.scale.data = self.scale.data.float()
        self.region_scale.data = self.region_scale.data.float()

    def forward(self, input_ids, atac_signal, region_signal, region_mask=None, labels=None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            inputs_embeds = self.base.get_input_embeddings()(input_ids)

        atac_embeds = self.atac_encoder(atac_signal)
        atac_embeds = atac_embeds.transpose(1, 2).to(torch.bfloat16)

        # region_signal is [B, L] -> [B, 1, L] -> [B, 1024, L] -> [B, L, 1024]
        reg = region_signal.to(dtype=torch.float32).unsqueeze(1)
        region_embeds = self.region_proj(reg).transpose(1, 2).to(torch.bfloat16)

        combined_embeds = inputs_embeds + atac_embeds + (F.softplus(self.region_scale) * region_embeds)

        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        if hasattr(self.base, "model") and hasattr(self.base.model, "rotary_emb"):
            rotary_emb = self.base.model.rotary_emb
        elif hasattr(self.base, "rotary_emb"):
            rotary_emb = self.base.rotary_emb
        else:
            raise RuntimeError("Cannot locate rotary_emb module in base model")

        with torch.no_grad():
            position_embeddings = rotary_emb(combined_embeds, position_ids)
            hidden_states = combined_embeds
            for layer in self.layers[:-1]:
                out = layer(hidden_states, position_embeddings=position_embeddings)
                hidden_states = out[0] if isinstance(out, tuple) else out

        last_out = self.layers[-1](hidden_states, position_embeddings=position_embeddings)
        last_hidden = last_out[0] if isinstance(last_out, tuple) else last_out

        dna_out = last_hidden.transpose(1, 2).float()

        x = self.fusion(dna_out)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d1 = self.up1(b)
        if d1.size(2) != e2.size(2):
            d1 = F.interpolate(d1, size=e2.size(2), mode="nearest")
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        if d2.size(2) != e1.size(2):
            d2 = F.interpolate(d2, size=e1.size(2), mode="nearest")
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        logits = self.final(d2)
        logits = F.softplus(logits) * F.softplus(self.scale)

        loss = None
        if labels is not None:
            if region_mask is None:
                raise ValueError("predictor.type=block requires region_mask when labels are provided")
            loss = _masked_mse_loss(logits, labels, region_mask)

        return {"loss": loss, "logits": logits}

