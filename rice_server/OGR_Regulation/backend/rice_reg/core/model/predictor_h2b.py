"""Multi-modal predictor: classification + gated regression over full region.

Variant of `predictor_h2.py`:
- predicted_value is always `cls_hard (0/1) * value_pred` (both train + inference)
- reg_loss is computed on predicted_value over the whole region (no non-zero mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.distributed import dist_print


class MultiModalPredictorH2B(nn.Module):
    def __init__(
        self,
        base_model,
        atac_encoder,
        lambda_reg: float = 1.0,
        classification_pos_weight: float = 10.0,
    ):
        super().__init__()
        self.base = base_model
        self.atac_encoder = atac_encoder
        self.lambda_reg = float(lambda_reg)
        self.classification_pos_weight = float(classification_pos_weight)

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
            nn.Conv1d(256, 4, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        self._ensure_float32()

    def _ensure_float32(self):
        for module in [
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

    def forward(self, input_ids, atac_signal, labels=None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            inputs_embeds = self.base.get_input_embeddings()(input_ids)

        atac_embeds = self.atac_encoder(atac_signal)
        atac_embeds = atac_embeds.transpose(1, 2).to(torch.bfloat16)
        combined_embeds = inputs_embeds + atac_embeds

        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
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

        head_out = self.final(d2)
        zero_logits = head_out[:, :2, :]
        value_pred = head_out[:, 2:, :]
        value_pred = F.softplus(value_pred) * F.softplus(self.scale)

        # cls_header gate (soft): sigmoid(-k * zero_logits), with k=10.
        cls_hard = torch.sigmoid(-10.0 * zero_logits).to(value_pred.dtype)
        predicted_value = value_pred * cls_hard

        loss = None
        cls_loss = None
        reg_loss = None
        if labels is not None:
            labels = labels.to(predicted_value.dtype)

            non_zero_target = (labels > 0).to(zero_logits.dtype)
            pos_w = torch.tensor(
                self.classification_pos_weight,
                device=labels.device,
                dtype=zero_logits.dtype,
            )
            cls_loss = F.binary_cross_entropy_with_logits(
                -zero_logits,
                non_zero_target,
                pos_weight=pos_w,
            )

            # Regression is weighted by cls_header gate and summed.
            reg_se = F.mse_loss(predicted_value, labels, reduction="none")
            # reg_loss = (reg_se * cls_hard).mean()
            reg_loss = reg_se.mean()
            loss = cls_loss + self.lambda_reg * reg_loss

        out = {
            "loss": loss,
            "logits": predicted_value,
            "predicted_value": predicted_value,
            "value_pred": value_pred,
            "zero_logits": zero_logits,
            "zero_prob": torch.sigmoid(zero_logits),
            "non_zero_prob": torch.sigmoid(-zero_logits),
            "cls_hard": cls_hard,
        }
        if cls_loss is not None:
            out["cls_loss"] = cls_loss
        if reg_loss is not None:
            out["reg_loss"] = reg_loss
        return out

