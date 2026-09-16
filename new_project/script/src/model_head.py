# -*- coding: utf-8 -*-
"""VariantAwareCausalLM：包裹 OGR(HF)，实现 Σ(w·CE)/Σw 加权 NTP（与 Megatron loss_func 语义一致）。"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger("variant_cpt.model_head")


class VariantAwareCausalLM(nn.Module):
    def __init__(self, base_model, aux_loss_coef=1e-3):
        super().__init__()
        self.base = base_model
        self.aux_loss_coef = aux_loss_coef

    # ---- 代理到 HF 基模，满足 Trainer 的要求 ----
    def _base(self):
        """取基模：nn.Module 会把 Module 属性放进 _modules 而非 __dict__。"""
        d = self.__dict__
        base = d.get("base")
        if base is None:
            mods = d.get("_modules") or {}
            base = mods.get("base")
        return base

    def gradient_checkpointing_enable(self, *a, **k):
        base = self._base()
        fn = getattr(base, "gradient_checkpointing_enable", None) if base is not None else None
        if fn is not None:
            fn(*a, **k)

    def gradient_checkpointing_disable(self, *a, **k):
        base = self._base()
        fn = getattr(base, "gradient_checkpointing_disable", None) if base is not None else None
        if fn is not None:
            fn(*a, **k)

    def __getattr__(self, name):
        """优先 nn.Module 常规逻辑（params/buffers/modules），再代理到 base。"""
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return super().__getattr__(name)   # 处理 _parameters/_buffers/_modules（含 self.base）
        except AttributeError:
            pass
        base = self._base()
        if base is not None:
            try:
                return getattr(base, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def forward(self, input_ids, loss_mask=None, **kwargs):
        out = self.base(input_ids=input_ids, **kwargs)   # labels=None → 返回 logits
        logits = out.logits
        # NTP shift：位置 i 预测 token[i+1]
        shift_logits = logits[:, :-1].contiguous()                # (B, L-1, V)
        shift_targets = input_ids[:, 1:].contiguous()             # (B, L-1)
        # cross_entropy 要求类别维在第 1 维 → (B, V, L-1)，target (B, L-1)
        ce = F.cross_entropy(shift_logits.transpose(1, 2), shift_targets, reduction="none")   # (B, L-1)
        variant_ce = None
        background_ce = None
        variant_frac = 0.0
        if loss_mask is not None:
            m = loss_mask[:, 1:].float()          # mask 同步 shift
            loss = (ce * m).sum() / m.sum().clamp(min=1e-8)
            # --- 分离信号：变异区（loss_mask>0.1）与背景区（其余被加权区）的 CE 分组统计 ---
            is_var = (m > 0.1).float()
            weighted = (m > 0).float()            # 参与 loss 的位点（排 padding）
            if is_var.sum() > 0:
                variant_ce = (ce * is_var).sum() / is_var.sum().clamp(min=1e-8)
            bg_mask = weighted * (1 - is_var)
            if bg_mask.sum() > 0:
                background_ce = (ce * bg_mask).sum() / bg_mask.sum().clamp(min=1e-8)
            variant_frac = is_var.mean().item()
        else:
            loss = ce.mean()
        # C4：沿用 MoE aux-loss（若 HF 模型暴露）
        aux = getattr(out, "aux_loss", None)
        if aux is None and isinstance(out, dict):
            aux = out.get("aux_loss", None)
        if aux is not None:
            loss = loss + self.aux_loss_coef * aux
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=out.past_key_values if hasattr(out, "past_key_values") else None,
        )
        if aux is not None:
            output.aux_loss = aux   # ModelOutput 是 OrderedDict，附加键即可，不传构造参数
        # 附加分离信号（ModelOutput 允许自由附加属性）
        output.variant_ce = variant_ce
        output.background_ce = background_ce
        output.variant_frac = variant_frac
        return output