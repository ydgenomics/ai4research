# -*- coding: utf-8 -*-
"""前向 + 可配置池化（mask-aware）+ 显存管理。

池化方式由 run yaml 的 `pooling` 指定, 可被 models.yaml 中单个模型的 `pooling` 覆盖。
为什么需要多选: **全 token 均值池化会把序列长度写进表征的前几个主成分**
(实测 rice_1B_stage2_8k 上 PC1 与 log10(seq_len) 的 Spearman 达 0.957、方差占比 72.7%)。
causal/decoder 模型改用 `last`(末位通过注意力已看到全部上文) 可显著削弱该效应。
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# 可选池化方式（run yaml 的 pooling / models.yaml 中该模型的 pooling）:
#   mean        全 token mask-aware 均值（默认, 旧行为; causal 下易把长度写进 PC1）
#   max         全 token mask-aware 最大值
#   first       第一个真实 token（kmer6_cls 的 <cls>; ESM 风格 encoder 推荐）
#   last        最后一个真实 token（causal/decoder 推荐）
#   mean_last_k 最后 k 个真实 token 的均值（k = pool_k）
#   mean_trim   去掉首尾各 pool_trim 比例后取均值（缓解首尾边界效应）
VALID_POOLING = ("mean", "max", "first", "last", "mean_last_k", "mean_trim")

# 需要额外超参的池化方式: 名称 -> (run yaml 中的键, 默认值)
POOLING_PARAMS = {
    "mean_last_k": ("pool_k", 64),
    "mean_trim": ("pool_trim", 0.25),
}


def validate_pooling(pooling: str, pool_k: int = 64, pool_trim: float = 0.25) -> None:
    """校验池化方式及其超参, 非法时抛 ValueError。"""
    if pooling not in VALID_POOLING:
        raise ValueError(f"pooling 非法: {pooling!r} (可选 {VALID_POOLING})")
    if pooling == "mean_last_k" and int(pool_k) < 1:
        raise ValueError(f"pool_k 必须 >= 1 (当前 {pool_k})")
    if pooling == "mean_trim" and not 0.0 <= float(pool_trim) < 0.5:
        raise ValueError(f"pool_trim 需满足 0 <= v < 0.5 (当前 {pool_trim})")


def _token_rank(mask: torch.Tensor) -> torch.Tensor:
    """[B, T] mask -> [B, T] 每个真实 token 的序号(0-based), pad 位置为 -1。

    用 cumsum 数「第几个真实 token」, 与 padding side 无关
    (HF 部分 causal 模型默认左 padding, 直接拿位置当下标会错位)。
    """
    rank = mask.long().cumsum(1) - 1
    return rank.masked_fill(mask == 0, -1)


def _select_mask(mask: torch.Tensor, mode: str, pool_k: int,
                 pool_trim: float) -> torch.Tensor:
    """返回 [B, T] bool: 哪些位置参与池化(只从真实 token 里挑)。"""
    L = mask.sum(1).clamp_min(1)                 # [B] 每条真实 token 数
    rank = _token_rank(mask)                     # [B, T]
    Lc = L.unsqueeze(1)                          # [B, 1]

    if mode == "first":
        sel = rank == 0
    elif mode == "last":
        sel = rank == (Lc - 1)
    elif mode == "mean_last_k":
        start = (L - int(pool_k)).clamp_min(0).unsqueeze(1)
        sel = rank >= start
    elif mode == "mean_trim":
        cut = (float(pool_trim) * L).floor().long().unsqueeze(1)   # 首尾各去掉几个
        sel = (rank >= cut) & (rank < (Lc - cut).clamp_min(cut + 1))
    else:                                        # mean / max: 全部真实 token
        sel = rank >= 0

    # 兜底: 极端长度下(如 L=1 + trim)若一行都没选到, 退回第一个真实 token
    empty = sel.sum(1) == 0
    if bool(empty.any()):
        sel = sel.clone()
        sel[empty, mask.long().argmax(1)[empty]] = True
    return sel


def pool_hidden(hs: torch.Tensor, mask: torch.Tensor, pooling: str = "mean",
                pool_k: int = 64, pool_trim: float = 0.25) -> torch.Tensor:
    """[B, T, H] hidden + [B, T] mask -> [B, H] 池化向量（fp32）。"""
    sel = _select_mask(mask, pooling, pool_k, pool_trim).unsqueeze(-1)   # [B, T, 1]
    if pooling == "max":
        neg = torch.finfo(hs.dtype).min
        return hs.masked_fill(~sel, neg).max(1).values
    denom = sel.sum(1).clamp_min(1).to(hs.dtype)                         # [B, 1]
    return (hs * sel).sum(1) / denom


@torch.no_grad()
def embed_batch(ids, mask, model, layer=-1, pooling: str = "mean",
                pool_k: int = 64, pool_trim: float = 0.25,
                l2_normalize: bool = False, use_attention_mask: bool = True):
    """输入 [B, T] 的 ids/mask -> [B, H] 池化向量（CPU float32）。

    use_attention_mask=False: 模型 forward 不接受 attention_mask(如 Caduceus/Mamba2
    SSM), 此时不传 mask; pooling 仍用 mask 区分真实 token。
    """
    fwd_kwargs = dict(input_ids=ids, output_hidden_states=True)
    if use_attention_mask:
        fwd_kwargs["attention_mask"] = mask
    try:
        out = model(**fwd_kwargs)
    except TypeError:
        # 自定义模型 forward 未实现 output_hidden_states(或模型不支持) -> 退化为返回
        # (logits, last_hidden); 用 last_hidden 代替层输出, 长度对齐日志里减一次层项
        if use_attention_mask:
            fwd_kwargs.pop("attention_mask", None)
        out = model(**fwd_kwargs)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        # ModelOutput 无 .hidden_states（如某些 MLM/自定义类只有 last_hidden_state）
        last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hs = last.float()
    else:
        hs = out.hidden_states[layer].float()   # [B, T, H] 升 fp32 再池化
    emb = pool_hidden(hs, mask, pooling, pool_k, pool_trim)  # [B, H]
    if l2_normalize:
        emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return emb.float().cpu().numpy()


def log_gpu_mem(tag: str = "") -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        logger.info(f"[GPU] {tag} 峰值显存 {peak:.1f} GB")