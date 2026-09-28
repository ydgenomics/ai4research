# -*- coding: utf-8 -*-
"""懒加载 Dataset + collate：批量读 GVL → OGR token_ids + 变异聚焦 loss_mask。"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("variant_cpt.variant_dataset")


class VariantCPTDataset(Dataset):
    """扁平 (row, col) 索引对；不持有序列，只持 gvl_ds 引用（内存最优）。"""

    def __init__(self, gvl_ds, rows, cols):
        self.gvl = gvl_ds
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.cols = torch.as_tensor(cols, dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # 兼容两种索引：int（查表）或直接 (row, col) 元组（两段采样自定义 sampler 用）
        if isinstance(i, tuple):
            return i
        return int(self.rows[i]), int(self.cols[i])   # 轻量，不碰 GVL


def build_char_token_map(tokenizer):
    """OGR 单碱基 tokenizer → 字符->token_id 映射（含 unknown 回退）。"""
    table = {}
    for ch in (b"A", b"C", b"G", b"T", b"N"):
        try:
            s = ch.decode()
            table[ch] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))[0]
        except Exception:
            table[ch] = 4   # 兜底字节序
    fallback = getattr(tokenizer, "pad_token_id", 4) or 4
    table[b"unknown"] = fallback
    return table


class VariantCollator:
    """collate：批量 gvl 索引（GVL 多线程最快路径）→ token_ids + loss_mask。

    注：NTP 的 shift 在 model 内完成；这里 loss_mask 与 input_ids 同坐标。
    """

    def __init__(self, gvl_ds, char_map, loss_cfg, seq_len):
        self.gvl = gvl_ds
        self.char_map = char_map
        self.l = loss_cfg
        self.seq_len = seq_len

    def __call__(self, batch):
        rows = [b[0] for b in batch]
        cols = [b[1] for b in batch]
        # GVL 批量索引 → AnnotatedHaps 具名对象（非元组！）
        out = self.gvl[rows, cols]
        haps = np.asarray(out.haps)
        try:
            var_idxs = np.asarray(out.var_idxs)
            ref_coords = np.asarray(out.ref_coords)
        except AttributeError:                    # 旧版字段兼容
            var_idxs = np.asarray(out.variant_indexes)
            ref_coords = np.asarray(out.reference_coordinates)
        # 形状 (batch, window=1, ploidy, length) → 取 (batch, 0, ploidy0, :) 压成
        # (batch, length)。未 phase → ploidy=0 链（deterministic 第一链，C2）。
        if haps.ndim >= 4:
            haps = haps[:, 0, 0, :]
            var_idxs = var_idxs[:, 0, 0, :]
            ref_coords = ref_coords[:, 0, 0, :]
        elif haps.ndim == 3:                      # 兜底 (batch, ploidy, len)
            haps = haps[:, 0, :]
            var_idxs = var_idxs[:, 0, :]
            ref_coords = ref_coords[:, 0, :]

        # bytes → token ids
        tok = np.full((len(rows), self.seq_len), self.char_map[b"unknown"], dtype=np.int64)
        for i in range(len(rows)):
            hap_i = haps[i]
            for j, b in enumerate(hap_i):
                if b in self.char_map:
                    tok[i, j] = self.char_map[b]
        input_ids = torch.from_numpy(tok)

        # loss_mask（输入坐标；shift 在 model 内做）
        mask = self._build_loss_mask(var_idxs, ref_coords)
        return {
            "input_ids": input_ids,
            "loss_mask": torch.from_numpy(mask),
        }

    def _build_loss_mask(self, var_idxs, ref_coords):
        L = self.seq_len
        B = var_idxs.shape[0]
        vw = self.l.get("variant_weight", 3.0)
        mw = self.l.get("motif_weight", 1.5)
        bw = self.l.get("background_weight", 0.02)
        fl = self.l.get("focus_left", 100)
        fr = self.l.get("focus_right", 25)
        mh = self.l.get("motif_halfwidth", 15)
        mask = np.full((B, L), bw, dtype=np.float32)
        # padding（ref_coords == -1）权重 0
        pad = ref_coords < 0
        mask[pad] = 0.0
        for b in range(B):
            vs = np.flatnonzero(var_idxs[b] != -1)   # 变异位点（在本链上）
            for v in vs:
                lo, hi = max(0, v - fl), min(L, v + fr + 1)
                mask[b, lo:hi] = np.maximum(mask[b, lo:hi], vw)
                # motif ±6-15bp（不含紧邻 v±5 的"变异本体"区域）
                m1, m2 = max(0, v - mh), max(0, v - 5)
                mask[b, m1:m2] = np.maximum(mask[b, m1:m2], mw)
                m3, m4 = min(L, v + 6), min(L, v + mh + 1)
                mask[b, m3:m4] = np.maximum(mask[b, m3:m4], mw)
        return mask