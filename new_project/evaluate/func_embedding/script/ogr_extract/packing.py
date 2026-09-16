# -*- coding: utf-8 -*-
"""打包策略：按序列长度降序 + 每批总 token 限额，padding 率最低、不 OOM。"""
from __future__ import annotations

import numpy as np


def pack_batches(lengths: np.ndarray, max_tokens: int) -> list[np.ndarray]:
    """贪心装包：长度降序, 每批总 token <= max_tokens。返回 batch 内的原始下标。"""
    order = np.argsort(lengths)[::-1]
    batches, cur, used = [], [], 0
    for i in order:
        if cur and used + lengths[i] > max_tokens:
            batches.append(np.array(cur))
            cur, used = [], 0
        cur.append(int(i))
        used += int(lengths[i])
    if cur:
        batches.append(np.array(cur))
    return batches


def pack_batches_by_count(lengths: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """固定条数打包：长度降序, 每批恰 <= batch_size 条(仍省 padding)。
    返回 batch 内的原始下标。"""
    assert batch_size >= 1, f"batch_size 必须 >=1, 收到 {batch_size}"
    order = np.argsort(lengths)[::-1]
    batches = [
        order[i:i + batch_size]
        for i in range(0, len(order), batch_size)
    ]
    return [np.asarray(b) for b in batches]


def padding_ratio(lengths: np.ndarray, batches: list[np.ndarray]) -> float:
    """整体 padding 率 = 1 - 真实 token / 批内最大长度*批数。"""
    total_real = int(lengths.sum())
    total_slot = sum(int(lengths[b].max()) * len(b) for b in batches)
    return 1.0 - total_real / total_slot if total_slot else 0.0