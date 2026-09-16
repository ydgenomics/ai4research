# -*- coding: utf-8 -*-
"""窗口网格 + 采样策略：生成 (row, col) 索引对与训练权重。"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("variant_cpt.window_sampling")


def build_index_pairs(cfg: dict, ds, chroms: list[str], max_rows=None) -> tuple[np.ndarray, np.ndarray]:
    """把 (window_row, sample) 展开成扁平索引对列表；返回 (row_idx, col_idx) 数组。"""
    if chroms:
        ridx = ds.regions["chrom"].is_in(chroms).to_numpy()
        ds_sub = ds.subset_to(regions=ridx)
    else:
        ds_sub = ds
    n_r, n_c = ds_sub.n_regions, ds_sub.shape[1]
    # meshgrid 展开（训集合集 ~24k×251，int64 可控）
    rows, cols = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
    rows = rows.reshape(-1).astype(np.int64)
    cols = cols.reshape(-1).astype(np.int64)
    if max_rows is not None:
        rng = np.random.default_rng(cfg.get("compute", {}).get("seed", 42))
        idx = rng.choice(len(rows), size=min(max_rows, len(rows)), replace=False)
        rows, cols = rows[idx], cols[idx]
    return rows, cols