"""渗入分析核心算法 —— 移植自 20.introgression_analysis/scripts/4.run_analysis.py。

默认使用与离线分析一致的参数：
- segment_size = 8000（8k 片段）
- window_size  = 256000（256k 窗口）
- window_step  = 64000（步长，75% 重叠）
- top_k        = 10（窗口内 top-10 片段聚合）
- 双阈值分组   [thr_jap, thr_ind]

为了支持 Web「固定标准窗口」查询，新增标准窗口网格（grid）概念：
- ``STANDARD_GRID``：以 0 为起点、window_step 为步长划分的 256k 标准窗口，
  网格与 FASTA 染色体长度、用户输入无关（保证 Web 结果与离线全基因组
  结果完全一致）。
- 用户输入 [start, end] 后，通过最大覆盖度匹配到最贴合的标准窗口。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# 与 4.run_analysis.py 保持一致的分组命名与配色
GROUP_IND = "Ind"
GROUP_JAP = "Jap"
GROUP_UNCERTAIN = "uncertain"
GROUP_ORDER = (GROUP_IND, GROUP_JAP, GROUP_UNCERTAIN)

JAP_COLOR = "#4874cb"
IND_COLOR = "#dc6b2d"
UNCERTAIN_COLOR = "#9e9e9e"

# 分析默认参数（可通过 .env 覆盖）
DEFAULT_SEGMENT_SIZE = 8000
DEFAULT_WINDOW_SIZE = 256000
DEFAULT_WINDOW_STEP = 64000
DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD_JAP = 0.55519
DEFAULT_THRESHOLD_IND = 0.53473


@dataclass
class AnalysisParams:
    """渗入分析参数集合。"""

    segment_size: int = DEFAULT_SEGMENT_SIZE
    window_size: int = DEFAULT_WINDOW_SIZE
    window_step: int = DEFAULT_WINDOW_STEP
    top_k: int = DEFAULT_TOP_K
    threshold_jap: float = DEFAULT_THRESHOLD_JAP
    threshold_ind: float = DEFAULT_THRESHOLD_IND

    def __post_init__(self):
        if self.window_size <= 0 or self.segment_size <= 0 or self.window_step <= 0:
            raise ValueError("window_size / segment_size / window_step must be positive")
        if self.window_size % self.segment_size != 0:
            raise ValueError(
                f"window_size({self.window_size}) must be divisible by segment_size({self.segment_size})"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


# ---------------------------------------------------------------------------
#  标准窗口网格
# ---------------------------------------------------------------------------
def standard_window_starts(
    params: AnalysisParams,
    chrom_len: int,
    trim_start: int = 0,
    n_segments: Optional[int] = None,
) -> list[int]:
    """返回某染色体上所有离线滑动窗口的起点列表（与 4.run_analysis 完全一致）。

    离线 aggregate_windows 的窗口起点 = ``starts[idx]``（第 idx 个片段的真实
    绝对坐标），其中：
    - 片段网格 = trim 后以 8000 步长切分的片段，starts[0] = trim_start
    - idx 网格  = 0, step_segments, 2*step_segments, ..., n_segments-window_segments

    因此窗口起点 = trim_start + idx * segment_size。默认 trim_start=0、
    n_segments=chrom_len//segment_size 时退化为 0-锚点网格（无 trim 的近似）。

    返回：窗口起点列表（绝对坐标，bp）。
    """
    seg = params.segment_size
    step = params.window_step
    n_seg_total = n_segments if n_segments is not None else chrom_len // seg
    window_segments = params.window_size // seg
    step_segments = step // seg

    if n_seg_total < window_segments:
        return [trim_start]

    last_start_idx = n_seg_total - window_segments
    return [trim_start + i * seg for i in range(0, last_start_idx + 1, step_segments)]


def match_window_to_grid(
    params: AnalysisParams,
    chrom_len: int,
    start: int,
    end: Optional[int],
    trim_start: int = 0,
    n_segments: Optional[int] = None,
) -> tuple[int, int]:
    """将用户输入 [start, end] 匹配到最大覆盖度的滑动窗口。

    返回 (win_start, win_end)。规则：
    1. 无窗口可匹配（染色体太短）时回退到 [0, min(chrom_len, window_size)]。
    2. 遍历所有离线滑动窗口（起点 = standard_window_starts），计算与用户
       区间的重叠长度，取重叠最大的窗口；重叠相同时取 start 较早的窗口
       （确定性）。
    """
    grid_starts = standard_window_starts(
        params, chrom_len, trim_start=trim_start, n_segments=n_segments
    )
    if not grid_starts:
        raise ValueError(f"chromosome length {chrom_len} too short for window {params.window_size}")

    ws = params.window_size
    user_start = int(start) if start is not None else 0
    if end is None:
        user_end = user_start + ws
    else:
        user_end = int(end)

    if user_end <= user_start:
        user_end = user_start + ws

    best_start = None
    best_overlap = -1
    for gs in grid_starts:
        ge = gs + ws
        overlap = max(0, min(ge, user_end) - max(gs, user_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_start = gs
        elif overlap == best_overlap and best_start is not None:
            # 重叠相等时取更早的窗口（确定性）
            if gs < best_start:
                best_start = gs

    if best_start is None:
        best_start = 0
    win_end = min(best_start + ws, chrom_len)
    win_start = best_start
    if win_end - win_start <= 0:
        win_start = max(0, chrom_len - ws)
        win_end = chrom_len
    return win_start, win_end


# ---------------------------------------------------------------------------
#  概率列名
# ---------------------------------------------------------------------------
def resolve_prob_columns(columns: list[str]) -> tuple[str, str]:
    """返回 (prob_jap_col, prob_ind_col)。"""
    col_set = set(columns)
    if {"prob_jap", "prob_ind"}.issubset(col_set):
        return "prob_jap", "prob_ind"
    if {"prob_0", "prob_1"}.issubset(col_set):
        return "prob_0", "prob_1"
    raise ValueError("window TSV must contain prob_jap/prob_ind or prob_0/prob_1")


# ---------------------------------------------------------------------------
#  分组判定
# ---------------------------------------------------------------------------
def assign_dual_group(
    prob_jap: float,
    prob_ind: float,
    threshold_jap: float,
    threshold_ind: float,
) -> str:
    """与 4.run_analysis.assign_dual_group / merge_jsonl_labels 完全一致。"""
    if prob_jap >= threshold_jap and prob_ind < threshold_ind:
        return GROUP_JAP
    if prob_jap < threshold_jap and prob_ind >= threshold_ind:
        return GROUP_IND
    return GROUP_UNCERTAIN


def threshold_rule_text(params: AnalysisParams) -> str:
    return (
        f"Ind: prob_jap < {params.threshold_jap:.4f} & prob_ind >= {params.threshold_ind:.4f}; "
        f"Jap: prob_jap >= {params.threshold_jap:.4f} & prob_ind < {params.threshold_ind:.4f}; "
        f"uncertain: otherwise"
    )


# ---------------------------------------------------------------------------
#  窗口内 top-k 聚合（由一段 8k 片段的概率数组计算单个 256k 窗口的聚合分数）
# ---------------------------------------------------------------------------
def aggregate_one_window(
    prob_jap: np.ndarray,
    prob_ind: np.ndarray,
    params: AnalysisParams,
    min_segments: int = 1,
) -> Optional[dict]:
    """对落在同一 256k 窗口内的 8k 片段求 top-k 均值（与离线完全一致）。

    prob_jap / prob_ind: 片段级概率数组（长度 = 窗口内片段数，通常 32）。
    Returns dict with keys:
        n_segments, topk_mean_jap, topk_mean_ind, group, center
    """
    n = len(prob_jap)
    if n < min_segments:
        return None
    k = min(params.top_k, n)
    top_jap = float(np.sort(np.asarray(prob_jap, dtype=float))[-k:].mean())
    top_ind = float(np.sort(np.asarray(prob_ind, dtype=float))[-k:].mean())
    group = assign_dual_group(
        top_jap, top_ind, params.threshold_jap, params.threshold_ind
    )
    return {
        "n_segments": n,
        "topk_mean_jap": top_jap,
        "topk_mean_ind": top_ind,
        "group": group,
    }


def aggregate_windows_for_chrom(
    segments_df: pd.DataFrame,
    prob_cols: tuple[str, str],
    params: AnalysisParams,
    min_segments: int = 1,
) -> pd.DataFrame:
    """对单条染色体的所有滑动窗口聚合 —— 与离线 4.run_analysis.aggregate_windows 逐位一致。

    segments_df 需含列：start, prob_jap/prob_ind（或 prob_0/prob_1，由 prob_cols 指定）。
    注意离线实现的关键细节（必须保持一致，否则 Web 结果与离线不一致）：
    - 片段按 start 升序排列；窗口起点 = starts[idx]（第 idx 片段的真实坐标）
    - 窗口内片段 = 排序数组的连续切片 [idx: idx+window_segments]（非坐标过滤）
    - 窗口起点网格 idx = 0, step_segments, 2*step_segments, ..., n_segments-window_segments
    - win_end = win_start + window_size（超出末端不做裁剪）

    返回列：chr_id, win_start, win_end, center, n_segments, topk_mean_jap,
    topk_mean_ind, group
    """
    if segments_df.empty:
        return pd.DataFrame(
            columns=["chr_id", "win_start", "win_end", "center", "n_segments",
                     "topk_mean_jap", "topk_mean_ind", "group"]
        )
    jap_col, ind_col = prob_cols
    out = segments_df.sort_values("start").reset_index(drop=True)
    prob_arrays = {
        col: out[col].to_numpy(dtype=float) for col in (jap_col, ind_col)
    }
    starts = out["start"].to_numpy(dtype=int)
    n_segments = len(starts)
    ws = params.window_size
    window_segments = ws // params.segment_size
    step_segments = params.window_step // params.segment_size

    if n_segments < min_segments:
        return pd.DataFrame(
            columns=["chr_id", "win_start", "win_end", "center", "n_segments",
                     "topk_mean_jap", "topk_mean_ind", "group"]
        )

    if n_segments < window_segments:
        window_indices = [0]
    else:
        last_start_idx = n_segments - window_segments
        window_indices = list(range(0, last_start_idx + 1, step_segments))

    rows = []
    for idx in window_indices:
        slice_jap = prob_arrays[jap_col][idx: idx + window_segments]
        slice_ind = prob_arrays[ind_col][idx: idx + window_segments]
        n_in_window = len(slice_jap)
        if n_in_window < min_segments:
            continue
        k = min(params.top_k, n_in_window)
        top_jap = float(np.sort(slice_jap)[-k:].mean())
        top_ind = float(np.sort(slice_ind)[-k:].mean())
        win_start = int(starts[idx])
        win_end = win_start + ws
        group = assign_dual_group(top_jap, top_ind, params.threshold_jap, params.threshold_ind)
        rows.append({
            "chr_id": out["chr_id"].iloc[0],
            "win_start": win_start,
            "win_end": win_end,
            "center": win_start + ws // 2,
            "n_segments": int(n_in_window),
            "topk_mean_jap": top_jap,
            "topk_mean_ind": top_ind,
            "group": group,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  区域融合（overlap merging）
# ---------------------------------------------------------------------------
def call_regions(windows_df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """将相邻/重叠的同组窗口融合为连续区域（与离线 call_regions 一致）。

    返回列：chr_id, start, end, topk_mean_jap, topk_mean_ind, n_windows
    """
    selected = windows_df[mask].copy()
    cols = ["chr_id", "start", "end", "topk_mean_jap", "topk_mean_ind", "n_windows"]
    if selected.empty:
        return pd.DataFrame(columns=cols)

    selected = selected.sort_values(["chr_id", "win_start"]).reset_index(drop=True)
    regions = []
    current = None
    for row in selected.itertuples(index=False):
        if current is None:
            current = {
                "chr_id": row.chr_id,
                "start": int(row.win_start),
                "end": int(row.win_end),
                "jap_vals": [float(row.topk_mean_jap)],
                "ind_vals": [float(row.topk_mean_ind)],
            }
            continue
        if int(row.win_start) <= current["end"]:
            current["end"] = max(current["end"], int(row.win_end))
            current["jap_vals"].append(float(row.topk_mean_jap))
            current["ind_vals"].append(float(row.topk_mean_ind))
        else:
            regions.append({
                "chr_id": current["chr_id"],
                "start": current["start"],
                "end": current["end"],
                "topk_mean_jap": float(np.mean(current["jap_vals"])),
                "topk_mean_ind": float(np.mean(current["ind_vals"])),
                "n_windows": len(current["jap_vals"]),
            })
            current = {
                "chr_id": row.chr_id,
                "start": int(row.win_start),
                "end": int(row.win_end),
                "jap_vals": [float(row.topk_mean_jap)],
                "ind_vals": [float(row.topk_mean_ind)],
            }
    if current is not None:
        regions.append({
            "chr_id": current["chr_id"],
            "start": current["start"],
            "end": current["end"],
            "topk_mean_jap": float(np.mean(current["jap_vals"])),
            "topk_mean_ind": float(np.mean(current["ind_vals"])),
            "n_windows": len(current["jap_vals"]),
        })
    return pd.DataFrame(regions, columns=cols)


def call_group_regions(windows_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        group: call_regions(windows_df, mask=windows_df["group"] == group)
        for group in GROUP_ORDER
    }