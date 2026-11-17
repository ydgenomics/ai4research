"""预测服务 —— 单窗口 / 整染色体渗入预测。

核心流程（与离线 0.run_fragments + 2.run_inference + 4.run_analysis 对齐）：
1. 用户输入 (genome, chrom, [start, end])；end 为空 -> start+256k；
   均空 -> 整条染色体。
2. 染色体级 trim 偏移（缓存）：离线对整条染色体先 trim_n（去头尾 N）
   再以 8000bp 无重叠步长切片段，片段绝对坐标 = trim_start + i*8000。
   这决定了离线窗口网格的锚点。
3. 窗口模式：读取 [win_start, win_end)（网格对齐），按绝对坐标网格直接
   切 8k 片段（不重复 trim），批量 GPU 推理得双概率。
   整条模式：读全染色体 -> trim -> 切片段（与离线完全一致）-> 推理。
4. 概率落到 8k 片段粒度后，窗口级聚合（top-k）、分组（Jap/Ind/uncertain）、
   相邻同组窗口融合为连续区域。
5. 返回：片段级 + 窗口级 + 区域级数据，供前端 Plotly 渲染。

窗口模式聚合只对匹配窗口内的 32 个片段做 top-k（离线同一窗口也是这
32 个片段），因此与离线全基因组分析逐位一致。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from rice_introgression import analysis as ana
from rice_introgression.cache_service import prediction_cache
from rice_introgression.genome_service import (
    get_chromosome_length,
    resolve_genome_config,
)
from rice_introgression.predictor import (
    read_fasta_sequence,
    require_predictor,
    trim_n,
)

logger = logging.getLogger(__name__)

# 染色体级 trim 偏移缓存：{ (fasta_path, chrom): (trim_start, trim_end) }
_TRIM_CACHE: dict[tuple[str, str], tuple[int, int]] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def analysis_params_from_env() -> ana.AnalysisParams:
    return ana.AnalysisParams(
        segment_size=_env_int("SEGMENT_SIZE", 8000),
        window_size=_env_int("WINDOW_SIZE", 256000),
        window_step=_env_int("WINDOW_STEP", 64000),
        top_k=_env_int("TOP_K", 10),
        threshold_jap=_env_float("THRESHOLD_JAP", 0.55519),
        threshold_ind=_env_float("THRESHOLD_IND", 0.53473),
    )


def max_windows_from_env() -> Optional[int]:
    """MAX_NUMBER_256W：无/空/0/负数 → 不限制（None）；正整数 → 每请求至多推理 N 个 256k 窗口。

    语义：限制「一次推理最多计算多少个 256k 标准窗口」以控制单请求 GPU 开销。
    """
    raw = str(os.getenv("MAX_NUMBER_256W", "")).strip()
    if not raw:
        return None
    try:
        v = int(float(raw))
    except (ValueError, TypeError):
        return None
    return v if v > 0 else None


def get_chromosome_trim(fasta_path: str, chrom: str, chrom_len: int) -> tuple[int, int]:
    """返回染色体级 trim 偏移 [trim_start, trim_end)（与离线先 trim 一致，缓存）。"""
    key = (fasta_path, chrom)
    if key in _TRIM_CACHE:
        return _TRIM_CACHE[key]
    seq = read_fasta_sequence(fasta_path, chrom, 0, chrom_len)
    _, span = trim_n(seq)
    span = (int(span[0]), int(span[1]))
    _TRIM_CACHE[key] = span
    logger.info("Chromosome %s trim span: %s", chrom, span)
    return span


def _n_segments_after_trim(trim_start: int, trim_end: int, seg: int) -> int:
    """离线片段数 = ceil(trimmed_len / segment_size)（最后一段可不足）。"""
    trimmed_len = max(0, trim_end - trim_start)
    return (trimmed_len + seg - 1) // seg


def _segments_window_grid(
    fasta_path: str, chrom: str, win_start: int, win_end: int, seg: int
) -> list[dict]:
    """窗口模式：按绝对坐标网格切 8k 片段（不再 trim）。

    离线整条染色体片段坐标 = trim_start + j*8000；匹配窗口起点 win_start
    已在网格上，故片段 = win_start + j*8000。直接读取 [win_start, win_end)
    序列并按 8000 无重叠滑窗切分即可自然对齐。
    """
    seq = read_fasta_sequence(fasta_path, chrom, win_start, win_end)
    segs: list[dict] = []
    for s in range(0, len(seq), seg):
        e = min(s + seg, len(seq))
        abs_start = win_start + s
        abs_end = win_start + e
        segs.append({
            "chrom": chrom,
            "start": abs_start,
            "end": abs_end,
            "sequence": seq[s:e],
            "index": abs_start // seg,
        })
    return segs


def _segments_chromosome(
    fasta_path: str, chrom: str, chrom_len: int, seg: int
) -> tuple[list[dict], tuple[int, int]]:
    """整条模式：读全染色体 -> trim -> 切 8k 片段（与离线完全一致）。"""
    seq = read_fasta_sequence(fasta_path, chrom, 0, chrom_len)
    trimmed, span = trim_n(seq)
    segs: list[dict] = []
    for s in range(0, len(trimmed), seg):
        e = min(s + seg, len(trimmed))
        abs_start = span[0] + s
        abs_end = span[0] + e
        segs.append({
            "chrom": chrom,
            "start": abs_start,
            "end": abs_end,
            "sequence": trimmed[s:e],
            "index": abs_start // seg,
        })
    return segs, (int(span[0]), int(span[1]))


def _existing_grid_starts(
    params: ana.AnalysisParams,
    chrom_len: int,
    trim_start: int,
    n_segments: Optional[int],
) -> list[int]:
    """标准窗口网格起点（与 match_window_to_grid 内部一致，避免重复实现）。"""
    return ana.standard_window_starts(
        params, chrom_len, trim_start=trim_start, n_segments=n_segments
    )


def _select_max_region_starts(
    wins: list[int], user_start: int, user_end: int, max_n: int, window_size: int
) -> list[int]:
    """从候选起点中选与用户区间 [user_start, user_end) 重叠最大的至多 max_n 个起点。

    - 覆盖度：overlap = min(gs+window_size, user_end) - max(gs, user_start)；
      必须用 window_size（窗口宽，通常 256k）而非窗口网格步长（64k）计算。
    - 负/0 重叠取 0；平局取起点更早者（确定性，与 match_window_to_grid 一致）。
    - 不足 max_n 时返回全部；没有正重叠时兜底返回前 max_n 个（确定性）。
    """
    if not wins:
        return []
    scored = [
        (
            max(0, min(gs + int(window_size), user_end) - max(gs, user_start)),
            -int(gs),
            gs,
        )
        for gs in wins
    ]
    ranked = sorted(scored, key=lambda t: (t[0], t[1]), reverse=True)
    if ranked[0][0] <= 0:
        return wins[:max_n]
    return [int(t[2]) for t in ranked[:max_n]]


def select_windows(
    params: ana.AnalysisParams,
    chrom_len: int,
    trim_start: int,
    n_seg: int,
    start: Optional[int],
    end: Optional[int],
    max_windows: Optional[int],
) -> list[int]:
    """决定本次请求实际计算的 256k 窗口起点列表（绝对坐标，网格起点）。

    语义（与离线标准窗口网格一致；max_windows=None 时行为与旧版逐位一致）：
    - 整条模式（start/end 都空）：
        max_windows is None -> 全部网格窗口（离线 aggregate_windows_for_chrom）
        max_windows = N      -> 只取前 N 个网格窗口（染色体 5' 端）
    - 窗口模式（给了 start 或 end）：
        只填 start（end 空）-> 始终 1 个「最大覆盖度」网格窗口（start 即用户
                              显式指定目标位点，忽略 MAX_NUMBER_256W）
        start/end 都填：
            max_windows is None -> 1 个「最大覆盖度」网格窗口（旧版 match_window_to_grid）
            max_windows = N      -> 覆盖度最大的 N 个网格窗口
    """
    seg = int(params.segment_size)
    grid_starts = _existing_grid_starts(params, chrom_len, trim_start, n_seg)

    if start is None and end is None:
        if max_windows is None:
            return list(grid_starts)
        return list(grid_starts[:max_windows])

    # 窗口模式
    ws = int(params.window_size)
    user_start = int(start) if start is not None else 0
    user_end = int(end) if end is not None else user_start + ws
    if user_end <= user_start:
        user_end = user_start + ws

    # 只填 start（end 空）：start 即用户显式指定的目标位点 → 默认只推 1 个
    # 覆盖度最大的 256k 窗口（无论 MAX_NUMBER_256W 是否设置）
    if end is None or max_windows is None:
        matched = ana.match_window_to_grid(
            params, chrom_len, user_start, user_end,
            trim_start=trim_start, n_segments=n_seg,
        )
        return [int(matched[0])]

    return _select_max_region_starts(
        grid_starts, user_start, user_end, int(max_windows),
        window_size=int(params.window_size),
    )


def _predict_segments(
    segs: list[dict],
    progress_key: Optional[str] = None,
) -> pd.DataFrame:
    """片段推理 -> DataFrame（含 prob_jap/prob_ind）。

    progress_key 非空时，把推理进度写入全局 progress_tracker（供 /progress 轮询）。
    """
    if not segs:
        return pd.DataFrame(
            columns=["chrom", "start", "end", "prob_jap", "prob_ind"]
        )
    result = require_predictor().predict_segments(
        segs,
        progress_cb=(
            (lambda done_segments, total_segments, done_batches, total_batches:
                _report_progress(progress_key, done_segments, total_segments,
                                 done_batches, total_batches))
            if progress_key else None
        ),
    )
    rows = []
    for seg, pj, pi in zip(segs, result["prob_jap"], result["prob_ind"]):
        rows.append({
            "chrom": seg["chrom"],
            "start": int(seg["start"]),
            "end": int(seg["end"]),
            "prob_jap": float(pj),
            "prob_ind": float(pi),
        })
    return pd.DataFrame(rows)


def _report_progress(
    key: str,
    done_segments: int,
    total_segments: int,
    done_batches: int,
    total_batches: int,
) -> None:
    """把 batch 回调进度写入全局 tracker（带 key 校验避免旧任务覆盖新任务）。"""
    from rice_introgression.progress import progress_tracker

    if not progress_tracker.is_current(key):
        return
    progress_tracker.update(
        done_segments=done_segments,
        total_segments=total_segments,
        done_batches=done_batches,
        total_batches=total_batches,
    )


def _aggregate_window_slice(
    segs_df: pd.DataFrame, params: ana.AnalysisParams, win_start: int
) -> Optional[dict]:
    """对单个窗口内的片段做 top-k 聚合（相邻 32 片段）。"""
    if segs_df.empty:
        return None
    jap = segs_df["prob_jap"].to_numpy(dtype=float)
    ind = segs_df["prob_ind"].to_numpy(dtype=float)
    k = min(params.top_k, len(jap))
    top_jap = float(np.sort(jap)[-k:].mean())
    top_ind = float(np.sort(ind)[-k:].mean())
    return {
        "chr_id": str(segs_df["chrom"].iloc[0]),
        "win_start": int(win_start),
        "win_end": int(win_start + params.window_size),
        "center": int(win_start) + params.window_size // 2,
        "n_segments": int(len(jap)),
        "topk_mean_jap": top_jap,
        "topk_mean_ind": top_ind,
        "group": ana.assign_dual_group(
            top_jap, top_ind, params.threshold_jap, params.threshold_ind
        ),
    }


def run_introgression(
    genome: str,
    chromosome: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> dict:
    """核心入口：按用户输入计算（滑动窗口对齐后的）渗入分析结果。"""
    params = analysis_params_from_env()
    seg = params.segment_size
    genome_config = resolve_genome_config(genome)
    fasta_path = genome_config["fasta"]
    chrom_len = get_chromosome_length(genome, chromosome, genome_config)

    # 染色体级 trim（缓存）
    trim_start, trim_end = get_chromosome_trim(fasta_path, chromosome, chrom_len)
    n_seg = _n_segments_after_trim(trim_start, trim_end, seg)

    # 1. 确定查询区间
    mode = "chromosome"
    max_windows = max_windows_from_env()
    try:
        query_start, query_end = (0, chrom_len) if start is None and end is None else (
            int(start) if start is not None else 0,
            int(end) if end is not None else int(start or 0) + params.window_size,
        )
    except (TypeError, ValueError):
        raise ValueError(f"Invalid start/end: start={start!r}, end={end!r}")

    if start is None and end is None:
        if max_windows is not None:
            mode = "window"
        wins = select_windows(
            params, chrom_len, trim_start, n_seg,
            start=None, end=None, max_windows=max_windows,
        )
        if max_windows is None:
            region_start, region_end = 0, chrom_len
        else:
            # 整条模式 + max_windows：region = 所选窗口覆盖的区间（不许超末端）
            region_start, region_end = wins[0], wins[-1] + params.window_size
    else:
        mode = "window"
        wins = select_windows(
            params, chrom_len, trim_start, n_seg,
            start=start, end=end, max_windows=max_windows,
        )
        if max_windows is None:
            # 旧版行为：单窗口（match_window_to_grid 结果）
            win_start, win_end = wins[0], wins[0] + params.window_size
            region_start, region_end = win_start, win_end
        else:
            if not wins:
                # 用户区间完全在染色体外或无可匹配窗口：回退到坐标 0
                wins = [0]
            region_start, region_end = wins[0], wins[-1] + params.window_size

    if any(w >= chrom_len for w in wins):
        raise ValueError(
            f"Start {wins[0]} exceeds chromosome length {chrom_len}"
        )
    region_end = min(region_end, chrom_len)

    # 2. 缓存命中检查
    cache_key = prediction_cache.build_key(
        "intro",
        genome=genome,
        chromosome=chromosome,
        start=region_start,
        end=region_end,
    )
    hit = prediction_cache.get(cache_key)
    if hit is not None:
        payload = dict(hit["payload"])
        payload["cached"] = True
        return payload

    # 2.5 初始化进度任务（供 /progress 轮询；缓存命中无需进度）
    from rice_introgression.progress import progress_tracker

    seg_param = seg
    if mode == "window":
        # 多窗口模式：进度总量 = 所选全部窗口覆盖的片段数（窗口间可能重叠，
        # 用「各窗口片段数之和」作为上界近似，避免低估）
        progress_total = sum(
            _n_segments_after_trim(w, w + params.window_size, seg_param) for w in wins
        )
    else:
        progress_total = n_seg
    progress_key = f"{genome}:{chromosome}:{region_start}:{region_end}"
    progress_tracker.start(
        key=progress_key,
        genome=genome,
        chromosome=chromosome,
        region_start=region_start,
        region_end=region_end,
        total_segments=progress_total,
        total_batches=(progress_total + require_predictor().batch_size - 1)
        // require_predictor().batch_size,
        message="读取序列 / 切片段",
    )

    # 3. 切片段 + 推理
    try:
        if mode == "window":
            # 窗口模式（单窗或多窗）：只推理所选窗口内的片段，窗口各自聚合
            win_segments: list[pd.DataFrame] = []
            all_segs: list[dict] = []
            # 单窗（max_windows is None）：保持旧版 payload 语义（1 个窗口）
            if max_windows is None:
                segs = _segments_window_grid(fasta_path, chromosome, region_start, region_end, seg)
                segs_df = _predict_segments(segs, progress_key=progress_key)
                all_segs = segs_df.to_dict("records")
                agg = _aggregate_window_slice(segs_df, params, region_start)
                if agg is not None:
                    windows_df = pd.DataFrame([agg])
                else:
                    windows_df = pd.DataFrame(columns=[
                        "chr_id", "win_start", "win_end", "center", "n_segments",
                        "topk_mean_jap", "topk_mean_ind", "group",
                    ])
            else:
                # 多窗：逐窗口读区段（含 N 补齐）→ 按绝对坐标网格切 8k → 推理 → 聚合
                all_segs = []
                win_frames: list[pd.DataFrame] = []
                for w in wins:
                    wseg = _segments_window_grid(fasta_path, chromosome, w, w + params.window_size, seg)
                    if not wseg:
                        continue
                    # 窗口级推理（进度累计到同一 progress_key）
                    wdf = _predict_segments(wseg, progress_key=progress_key)
                    agg = _aggregate_window_slice(wdf, params, w)
                    all_segs.extend(wdf.to_dict("records"))
                    if agg is not None:
                        win_frames.append(pd.DataFrame([agg]))
                windows_df = (
                    pd.concat(win_frames, ignore_index=True)
                    if win_frames
                    else pd.DataFrame(columns=[
                        "chr_id", "win_start", "win_end", "center", "n_segments",
                        "topk_mean_jap", "topk_mean_ind", "group",
                    ])
                )
                segs_df = pd.DataFrame(all_segs)
        else:
            segs, _ = _segments_chromosome(fasta_path, chromosome, chrom_len, seg)
            segs_df = _predict_segments(segs, progress_key=progress_key)
            segs_for_agg = segs_df.copy()
            segs_for_agg["chr_id"] = chromosome
            windows_df = ana.aggregate_windows_for_chrom(
                segs_for_agg,
                prob_cols=("prob_jap", "prob_ind"),
                params=params,
                min_segments=1,
            )
    except Exception:
        progress_tracker.fail("推理失败")
        raise

    # 4. 区域融合
    if windows_df.empty:
        group_regions = {g: pd.DataFrame() for g in ana.GROUP_ORDER}
    else:
        group_regions = ana.call_group_regions(windows_df)

    # 5. 组装 payload
    seg_list = segs_df[["start", "end", "prob_jap", "prob_ind"]].to_dict("records")
    win_list = windows_df.to_dict("records") if not windows_df.empty else []
    regions_out = {
        g: group_regions[g].to_dict("records") if not group_regions[g].empty else []
        for g in ana.GROUP_ORDER
    }

    payload = {
        "genome": genome,
        "chromosome": chromosome,
        "chrom_len": int(chrom_len),
        "query_start": int(query_start),
        "query_end": int(query_end),
        "mode": mode,
        "win_start": int(region_start) if mode == "window" else None,
        "win_end": int(region_end) if mode == "window" else None,
        "trim_start": int(trim_start),
        "trim_end": int(trim_end),
        "segments": seg_list,
        "windows": win_list,
        "regions": regions_out,
        "params": {
            "segment_size": params.segment_size,
            "window_size": params.window_size,
            "window_step": params.window_step,
            "top_k": params.top_k,
            "threshold_jap": params.threshold_jap,
            "threshold_ind": params.threshold_ind,
        },
        "threshold_rule": ana.threshold_rule_text(params),
        "cached": False,
    }
    progress_tracker.finish("推理完成")
    prediction_cache.put(cache_key, {"payload": payload})
    return payload


# ---------------------------------------------------------------------------
#  全基因组渗入分析（12 条染色体全景，与离线 4.run_analysis 一致）
# ---------------------------------------------------------------------------
def _chromosome_order(chrs: list[str]) -> list[str]:
    """把染色体名按数字排序（Chr01..Chr12 / GWHBKAR00000001..12）。不认识的放最后保持原序。"""
    def num_key(c):
        m = re.search(r"(\d+)", str(c))
        return (0, int(m.group(1))) if m else (1, str(c))

    return sorted(list(dict.fromkeys(chrs)), key=num_key)


def inject_genome_context(payload: dict) -> dict:
    """把单染色体/窗口预测 payload 提升为「全基因组展示」payload。

    推理只跑用户请求的区域（单次，不合并历史缓存），但展示恒为全基因组视图：
    - chromosomes：该基因组全部染色体（数字排序，Chr01..Chr12）
    - chromosome_lengths：各染色体真实长度（来自 FASTA，画车道骨架）
    - windows / regions：保持本次推断结果，并补 chromosome 字段
    前端据此绘制 12 条车道：uninferenced 浅灰底色 + 已推断区按类叠涂。
    """
    from rice_introgression.genome_service import (
        get_chromosome_length,
        get_genome_chromosomes,
        resolve_genome_config,
    )

    genome = payload["genome"]
    genome_config = resolve_genome_config(genome)
    chromosomes = _chromosome_order(get_genome_chromosomes(genome, genome_config))
    chrom_lengths = {
        str(c): int(get_chromosome_length(genome, c, genome_config))
        for c in chromosomes
    }

    windows = []
    for w in payload.get("windows", []):
        w = dict(w)
        w.setdefault("chromosome", payload.get("chromosome"))
        windows.append(w)

    regions = {}
    for g, recs in payload.get("regions", {}).items():
        out = []
        for r in recs:
            r = dict(r)
            r.setdefault("chromosome", payload.get("chromosome"))
            out.append(r)
        regions[g] = out

    return {
        **payload,
        "mode": "genome",
        "chromosomes": chromosomes,
        "chromosome_lengths": chrom_lengths,
        "windows": windows,
        "regions": regions,
    }


def run_genome_introgression(
    genome: str,
) -> dict:
    """全染色体渗入分析：逐条调用 run_introgression（整染色体模式）。

    结果按染色体顺序拼装，返回与 /analyze 单染色体兼容的 payload，并附带
    chromosome_lengths（供展示）与逐染色体的 elapsed。

    策略：不重复推理——每条染色体先检查 prediction_cache（磁盘+内存）；
    未命中才触发 GPU 推理。返回的 payload 同时纳入各染色体缓存。
    """
    from rice_introgression.genome_service import (
        get_chromosome_length,
        get_genome_chromosomes,
        resolve_genome_config,
    )

    params = analysis_params_from_env()
    genome_config = resolve_genome_config(genome)
    chrs = get_genome_chromosomes(genome, genome_config)
    chromosomes = _chromosome_order(chrs)

    chrom_payloads: dict[str, dict] = {}
    chrom_lengths: dict[str, int] = {}
    all_segments: list[dict] = []
    all_windows: list[dict] = []
    all_regions: dict[str, list] = {g: [] for g in ana.GROUP_ORDER}
    totals: dict[str, int] = {g: 0 for g in ana.GROUP_ORDER}

    for chrom in chromosomes:
        length = get_chromosome_length(genome, chrom, genome_config)
        chrom_lengths[str(chrom)] = int(length)
        payload = run_introgression(genome=genome, chromosome=chrom, start=None, end=None)
        chrom_payloads[str(chrom)] = payload

        for seg in payload.get("segments", []):
            seg = dict(seg)
            seg["chromosome"] = chrom
            all_segments.append(seg)
        for w in payload.get("windows", []):
            w = dict(w)
            w["chromosome"] = chrom
            all_windows.append(w)
        for g in ana.GROUP_ORDER:
            for r in payload.get("regions", {}).get(g, []):
                r = dict(r)
                r["chromosome"] = chrom
                all_regions[g].append(r)
            totals[g] += len(payload.get("regions", {}).get(g, []))

    return {
        "genome": genome,
        "mode": "genome",
        "chromosomes": chromosomes,
        "chromosome_lengths": chrom_lengths,
        "regions": all_regions,
        "windows": all_windows,
        "segments": all_segments,
        "totals": totals,
        "params": {
            "segment_size": params.segment_size,
            "window_size": params.window_size,
            "window_step": params.window_step,
            "top_k": params.top_k,
            "threshold_jap": params.threshold_jap,
            "threshold_ind": params.threshold_ind,
        },
        "threshold_rule": ana.threshold_rule_text(params),
        "cached": False,
    }