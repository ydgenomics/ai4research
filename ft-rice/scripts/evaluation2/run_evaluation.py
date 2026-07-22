#!/usr/bin/env python3
"""
run_evaluation.py — 基因表达预测评估统一入口（配置驱动版）

所有输入文件在 config.yaml 中显式指定，不做目录扫描和正则解析。

执行流程:
  Step 1: 读取 config.yaml 中的 tasks 列表
  Step 2: 逐任务（按染色体）计算：
    - Track级（碱基级）指标 → 01_track_level.csv
    - Segment级（窗口级）指标 → 02_segment_level.csv + 03_segment_per_window.csv
    - Feature级（基因/外显子级）指标 → 04_feature_level.csv + 05_feature_per_gene.csv
  Step 3: 计算表达分桶指标（含 delta_pcc）→ 06_feature_stratified.csv
  Step 4: 跨染色体全局汇总（按品种-组织）→ 00_global_summary.csv
  Step 5: 计算跨品种差异表达 → 00_cross_variety_delta_summary.csv
  Step 6: 写入清单 → 08_run_manifest.csv

用法:
  python run_evaluation.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# 将 evaluation2 目录加入 path，确保直接运行也能导入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from metrics_core import (
    compute_track_metrics,
    compute_segment_metrics_per_window,
    compute_segment_level_summary,
    compute_feature_mean_correlation,
    assign_expression_buckets,
    compute_stratified_feature_metrics,
    build_per_gene_reference,
    compute_pairwise_delta_metrics,
)
from utils import (
    EvalConfig,
    EvalTask,
    load_config,
    load_prediction_csv,
    flatten_to_genome_array,
    load_features_from_gff,
    aggregate_predictions_to_features,
)


# =============================================================================
# Step 1: 任务列表（从 config.tasks 读取，不做目录扫描和正则）
# =============================================================================

def print_tasks(tasks: list[EvalTask]):
    """打印任务摘要。"""
    print(f"\n📂 {len(tasks)} evaluation task(s) from config:")
    for t in tasks:
        gff_info = f"  gff={t.gff.name}" if t.gff else ""
        print(f"   [{t.split:5s}] {t.tissue}/{t.species}/{t.chromosome}  {t.predict_csv.name}{gff_info}")


# =============================================================================
# Step 2: 单任务评估
# =============================================================================

def evaluate_one_task(
    task: EvalTask,
    config: EvalConfig,
    features_cache: Optional[dict] = None,
) -> dict:
    """对单个评估任务执行全部三个层级的评估。

    Returns dict with keys: task, context, track_metrics, segment_summary,
      per_window_df, feature_df, feature_summary.
    """
    context = task.to_context()
    csv_path = task.predict_csv

    print(f"\n{'='*60}")
    print(f"📊 Evaluating: [{task.split}] {task.tissue}/{task.species}/{task.chromosome}")
    print(f"   CSV: {csv_path}")
    if task.gff:
        print(f"   gff: {task.gff}")

    # 读取并解析
    df = load_prediction_csv(csv_path)
    if len(df) == 0:
        return {"csv_path": csv_path, "context": context, "error": "empty dataframe"}

    # ---- Track级：拼接全染色体逐碱基 ----
    print("  📐 Track-level (base-pair)...")
    pred_dict = flatten_to_genome_array(df, "parsed_pred")
    true_dict = flatten_to_genome_array(df, "parsed_true")

    all_preds = []
    all_trues = []
    for chrom in sorted(set(list(pred_dict.keys()) + list(true_dict.keys()))):
        if chrom in pred_dict and chrom in true_dict:
            all_preds.append(pred_dict[chrom])
            all_trues.append(true_dict[chrom])

    global_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float32)
    global_true = np.concatenate(all_trues) if all_trues else np.array([], dtype=np.float32)

    track_metrics = compute_track_metrics(global_pred, global_true)

    # ---- Segment级：逐窗口 ----
    print("  📐 Segment-level (per-window)...")
    pred_arrays = [np.asarray(v, dtype=float) for v in df["parsed_pred"]]
    true_arrays = [np.asarray(v, dtype=float) for v in df["parsed_true"]]
    chromosomes = df["chromosome"].tolist()
    starts = df["start"].tolist()
    ends = df["end"].tolist()

    per_window_df = compute_segment_metrics_per_window(pred_arrays, true_arrays, chromosomes, starts, ends)
    segment_summary = compute_segment_level_summary(per_window_df, global_pred, global_true)

    # ---- Feature级（如果有注释） ----
    feature_df = pd.DataFrame()
    feature_summary = {}

    if task.gff is not None and task.gff.is_file() and features_cache is not None:
        try:
            print(f"  📐 Feature-level (gene/exon)...")
            gff_path = task.gff
            print(f"     Annotation: {gff_path.name}")

            features_by_chrom = features_cache.get(task.species)
            if features_by_chrom is None:
                features_by_chrom = load_features_from_gff(gff_path, set(config.feature_types), config.feature_flank_bp)
                features_cache[task.species] = features_by_chrom

            feature_df = aggregate_predictions_to_features(df, features_by_chrom, config.min_overlap_bp)
            if not feature_df.empty:
                # 写入上下文信息
                for key, value in context.items():
                    feature_df.insert(0, key, value)

                # 按 feature_type 汇总
                for ftype in feature_df["feature_type"].unique():
                    part = feature_df[feature_df["feature_type"] == ftype]
                    corr = compute_feature_mean_correlation(part)
                    feature_summary[ftype] = {
                        "n_features": len(part),
                        "n_valid": len(part.dropna(subset=["pred_mean", "true_mean"])),
                        **corr,
                        "pearson_mean": round(float(part["pearson"].mean()), 6),
                        "pearson_median": round(float(part["pearson"].median()), 6),
                    }
        except FileNotFoundError as e:
            print(f"     ⚠️  Skipping feature-level: {e}")

    return {
        "csv_path": csv_path,
        "context": context,
        "track_metrics": track_metrics,
        "segment_summary": segment_summary,
        "per_window_df": per_window_df,
        "feature_df": feature_df,
        "feature_summary": feature_summary,
        # 保留逐碱基数组，用于跨染色体全局聚合
        "global_pred": global_pred,
        "global_true": global_true,
    }


# =============================================================================
# Step 3-5: 汇总与输出
# =============================================================================

def build_track_level_df(all_results: list[dict]) -> pd.DataFrame:
    """构建 01_track_level.csv。"""
    rows = []
    for r in all_results:
        if r.get("error"):
            continue
        row = {**r["context"], **r["track_metrics"]}
        rows.append(row)
    return pd.DataFrame(rows)


def build_segment_level_df(all_results: list[dict]) -> pd.DataFrame:
    """构建 02_segment_level.csv。"""
    rows = []
    for r in all_results:
        if r.get("error"):
            continue
        row = {**r["context"], **r["segment_summary"]}
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_summary_df(
    all_results: list[dict],
    ref_table: pd.DataFrame | None = None,
    normalize: str = "global_mean",
) -> pd.DataFrame:
    """构建 04_feature_level.csv（gene 行含 delta_pcc）。"""
    rows = []
    for r in all_results:
        if r.get("error"):
            continue
        fdf = r.get("feature_df")
        biosample = r["context"].get("biosample", "")
        for ftype, summary in r.get("feature_summary", {}).items():
            row = {
                **r["context"],
                "feature_type": ftype,
                **summary,
            }
            # 对 gene 类型计算 delta_pcc
            if ref_table is not None and not ref_table.empty and ftype == "gene" and fdf is not None and not fdf.empty:
                gene_part = fdf[fdf["feature_type"] == "gene"]
                if not gene_part.empty:
                    from metrics_core import compute_all_delta_pearson
                    delta = compute_all_delta_pearson(gene_part, ref_table, normalize, biosample)
                    row["delta_pcc_zero"] = delta.get("gene_delta_pcc_zero", np.nan)
                    row["delta_pcc_feature_ref"] = delta.get("gene_delta_pcc_feature_ref", np.nan)
                    row["delta_pcc_global_ref"] = delta.get("gene_delta_pcc_global_ref", np.nan)
                    row["delta_rmse"] = delta.get("gene_delta_rmse", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_feature_stratified_df(
    all_results: list[dict],
    bucket_thresholds=None,
    ref_table: pd.DataFrame | None = None,
    normalize: str = "global_mean",
) -> pd.DataFrame:
    """构建 00_stratified_summary.csv（按 feature_type 分别分箱，含 delta_pcc）。"""
    rows = []
    for r in all_results:
        if r.get("error"):
            continue
        fdf = r.get("feature_df")
        if fdf is None or fdf.empty:
            continue

        # 对所有训练集基因的 true_mean 统一分桶（仅用 gene 行确定阈值）
        all_train_true = []
        for r2 in all_results:
            if r2.get("error"):
                continue
            if r2["context"].get("split") == "train":
                fdf2 = r2.get("feature_df")
                if fdf2 is not None and not fdf2.empty:
                    gene_part = fdf2[fdf2["feature_type"] == "gene"]
                    if not gene_part.empty:
                        all_train_true.append(gene_part["true_mean"].dropna())

        if all_train_true and bucket_thresholds is None:
            combined = pd.concat([pd.Series(a) for a in all_train_true])
            perc = 100.0 / 3
            bucket_thresholds = (
                np.percentile(combined, perc),
                np.percentile(combined, 100 - perc),
            )

        # 按 feature_type 分组，分别计算分层指标
        biosample = r["context"].get("biosample", "")
        for ftype in sorted(fdf["feature_type"].dropna().unique()):
            part = fdf[fdf["feature_type"] == ftype]
            if part.empty:
                continue
            fdf_bucketed, _ = assign_expression_buckets(
                part, "true_mean", n_buckets=3, thresholds=bucket_thresholds
            )
            stratified = compute_stratified_feature_metrics(
                fdf_bucketed, feature_type=ftype,
                ref_table=ref_table, biosample=biosample, normalize=normalize,
            )

            for _, srow in stratified.iterrows():
                rows.append({
                    **r["context"],
                    **srow.to_dict(),
                })

    df = pd.DataFrame(rows)
    if not df.empty and bucket_thresholds:
        df["bucket_low_threshold"] = bucket_thresholds[0]
        df["bucket_high_threshold"] = bucket_thresholds[1]
    return df





def _task_dir_name(context: dict) -> str:
    """从 context 构建子目录名: {split}-{species}-{tissue}-{chromosome}"""
    return f"{context['split']}-{context['species']}-{context['tissue']}-{context['chrom_unit']}"


def write_one_task(
    result: dict,
    base_outdir: Path,
    bucket_thresholds=None,
    ref_table: pd.DataFrame | None = None,
    n_buckets: int = 3,
):
    """将单个 task 的评估结果写入独立子目录（01~07）。"""
    if result.get("error"):
        return

    ctx = result["context"]
    subdir = base_outdir / _task_dir_name(ctx)
    subdir.mkdir(parents=True, exist_ok=True)

    # 01: Track
    row = {**ctx, **result["track_metrics"]}
    pd.DataFrame([row]).to_csv(subdir / "01_track_level.csv", index=False)

    # 02: Segment summary
    row = {**ctx, **result["segment_summary"]}
    pd.DataFrame([row]).to_csv(subdir / "02_segment_level.csv", index=False)

    # 03: Per-window
    pdf = result.get("per_window_df")
    if pdf is not None and not pdf.empty:
        for key, value in ctx.items():
            pdf.insert(0, key, value)
        pdf.to_csv(subdir / "03_segment_per_window.csv", index=False)

    # 04: Feature summary（gene 行含 delta_pcc）
    feat_sum = result.get("feature_summary", {})
    if feat_sum:
        rows = []
        fdf = result.get("feature_df")
        biosample = ctx.get("biosample", "")
        for ftype, summary in feat_sum.items():
            row = {**ctx, "feature_type": ftype, **summary}
            # 对 gene 类型计算 delta_pcc
            if ref_table is not None and not ref_table.empty and ftype == "gene" and fdf is not None and not fdf.empty:
                gene_part = fdf[fdf["feature_type"] == "gene"]
                if not gene_part.empty:
                    from metrics_core import compute_all_delta_pearson
                    delta = compute_all_delta_pearson(gene_part, ref_table, "global_mean", biosample)
                    row["delta_pcc_zero"] = delta.get("gene_delta_pcc_zero", np.nan)
                    row["delta_pcc_feature_ref"] = delta.get("gene_delta_pcc_feature_ref", np.nan)
                    row["delta_pcc_global_ref"] = delta.get("gene_delta_pcc_global_ref", np.nan)
                    row["delta_rmse"] = delta.get("gene_delta_rmse", np.nan)
            rows.append(row)
        pd.DataFrame(rows).to_csv(subdir / "04_feature_level.csv", index=False)

    # 05: Per-feature
    fdf = result.get("feature_df")
    if fdf is not None and not fdf.empty:
        fdf.to_csv(subdir / "05_feature_per_gene.csv", index=False)

    # 06: Stratified (依赖全局分桶阈值，按 feature_type 分组，含 delta_pcc)
    if fdf is not None and not fdf.empty and bucket_thresholds is not None:
        all_stratified = []
        biosample = ctx.get("biosample", "")
        for ftype in sorted(fdf["feature_type"].dropna().unique()):
            ftype_part = fdf[fdf["feature_type"] == ftype]
            if ftype_part.empty:
                continue
            fdf_bucketed, _ = assign_expression_buckets(ftype_part, "true_mean", n_buckets=n_buckets, thresholds=bucket_thresholds)
            stratified = compute_stratified_feature_metrics(
                fdf_bucketed, feature_type=ftype,
                ref_table=ref_table, biosample=biosample, normalize="global_mean",
            )
            if not stratified.empty:
                for col, val in ctx.items():
                    stratified.insert(0, col, val)
                all_stratified.append(stratified)
        if all_stratified:
            combined = pd.concat(all_stratified, ignore_index=True)
            combined.to_csv(subdir / "06_feature_stratified.csv", index=False)

    print(f"  ✅ {subdir.name}/  (01~06)")


def _compute_global_thresholds(all_results: list[dict], n_buckets: int = 3):
    """从所有训练集的 true_mean 计算全局分桶阈值。"""
    all_train_true = []
    for r in all_results:
        if r.get("error"):
            continue
        if r["context"].get("split") == "train":
            fdf = r.get("feature_df")
            if fdf is not None and not fdf.empty:
                all_train_true.append(fdf["true_mean"].dropna())
    if not all_train_true:
        return None
    combined = pd.concat([pd.Series(a) for a in all_train_true])
    perc = 100.0 / n_buckets
    return (
        float(np.percentile(combined, perc)),
        float(np.percentile(combined, 100 - perc)),
    )


def build_cross_variety_delta_df(all_results: list[dict]) -> pd.DataFrame:
    """构建 00_cross_variety_delta_summary.csv。

    对每一对品种（同 tissue, chromosome），在匹配的同源基因上计算：
      - Δ expression Pearson/Spearman（品种间表达差异的预测精度）
      - log2FC Pearson/Spearman（差异表达分析标准指标）
      - 方向一致性 sign_accuracy
      - Up/Down 精确率

    品种对按 pair_type 分类：train_train / train_test。
    """
    from itertools import combinations

    # Group results by (tissue, chrom_unit)
    groups: dict[tuple, list[dict]] = {}
    for r in all_results:
        if r.get("error"):
            continue
        ctx = r["context"]
        key = (ctx["tissue"], ctx["chrom_unit"])
        groups.setdefault(key, []).append(r)

    rows = []
    for (tissue, chrom), group_results in groups.items():
        # Build dict: species -> gene-level DataFrame (indexed by feature_id)
        species_dfs: dict[str, pd.DataFrame] = {}
        species_splits: dict[str, str] = {}
        for r in group_results:
            fdf = r.get("feature_df")
            if fdf is None or fdf.empty:
                continue
            sp = r["context"]["species"]
            gene_df = fdf[fdf["feature_type"] == "gene"].set_index("feature_id")
            if not gene_df.empty:
                species_dfs[sp] = gene_df
                species_splits[sp] = r["context"]["split"]

        species_list = sorted(species_dfs.keys())
        if len(species_list) < 2:
            continue

        for sp_a, sp_b in combinations(species_list, 2):
            df_a = species_dfs[sp_a]
            df_b = species_dfs[sp_b]

            # Only genes present in both cultivars
            common_ids = df_a.index.intersection(df_b.index)
            if len(common_ids) < 10:
                continue

            df_a_m = df_a.loc[common_ids]
            df_b_m = df_b.loc[common_ids]

            # Determine pair type
            split_a = species_splits.get(sp_a, "unknown")
            split_b = species_splits.get(sp_b, "unknown")
            if split_a == "train" and split_b == "train":
                pair_type = "train_train"
            else:
                pair_type = "train_test"

            metrics = compute_pairwise_delta_metrics(
                df_a_m["pred_mean"].to_numpy(dtype=float),
                df_a_m["true_mean"].to_numpy(dtype=float),
                df_b_m["pred_mean"].to_numpy(dtype=float),
                df_b_m["true_mean"].to_numpy(dtype=float),
            )

            rows.append({
                "tissue": tissue,
                "chrom_unit": chrom,
                "cultivar_a": sp_a,
                "cultivar_b": sp_b,
                "pair_type": pair_type,
                **{k: round(v, 6) if isinstance(v, float) else v
                   for k, v in metrics.items()},
            })

    return pd.DataFrame(rows)


def _build_ref_table(all_results: list[dict], normalize: str = "global_mean") -> pd.DataFrame:
    """从所有训练集的 feature_df 构建基因参考表（用于 Delta Pearson）。

    key 包含 biosample，避免不同组织互相覆盖。
    """
    train_features: dict[str, pd.DataFrame] = {}
    for r in all_results:
        if r.get("error"):
            continue
        if r["context"].get("split") != "train":
            continue
        fdf = r.get("feature_df")
        if fdf is None or fdf.empty:
            continue
        key = f"{r['context']['species']}/{r['context']['chrom_unit']}/{r['context']['biosample']}"
        train_features[key] = fdf
    if not train_features:
        return pd.DataFrame()
    return build_per_gene_reference(train_features, normalize)


def build_global_summary_df(
    all_results: list[dict],
    config: EvalConfig,
    bucket_thresholds=None,
    ref_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """跨染色体全局汇总（按品种-组织-数据集，而非按染色体）。

    对同一 (species, tissue, split) 的所有染色体数据做聚合：
      - bp: 拼接所有染色体的逐碱基数组，计算 track 级指标
      - gene/exon: 拼接所有染色体的 feature_df，计算跨基因均值相关性
      - gene-low/medium/high: 拼接 feature_df 后分桶统计

    Returns: DataFrame with columns [species, tissue, split, resolution, pcc,
      log1p_pcc, nozero_pcc, r2, delta_pcc_zero, delta_pcc_feature_ref, ...]
    """
    # 按 (species, tissue, split) 分组
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in all_results:
        if r.get("error"):
            continue
        ctx = r["context"]
        key = (ctx["species"], ctx["tissue"], ctx["split"])
        groups.setdefault(key, []).append(r)

    rows = []

    for (species, tissue, split), group_results in groups.items():
        biosample = group_results[0]["context"].get("biosample", "")

        # === Track-level (bp): 拼接所有染色体的逐碱基数组 ===
        all_pred = np.concatenate([r["global_pred"] for r in group_results if "global_pred" in r]) if group_results else np.array([], dtype=np.float32)
        all_true = np.concatenate([r["global_true"] for r in group_results if "global_true" in r]) if group_results else np.array([], dtype=np.float32)

        if len(all_pred) > 0:
            track = compute_track_metrics(all_pred, all_true)
            rows.append({
                "species": species, "tissue": tissue, "split": split,
                "resolution": "bp",
                "pcc": track["pearson"],
                "log1p_pcc": track["log_pearson"],
                "nozero_pcc": track["nonzero_pearson"],
                "r2": track["r2"],
                "n_positions": track["n_positions"],
                "mae": track["mae"],
                "rmse": track["rmse"],
                "spearman": track["spearman"],
                "zero_auroc": track["zero_auroc"],
                "zero_auprc": track["zero_auprc"],
                "nonzero_spearman": track["nonzero_spearman"],
                "zero_ratio": track["zero_ratio"],
            })

        # === Feature-level (gene/exon): 拼接所有染色体的 feature_df ===
        # 检查是否有 feature_df
        has_features = any(
            r.get("feature_df") is not None and not r["feature_df"].empty
            for r in group_results
        )
        if has_features:
            all_feature = pd.concat(
                [r["feature_df"] for r in group_results if r.get("feature_df") is not None and not r["feature_df"].empty],
                ignore_index=True,
            )

            for ftype in config.feature_types:
                part = all_feature[all_feature["feature_type"] == ftype]
                if part.empty:
                    continue

                # 跨基因均值相关性（所有基因的 pred_mean vs true_mean）
                corr = compute_feature_mean_correlation(part)
                # 逐基因 Pearson 的均值/中位数
                pearson_mean = round(float(part["pearson"].dropna().mean()), 6)
                pearson_median = round(float(part["pearson"].median()), 6)
                n_features = len(part)
                n_valid = len(part.dropna(subset=["pred_mean", "true_mean"]))

                row = {
                    "species": species, "tissue": tissue, "split": split,
                    "resolution": ftype,
                    "pcc": corr.get("feature_mean_pearson", np.nan),
                    "log1p_pcc": corr.get("feature_mean_log_pearson", np.nan),
                    "nozero_pcc": corr.get("feature_mean_nonzero_pearson", np.nan),
                    "r2": corr.get("feature_mean_r2", np.nan),
                    "spearman": corr.get("feature_mean_spearman", np.nan),
                    "pearson_mean": pearson_mean,
                    "pearson_median": pearson_median,
                    "n_features": n_features,
                    "n_valid": n_valid,
                }

                # 对 gene 类型计算 delta_pcc
                if ref_table is not None and not ref_table.empty and ftype == "gene":
                    from metrics_core import compute_all_delta_pearson
                    delta = compute_all_delta_pearson(part, ref_table, config.delta_normalize, biosample)
                    row["delta_pcc_zero"] = delta.get("gene_delta_pcc_zero", np.nan)
                    row["delta_pcc_feature_ref"] = delta.get("gene_delta_pcc_feature_ref", np.nan)
                    row["delta_pcc_global_ref"] = delta.get("gene_delta_pcc_global_ref", np.nan)
                    row["delta_rmse"] = delta.get("gene_delta_rmse", np.nan)

                rows.append(row)

            # === Stratified: 分桶统计 ===
            if bucket_thresholds is not None:
                for ftype in config.feature_types:
                    part = all_feature[all_feature["feature_type"] == ftype]
                    if part.empty:
                        continue
                    bucketed, _ = assign_expression_buckets(
                        part, "true_mean", n_buckets=config.n_expression_buckets,
                        thresholds=bucket_thresholds,
                    )
                    stratified = compute_stratified_feature_metrics(
                        bucketed, feature_type=ftype,
                        ref_table=ref_table, biosample=biosample,
                        normalize=config.delta_normalize,
                    )
                    for _, srow in stratified.iterrows():
                        bucket_label = srow.get("expression_bucket", "unknown")
                        resolution = f"{ftype}-{bucket_label}"
                        rows.append({
                            "species": species, "tissue": tissue, "split": split,
                            "resolution": resolution,
                            "pcc": srow.get("feature_mean_pearson", np.nan),
                            "log1p_pcc": srow.get("feature_mean_log_pearson", np.nan),
                            "nozero_pcc": srow.get("feature_mean_nonzero_pearson", np.nan),
                            "r2": srow.get("feature_mean_r2", np.nan),
                            "spearman": srow.get("feature_mean_spearman", np.nan),
                            "n_features": srow.get("number", np.nan),
                            "delta_pcc_zero": srow.get("delta_pcc_zero", np.nan),
                            "delta_pcc_feature_ref": srow.get("delta_pcc_feature_ref", np.nan),
                            "delta_pcc_global_ref": srow.get("delta_pcc_global_ref", np.nan),
                            "delta_rmse": srow.get("delta_rmse", np.nan),
                        })

    return pd.DataFrame(rows)


def write_top_level_summaries(
    all_results: list[dict],
    config: EvalConfig,
    bucket_thresholds=None,
):
    """在顶层目录写入跨任务汇总（00_* 汇总表 + 08 manifest）。"""
    outdir = config.output_dir

    # 00a: Track summary (all tasks)
    df = build_track_level_df(all_results)
    if not df.empty:
        df.to_csv(outdir / "00_track_summary.csv", index=False)
        print(f"  ✅ 00_track_summary.csv ({len(df)} rows)")

    # 00b: Segment summary (all tasks)
    df = build_segment_level_df(all_results)
    if not df.empty:
        df.to_csv(outdir / "00_segment_summary.csv", index=False)
        print(f"  ✅ 00_segment_summary.csv ({len(df)} rows)")

    # 00c: Feature summary (all tasks, gene 行含 delta_pcc)
    ref_table = _build_ref_table(all_results, config.delta_normalize)
    df = build_feature_summary_df(all_results, ref_table=ref_table, normalize=config.delta_normalize)
    if not df.empty:
        df.to_csv(outdir / "00_feature_summary.csv", index=False)
        print(f"  ✅ 00_feature_summary.csv ({len(df)} rows)")

    # 00d: Stratified summary (all tasks, uses global thresholds,含 delta_pcc)
    df = build_feature_stratified_df(
        all_results, bucket_thresholds,
        ref_table=ref_table, normalize=config.delta_normalize,
    )
    if not df.empty:
        df.to_csv(outdir / "00_stratified_summary.csv", index=False)
        print(f"  ✅ 00_stratified_summary.csv ({len(df)} rows)")

    # 00e: Global summary (跨染色体, 按品种-组织聚合)
    df = build_global_summary_df(all_results, config, bucket_thresholds, ref_table)
    if not df.empty:
        df.to_csv(outdir / "00_global_summary.csv", index=False)
        print(f"  ✅ 00_global_summary.csv ({len(df)} rows)")

    # 00f: Cross-variety differential expression summary
    df = build_cross_variety_delta_df(all_results)
    if not df.empty:
        df.to_csv(outdir / "00_cross_variety_delta_summary.csv", index=False)
        print(f"  ✅ 00_cross_variety_delta_summary.csv ({len(df)} rows)")

    # 08: Manifest
    rows = []
    for r in all_results:
        ctx = r.get("context", {})
        if r.get("error"):
            rows.append({**ctx, "status": "error", "error": r["error"]})
        else:
            rows.append({**ctx, "status": "ok"})
    pd.DataFrame(rows).to_csv(outdir / "08_run_manifest.csv", index=False)
    print(f"  ✅ 08_run_manifest.csv ({len(rows)} rows)")

    print(f"\n🎉 All outputs saved to {outdir}/")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gene Expression Prediction Evaluation (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --config config.yaml
  python run_evaluation.py --config config.yaml -o /path/to/output
  python run_evaluation.py --config config.yaml --skip-features
        """,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML configuration file (tasks + varieties + output)",
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip gene/exon feature-level evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    # Step 1: 任务列表
    tasks = config.tasks
    if not tasks:
        print("❌ No evaluation tasks defined in config. Exiting.")
        sys.exit(1)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧬 Gene Expression Prediction Evaluation (v2)")
    print("=" * 60)
    print(f"  Output dir:      {config.output_dir}")
    print(f"  Feature types:   {config.feature_types}")
    print(f"  Delta normalize: {config.delta_normalize}")
    print_tasks(tasks)

    # Step 2: 逐任务评估（只计算，暂不写文件）
    features_cache = {} if not args.skip_features else None
    all_results = []
    for task in tqdm(tasks, desc="Evaluating tasks", unit="task"):
        result = evaluate_one_task(task, config, features_cache)
        all_results.append(result)

    # Step 3: 计算全局阈值和参考表（用于 06/07）
    bucket_thresholds = _compute_global_thresholds(all_results, config.n_expression_buckets)
    ref_table = _build_ref_table(all_results, config.delta_normalize)
    if not ref_table.empty:
        print(f"  Built reference table: {len(ref_table)} genes")

    # Step 4: 逐 task 写入完整输出（01~06）
    print(f"\n{'='*60}")
    print("📝 Writing per-task outputs...")
    for result in all_results:
        write_one_task(
            result, config.output_dir,
            bucket_thresholds=bucket_thresholds,
            ref_table=ref_table,
            n_buckets=config.n_expression_buckets,
        )

    # Step 5: 顶层跨任务汇总
    write_top_level_summaries(all_results, config, bucket_thresholds)


if __name__ == "__main__":
    main()
