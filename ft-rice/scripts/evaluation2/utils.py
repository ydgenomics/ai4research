#!/usr/bin/env python3
"""
utils.py — 评估工具函数（CSV解析、GFF加载、目录扫描、特征聚合等）
"""

from __future__ import annotations

import json
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# 支持两种导入方式：包内相对导入 和 独立脚本直接导入
try:
    from .metrics_core import parse_expression_column
except ImportError:
    from metrics_core import parse_expression_column


# =============================================================================
# 0. 数据类
# =============================================================================

@dataclass(frozen=True)
class Feature:
    """GFF 中的基因/外显子特征。"""
    chrom: str
    start0: int       # 0-based start
    end0: int         # 0-based end (exclusive)
    feature_type: str  # gene / exon
    feature_id: str
    parent_id: str
    strand: str


@dataclass(frozen=True)
class EvalTask:
    """单个评估任务。所有路径和元数据在 config 中显式指定。"""
    predict_csv: Path
    species: str          # 物种/品种 (如 P1, P7)
    tissue: str           # 组织 (如 CSQ)
    gff: Optional[Path]   # GFF 注释文件路径 (为 None 则跳过 feature 级评估)
    split: str            # train / valid / test
    chromosome: str       # 染色体 (如 Chr01)
    biosample: str        # 生物样本名
    modality: str         # 测序模态 (如 total_RNA-seq_+)

    def to_context(self) -> dict[str, str]:
        """转为评估上下文 dict。"""
        return {
            "split": self.split,
            "species": self.species,
            "tissue": self.tissue,
            "chrom_unit": self.chromosome,
            "biosample": self.biosample,
            "modality": self.modality,
        }


@dataclass
class EvalConfig:
    """评估配置。"""
    predict_dir: Path
    ref_dir: Path
    output_dir: Path

    # 评估任务列表（从 config.yaml 的 tasks 显式列表填充）
    tasks: list[EvalTask] = None  # type: ignore[assignment]

    # Feature-level
    feature_types: tuple[str, ...] = ("gene", "exon")
    feature_flank_bp: int = 0
    min_overlap_bp: int = 1
    min_nonzero_bp: int = 2
    min_r2_variance: float = 1e-8

    # Delta Pearson
    delta_normalize: str = "global_mean"         # "none" / "global_mean" / "nonzero_mean"
    delta_ref_modes: tuple[str, ...] = ("zero", "true_global_mean", "feature_normalized_mean")

    # Expression bucket
    n_expression_buckets: int = 3                # low / medium / high
    bucket_thresholds: Optional[tuple[float, float]] = None  # 使用训练集阈值

    # CSV discovery
    csv_glob: str = "*_multitrack/*/*_predictions.csv"
    max_csv: Optional[int] = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []


# =============================================================================
# 1. CSV 读取与解析
# =============================================================================

def load_prediction_csv(csv_path: Path) -> pd.DataFrame:
    """读取预测 CSV 并解析 expression 列。

    Returns DataFrame with added columns:
      - parsed_pred: np.ndarray of float32
      - parsed_true: np.ndarray of float32
      - calc_length: int (end - start)
    """
    df = pd.read_csv(csv_path)

    required = {"chromosome", "start", "end", "predicted_expression", "true_expression"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df["parsed_pred"] = df["predicted_expression"].apply(parse_expression_column)
    df["parsed_true"] = df["true_expression"].apply(parse_expression_column)

    # 验证长度一致性
    df["calc_length"] = df["end"] - df["start"]
    df["parsed_length"] = df["parsed_pred"].apply(len)

    mismatch = df["calc_length"] != df["parsed_length"]
    if mismatch.any():
        print(f"  ⚠️  {csv_path.name}: {mismatch.sum()} rows with length mismatch, filtered")
        df = df[~mismatch].copy()

    return df


def flatten_to_genome_array(df: pd.DataFrame, value_col: str = "parsed_pred") -> dict[str, np.ndarray]:
    """将逐窗口的值按染色体位置平均后拼接。

    Returns: dict[chromosome] → np.ndarray (逐碱基平均值)
    """
    pos_data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    for _, row in df.iterrows():
        chrom = str(row["chromosome"])
        start = int(row["start"])
        values = np.asarray(row[value_col], dtype=float)
        for j, val in enumerate(values):
            pos = start + j
            pos_data[chrom][pos][0] += val
            pos_data[chrom][pos][1] += 1

    result = {}
    for chrom in sorted(pos_data.keys()):
        positions = sorted(pos_data[chrom].keys())
        result[chrom] = np.array(
            [pos_data[chrom][p][0] / pos_data[chrom][p][1] for p in positions],
            dtype=np.float32,
        )
    return result


# =============================================================================
# 3. GFF 加载
# =============================================================================

def parse_gff_attr(attr_text: str) -> dict[str, str]:
    """解析 GFF/GTF 属性列。"""
    attrs: dict[str, str] = {}
    for item in attr_text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            key, value = item.split(" ", 1)
            value = value.strip().strip('"')
        else:
            continue
        from urllib.parse import unquote
        attrs[key.strip()] = unquote(value.strip())
    return attrs


def pick_feature_id(attrs: dict[str, str], feature_type: str) -> str:
    """从 GFF 属性中提取 feature ID。"""
    for key in ("ID", "gene_id", "transcript_id", "Name"):
        value = attrs.get(key)
        if value:
            return value
    return f"{feature_type}:unknown"


def pick_parent(attrs: dict[str, str]) -> str:
    """从 GFF 属性中提取 Parent ID。"""
    for key in ("Parent", "gene_id", "transcript_id"):
        value = attrs.get(key)
        if value:
            return value.split(",")[0]
    return ""


def load_features_from_gff(
    gff_path: Path,
    feature_types: set[str],
    flank_bp: int = 0,
) -> dict[str, list[Feature]]:
    """从 GFF/GTF 文件加载指定类型的特征并按染色体分组。

    当同时请求 "gene" 和 "exon" 类型时，gene 的区间会自动替换为
    该基因所有 exon 的并集，使基因表达只计算外显子区域（不含 intron）。
    内部会自动解析 mRNA/transcript 特征来建立 exon → gene 的映射关系。

    Returns: dict[chromosome] → sorted list of Feature
    """
    features_by_chrom: dict[str, list[Feature]] = defaultdict(list)
    transcript_to_gene: dict[str, str] = {}

    with gff_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, ftype, start, end, _, strand, _, attr_text = parts[:9]

            attrs = parse_gff_attr(attr_text)

            # 构建转录本→基因映射（内部使用，不输出到 feature_types）
            if ftype in ("mRNA", "transcript"):
                parent = pick_parent(attrs)
                tid = pick_feature_id(attrs, ftype)
                if parent and tid and tid != parent:
                    transcript_to_gene[tid] = parent
                continue  # 不存储 mRNA 特征

            if ftype not in feature_types:
                continue

            start0 = int(start) - 1
            end0 = int(end)
            if start0 < 0 or end0 <= start0:
                continue

            eval_start0 = max(0, start0 - flank_bp)
            eval_end0 = end0 + flank_bp

            feature = Feature(
                chrom=chrom,
                start0=start0,
                end0=end0,
                feature_type=ftype,
                feature_id=pick_feature_id(attrs, ftype),
                parent_id=pick_parent(attrs),
                strand=strand,
            )

            # 将 eval_start0/eval_end0 存储到 feature 中（通过动态属性）
            # 这里使用 composition 方式：用 dict 而非修改 frozen dataclass
            features_by_chrom[chrom].append((feature, eval_start0, eval_end0))

    # Sort and convert
    result: dict[str, list[Feature]] = {}
    for chrom in features_by_chrom:
        features_by_chrom[chrom].sort(key=lambda x: (x[1], x[2]))
        result[chrom] = [f for f, _, _ in features_by_chrom[chrom]]

    # 如果同时包含 gene 和 exon，将 gene 区间替换为对应 exon 的并集
    # 使基因表达只计算外显子区域，排除 intron 零值稀释
    if "gene" in feature_types and "exon" in feature_types:
        # 通过转录本→基因映射，将 exon parent_id 解析为 gene_id
        def _resolve_gene_id(exon: Feature) -> str:
            if exon.parent_id and exon.parent_id in transcript_to_gene:
                return transcript_to_gene[exon.parent_id]
            return exon.parent_id or exon.feature_id

        # 按 gene_id 收集所有 exon 区间
        exon_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chrom, features in result.items():
            for f in features:
                if f.feature_type == "exon":
                    gene_id = _resolve_gene_id(f)
                    exon_intervals[gene_id].append((f.start0, f.end0))

        new_result: dict[str, list[Feature]] = {}
        for chrom, features in result.items():
            new_features = []
            for f in features:
                if f.feature_type == "gene" and f.feature_id in exon_intervals:
                    # 用 exon 区间替换 gene 全长区间（同一 gene_id，多条记录自动合并）
                    intervals = sorted(exon_intervals[f.feature_id], key=lambda x: (x[0], x[1]))
                    for ex_start, ex_end in intervals:
                        new_features.append(Feature(
                            chrom=f.chrom, start0=ex_start, end0=ex_end,
                            feature_type="gene",
                            feature_id=f.feature_id,
                            parent_id=f.parent_id,
                            strand=f.strand,
                        ))
                else:
                    new_features.append(f)
            new_result[chrom] = new_features
        result = new_result

    return result


# =============================================================================
# 4. Feature 聚合
# =============================================================================

def aggregate_predictions_to_features(
    df: pd.DataFrame,
    features_by_chrom: dict[str, list[Feature]],
    min_overlap_bp: int = 1,
) -> pd.DataFrame:
    """将逐窗口预测值与GFF特征做overlap聚合，输出逐基因/外显子指标。

    注意：需要在外部 load_features_from_gff 中传递 eval_start/eval_end。
    为简化，这里直接用 feature.start0/end0 作为评估区间。
    """
    try:
        from .metrics_core import compute_feature_basic_metrics
    except ImportError:
        from metrics_core import compute_feature_basic_metrics

    # 构建染色体→feature映射（带 eval 区间）
    feature_data: dict[str, list[tuple[Feature, int, int]]] = defaultdict(list)
    for chrom, features in features_by_chrom.items():
        for f in features:
            feature_data[chrom].append((f, f.start0, f.end0))

    # 对每条预测窗口聚合到 feature
    feature_values: dict[tuple, dict] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating to features", leave=False):
        chrom = str(row["chromosome"])
        window_start = int(row["start"])
        window_end = int(row["end"])
        pred = np.asarray(row["parsed_pred"], dtype=float)
        true = np.asarray(row["parsed_true"], dtype=float)

        feats = feature_data.get(chrom, [])
        if not feats:
            continue

        for feat, eval_start, eval_end in feats:
            if eval_end <= window_start:
                continue
            if eval_start >= window_end:
                break

            overlap_start = max(window_start, eval_start)
            overlap_end = min(window_end, eval_end)
            if overlap_end <= overlap_start:
                continue

            src_start = overlap_start - window_start
            src_end = overlap_end - window_start

            key = (feat.feature_type, feat.feature_id, feat.chrom)
            entry = feature_values.setdefault(key, {
                "feature": feat,
                "pred_sum": defaultdict(float),
                "true_sum": defaultdict(float),
                "counts": defaultdict(int),
            })
            for offset, pos in enumerate(range(overlap_start, overlap_end)):
                entry["pred_sum"][pos] += float(pred[src_start + offset])
                entry["true_sum"][pos] += float(true[src_start + offset])
                entry["counts"][pos] += 1

    rows = []
    for entry in feature_values.values():
        feat = entry["feature"]
        covered = sorted(entry["counts"].keys())
        overlap_bp = len(covered)
        if overlap_bp < min_overlap_bp:
            continue

        pred_vals = np.array([entry["pred_sum"][p] / entry["counts"][p] for p in covered], dtype=np.float32)
        true_vals = np.array([entry["true_sum"][p] / entry["counts"][p] for p in covered], dtype=np.float32)

        metrics = compute_feature_basic_metrics(pred_vals, true_vals)
        nonzero_mask = (pred_vals > 0) & (true_vals > 0)

        feature_length = feat.end0 - feat.start0
        eval_length = covered[-1] - covered[0] + 1 if covered else 0

        rows.append({
            "feature_type": feat.feature_type,
            "feature_id": feat.feature_id,
            "parent_id": feat.parent_id,
            "chromosome": feat.chrom,
            "start": feat.start0,
            "end": feat.end0,
            "strand": feat.strand,
            "feature_length": feature_length,
            "eval_length": eval_length,
            "overlap_bp": overlap_bp,
            "coverage_fraction": round(overlap_bp / eval_length, 4) if eval_length else np.nan,
            "pred_sum": float(np.sum(pred_vals)),
            "true_sum": float(np.sum(true_vals)),
            "pred_mean": float(np.mean(pred_vals)),
            "true_mean": float(np.mean(true_vals)),
            "pred_max": float(np.max(pred_vals)),
            "true_max": float(np.max(true_vals)),
            "pred_zero_ratio": round(float(np.mean(pred_vals == 0)), 4),
            "true_zero_ratio": round(float(np.mean(true_vals == 0)), 4),
            "nonzero_bp": int(np.sum(nonzero_mask)),
            **metrics,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 5. 配置文件
# =============================================================================

def load_config(config_path: Optional[Path] = None) -> EvalConfig:
    """加载评估配置（优先级：命令行 > config YAML > 默认值）。"""
    import yaml

    config = EvalConfig(
        predict_dir=Path("./outputs/predict/latest"),
        ref_dir=Path("./ref"),
        output_dir=Path("./outputs/predict/latest/evaluation"),
    )

    if config_path and config_path.is_file():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        if "predict_dir" in data:
            config.predict_dir = Path(data["predict_dir"])
        if "ref_dir" in data:
            config.ref_dir = Path(data["ref_dir"])
        if "output_dir" in data and data["output_dir"]:
            config.output_dir = Path(data["output_dir"])
        else:
            config.output_dir = config.predict_dir / "evaluation"

        # 评估任务列表
        config.tasks = []
        tl = data.get("tasks", [])
        for tdata in tl:
            config.tasks.append(EvalTask(
                predict_csv=Path(tdata["predict_csv"]),
                species=tdata["species"],
                tissue=tdata["tissue"],
                gff=Path(tdata["gff"]) if tdata.get("gff") else None,
                split=tdata["split"],
                chromosome=tdata["chromosome"],
                biosample=tdata["biosample"],
                modality=tdata["modality"],
            ))

        fl = data.get("feature_level", {})
        if fl:
            config.feature_types = tuple(fl.get("feature_types", ["gene", "exon"]))
            config.min_overlap_bp = fl.get("min_overlap_bp", 1)
            config.min_nonzero_bp = fl.get("min_nonzero_bp", 2)

        dl = data.get("delta", {})
        if dl:
            config.delta_normalize = dl.get("normalize", "global_mean")
            config.delta_ref_modes = tuple(dl.get("ref_modes", ["zero", "feature_normalized_mean"]))

        bk = data.get("buckets", {})
        if bk:
            config.n_expression_buckets = bk.get("n_buckets", 3)
        if config.bucket_thresholds is None:
            lt = bk.get("low_threshold")
            ht = bk.get("high_threshold")
            if lt is not None and ht is not None:
                config.bucket_thresholds = (float(lt), float(ht))

    else:
        config.output_dir = config.predict_dir / "evaluation"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config
