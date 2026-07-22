#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def nb_cell(cell_type: str, source: str) -> dict:
    result = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }
    if cell_type == "code":
        result["execution_count"] = None
        result["outputs"] = []
    return result


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")


COMMON_IMPORTS = r'''
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import pyBigWig
except ImportError as exc:
    raise ImportError("Install pyBigWig before running this notebook.") from exc
'''


COMMON_HELPERS = r'''
PROJECT_DIR = Path("/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding")
HUMAN_DIR = Path("/mnt/rice/default/Workspace/xuxiaolong/human")
EMBEDDING_ROOT = Path("/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_res_260708")
GENE_BED = PROJECT_DIR / "test.10gene.bed"
GENCODE_GTF = HUMAN_DIR / "gencode.v49.annotation.sorted.gtf"
SAMPLE_NAME_FILE = PROJECT_DIR / "101samples.name.txt"
BIGWIG_DIR = Path("/mnt/genos100-new/Public/CIMA/norm_CIMA_bw_101")
REFERENCE_AVERAGE_BW = PROJECT_DIR / "reference_from_train81" / "train81.average.bw"

VAL_INDIVIDUALS = ["H005", "H010", "H055", "H102", "H103", "H137", "H198", "H202", "H276", "H319"]
TEST_INDIVIDUALS = ["H030", "H117", "H118", "H129", "H195", "H197", "H215", "H225", "H261", "H309"]

RANDOM_STATE = 20260712
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def read_sample_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["accession", "sample"])
    df["accession"] = df["accession"].astype(str)
    df["sample"] = df["sample"].astype(str)
    return df

sample_table = read_sample_table(SAMPLE_NAME_FILE)
sample_to_accession = dict(zip(sample_table["sample"], sample_table["accession"]))
all_individuals = sample_table["sample"].tolist()
val_set = set(VAL_INDIVIDUALS)
test_set = set(TEST_INDIVIDUALS)
train_individuals = [x for x in all_individuals if x not in val_set and x not in test_set]

assert len(train_individuals) == len(all_individuals) - len(VAL_INDIVIDUALS) - len(TEST_INDIVIDUALS)
assert not (val_set & test_set)
assert REFERENCE_AVERAGE_BW.exists(), REFERENCE_AVERAGE_BW

def bigwig_path_for_sample(sample: str) -> Path:
    accession = sample_to_accession[sample]
    return BIGWIG_DIR / accession / "re-normalized_Monocyte.total.bw"

missing_bigwig = [sample for sample in all_individuals if not bigwig_path_for_sample(sample).exists()]
assert not missing_bigwig, f"Samples missing normalized bigWig: {missing_bigwig}"

@dataclass(frozen=True)
class GeneRecord:
    chrom: str
    start: int
    end: int
    name: str
    embedding_name: str
    strand: str
    tss: int
    transcript_id: str
    expression_regions: tuple[tuple[str, int, int], ...]

def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"

def strip_ensembl_version(value: str) -> str:
    return str(value).split(".", 1)[0]

def transcript_id_from_name(name: str) -> str:
    for token in re.split(r"[|_]", str(name)):
        if token.startswith("ENST"):
            return token
    raise ValueError(f"Could not extract ENST transcript id from {name!r}")

def embedding_name_from_bed_name(name: str) -> str:
    parts = str(name).split("|")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return str(name).replace("|", "_")

def parse_gtf_attributes(attr_text: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r'(\S+) "([^"]*)"', attr_text)}

def load_transcript_exons(gtf_path: Path, transcript_ids: Iterable[str]) -> dict[str, list[tuple[str, int, int, str]]]:
    requested = {str(tid) for tid in transcript_ids}
    requested_base = {strip_ensembl_version(tid) for tid in requested}
    exons: dict[str, list[tuple[str, int, int, str]]] = {tid: [] for tid in requested}
    with Path(gtf_path).open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "exon":
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = attrs.get("transcript_id")
            if not transcript_id:
                continue
            transcript_base = strip_ensembl_version(transcript_id)
            if transcript_id in requested:
                key = transcript_id
            elif transcript_base in requested_base:
                matches = [tid for tid in requested if strip_ensembl_version(tid) == transcript_base]
                key = matches[0]
            else:
                continue
            exons.setdefault(key, []).append((normalize_chrom(fields[0]), int(fields[3]) - 1, int(fields[4]), fields[6]))
    missing = [tid for tid in requested if not exons.get(tid)]
    if missing:
        raise KeyError(f"No exon records found for transcript(s): {missing[:10]}")
    return exons

def select_three_prime_exon(exons: list[tuple[str, int, int, str]]) -> tuple[tuple[str, int, int], ...]:
    strands = {exon[3] for exon in exons}
    if len(strands) != 1:
        raise ValueError(f"Expected one strand per transcript, got {sorted(strands)}")
    strand = next(iter(strands))
    if strand == "+":
        chrom, start, end, _ = max(exons, key=lambda item: (item[2], item[1]))
    elif strand == "-":
        chrom, start, end, _ = min(exons, key=lambda item: (item[1], item[2]))
    else:
        raise ValueError(f"Unsupported strand: {strand!r}")
    return ((chrom, start, end),)

def read_test10_genes() -> list[GeneRecord]:
    raw = []
    with GENE_BED.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, start, end, name = fields[:4]
            strand = fields[5] if len(fields) > 5 else "+"
            transcript_id = transcript_id_from_name(name)
            raw.append((normalize_chrom(chrom), int(start), int(end), name, embedding_name_from_bed_name(name), strand, transcript_id))
    exons = load_transcript_exons(GENCODE_GTF, [x[6] for x in raw])
    return [
        GeneRecord(chrom, start, end, name, embedding_name, exons[transcript_id][0][3], (start + end) // 2, transcript_id, select_three_prime_exon(exons[transcript_id]))
        for chrom, start, end, name, embedding_name, strand, transcript_id in raw
    ]

def read_all_embedding_genes() -> list[GeneRecord]:
    metas = []
    for meta_path in sorted(EMBEDDING_ROOT.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        transcript_id = transcript_id_from_name(meta["name"])
        metas.append((meta_path.parent.name, meta, transcript_id))
    exons = load_transcript_exons(GENCODE_GTF, [x[2] for x in metas])
    genes = []
    for embedding_name, meta, transcript_id in metas:
        start = int(meta["start"])
        end = int(meta["end"])
        chrom = normalize_chrom(meta["chrom"])
        expr = select_three_prime_exon(exons[transcript_id])
        gene_name = meta["name"].replace("_", "|", 2)
        genes.append(GeneRecord(chrom, start, end, gene_name, embedding_name, exons[transcript_id][0][3], (start + end) // 2, transcript_id, expr))
    return genes

def bw_mean_regions(bw, regions: Iterable[tuple[str, int, int]]) -> float:
    weighted_sum = 0.0
    total_bp = 0
    for chrom, start, end in regions:
        values = np.array(bw.values(chrom, max(0, int(start)), max(0, int(end)), numpy=True), dtype=np.float32)
        if values.size == 0:
            continue
        values = np.nan_to_num(values, nan=0.0)
        weighted_sum += float(values.sum())
        total_bp += int(values.size)
    if total_bp == 0:
        return float("nan")
    return weighted_sum / total_bp

def compute_targets(samples: list[str], gene: GeneRecord) -> np.ndarray:
    values = []
    with pyBigWig.open(str(REFERENCE_AVERAGE_BW)) as ref_bw:
        reference_value = bw_mean_regions(ref_bw, gene.expression_regions)
    for sample in samples:
        with pyBigWig.open(str(bigwig_path_for_sample(sample))) as sample_bw:
            sample_value = bw_mean_regions(sample_bw, gene.expression_regions)
        values.append(sample_value - reference_value)
    return np.asarray(values, dtype=np.float32)

def sample_pt_path(gene: GeneRecord, sample: str) -> Path:
    path = EMBEDDING_ROOT / gene.embedding_name / f"CIMA-{sample}_CIMA-{sample}.vcf.pt"
    if path.exists():
        return path
    matches = sorted((EMBEDDING_ROOT / gene.embedding_name).glob(f"*{sample}*.vcf.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one .vcf.pt for {gene.embedding_name}/{sample}, found {len(matches)}")
    return matches[0]

def hidden_states_for_sample(gene: GeneRecord, sample: str) -> np.ndarray:
    payload = torch.load(sample_pt_path(gene, sample), map_location="cpu", weights_only=False)
    arr = payload["hidden_states"].detach().cpu().float().numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D hidden_states, got {arr.shape} for {gene.embedding_name}/{sample}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def pooled_snp_features(states: np.ndarray) -> np.ndarray:
    return np.concatenate([
        states.mean(axis=0),
        states.std(axis=0),
        states.max(axis=0),
        np.abs(states).mean(axis=0),
    ]).astype(np.float32, copy=False)

def gene_feature_matrix(gene: GeneRecord, samples: list[str], feature_mode: str) -> tuple[np.ndarray, dict[str, int]]:
    rows = []
    shapes = []
    for sample in samples:
        states = hidden_states_for_sample(gene, sample)
        shapes.append(tuple(states.shape))
        if feature_mode == "pooled":
            rows.append(pooled_snp_features(states))
        elif feature_mode == "flatten":
            rows.append(states.reshape(-1))
        else:
            raise ValueError(f"Unknown feature_mode={feature_mode!r}")
    if feature_mode == "flatten" and len(set(shapes)) != 1:
        raise ValueError(f"Flatten feature requires equal shapes for {gene.embedding_name}, got {sorted(set(shapes))[:5]}")
    X = np.vstack(rows).astype(np.float32, copy=False)
    return X, {"n_snps": int(shapes[0][0]), "embedding_dim": int(shapes[0][1]), "n_features": int(X.shape[1])}

def safe_pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0 or np.var(y_true) == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot)

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "pearson": safe_pearson(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
        "target_std": float(np.std(y_true)),
        "prediction_std": float(np.std(y_pred)),
    }

def prefixed_metrics(prefix: str, y_true, y_pred) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in regression_metrics(y_true, y_pred).items()}
'''


def single_gene_notebook() -> list[dict]:
    return [
        nb_cell("markdown", """
# Single-Gene SNP Embedding Models

This notebook trains downstream neural models for each gene independently.
It intentionally does not compute the separate linear comparison; use `run_elasticnet_baselines.ipynb` for that.
"""),
        nb_cell("code", COMMON_IMPORTS),
        nb_cell("code", COMMON_HELPERS),
        nb_cell("code", r'''
OUTPUT_DIR = PROJECT_DIR / "single_gene_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_SET_MODE = "test10"  # "test10" or "all97"
FEATURE_MODE = "pooled"  # "pooled" is compact; "flatten" trains a true full downstream linear/MLP head per gene.
MODEL_KIND = "linear"  # "linear" or "mlp"
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.2
PATIENCE = 30

genes = read_test10_genes() if GENE_SET_MODE == "test10" else read_all_embedding_genes()
print(f"device={DEVICE} genes={len(genes)} train={len(train_individuals)} val={len(VAL_INDIVIDUALS)} test={len(TEST_INDIVIDUALS)}")
'''),
        nb_cell("code", r'''
class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

class SingleGeneRegressor(nn.Module):
    def __init__(self, input_dim: int, model_kind: str = MODEL_KIND, dropout: float = DROPOUT):
        super().__init__()
        if model_kind == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif model_kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            raise ValueError(f"Unknown model_kind={model_kind!r}")

    def forward(self, x):
        return self.net(x)

def standardize_from_train(X_train, X_val, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

def train_torch_regressor(model, X_train, y_train, X_val, y_val):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    yv = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    best_state = None
    best_val = math.inf
    bad_epochs = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv), yv).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def predict_model(model, X):
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy().reshape(-1)
    return pred
'''),
        nb_cell("code", r'''
def run_single_gene_model(gene: GeneRecord) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    samples = all_individuals
    X, info = gene_feature_matrix(gene, samples, FEATURE_MODE)
    y = compute_targets(samples, gene)
    sample_to_idx = {sample: i for i, sample in enumerate(samples)}
    train_idx = [sample_to_idx[s] for s in train_individuals]
    val_idx = [sample_to_idx[s] for s in VAL_INDIVIDUALS]
    test_idx = [sample_to_idx[s] for s in TEST_INDIVIDUALS]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_train, X_val, X_test = standardize_from_train(X_train, X_val, X_test)

    model = SingleGeneRegressor(input_dim=X_train.shape[1])
    model, history = train_torch_regressor(model, X_train, y_train, X_val, y_val)
    pred_train = predict_model(model, X_train)
    pred_val = predict_model(model, X_val)
    pred_test = predict_model(model, X_test)

    row = {
        "gene": gene.name,
        "embedding_name": gene.embedding_name,
        "model_kind": MODEL_KIND,
        "feature_mode": FEATURE_MODE,
        **info,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        **prefixed_metrics("train", y_train, pred_train),
        **prefixed_metrics("val", y_val, pred_val),
        **prefixed_metrics("test", y_test, pred_test),
    }
    prediction_rows = []
    for split, idxs, preds in [("train", train_idx, pred_train), ("val", val_idx, pred_val), ("test", test_idx, pred_test)]:
        for i, pred in zip(idxs, preds):
            prediction_rows.append({
                "split": split,
                "sample": samples[i],
                "gene": gene.name,
                "embedding_name": gene.embedding_name,
                "prediction_diff": float(pred),
                "target_diff": float(y[i]),
            })
    history["gene"] = gene.name
    history["embedding_name"] = gene.embedding_name
    return row, pd.DataFrame(prediction_rows), history
'''),
        nb_cell("code", r'''
metric_rows = []
prediction_frames = []
history_frames = []

for gene in genes:
    print(f"Training {MODEL_KIND} single-gene model: {gene.embedding_name}")
    row, pred_df, history_df = run_single_gene_model(gene)
    metric_rows.append(row)
    prediction_frames.append(pred_df)
    history_frames.append(history_df)
    print({"gene": gene.embedding_name, "test_pearson": row["test_pearson"], "test_r2": row["test_r2"], "test_rmse": row["test_rmse"]})

metrics = pd.DataFrame(metric_rows)
predictions = pd.concat(prediction_frames, ignore_index=True)
history = pd.concat(history_frames, ignore_index=True)

metrics_path = OUTPUT_DIR / "model_single_gene_metrics.csv"
predictions_path = OUTPUT_DIR / "model_single_gene_predictions.csv"
history_path = OUTPUT_DIR / "model_single_gene_history.csv"
metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False)
history.to_csv(history_path, index=False)
print(metrics_path)
print(predictions_path)
print(history_path)
metrics.sort_values("test_pearson", ascending=False)
'''),
    ]


def multi_gene_notebook() -> list[dict]:
    return [
        nb_cell("markdown", """
# Multi-Gene SNP Embedding Models

This notebook trains shared downstream models across 10, 50, and 97 gene sets.
It intentionally does not compute the separate linear comparison; use `run_elasticnet_baselines.ipynb` for that.
"""),
        nb_cell("code", COMMON_IMPORTS),
        nb_cell("code", COMMON_HELPERS),
        nb_cell("code", r'''
OUTPUT_DIR = PROJECT_DIR / "multi_gene_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_SET_SIZES = [10, 50, 97]
FEATURE_MODE = "pooled"
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.2
GENE_EMBEDDING_DIM = 32
PATIENCE = 30

all_embedding_genes = read_all_embedding_genes()
test10_names = {gene.embedding_name for gene in read_test10_genes()}
test10_genes = [gene for gene in all_embedding_genes if gene.embedding_name in test10_names]
remaining_genes = [gene for gene in all_embedding_genes if gene.embedding_name not in test10_names]
genes_by_size = {
    10: test10_genes,
    50: test10_genes + remaining_genes[: max(0, 50 - len(test10_genes))],
    97: all_embedding_genes,
}
print({size: len(genes) for size, genes in genes_by_size.items()})
print(f"device={DEVICE} train={len(train_individuals)} val={len(VAL_INDIVIDUALS)} test={len(TEST_INDIVIDUALS)}")
'''),
        nb_cell("code", r'''
class MultiGeneDataset(Dataset):
    def __init__(self, X: np.ndarray, gene_ids: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.gene_ids = torch.as_tensor(gene_ids, dtype=torch.long)
        self.y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, index: int):
        return self.X[index], self.gene_ids[index], self.y[index]

class MultiGeneRegressor(nn.Module):
    def __init__(self, input_dim: int, n_genes: int, gene_embedding_dim: int = GENE_EMBEDDING_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, gene_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + gene_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, gene_ids):
        gene_vec = self.gene_embedding(gene_ids)
        return self.net(torch.cat([x, gene_vec], dim=1))

def build_multi_gene_table(genes: list[GeneRecord]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    X_rows = []
    y_rows = []
    gene_id_rows = []
    gene_to_id = {gene.embedding_name: idx for idx, gene in enumerate(genes)}
    for gene in genes:
        X_gene, info = gene_feature_matrix(gene, all_individuals, FEATURE_MODE)
        y_gene = compute_targets(all_individuals, gene)
        for sample_idx, sample in enumerate(all_individuals):
            split = "train"
            if sample in VAL_INDIVIDUALS:
                split = "val"
            elif sample in TEST_INDIVIDUALS:
                split = "test"
            rows.append({
                "sample": sample,
                "split": split,
                "gene": gene.name,
                "embedding_name": gene.embedding_name,
                "gene_id": gene_to_id[gene.embedding_name],
                **info,
            })
            X_rows.append(X_gene[sample_idx])
            y_rows.append(y_gene[sample_idx])
            gene_id_rows.append(gene_to_id[gene.embedding_name])
    return pd.DataFrame(rows), np.vstack(X_rows).astype(np.float32), np.asarray(gene_id_rows, dtype=np.int64), np.asarray(y_rows, dtype=np.float32)

def standardize_multi(X, train_mask):
    mean = X[train_mask].mean(axis=0, keepdims=True)
    std = X[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std
'''),
        nb_cell("code", r'''
def train_multi_model(model, X, gene_ids, y, train_mask, val_mask):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    loader = DataLoader(MultiGeneDataset(X[train_mask], gene_ids[train_mask], y[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
    Xv = torch.as_tensor(X[val_mask], dtype=torch.float32, device=DEVICE)
    gv = torch.as_tensor(gene_ids[val_mask], dtype=torch.long, device=DEVICE)
    yv = torch.as_tensor(y[val_mask].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    best_state = None
    best_val = math.inf
    bad_epochs = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, gb, yb in loader:
            xb = xb.to(DEVICE)
            gb = gb.to(DEVICE)
            yb = yb.to(DEVICE)
            loss = criterion(model(xb, gb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv, gv), yv).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def predict_multi_model(model, X, gene_ids):
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.as_tensor(X, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(gene_ids, dtype=torch.long, device=DEVICE),
        ).detach().cpu().numpy().reshape(-1)
    return pred

def summarize_group_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, df in predictions.groupby(group_col, sort=True):
        metric = regression_metrics(df["target_diff"].to_numpy(), df["prediction_diff"].to_numpy())
        rows.append({group_col: group_value, "n": len(df), **metric})
    return pd.DataFrame(rows)
'''),
        nb_cell("code", r'''
def run_multi_gene_model(gene_set_size: int, genes: list[GeneRecord]):
    table, X, gene_ids, y = build_multi_gene_table(genes)
    train_mask = table["split"].eq("train").to_numpy()
    val_mask = table["split"].eq("val").to_numpy()
    test_mask = table["split"].eq("test").to_numpy()
    X = standardize_multi(X, train_mask)
    model = MultiGeneRegressor(input_dim=X.shape[1], n_genes=len(genes))
    model, history = train_multi_model(model, X, gene_ids, y, train_mask, val_mask)
    pred = predict_multi_model(model, X, gene_ids)
    predictions = table.copy()
    predictions["target_diff"] = y.astype(float)
    predictions["prediction_diff"] = pred.astype(float)
    overall_rows = []
    for split in ["train", "val", "test"]:
        df = predictions[predictions["split"].eq(split)]
        overall_rows.append({"gene_set_size": gene_set_size, "split": split, "n": len(df), **regression_metrics(df["target_diff"], df["prediction_diff"])})
    metrics_overall = pd.DataFrame(overall_rows)
    metrics_by_gene = summarize_group_metrics(predictions[predictions["split"].eq("test")], "embedding_name")
    metrics_by_gene.insert(0, "gene_set_size", gene_set_size)
    metrics_by_individual = summarize_group_metrics(predictions[predictions["split"].eq("test")], "sample")
    metrics_by_individual.insert(0, "gene_set_size", gene_set_size)
    history["gene_set_size"] = gene_set_size
    predictions.insert(0, "gene_set_size", gene_set_size)
    return metrics_overall, metrics_by_gene, metrics_by_individual, predictions, history
'''),
        nb_cell("code", r'''
overall_frames = []
by_gene_frames = []
by_individual_frames = []
prediction_frames = []
history_frames = []

for gene_set_size in GENE_SET_SIZES:
    genes = genes_by_size[gene_set_size]
    print(f"Training multi-gene model: {gene_set_size} genes")
    overall, metrics_by_gene, metrics_by_individual, predictions, history = run_multi_gene_model(gene_set_size, genes)
    overall_frames.append(overall)
    by_gene_frames.append(metrics_by_gene)
    by_individual_frames.append(metrics_by_individual)
    prediction_frames.append(predictions)
    history_frames.append(history)
    print(overall[overall["split"].eq("test")].to_dict("records")[0])

metrics_overall = pd.concat(overall_frames, ignore_index=True)
metrics_by_gene = pd.concat(by_gene_frames, ignore_index=True)
metrics_by_individual = pd.concat(by_individual_frames, ignore_index=True)
predictions = pd.concat(prediction_frames, ignore_index=True)
history = pd.concat(history_frames, ignore_index=True)

metrics_overall.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_overall.csv", index=False)
metrics_by_gene.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_by_gene.csv", index=False)
metrics_by_individual.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_by_individual.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "model_multi_gene_predictions.csv", index=False)
history.to_csv(OUTPUT_DIR / "model_multi_gene_history.csv", index=False)
metrics_overall
'''),
    ]


def position_aware_multi_gene_notebook() -> list[dict]:
    helper_start = COMMON_HELPERS.index("def hidden_states_for_sample")
    helper_end = COMMON_HELPERS.index("def safe_pearson")
    multi_helpers = COMMON_HELPERS[:helper_start] + COMMON_HELPERS[helper_end:]
    return [
        nb_cell("markdown", """
# Position-Aware Multi-Gene SNP Embedding Model

This notebook trains a frozen-input, position-binned downstream model across
10, 50, and 97 gene sets. Primary evaluation uses within-gene metrics across
individuals; pooled raw-scale metrics are secondary diagnostics.
"""),
        nb_cell("code", COMMON_IMPORTS + r'''

from position_binned_model import (
    PositionBinnedRegressor,
    bin_snp_hidden_states,
    build_checkpoint,
    fit_gene_target_scalers,
    inverse_scale_gene_targets,
    macro_gene_metrics,
    per_gene_metrics,
    scale_gene_targets,
)
'''),
        nb_cell("code", multi_helpers),
        nb_cell("code", r'''
OUTPUT_DIR = PROJECT_DIR / "multi_gene_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_SET_SIZES = [10, 50, 97]
N_POSITION_BINS = 32
BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
PROJECTION_DIM = 128
GENE_EMBEDDING_DIM = 32
PATIENCE = 30
TARGET_STD_EPS = 1e-6
MODEL_VARIANT = "position_binned"
CHECKPOINT_FILENAMES = {
    10: "model_multi_gene_position_binned_10.pt",
    50: "model_multi_gene_position_binned_50.pt",
    97: "model_multi_gene_position_binned_97.pt",
}

all_embedding_genes = read_all_embedding_genes()
test10_names = {gene.embedding_name for gene in read_test10_genes()}
test10_genes = [gene for gene in all_embedding_genes if gene.embedding_name in test10_names]
remaining_genes = [gene for gene in all_embedding_genes if gene.embedding_name not in test10_names]
genes_by_size = {
    10: test10_genes,
    50: test10_genes + remaining_genes[: max(0, 50 - len(test10_genes))],
    97: all_embedding_genes,
}
assert {size: len(genes) for size, genes in genes_by_size.items()} == {10: 10, 50: 50, 97: 97}
print({size: len(genes) for size, genes in genes_by_size.items()})
print(f"device={DEVICE} train={len(train_individuals)} val={len(VAL_INDIVIDUALS)} test={len(TEST_INDIVIDUALS)}")
'''),
        nb_cell("code", r'''
def load_binned_snp_features(gene: GeneRecord, sample: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    payload = torch.load(sample_pt_path(gene, sample), map_location="cpu", weights_only=False)
    states = payload["hidden_states"].detach().cpu().float().numpy()
    positions = payload["positions"].detach().cpu().long().numpy()
    if states.ndim != 2 or positions.ndim != 1 or len(states) != len(positions):
        raise ValueError(
            f"Expected aligned hidden_states/positions for {gene.embedding_name}/{sample}, "
            f"got {states.shape} and {positions.shape}"
        )
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    features, bin_mask, counts = bin_snp_hidden_states(
        states,
        positions,
        gene.start,
        gene.end,
        N_POSITION_BINS,
    )
    # Align bins to transcriptional orientation: upstream -> downstream.
    if gene.strand == "-":
        features = features[::-1].copy()
        bin_mask = bin_mask[::-1].copy()
        counts = counts[::-1].copy()
    return features.astype(np.float16), bin_mask, {
        "n_snps": int(len(positions)),
        "embedding_dim": int(states.shape[1]),
        "nonempty_bins": int(bin_mask.sum()),
        "max_bin_snps": int(counts.max()),
    }


class PositionBinnedDataset(Dataset):
    def __init__(self, X, bin_mask, gene_ids, y_scaled):
        # X contains precomputed embeddings and is immutable model input.
        self.X = torch.as_tensor(X, dtype=torch.float16)
        self.bin_mask = torch.as_tensor(bin_mask, dtype=torch.bool)
        self.gene_ids = torch.as_tensor(gene_ids, dtype=torch.long)
        self.y = torch.as_tensor(np.asarray(y_scaled).reshape(-1, 1), dtype=torch.float32)
        self.X.requires_grad_(False)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, index):
        return self.X[index], self.bin_mask[index], self.gene_ids[index], self.y[index]


def build_multi_gene_data(genes: list[GeneRecord]):
    rows, X_rows, mask_rows, y_rows, gene_id_rows = [], [], [], [], []
    gene_to_id = {gene.embedding_name: idx for idx, gene in enumerate(genes)}
    for gene in genes:
        y_gene = compute_targets(all_individuals, gene)
        for sample_idx, sample in enumerate(all_individuals):
            features, bin_mask, info = load_binned_snp_features(gene, sample)
            split = "val" if sample in VAL_INDIVIDUALS else "test" if sample in TEST_INDIVIDUALS else "train"
            rows.append({
                "sample": sample,
                "split": split,
                "gene": gene.name,
                "embedding_name": gene.embedding_name,
                "gene_id": gene_to_id[gene.embedding_name],
                **info,
            })
            X_rows.append(features)
            mask_rows.append(bin_mask)
            y_rows.append(y_gene[sample_idx])
            gene_id_rows.append(gene_to_id[gene.embedding_name])
    return (
        pd.DataFrame(rows),
        np.stack(X_rows),
        np.stack(mask_rows),
        np.asarray(gene_id_rows, dtype=np.int64),
        np.asarray(y_rows, dtype=np.float32),
        gene_to_id,
    )
'''),
        nb_cell("code", r'''
def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_position_binned_model(model, X, bin_mask, gene_ids, y_scaled, train_mask, val_mask):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=1.0)
    loader = DataLoader(
        PositionBinnedDataset(X[train_mask], bin_mask[train_mask], gene_ids[train_mask], y_scaled[train_mask]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    Xv = torch.as_tensor(X[val_mask], dtype=torch.float32, device=DEVICE)
    mv = torch.as_tensor(bin_mask[val_mask], dtype=torch.bool, device=DEVICE)
    gv = torch.as_tensor(gene_ids[val_mask], dtype=torch.long, device=DEVICE)
    yv = torch.as_tensor(y_scaled[val_mask].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    best_val = {"loss": math.inf, "epoch": 0, "state": None}
    best_train = {"loss": math.inf, "epoch": 0, "state": None}
    bad_epochs = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, mb, gb, yb in loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            mb = mb.to(DEVICE)
            gb = gb.to(DEVICE)
            yb = yb.to(DEVICE)
            loss = criterion(model(xb, mb, gb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv, mv, gv), yv).detach().cpu())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if train_loss < best_train["loss"]:
            best_train = {"loss": train_loss, "epoch": epoch, "state": cpu_state_dict(model)}
        if val_loss < best_val["loss"]:
            best_val = {"loss": val_loss, "epoch": epoch, "state": cpu_state_dict(model)}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break
    if best_val["state"] is None or best_train["state"] is None:
        raise RuntimeError("Training did not produce a restorable state")
    model.load_state_dict(best_val["state"])
    return model, pd.DataFrame(history), best_val, best_train


def predict_scaled(model, X, bin_mask, gene_ids):
    model.eval()
    output = []
    for start in range(0, len(X), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(X))
        with torch.no_grad():
            pred = model(
                torch.as_tensor(X[start:stop], dtype=torch.float32, device=DEVICE),
                torch.as_tensor(bin_mask[start:stop], dtype=torch.bool, device=DEVICE),
                torch.as_tensor(gene_ids[start:stop], dtype=torch.long, device=DEVICE),
            )
        output.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(output).astype(np.float32)
'''),
        nb_cell("code", r'''
def run_multi_gene_model(gene_set_size: int, genes: list[GeneRecord]):
    table, X, bin_mask, gene_ids, y, gene_to_id = build_multi_gene_data(genes)
    train_mask = table["split"].eq("train").to_numpy()
    val_mask = table["split"].eq("val").to_numpy()
    target_scalers = fit_gene_target_scalers(y, gene_ids, train_mask, len(genes), TARGET_STD_EPS)
    y_scaled = scale_gene_targets(y, gene_ids, target_scalers)
    model_config = {
        "bin_feature_dim": int(X.shape[2]),
        "n_genes": len(genes),
        "projection_dim": PROJECTION_DIM,
        "gene_embedding_dim": GENE_EMBEDDING_DIM,
        "dropout": DROPOUT,
    }
    model = PositionBinnedRegressor(**model_config)
    model, history, best_val, best_train = train_position_binned_model(
        model, X, bin_mask, gene_ids, y_scaled, train_mask, val_mask
    )
    pred_scaled = predict_scaled(model, X, bin_mask, gene_ids)
    pred = inverse_scale_gene_targets(pred_scaled, gene_ids, target_scalers)
    predictions = table.copy()
    predictions["model_variant"] = MODEL_VARIANT
    predictions["target_scaled"] = y_scaled.astype(float)
    predictions["prediction_scaled"] = pred_scaled.astype(float)
    predictions["target_diff"] = y.astype(float)
    predictions["prediction_diff"] = pred.astype(float)

    overall_rows, by_gene_frames, macro_rows = [], [], []
    for split in ["train", "val", "test"]:
        split_frame = predictions[predictions["split"].eq(split)]
        overall_rows.append({
            "gene_set_size": gene_set_size,
            "model_variant": MODEL_VARIANT,
            "model_state": "best_validation",
            "split": split,
            "n": len(split_frame),
            **regression_metrics(split_frame["target_diff"], split_frame["prediction_diff"]),
        })
        by_gene = per_gene_metrics(split_frame, target_scalers["variable"])
        by_gene.insert(0, "split", split)
        by_gene.insert(0, "model_variant", MODEL_VARIANT)
        by_gene.insert(0, "gene_set_size", gene_set_size)
        by_gene_frames.append(by_gene)
        macro_rows.append({
            "gene_set_size": gene_set_size,
            "model_variant": MODEL_VARIANT,
            "model_state": "best_validation",
            "split": split,
            **macro_gene_metrics(by_gene),
        })

    # Report best-training fit separately without using it for validation/test.
    best_val_state = cpu_state_dict(model)
    model.load_state_dict(best_train["state"])
    train_pred_scaled = predict_scaled(model, X[train_mask], bin_mask[train_mask], gene_ids[train_mask])
    train_pred = inverse_scale_gene_targets(train_pred_scaled, gene_ids[train_mask], target_scalers)
    train_true = y[train_mask]
    overall_rows.append({
        "gene_set_size": gene_set_size,
        "model_variant": MODEL_VARIANT,
        "model_state": "best_training",
        "split": "train",
        "n": int(train_mask.sum()),
        **regression_metrics(train_true, train_pred),
    })
    model.load_state_dict(best_val_state)

    checkpoint = build_checkpoint(
        model=model,
        model_config=model_config,
        ordered_genes=[{
            "name": gene.name,
            "embedding_name": gene.embedding_name,
            "chrom": gene.chrom,
            "start": gene.start,
            "end": gene.end,
            "strand": gene.strand,
            "transcript_id": gene.transcript_id,
        } for gene in genes],
        gene_to_id=gene_to_id,
        target_scalers=target_scalers,
        n_bins=N_POSITION_BINS,
        splits={"train": train_individuals, "val": VAL_INDIVIDUALS, "test": TEST_INDIVIDUALS},
        training_config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "loss": "HuberLoss",
            "random_state": RANDOM_STATE,
            "embedding_frozen": True,
        },
        best_epochs={
            "validation": {"epoch": best_val["epoch"], "loss": best_val["loss"]},
            "training": {"epoch": best_train["epoch"], "loss": best_train["loss"]},
        },
    )
    checkpoint["best_training_state_dict"] = best_train["state"]
    torch.save(checkpoint, OUTPUT_DIR / CHECKPOINT_FILENAMES[gene_set_size])
    history["gene_set_size"] = gene_set_size
    history["model_variant"] = MODEL_VARIANT
    predictions.insert(0, "gene_set_size", gene_set_size)
    return (
        pd.DataFrame(overall_rows),
        pd.concat(by_gene_frames, ignore_index=True),
        pd.DataFrame(macro_rows),
        predictions,
        history,
    )
'''),
        nb_cell("code", r'''
overall_frames, by_gene_frames, macro_frames = [], [], []
prediction_frames, history_frames = [], []

for gene_set_size in GENE_SET_SIZES:
    print(f"Training {MODEL_VARIANT}: {gene_set_size} genes")
    overall, by_gene, macro, predictions, history = run_multi_gene_model(
        gene_set_size, genes_by_size[gene_set_size]
    )
    overall_frames.append(overall)
    by_gene_frames.append(by_gene)
    macro_frames.append(macro)
    prediction_frames.append(predictions)
    history_frames.append(history)
    print(macro[macro["split"].eq("test")].to_dict("records")[0])

metrics_overall = pd.concat(overall_frames, ignore_index=True)
metrics_by_gene = pd.concat(by_gene_frames, ignore_index=True)
metrics_macro = pd.concat(macro_frames, ignore_index=True)
predictions = pd.concat(prediction_frames, ignore_index=True)
history = pd.concat(history_frames, ignore_index=True)
collapse = metrics_by_gene[[
    "gene_set_size", "model_variant", "split", "gene_id", "embedding_name", "n",
    "target_std", "prediction_std", "prediction_target_std_ratio", "eligible_correlation",
]].copy()

metrics_overall.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_overall.csv", index=False)
metrics_by_gene.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_by_gene.csv", index=False)
metrics_macro.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_macro.csv", index=False)
collapse.to_csv(OUTPUT_DIR / "model_multi_gene_prediction_collapse_by_gene.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "model_multi_gene_predictions.csv", index=False)
history.to_csv(OUTPUT_DIR / "model_multi_gene_history.csv", index=False)
metrics_macro
'''),
    ]


def specific_snp_delta_multi_gene_notebook() -> list[dict]:
    helper_start = COMMON_HELPERS.index("def hidden_states_for_sample")
    helper_end = COMMON_HELPERS.index("def safe_pearson")
    multi_helpers = COMMON_HELPERS[:helper_start] + COMMON_HELPERS[helper_end:]
    return [
        nb_cell("markdown", """
# 具体 SNP Delta 多基因表达量模型

本 notebook 只使用现有二维 hap1 SNP hidden states。每个具体 SNP 先减去
81个训练个体的均值，再通过带 padding mask 的 attention 聚合。默认先训练
10个基因验证训练集能否被拟合；确认后将 `GENE_SET_SIZES` 改为 `[50]`。
"""),
        nb_cell("code", COMMON_IMPORTS + r'''

from specific_snp_model import (
    SameGenePairBatchSampler,
    SpecificSNPRegressor,
    build_specific_snp_checkpoint,
    center_and_pad_snp_hidden,
    fit_gene_target_scalers,
    inverse_scale_gene_targets,
    macro_gene_metrics,
    normalize_snp_positions,
    pairwise_difference_loss,
    per_gene_metrics,
    scale_gene_targets,
)
'''),
        nb_cell("code", multi_helpers),
        nb_cell("code", r'''
OUTPUT_DIR = PROJECT_DIR / "multi_gene_models_specific_snp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_SET_SIZES = [10]  # 训练集拟合成功后改为 [50]
BATCH_SIZE = 2
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
PROJECTION_DIM = 64
GENE_EMBEDDING_DIM = 32
PATIENCE = 40
TARGET_STD_EPS = 1e-6
PAIRWISE_LOSS_WEIGHT = 1.0
GRAD_ACCUM_STEPS = 8
MODEL_VARIANT = "specific_snp_delta_hap1"
ALL_GENE_BED = PROJECT_DIR / "top100_robust_cv_16k_windows.tss_1mb.bed"
CHECKPOINT_FILENAMES = {
    10: "model_multi_gene_specific_snp_delta_10.pt",
    50: "model_multi_gene_specific_snp_delta_50.pt",
    97: "model_multi_gene_specific_snp_delta_97.pt",
}

def order_genes_by_bed(genes, bed_path):
    by_name = {gene.embedding_name: gene for gene in genes}
    ordered_names = []
    with Path(bed_path).open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                ordered_names.append(embedding_name_from_bed_name(line.rstrip("\n").split("\t")[3]))
    return [by_name[name] for name in ordered_names if name in by_name]

all_embedding_genes = order_genes_by_bed(read_all_embedding_genes(), ALL_GENE_BED)
test10_order = [gene.embedding_name for gene in read_test10_genes()]
all_by_name = {gene.embedding_name: gene for gene in all_embedding_genes}
test10_genes = [all_by_name[name] for name in test10_order if name in all_by_name]
test10_names = set(test10_order)
remaining_genes = [gene for gene in all_embedding_genes if gene.embedding_name not in test10_names]
genes_by_size = {
    10: test10_genes,
    50: test10_genes + remaining_genes[: max(0, 50 - len(test10_genes))],
    97: all_embedding_genes,
}
assert len(test10_genes) == 10
print({size: len(genes_by_size[size]) for size in GENE_SET_SIZES})
print(f"device={DEVICE} train={len(train_individuals)} val={len(VAL_INDIVIDUALS)} test={len(TEST_INDIVIDUALS)}")
'''),
        nb_cell("code", r'''
def load_hap1_payload(gene: GeneRecord, sample: str):
    payload = torch.load(sample_pt_path(gene, sample), map_location="cpu", weights_only=False)
    states = payload["hidden_states"].detach().cpu().float().numpy()
    positions = payload["positions"].detach().cpu().long().numpy().reshape(-1)
    variant_keys = tuple(str(value) for value in payload["variant_keys"])
    if states.ndim != 2 or len(states) != len(positions) or len(states) != len(variant_keys):
        raise ValueError(f"Misaligned SNP payload for {gene.embedding_name}/{sample}")
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite hap1 hidden states for {gene.embedding_name}/{sample}")
    return states.astype(np.float32, copy=False), positions, variant_keys


def fit_gene_snp_preprocessing(gene: GeneRecord):
    reference_positions = None
    reference_keys = None
    center_sum = None
    n_train = 0
    for sample in all_individuals:
        states, positions, variant_keys = load_hap1_payload(gene, sample)
        if reference_positions is None:
            reference_positions = positions.copy()
            reference_keys = variant_keys
            center_sum = np.zeros(states.shape, dtype=np.float64)
        else:
            if states.shape != center_sum.shape:
                raise ValueError(f"hidden_states shape mismatch for {gene.embedding_name}/{sample}")
            if not np.array_equal(positions, reference_positions):
                raise ValueError(f"positions mismatch for {gene.embedding_name}/{sample}")
            if variant_keys != reference_keys:
                raise ValueError(f"variant_keys mismatch for {gene.embedding_name}/{sample}")
        if sample in train_individuals:
            center_sum += states
            n_train += 1
    if n_train != len(train_individuals):
        raise RuntimeError(f"Expected {len(train_individuals)} train samples, got {n_train}")
    return (center_sum / n_train).astype(np.float32), reference_positions, reference_keys


class SpecificSNPDataset(Dataset):
    def __init__(self, table, genes, centers, normalized_positions, y_scaled, max_snps):
        self.table = table.reset_index(drop=True)
        self.genes = genes
        self.centers = centers
        self.normalized_positions = normalized_positions
        self.y_scaled = np.asarray(y_scaled, dtype=np.float32)
        self.max_snps = int(max_snps)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        gene_id = int(row["gene_id"])
        states, _, _ = load_hap1_payload(self.genes[gene_id], str(row["sample"]))
        return states, gene_id, float(self.y_scaled[index]), int(index)

    def collate(self, batch):
        states = [item[0][None, :, :] for item in batch]
        gene_ids = np.asarray([item[1] for item in batch], dtype=np.int64)
        centers = [self.centers[gene_id] for gene_id in gene_ids]
        delta, snp_mask = center_and_pad_snp_hidden(states, centers, max_snps=self.max_snps)
        positions = self.normalized_positions[gene_ids]
        targets = np.asarray([item[2] for item in batch], dtype=np.float32).reshape(-1, 1)
        row_indices = np.asarray([item[3] for item in batch], dtype=np.int64)
        return (
            torch.as_tensor(delta[:, 0], dtype=torch.float16),
            torch.as_tensor(snp_mask, dtype=torch.bool),
            torch.as_tensor(positions, dtype=torch.float32),
            torch.as_tensor(gene_ids, dtype=torch.long),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(row_indices, dtype=torch.long),
        )


def build_multi_gene_data(genes: list[GeneRecord]):
    rows, y_rows, gene_id_rows = [], [], []
    centers, positions_by_gene, variant_keys_by_gene = [], [], []
    gene_to_id = {gene.embedding_name: idx for idx, gene in enumerate(genes)}
    for gene in genes:
        print(f"scan train-only SNP center: {gene.embedding_name}", flush=True)
        center, positions, variant_keys = fit_gene_snp_preprocessing(gene)
        centers.append(center)
        positions_by_gene.append(positions)
        variant_keys_by_gene.append(variant_keys)
        targets = compute_targets(all_individuals, gene)
        gene_id = gene_to_id[gene.embedding_name]
        for sample_idx, sample in enumerate(all_individuals):
            split = "val" if sample in VAL_INDIVIDUALS else "test" if sample in TEST_INDIVIDUALS else "train"
            rows.append({
                "sample": sample, "split": split, "gene": gene.name,
                "embedding_name": gene.embedding_name, "gene_id": gene_id,
                "n_snps": len(positions), "embedding_dim": center.shape[1],
            })
            y_rows.append(targets[sample_idx])
            gene_id_rows.append(gene_id)
    table = pd.DataFrame(rows)
    y = np.asarray(y_rows, dtype=np.float32)
    gene_ids = np.asarray(gene_id_rows, dtype=np.int64)
    max_snps = max(len(values) for values in positions_by_gene)
    normalized_positions = normalize_snp_positions(
        positions_by_gene,
        [gene.start for gene in genes],
        [gene.end for gene in genes],
        max_snps,
    )
    return table, y, gene_ids, gene_to_id, centers, positions_by_gene, variant_keys_by_gene, normalized_positions, max_snps
'''),
        nb_cell("code", r'''
def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_loader(dataset, indices, shuffle):
    return DataLoader(
        torch.utils.data.Subset(dataset, np.flatnonzero(indices).tolist()),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dataset.collate,
    )


def make_same_gene_pair_loader(dataset, gene_ids, train_mask):
    sampler = SameGenePairBatchSampler(
        gene_ids=gene_ids,
        selected_indices=np.flatnonzero(train_mask),
        seed=RANDOM_STATE,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=dataset.collate)


def predict_scaled(model, loader):
    model.eval()
    predictions, row_indices = [], []
    with torch.no_grad():
        for xb, mb, pb, gb, _, ib in loader:
            pred = model(
                xb.to(DEVICE, dtype=torch.float32), mb.to(DEVICE),
                pb.to(DEVICE), gb.to(DEVICE),
            )
            predictions.append(pred.detach().cpu().numpy().reshape(-1))
            row_indices.append(ib.numpy())
    return np.concatenate(predictions), np.concatenate(row_indices)


def macro_for_rows(table, row_indices, target, prediction, variable):
    frame = table.iloc[row_indices].copy()
    frame["target_diff"] = target
    frame["prediction_diff"] = prediction
    return macro_gene_metrics(per_gene_metrics(frame, variable))


def train_specific_snp_model(model, dataset, table, y, gene_ids, scalers, train_mask, val_mask):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=1.0)
    train_loader = make_same_gene_pair_loader(dataset, gene_ids, train_mask)
    train_eval_loader = make_loader(dataset, train_mask, False)
    val_loader = make_loader(dataset, val_mask, False)
    best_val = {"loss": math.inf, "epoch": 0, "state": None}
    best_train = {"loss": math.inf, "epoch": 0, "state": None}
    bad_epochs, history = 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses, absolute_losses, pairwise_losses = [], [], []
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        n_train_batches = len(train_loader)
        for batch_index, (xb, mb, pb, gb, yb, _) in enumerate(train_loader):
            gb = gb.to(DEVICE)
            yb = yb.to(DEVICE)
            prediction = model(
                xb.to(DEVICE, dtype=torch.float32), mb.to(DEVICE), pb.to(DEVICE), gb,
            )
            absolute_loss = criterion(prediction, yb)
            pairwise_loss = pairwise_difference_loss(prediction, yb, gb, delta=1.0)
            loss = absolute_loss + PAIRWISE_LOSS_WEIGHT * pairwise_loss
            group_start = (batch_index // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
            accumulation_group_size = min(GRAD_ACCUM_STEPS, n_train_batches - group_start)
            (loss / accumulation_group_size).backward()
            should_update = (batch_index + 1) % GRAD_ACCUM_STEPS == 0 or (batch_index + 1) == n_train_batches
            if should_update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            batch_losses.append(float(loss.detach().cpu()))
            absolute_losses.append(float(absolute_loss.detach().cpu()))
            pairwise_losses.append(float(pairwise_loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses))
        val_scaled, val_rows = predict_scaled(model, val_loader)
        val_prediction_tensor = torch.as_tensor(val_scaled, dtype=torch.float32)
        val_target_tensor = torch.as_tensor(dataset.y_scaled[val_rows], dtype=torch.float32)
        val_gene_tensor = torch.as_tensor(gene_ids[val_rows], dtype=torch.long)
        val_absolute_loss = float(criterion(val_prediction_tensor, val_target_tensor).item())
        val_pairwise_loss = float(pairwise_difference_loss(
            val_prediction_tensor, val_target_tensor, val_gene_tensor, delta=1.0
        ).item())
        val_loss = val_absolute_loss + PAIRWISE_LOSS_WEIGHT * val_pairwise_loss
        train_scaled, train_rows = predict_scaled(model, train_eval_loader)
        train_pred = inverse_scale_gene_targets(train_scaled, gene_ids[train_rows], scalers)
        val_pred = inverse_scale_gene_targets(val_scaled, gene_ids[val_rows], scalers)
        train_macro = macro_for_rows(table, train_rows, y[train_rows], train_pred, scalers["variable"])
        val_macro = macro_for_rows(table, val_rows, y[val_rows], val_pred, scalers["variable"])
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_absolute_loss": float(np.mean(absolute_losses)),
            "train_pairwise_loss": float(np.mean(pairwise_losses)),
            "val_absolute_loss": val_absolute_loss,
            "val_pairwise_loss": val_pairwise_loss,
            "optimizer_steps": optimizer_steps,
            "train_macro_pearson": train_macro["macro_pearson"], "train_macro_r2": train_macro["macro_r2"],
            "val_macro_pearson": val_macro["macro_pearson"], "val_macro_r2": val_macro["macro_r2"],
        })
        if train_loss < best_train["loss"]:
            best_train = {"loss": train_loss, "epoch": epoch, "state": cpu_state_dict(model)}
        if val_loss < best_val["loss"]:
            best_val = {"loss": val_loss, "epoch": epoch, "state": cpu_state_dict(model)}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(history[-1], flush=True)
        if bad_epochs >= PATIENCE:
            break
    if best_val["state"] is None or best_train["state"] is None:
        raise RuntimeError("Training did not produce restorable states")
    return pd.DataFrame(history), best_val, best_train
'''),
        nb_cell("code", r'''
def prediction_frame_for_state(model, state, dataset, table, y, gene_ids, scalers, state_name):
    model.load_state_dict(state)
    loader = make_loader(dataset, np.ones(len(table), dtype=bool), False)
    pred_scaled, row_indices = predict_scaled(model, loader)
    prediction = inverse_scale_gene_targets(pred_scaled, gene_ids[row_indices], scalers)
    frame = table.iloc[row_indices].copy()
    frame["model_state"] = state_name
    frame["target_scaled"] = dataset.y_scaled[row_indices]
    frame["prediction_scaled"] = pred_scaled
    frame["target_diff"] = y[row_indices]
    frame["prediction_diff"] = prediction
    return frame


def summarize_predictions(predictions, variable, gene_set_size):
    overall_rows, by_gene_frames, macro_rows = [], [], []
    for state_name in predictions["model_state"].unique():
        state_frame = predictions[predictions["model_state"].eq(state_name)]
        allowed_splits = ["train"] if state_name == "best_training" else ["train", "val", "test"]
        for split in allowed_splits:
            split_frame = state_frame[state_frame["split"].eq(split)]
            overall_rows.append({
                "gene_set_size": gene_set_size, "model_variant": MODEL_VARIANT,
                "model_state": state_name, "split": split, "n": len(split_frame),
                **regression_metrics(split_frame["target_diff"], split_frame["prediction_diff"]),
            })
            by_gene = per_gene_metrics(split_frame, variable)
            by_gene.insert(0, "model_state", state_name)
            by_gene.insert(0, "split", split)
            by_gene.insert(0, "model_variant", MODEL_VARIANT)
            by_gene.insert(0, "gene_set_size", gene_set_size)
            by_gene_frames.append(by_gene)
            macro_rows.append({
                "gene_set_size": gene_set_size, "model_variant": MODEL_VARIANT,
                "model_state": state_name, "split": split, **macro_gene_metrics(by_gene),
            })
    return pd.DataFrame(overall_rows), pd.concat(by_gene_frames, ignore_index=True), pd.DataFrame(macro_rows)


def run_multi_gene_model(gene_set_size, genes):
    (table, y, gene_ids, gene_to_id, centers, positions, variant_keys,
     normalized_positions, max_snps) = build_multi_gene_data(genes)
    train_mask = table["split"].eq("train").to_numpy()
    val_mask = table["split"].eq("val").to_numpy()
    scalers = fit_gene_target_scalers(y, gene_ids, train_mask, len(genes), TARGET_STD_EPS)
    y_scaled = scale_gene_targets(y, gene_ids, scalers)
    dataset = SpecificSNPDataset(table, genes, centers, normalized_positions, y_scaled, max_snps)
    model_config = {
        "hidden_dim": int(centers[0].shape[1]), "n_genes": len(genes),
        "projection_dim": PROJECTION_DIM, "gene_embedding_dim": GENE_EMBEDDING_DIM,
        "dropout": DROPOUT,
    }
    model = SpecificSNPRegressor(**model_config)
    history, best_val, best_train = train_specific_snp_model(
        model, dataset, table, y, gene_ids, scalers, train_mask, val_mask
    )
    best_validation_predictions = prediction_frame_for_state(
        model, best_val["state"], dataset, table, y, gene_ids, scalers, "best_validation"
    )
    best_training_predictions = prediction_frame_for_state(
        model, best_train["state"], dataset, table, y, gene_ids, scalers, "best_training"
    )
    predictions = pd.concat([best_validation_predictions, best_training_predictions], ignore_index=True)
    overall, by_gene, macro = summarize_predictions(predictions, scalers["variable"], gene_set_size)
    model.load_state_dict(best_val["state"])
    checkpoint = build_specific_snp_checkpoint(
        model=model, model_config=model_config,
        ordered_genes=[{
            "name": gene.name, "embedding_name": gene.embedding_name, "chrom": gene.chrom,
            "start": gene.start, "end": gene.end, "strand": gene.strand,
            "transcript_id": gene.transcript_id,
        } for gene in genes],
        gene_to_id=gene_to_id, target_scalers=scalers, snp_centers=centers,
        positions=positions, variant_keys=variant_keys, max_snps=max_snps,
        splits={"train": train_individuals, "val": VAL_INDIVIDUALS, "test": TEST_INDIVIDUALS},
        training_config={
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "patience": PATIENCE,
            "loss": "HuberLoss", "random_state": RANDOM_STATE,
            "pairwise_loss": "same_gene_HuberLoss",
            "pairwise_loss_weight": PAIRWISE_LOSS_WEIGHT,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "embedding_source": "hap1", "embedding_frozen": True,
        },
        best_validation=best_val, best_training=best_train,
    )
    assert "best_validation_state_dict" in checkpoint
    assert "best_training_state_dict" in checkpoint
    torch.save(checkpoint, OUTPUT_DIR / CHECKPOINT_FILENAMES[gene_set_size])
    history["gene_set_size"] = gene_set_size
    history["model_variant"] = MODEL_VARIANT
    predictions.insert(0, "gene_set_size", gene_set_size)
    return overall, by_gene, macro, predictions, history
'''),
        nb_cell("code", r'''
overall_frames, by_gene_frames, macro_frames = [], [], []
prediction_frames, history_frames = [], []
for gene_set_size in GENE_SET_SIZES:
    print(f"Training {MODEL_VARIANT}: {gene_set_size} genes")
    result = run_multi_gene_model(gene_set_size, genes_by_size[gene_set_size])
    overall, by_gene, macro, predictions, history = result
    overall_frames.append(overall); by_gene_frames.append(by_gene); macro_frames.append(macro)
    prediction_frames.append(predictions); history_frames.append(history)
    print(macro.to_dict("records"))

metrics_overall = pd.concat(overall_frames, ignore_index=True)
metrics_by_gene = pd.concat(by_gene_frames, ignore_index=True)
metrics_macro = pd.concat(macro_frames, ignore_index=True)
predictions = pd.concat(prediction_frames, ignore_index=True)
history = pd.concat(history_frames, ignore_index=True)
collapse = metrics_by_gene[[
    "gene_set_size", "model_variant", "model_state", "split", "gene_id",
    "embedding_name", "n", "target_std", "prediction_std",
    "prediction_target_std_ratio", "eligible_correlation",
]].copy()

metrics_overall.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_overall.csv", index=False)
metrics_by_gene.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_by_gene.csv", index=False)
metrics_macro.to_csv(OUTPUT_DIR / "model_multi_gene_metrics_macro.csv", index=False)
collapse.to_csv(OUTPUT_DIR / "model_multi_gene_prediction_collapse_by_gene.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "model_multi_gene_predictions.csv", index=False)
history.to_csv(OUTPUT_DIR / "model_multi_gene_history.csv", index=False)
metrics_macro
'''),
    ]


def multi_gene_notebook() -> list[dict]:
    return specific_snp_delta_multi_gene_notebook()


def specific_snp_transformer_cnn_multi_gene_notebook() -> list[dict]:
    cells = specific_snp_delta_multi_gene_notebook()
    cells[0] = nb_cell("markdown", """
# 具体 SNP Delta Transformer + CNN 多基因表达量模型

本 notebook 只使用现有二维 hap1 SNP hidden states。每个具体 SNP 先减去
81个训练个体的均值，再严格按照参考 workflow 使用全长 Transformer 建模
远距离关系，并用多尺度 CNN 提取局部模式。padding mask 贯穿 Transformer、
CNN、残差融合和池化阶段，不进行 SNP 分箱或下采样。
""")
    replacements = [
        ("    SpecificSNPRegressor,\n", "    SpecificSNPTransformerCNN,\n"),
        (
            'OUTPUT_DIR = PROJECT_DIR / "multi_gene_models_specific_snp"',
            'OUTPUT_DIR = PROJECT_DIR / "multi_gene_models_specific_snp_transformer_cnn"',
        ),
        ("GENE_SET_SIZES = [10]", "GENE_SET_SIZES = [50]"),
        ("WEIGHT_DECAY = 0.0", "WEIGHT_DECAY = 1e-4"),
        ("DROPOUT = 0.0", "DROPOUT = 0.2"),
        (
            "PROJECTION_DIM = 64",
            "MODEL_DIM = 64\nN_HEADS = 4\nN_LAYERS = 2\nCNN_KERNELS = (3, 5, 9, 15)",
        ),
        ("PATIENCE = 40", "PATIENCE = 15"),
        ("PAIRWISE_LOSS_WEIGHT = 1.0", "PAIRWISE_LOSS_WEIGHT = 0.25"),
        (
            'MODEL_VARIANT = "specific_snp_delta_hap1"',
            'MODEL_VARIANT = "specific_snp_delta_hap1_transformer_cnn"',
        ),
        (
            '10: "model_multi_gene_specific_snp_delta_10.pt",\n'
            '    50: "model_multi_gene_specific_snp_delta_50.pt",\n'
            '    97: "model_multi_gene_specific_snp_delta_97.pt",',
            '10: "model_multi_gene_specific_snp_transformer_cnn_delta_10.pt",\n'
            '    50: "model_multi_gene_specific_snp_transformer_cnn_delta_50.pt",\n'
            '    97: "model_multi_gene_specific_snp_transformer_cnn_delta_97.pt",',
        ),
        (
            '        "hidden_dim": int(centers[0].shape[1]), "n_genes": len(genes),\n'
            '        "projection_dim": PROJECTION_DIM, "gene_embedding_dim": GENE_EMBEDDING_DIM,\n'
            '        "dropout": DROPOUT,',
            '        "hidden_dim": int(centers[0].shape[1]), "n_genes": len(genes),\n'
            '        "max_snps": max_snps, "model_dim": MODEL_DIM,\n'
            '        "n_heads": N_HEADS, "n_layers": N_LAYERS,\n'
            '        "gene_embedding_dim": GENE_EMBEDDING_DIM, "dropout": DROPOUT,\n'
            '        "cnn_kernels": CNN_KERNELS,',
        ),
        (
            "    model = SpecificSNPRegressor(**model_config)",
            "    model = SpecificSNPTransformerCNN(**model_config)",
        ),
        (
            "        training_config={\n"
            "            \"batch_size\": BATCH_SIZE,",
            "        training_config={\n"
            "            \"model_variant\": MODEL_VARIANT,\n"
            "            \"architecture\": \"full_length_transformer_multiscale_cnn\",\n"
            "            \"model_dim\": MODEL_DIM, \"n_heads\": N_HEADS, \"n_layers\": N_LAYERS,\n"
            "            \"cnn_kernels\": CNN_KERNELS,\n"
            "            \"batch_size\": BATCH_SIZE,",
        ),
    ]
    for old, new in replacements:
        matches = 0
        for cell in cells:
            source = "".join(cell.get("source", []))
            if old in source:
                source = source.replace(old, new)
                cell["source"] = source.splitlines(keepends=True)
                matches += 1
        if matches != 1:
            raise RuntimeError(f"expected one notebook template match for {old!r}, found {matches}")
    loss_replacements = [
        ("pairwise_difference_loss", "pairwise_difference_mse_loss", 3),
        ("criterion = nn.HuberLoss(delta=1.0)", "criterion = nn.MSELoss()", 1),
        ("pairwise_difference_mse_loss(prediction, yb, gb, delta=1.0)", "pairwise_difference_mse_loss(prediction, yb, gb)", 1),
        ("val_gene_tensor, delta=1.0", "val_gene_tensor", 1),
        ('"loss": "HuberLoss"', '"loss": "MSELoss"', 1),
        ('"pairwise_loss": "same_gene_HuberLoss"', '"pairwise_loss": "same_gene_MSELoss"', 1),
    ]
    for old, new, expected_matches in loss_replacements:
        matches = 0
        for cell in cells:
            source = "".join(cell.get("source", []))
            matches += source.count(old)
            if old in source:
                cell["source"] = source.replace(old, new).splitlines(keepends=True)
        if matches != expected_matches:
            raise RuntimeError(
                f"expected {expected_matches} notebook template matches for {old!r}, found {matches}"
            )
    return cells


def main() -> int:
    existing_notebooks = [
        (PROJECT_DIR / "run_single_gene_models.ipynb", single_gene_notebook),
        (PROJECT_DIR / "run_multi_gene_models.ipynb", multi_gene_notebook),
    ]
    for path, factory in existing_notebooks:
        if not path.exists():
            write_notebook(path, factory())
            print(path)
    transformer_path = PROJECT_DIR / "run_multi_gene_transformer_cnn_models.ipynb"
    write_notebook(transformer_path, specific_snp_transformer_cnn_multi_gene_notebook())
    print(transformer_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
