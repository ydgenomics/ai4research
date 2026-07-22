#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_ROOT = Path("/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_res_260708")
DEFAULT_SAMPLE_FILE = PROJECT_DIR / "101samples.name.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "hidden_state_variance_decomposition.csv"

VAL_INDIVIDUALS = {"H005", "H010", "H055", "H102", "H103", "H137", "H198", "H202", "H276", "H319"}
TEST_INDIVIDUALS = {"H030", "H117", "H118", "H129", "H195", "H197", "H215", "H225", "H261", "H309"}
REQUIRED_KEYS = ("hidden_states", "alt_probability", "positions", "variant_keys")


def _as_numpy(value, dtype) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _variance_metrics(within_variance: float, between_variance: float, prefix: str) -> dict[str, float]:
    total = within_variance + between_variance
    individual_fraction = within_variance / total if total > 0 else float("nan")
    icc = between_variance / total if total > 0 else float("nan")
    return {
        f"{prefix}_within_variance": within_variance,
        f"{prefix}_between_site_variance": between_variance,
        f"{prefix}_individual_fraction": individual_fraction,
        f"{prefix}_icc": icc,
    }


def analyze_payload_paths(
    paths: Iterable[str | Path],
    *,
    accumulator_dtype=np.float64,
) -> dict[str, float | int | bool]:
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one payload path is required")

    reference_positions = None
    reference_variant_keys = None
    hidden_sum = hidden_square_sum = None
    alt_sum = alt_square_sum = None
    n_samples = 0

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload: {path}")
        for key in REQUIRED_KEYS:
            if key not in payload:
                raise KeyError(f"{path} is missing required key {key!r}")

        hidden = _as_numpy(payload["hidden_states"], accumulator_dtype)
        alt = _as_numpy(payload["alt_probability"], accumulator_dtype).reshape(-1)
        positions = _as_numpy(payload["positions"], np.int64).reshape(-1)
        variant_keys = tuple(str(value) for value in payload["variant_keys"])
        if hidden.ndim != 2:
            raise ValueError(f"hidden_states must be 2D in {path}, got {hidden.shape}")
        if len(hidden) != len(alt) or len(hidden) != len(positions) or len(hidden) != len(variant_keys):
            raise ValueError(f"SNP arrays do not align in {path}")

        if reference_positions is None:
            reference_positions = positions.copy()
            reference_variant_keys = variant_keys
            hidden_sum = np.zeros_like(hidden, dtype=accumulator_dtype)
            hidden_square_sum = np.zeros_like(hidden, dtype=accumulator_dtype)
            alt_sum = np.zeros_like(alt, dtype=accumulator_dtype)
            alt_square_sum = np.zeros_like(alt, dtype=accumulator_dtype)
        else:
            if hidden.shape != hidden_sum.shape:
                raise ValueError(f"hidden_states shapes are not aligned: {path}")
            if not np.array_equal(positions, reference_positions):
                raise ValueError(f"positions are not aligned: {path}")
            if variant_keys != reference_variant_keys:
                raise ValueError(f"variant_keys are not aligned: {path}")

        hidden_sum += hidden
        hidden_square_sum += hidden * hidden
        alt_sum += alt
        alt_square_sum += alt * alt
        n_samples += 1

    hidden_mean = hidden_sum / n_samples
    hidden_within_by_feature = np.maximum(hidden_square_sum / n_samples - hidden_mean * hidden_mean, 0.0)
    hidden_within = float(hidden_within_by_feature.mean())
    hidden_gene_mean = hidden_mean.mean(axis=0, keepdims=True)
    hidden_between = float(np.mean((hidden_mean - hidden_gene_mean) ** 2))

    alt_mean = alt_sum / n_samples
    alt_within_by_snp = np.maximum(alt_square_sum / n_samples - alt_mean * alt_mean, 0.0)
    alt_within = float(alt_within_by_snp.mean())
    alt_between = float(np.var(alt_mean))

    return {
        "n_train": n_samples,
        "n_snps": int(hidden_mean.shape[0]),
        "embedding_dim": int(hidden_mean.shape[1]),
        "positions_aligned": True,
        "variant_keys_aligned": True,
        **_variance_metrics(hidden_within, hidden_between, "hidden"),
        **_variance_metrics(alt_within, alt_between, "alt"),
    }


def read_train_individuals(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", header=None, names=["accession", "sample"], dtype=str)
    samples = frame["sample"].tolist()
    train = [sample for sample in samples if sample not in VAL_INDIVIDUALS and sample not in TEST_INDIVIDUALS]
    if not train:
        raise ValueError(f"No training individuals found in {path}")
    return train


def sample_pt_path(gene_dir: Path, sample: str) -> Path:
    exact = gene_dir / f"CIMA-{sample}_CIMA-{sample}.vcf.pt"
    if exact.exists():
        return exact
    matches = sorted(gene_dir.glob(f"*{sample}*.vcf.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one .vcf.pt for {gene_dir.name}/{sample}, found {len(matches)}")
    return matches[0]


def write_results(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose frozen SNP hidden-state variation into fixed site and between-individual components."
    )
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gene", action="append", default=[], help="Embedding directory name; repeat to select genes")
    parser.add_argument("--limit", type=int, default=None, help="Analyze only the first N selected genes")
    parser.add_argument("--accumulator-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--resume", action="store_true", help="Keep successful rows already present in --output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first invalid gene instead of recording an error row")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_individuals = read_train_individuals(args.sample_file)
    requested = set(args.gene)
    gene_dirs = sorted(path for path in args.embedding_root.iterdir() if path.is_dir())
    if requested:
        gene_dirs = [path for path in gene_dirs if path.name in requested]
        missing = sorted(requested - {path.name for path in gene_dirs})
        if missing:
            raise FileNotFoundError(f"Requested gene directories not found: {missing}")
    if args.limit is not None:
        gene_dirs = gene_dirs[: args.limit]
    if not gene_dirs:
        raise ValueError("No gene directories selected")

    rows = []
    completed = set()
    if args.resume and args.output.exists():
        existing = pd.read_csv(args.output).to_dict("records")
        rows.extend(existing)
        completed = {
            str(row["embedding_name"])
            for row in existing
            if str(row.get("status", "")) == "ok"
        }

    accumulator_dtype = np.float64 if args.accumulator_dtype == "float64" else np.float32
    total = len(gene_dirs)
    for index, gene_dir in enumerate(gene_dirs, start=1):
        if gene_dir.name in completed:
            print(f"[{index}/{total}] skip completed {gene_dir.name}", flush=True)
            continue
        print(f"[{index}/{total}] analyze {gene_dir.name}", flush=True)
        meta_path = gene_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        base_row = {
            "embedding_name": gene_dir.name,
            "chrom": meta.get("chrom"),
            "window_start": meta.get("start"),
            "window_end": meta.get("end"),
        }
        try:
            paths = [sample_pt_path(gene_dir, sample) for sample in train_individuals]
            result = analyze_payload_paths(paths, accumulator_dtype=accumulator_dtype)
            row = {**base_row, "status": "ok", "error": "", **result}
        except Exception as exc:
            if args.fail_fast:
                raise
            row = {
                **base_row,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "n_train": 0,
                "positions_aligned": False,
                "variant_keys_aligned": False,
            }
            print(f"  ERROR {row['error']}", flush=True)
        rows = [old for old in rows if str(old.get("embedding_name")) != gene_dir.name]
        rows.append(row)
        write_results(rows, args.output)

    ok = sum(str(row.get("status")) == "ok" for row in rows)
    errors = sum(str(row.get("status")) == "error" for row in rows)
    print(f"Wrote {args.output} (ok={ok}, errors={errors})")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
