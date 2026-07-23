"""Data loading: GeneRecord, BigWig targets, embedding payloads, Dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import center_and_pad_snp_hidden, fit_snp_centers, normalize_snp_positions
from .scaling import fit_gene_target_scalers, scale_gene_targets


# ---------------------------------------------------------------------------
# Gene annotation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneRecord:
    """Metadata for one gene window."""
    chrom: str
    start: int
    end: int
    name: str
    embedding_name: str
    strand: str
    tss: int
    transcript_id: str
    expression_regions: tuple[tuple[str, int, int], ...]


# ---------------------------------------------------------------------------
# Path & name helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def parse_gtf_attributes(attr_text: str) -> dict[str, str]:
    return {k: v for k, v in re.findall(r'(\S+) "([^"]*)"', attr_text)}


def load_transcript_exons(
    gtf_path: Path, transcript_ids: Iterable[str]
) -> dict[str, list[tuple[str, int, int, str]]]:
    """Extract exon coordinates for given transcript IDs from a GTF file."""
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
            tid = attrs.get("transcript_id")
            if not tid:
                continue
            tid_base = strip_ensembl_version(tid)
            if tid in requested:
                key = tid
            elif tid_base in requested_base:
                matches = [t for t in requested if strip_ensembl_version(t) == tid_base]
                key = matches[0]
            else:
                continue
            exons.setdefault(key, []).append(
                (normalize_chrom(fields[0]), int(fields[3]) - 1, int(fields[4]), fields[6])
            )

    missing = [tid for tid in requested if not exons.get(tid)]
    if missing:
        raise KeyError(f"No exon records for transcript(s): {missing[:10]}")
    return exons


def select_three_prime_exon(
    exons: list[tuple[str, int, int, str]],
) -> tuple[tuple[str, int, int], ...]:
    """Select the 3'-most exon for expression quantification."""
    strands = {e[3] for e in exons}
    if len(strands) != 1:
        raise ValueError(f"Expected one strand per transcript, got {sorted(strands)}")
    strand = next(iter(strands))
    if strand == "+":
        _, start, end, _ = max(exons, key=lambda item: (item[2], item[1]))
    elif strand == "-":
        _, start, end, _ = min(exons, key=lambda item: (item[1], item[2]))
    else:
        raise ValueError(f"Unsupported strand: {strand!r}")
    return ((normalize_chrom(exons[0][0]), start, end),)


# ---------------------------------------------------------------------------
# Gene list parsing
# ---------------------------------------------------------------------------

def read_genes_from_bed(bed_path: Path, gencode_gtf: Path) -> list[GeneRecord]:
    """Parse gene records from a BED file + GTF annotation."""
    raw = []
    with bed_path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, start, end, name = fields[:4]
            strand = fields[5] if len(fields) > 5 else "+"
            tid = transcript_id_from_name(name)
            raw.append((
                normalize_chrom(chrom), int(start), int(end), name,
                embedding_name_from_bed_name(name), strand, tid,
            ))

    exons = load_transcript_exons(gencode_gtf, [r[6] for r in raw])
    return [
        GeneRecord(
            chrom, start, end, name, emb_name, exons[tid][0][3],
            (start + end) // 2, tid,
            select_three_prime_exon(exons[tid]),
        )
        for chrom, start, end, name, emb_name, strand, tid in raw
    ]


def read_genes_from_embedding_dir(
    embedding_root: Path, gencode_gtf: Path
) -> list[GeneRecord]:
    """Discover all genes that have embedding directories."""
    metas = []
    for meta_path in sorted(embedding_root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        tid = transcript_id_from_name(meta["name"])
        metas.append((meta_path.parent.name, meta, tid))

    exons = load_transcript_exons(gencode_gtf, [m[2] for m in metas])
    genes = []
    for emb_name, meta, tid in metas:
        start = int(meta["start"])
        end = int(meta["end"])
        chrom = normalize_chrom(meta["chrom"])
        gene_name = meta["name"].replace("_", "|", 2)
        genes.append(
            GeneRecord(
                chrom, start, end, gene_name, emb_name,
                exons[tid][0][3], (start + end) // 2, tid,
                select_three_prime_exon(exons[tid]),
            )
        )
    return genes


def order_genes_by_bed(
    genes: list[GeneRecord], bed_path: Path
) -> list[GeneRecord]:
    """Reorder genes to match the order in a BED file."""
    by_name = {g.embedding_name: g for g in genes}
    ordered_names = []
    with Path(bed_path).open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                ordered_names.append(
                    embedding_name_from_bed_name(line.rstrip("\n").split("\t")[3])
                )
    return [by_name[n] for n in ordered_names if n in by_name]


# ---------------------------------------------------------------------------
# Sample table
# ---------------------------------------------------------------------------

def read_sample_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["accession", "sample"])
    df["accession"] = df["accession"].astype(str)
    df["sample"] = df["sample"].astype(str)
    return df


# ---------------------------------------------------------------------------
# BigWig target computation
# ---------------------------------------------------------------------------

def bw_mean_regions(
    bw, regions: Iterable[tuple[str, int, int]]
) -> float:
    """Compute mean BigWig signal across multiple genomic regions."""
    weighted_sum = 0.0
    total_bp = 0
    for chrom, start, end in regions:
        values = np.array(
            bw.values(chrom, max(0, int(start)), max(0, int(end)), numpy=True),
            dtype=np.float32,
        )
        if values.size == 0:
            continue
        values = np.nan_to_num(values, nan=0.0)
        weighted_sum += float(values.sum())
        total_bp += int(values.size)
    if total_bp == 0:
        return float("nan")
    return weighted_sum / total_bp


def compute_targets(
    samples: list[str],
    gene: GeneRecord,
    bigwig_dir: Path,
    reference_bw_path: Path,
    sample_to_accession: dict[str, str],
) -> np.ndarray:
    """Compute per-sample expression difference from reference mean.

    target[sample] = personal_BigWig_mean - reference_BigWig_mean
    """
    import pyBigWig

    with pyBigWig.open(str(reference_bw_path)) as ref_bw:
        reference_value = bw_mean_regions(ref_bw, gene.expression_regions)

    values = []
    for sample in samples:
        accession = sample_to_accession[sample]
        bw_path = bigwig_dir / accession / "re-normalized_Monocyte.total.bw"
        with pyBigWig.open(str(bw_path)) as sample_bw:
            sample_value = bw_mean_regions(sample_bw, gene.expression_regions)
        values.append(sample_value - reference_value)
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding payload loading
# ---------------------------------------------------------------------------

def load_hap1_payload(
    gene: GeneRecord, sample: str, embedding_root: Path
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load one (gene, sample) SNP embedding payload.

    Accepts ``sample`` as either ``"H005"`` or ``"CIMA-H005"`` format;
    the ``CIMA-`` prefix is stripped automatically for path construction.
    File on disk is always ``CIMA-{H编号}_CIMA-{H编号}.vcf.pt``.

    Returns:
        states:      [n_snps, hidden_dim]
        positions:   [n_snps] 1-based genomic coordinates
        variant_keys: (n_snps,) "chr:pos:ref:alt" strings
    """
    # Strip optional "CIMA-" prefix
    short_id = sample[5:] if sample.startswith("CIMA-") else sample
    # Try exact match first, then glob
    exact_path = (
        embedding_root / gene.embedding_name / f"CIMA-{short_id}_CIMA-{short_id}.vcf.pt"
    )
    if exact_path.exists():
        path = exact_path
    else:
        matches = sorted(
            (embedding_root / gene.embedding_name).glob(f"*{short_id}*.vcf.pt")
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected 1 .vcf.pt for {gene.embedding_name}/{sample}, "
                f"found {len(matches)}"
            )
        path = matches[0]

    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload["hidden_states"].detach().cpu().float().numpy()
    positions = payload["positions"].detach().cpu().long().numpy().reshape(-1)
    variant_keys = tuple(str(v) for v in payload["variant_keys"])

    if states.ndim != 2 or len(states) != len(positions) or len(states) != len(variant_keys):
        raise ValueError(f"Misaligned payload for {gene.embedding_name}/{sample}")
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite states for {gene.embedding_name}/{sample}")
    return states.astype(np.float32, copy=False), positions, variant_keys


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpecificSNPDataset(Dataset):
    """Dataset yielding (snp_delta, gene_id, y_scaled) tuples.

    Collation handles center subtraction, padding, and position injection.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        genes: list[GeneRecord],
        centers: list[np.ndarray],
        normalized_positions: np.ndarray,
        y_scaled: np.ndarray,
        max_snps: int,
        embedding_root: Path,
    ):
        self.table = table.reset_index(drop=True)
        self.genes = genes
        self.centers = centers
        self.normalized_positions = normalized_positions
        self.y_scaled = np.asarray(y_scaled, dtype=np.float32)
        self.max_snps = int(max_snps)
        self.embedding_root = embedding_root

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        gene_id = int(row["gene_id"])
        states, _, _ = load_hap1_payload(
            self.genes[gene_id], str(row["sample"]), self.embedding_root
        )
        return states, gene_id, float(self.y_scaled[index]), int(index)

    def collate(self, batch):
        states = [item[0][None, :, :] for item in batch]
        gene_ids = np.asarray([item[1] for item in batch], dtype=np.int64)
        centers = [self.centers[gid] for gid in gene_ids]
        delta, snp_mask = center_and_pad_snp_hidden(
            states, centers, max_snps=self.max_snps
        )
        positions = self.normalized_positions[gene_ids]
        targets = np.asarray(
            [item[2] for item in batch], dtype=np.float32
        ).reshape(-1, 1)
        row_indices = np.asarray([item[3] for item in batch], dtype=np.int64)
        return (
            torch.as_tensor(delta[:, 0], dtype=torch.float16),
            torch.as_tensor(snp_mask, dtype=torch.bool),
            torch.as_tensor(positions, dtype=torch.float32),
            torch.as_tensor(gene_ids, dtype=torch.long),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(row_indices, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data assembly pipeline
# ---------------------------------------------------------------------------

def fit_gene_snp_preprocessing(
    gene: GeneRecord,
    all_individuals: list[str],
    train_individuals: list[str],
    embedding_root: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Compute per-SNP center from training individuals for one gene."""
    train_set = set(train_individuals)
    ref_positions = None
    ref_keys = None
    center_sum = None
    n_train = 0

    for sample in all_individuals:
        states, positions, variant_keys = load_hap1_payload(
            gene, sample, embedding_root
        )
        if ref_positions is None:
            ref_positions = positions.copy()
            ref_keys = variant_keys
            center_sum = np.zeros(states.shape, dtype=np.float64)
        else:
            if states.shape != center_sum.shape:
                raise ValueError(f"shape mismatch for {gene.embedding_name}/{sample}")
            if not np.array_equal(positions, ref_positions):
                raise ValueError(f"positions mismatch for {gene.embedding_name}/{sample}")
            if variant_keys != ref_keys:
                raise ValueError(f"variant_keys mismatch for {gene.embedding_name}/{sample}")
        if sample in train_set:
            center_sum += states
            n_train += 1

    if n_train != len(train_individuals):
        raise RuntimeError(
            f"Expected {len(train_individuals)} train samples, got {n_train}"
        )
    return (center_sum / n_train).astype(np.float32), ref_positions, ref_keys


def build_multi_gene_data(
    genes: list[GeneRecord],
    all_individuals: list[str],
    train_individuals: list[str],
    val_individuals: set[str],
    test_individuals: set[str],
    embedding_root: Path,
    bigwig_dir: Path,
    reference_bw_path: Path,
    sample_to_accession: dict[str, str],
) -> tuple:
    """Full data assembly pipeline (BigWig).

    Returns:
        table, y, gene_ids, gene_to_id, centers, positions_by_gene,
        variant_keys_by_gene, normalized_positions, max_snps,
        cached_delta (None), cached_mask (None)
    """
    rows, y_rows, gene_id_rows = [], [], []
    centers, positions_by_gene, variant_keys_by_gene = [], [], []
    gene_to_id = {g.embedding_name: idx for idx, g in enumerate(genes)}

    for gene in genes:
        print(f"  Processing: {gene.embedding_name}", flush=True)

        # Preprocessing
        center, positions, variant_keys = fit_gene_snp_preprocessing(
            gene, all_individuals, train_individuals, embedding_root
        )
        centers.append(center)
        positions_by_gene.append(positions)
        variant_keys_by_gene.append(variant_keys)

        # Targets
        targets = compute_targets(
            all_individuals, gene, bigwig_dir, reference_bw_path, sample_to_accession
        )

        gene_id = gene_to_id[gene.embedding_name]
        for sample_idx, sample in enumerate(all_individuals):
            split = (
                "val"
                if sample in val_individuals
                else "test"
                if sample in test_individuals
                else "train"
            )
            rows.append(
                {
                    "sample": sample,
                    "split": split,
                    "gene": gene.name,
                    "embedding_name": gene.embedding_name,
                    "gene_id": gene_id,
                    "n_snps": len(positions),
                    "embedding_dim": center.shape[1],
                }
            )
            y_rows.append(targets[sample_idx])
            gene_id_rows.append(gene_id)

    table = pd.DataFrame(rows)
    y = np.asarray(y_rows, dtype=np.float32)
    gene_ids_arr = np.asarray(gene_id_rows, dtype=np.int64)
    max_snps = max(len(v) for v in positions_by_gene)
    normalized_positions = normalize_snp_positions(
        positions_by_gene,
        [g.start for g in genes],
        [g.end for g in genes],
        max_snps,
    )
    return (
        table, y, gene_ids_arr, gene_to_id, centers,
        positions_by_gene, variant_keys_by_gene, normalized_positions, max_snps,
        None, None, genes,  # cached_delta, cached_mask (BigWig: no cache), genes
    )


# ---------------------------------------------------------------------------
# Expression matrix target loading
# ---------------------------------------------------------------------------

def extract_gene_symbol(embedding_name: str) -> str:
    """Extract HGNC symbol from embedding directory name.

    Compatible with two naming conventions:
    - Old: "RETN_ENST00000442999.3_win_4" → "RETN"
    - New: "001_RETN" → "RETN"
    """
    import re
    # Strip numeric prefix if present (e.g. "001_RETN" → "RETN")
    m = re.match(r"^\d+_(.+)", embedding_name)
    if m:
        return m.group(1)
    return embedding_name.split("_", 1)[0]


def load_expression_matrix(
    matrix_path: str | Path,
    individual_ids: list[str],
    gene_symbols: list[str],
) -> pd.DataFrame:
    """Load expression matrix and extract subset for given individuals × genes.

    Args:
        matrix_path: path to Monocyte_matrix_log2_TPM_annot.tsv.gz
        individual_ids: list of "CIMA-H005" style IDs (hyphen)
        gene_symbols: list of HGNC gene symbols (e.g. ["RETN", "NT5C3B"])

    Returns:
        DataFrame with gene_id as index, individual IDs as columns,
        log2(TPM+0.01) values. Shape: [n_genes, n_individuals].
    """
    import gzip

    matrix_path = Path(matrix_path)

    # Read with gzip if compressed
    if matrix_path.suffix == ".gz":
        df = pd.read_csv(gzip.open(matrix_path, "rt"), sep="\t")
    else:
        df = pd.read_csv(matrix_path, sep="\t")

    # Harmonize individual ID format: CIMA_H056 → CIMA-H056
    rename_map = {}
    for col in df.columns:
        if col.startswith("CIMA_"):
            rename_map[col] = col.replace("_", "-", 1)  # only first underscore
    df = df.rename(columns=rename_map)

    # Subset by gene
    gene_subset = df[df["gene_id"].isin(gene_symbols)].copy()
    found = set(gene_subset["gene_id"])
    missing = set(gene_symbols) - found
    if missing:
        print(f"  Warning: {len(missing)} genes not found in expression matrix: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    # Subset by individual
    available_cols = [c for c in gene_subset.columns if c in set(individual_ids)]
    missing_ind = set(individual_ids) - set(available_cols)
    if missing_ind:
        print(f"  Warning: {len(missing_ind)} individuals not found: {sorted(missing_ind)[:5]}...")

    # Build gene × individual matrix
    expr = gene_subset.set_index("gene_id")[available_cols].astype(np.float32)

    print(f"  Expression matrix subset: {expr.shape[0]} genes × {expr.shape[1]} individuals")
    return expr


def build_multi_gene_data_from_matrix(
    genes: list[GeneRecord],
    all_individuals: list[str],
    train_individuals: list[str],
    val_individuals: set[str],
    test_individuals: set[str],
    embedding_root: Path,
    expression_matrix: str | Path | pd.DataFrame,
    cache_deltas: bool = True,
) -> tuple:
    """Full data assembly pipeline using expression matrix for targets.

    When ``expression_matrix`` is a file path (``str`` or ``Path``):
      - ``all_individuals`` should be short IDs like ``"H005"``
      - they will be converted to ``"CIMA-H005"`` for matrix lookup
    When ``expression_matrix`` is a pre-loaded ``pd.DataFrame``:
      - ``all_individuals`` must already match the DataFrame's column names
      - no prefix conversion is done

    When cache_deltas=True, precomputes all SNP deltas in memory to avoid
    repeated disk I/O during training. Returns cached_delta and cached_mask
    as the last two tuple elements.

    Returns:
        table, y, gene_ids, gene_to_id, centers, positions_by_gene,
        variant_keys_by_gene, normalized_positions, max_snps,
        cached_delta, cached_mask
    """
    rows, y_rows, gene_id_rows = [], [], []
    centers, positions_by_gene, variant_keys_by_gene = [], [], []
    gene_to_id = {g.embedding_name: idx for idx, g in enumerate(genes)}

    # ── Prepare individual IDs & expression matrix ──
    gene_symbols = [extract_gene_symbol(g.embedding_name) for g in genes]
    if isinstance(expression_matrix, pd.DataFrame):
        # Pre-loaded DataFrame: use IDs as-is (must match column names)
        cima_ids = all_individuals
        expr = expression_matrix
        print(f"  Using pre-loaded expression matrix: {expr.shape[0]} genes × {expr.shape[1]} individuals")
    else:
        # File path: convert short IDs ("H005") to CIMA format ("CIMA-H005")
        cima_ids = [f"CIMA-{s}" for s in all_individuals]
        expr = load_expression_matrix(
            expression_matrix, cima_ids, gene_symbols
        )

    for gene in genes:
        gene_symbol = extract_gene_symbol(gene.embedding_name)
        print(f"  Processing: {gene.embedding_name}  ({gene_symbol})", flush=True)

        # Preprocessing
        center, positions, variant_keys = fit_gene_snp_preprocessing(
            gene, all_individuals, train_individuals, embedding_root
        )
        centers.append(center)
        positions_by_gene.append(positions)
        variant_keys_by_gene.append(variant_keys)

        # Expression targets from matrix
        targets = np.full(len(all_individuals), np.nan, dtype=np.float32)
        if gene_symbol in expr.index:
            for sample_idx, cima_sample in enumerate(cima_ids):
                if cima_sample in expr.columns:
                    val = expr.at[gene_symbol, cima_sample]
                    if not np.isnan(val):
                        targets[sample_idx] = val

        gene_id = gene_to_id[gene.embedding_name]
        for sample_idx, sample in enumerate(all_individuals):
            split = (
                "val"
                if sample in val_individuals
                else "test"
                if sample in test_individuals
                else "train"
            )
            rows.append(
                {
                    "sample": sample,
                    "split": split,
                    "gene": gene.name,
                    "embedding_name": gene.embedding_name,
                    "gene_id": gene_id,
                    "n_snps": len(positions),
                    "embedding_dim": center.shape[1],
                }
            )
            y_rows.append(targets[sample_idx])
            gene_id_rows.append(gene_id)

    table = pd.DataFrame(rows)
    y = np.asarray(y_rows, dtype=np.float32)

    # Remove samples with NaN targets
    valid = np.isfinite(y)
    n_nan = (~valid).sum()
    if n_nan > 0:
        print(f"  Removing {n_nan} samples with NaN expression values")
        table = table.loc[valid].reset_index(drop=True)
        y = y[valid]
        gene_ids_arr = np.asarray(gene_id_rows, dtype=np.int64)[valid]
    else:
        gene_ids_arr = np.asarray(gene_id_rows, dtype=np.int64)

    # ── Remove genes that have zero training samples ──
    train_mask = table["split"].eq("train").to_numpy()
    train_gene_counts = {}
    for gid in np.unique(gene_ids_arr[train_mask]):
        train_gene_counts[int(gid)] = int(train_mask[gene_ids_arr == gid].sum())
    n_genes_orig = len(genes)
    missing_train = [
        i for i in range(n_genes_orig) if i not in train_gene_counts
    ]
    if missing_train:
        print(
            f"  Removing {len(missing_train)} gene(s) with zero training targets: "
            f"{[genes[i].embedding_name for i in missing_train]}"
        )
        keep_genes = [i for i in range(n_genes_orig) if i not in missing_train]
        old_to_new = {old: new for new, old in enumerate(keep_genes)}
        # Filter table rows
        keep_rows = np.isin(gene_ids_arr, keep_genes)
        table = table.loc[keep_rows].reset_index(drop=True)
        y = y[keep_rows]
        gene_ids_arr = gene_ids_arr[keep_rows]
        # Remap gene IDs in both gene_ids_arr and table
        gene_ids_arr = np.array([old_to_new[int(g)] for g in gene_ids_arr], dtype=np.int64)
        table["gene_id"] = gene_ids_arr.astype(int)
        # Filter per-gene arrays
        genes = [genes[i] for i in keep_genes]
        centers = [centers[i] for i in keep_genes]
        positions_by_gene = [positions_by_gene[i] for i in keep_genes]
        variant_keys_by_gene = [variant_keys_by_gene[i] for i in keep_genes]
        gene_to_id = {g.embedding_name: idx for idx, g in enumerate(genes)}
        print(f"  Remaining genes: {len(genes)}")

    max_snps = max(len(v) for v in positions_by_gene)
    normalized_positions = normalize_snp_positions(
        positions_by_gene,
        [g.start for g in genes],
        [g.end for g in genes],
        max_snps,
    )

    # ── Precompute cached deltas (optional) ──
    cached_delta = None
    cached_mask = None
    if cache_deltas:
        print(f"  Precomputing SNP deltas for {len(table)} samples...")
        cached_delta, cached_mask = precompute_snp_deltas_from_table(
            table, genes, centers, max_snps, embedding_root,
        )
        print(f"    cached_delta: {cached_delta.shape} ({cached_delta.dtype})")

    return (
        table, y, gene_ids_arr, gene_to_id, centers,
        positions_by_gene, variant_keys_by_gene, normalized_positions, max_snps,
        cached_delta, cached_mask, genes,
    )


# ---------------------------------------------------------------------------
# Precomputed delta cache
# ---------------------------------------------------------------------------

def precompute_snp_deltas_from_table(
    table: pd.DataFrame,
    genes: list[GeneRecord],
    centers: list[np.ndarray],
    max_snps: int,
    embedding_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute SNP deltas aligned to table rows (one row = one sample).

    Unlike precompute_snp_deltas (which builds n_genes × n_individuals),
    this builds len(table) rows, matching the table after NaN/gene filtering.

    Returns:
        cached_delta: [len(table), max_snps, hidden_dim] float16
        cached_mask:  [len(table), max_snps] bool
    """
    n_rows = len(table)
    hidden_dim = centers[0].shape[1]
    delta = np.zeros((n_rows, max_snps, hidden_dim), dtype=np.float16)
    mask = np.zeros((n_rows, max_snps), dtype=bool)

    for idx in range(n_rows):
        row = table.iloc[idx]
        gene_id = int(row["gene_id"])
        sample = str(row["sample"])
        center = centers[gene_id]
        states, _, _ = load_hap1_payload(genes[gene_id], sample, embedding_root)
        n_snps = states.shape[0]
        delta[idx, :n_snps] = (states - center).astype(np.float16)
        mask[idx, :n_snps] = True
        if (idx + 1) % 500 == 0 or idx == n_rows - 1:
            print(f"    cached {idx + 1}/{n_rows} samples")

    return delta, mask


# Backwards compatibility wrapper for older imports
def precompute_snp_deltas(*args, **kwargs):
    """Compatibility wrapper for legacy name `precompute_snp_deltas`.

    Delegates to `precompute_snp_deltas_from_table`, which builds a cache
    aligned to the (possibly filtered) `table` rows.
    """
    return precompute_snp_deltas_from_table(*args, **kwargs)


class CachedSpecificSNPDataset(Dataset):
    """Memory-cached SNP delta Dataset — no disk I/O during training.

    All deltas are precomputed into a single [total_samples, max_snps, hidden_dim]
    float16 buffer. Collation is a fast slice-and-cast operation.

    Memory estimate (97 genes × 101 ind):
      max_snps ≈ 2500, hidden_dim = 1024
      9797 × 2500 × 1024 × 2 bytes ≈ 47 GB (float16) — borderline
      → if too large, reduce max_snps or use fewer genes

    For 10 genes:
      1010 × 2500 × 1024 × 2 bytes ≈ 5 GB — very comfortable
    """

    def __init__(
        self,
        table: pd.DataFrame,
        cached_delta: np.ndarray,       # [n_samples, max_snps, hidden_dim] float16
        cached_mask: np.ndarray,         # [n_samples, max_snps] bool
        normalized_positions: np.ndarray, # [n_genes, max_snps] float32
        y_scaled: np.ndarray,            # [n_samples] float32
    ):
        self.table = table.reset_index(drop=True)
        self.cached_delta = cached_delta
        self.cached_mask = cached_mask
        self.normalized_positions = normalized_positions
        self.y_scaled = np.asarray(y_scaled, dtype=np.float32)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        gene_id = int(row["gene_id"])
        return (
            index,  # used to look up cached delta
            gene_id,
            float(self.y_scaled[index]),
            int(index),  # original row index for tracking
        )

    def collate(self, batch):
        indices = [item[0] for item in batch]
        gene_ids = np.asarray([item[1] for item in batch], dtype=np.int64)
        positions = self.normalized_positions[gene_ids]
        targets = np.asarray(
            [item[2] for item in batch], dtype=np.float32
        ).reshape(-1, 1)
        row_indices = np.asarray([item[3] for item in batch], dtype=np.int64)

        # Fast slice from precomputed cache
        delta = torch.as_tensor(self.cached_delta[indices], dtype=torch.float16)
        snp_mask = torch.as_tensor(self.cached_mask[indices], dtype=torch.bool)

        return (
            delta,
            snp_mask,
            torch.as_tensor(positions, dtype=torch.float32),
            torch.as_tensor(gene_ids, dtype=torch.long),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(row_indices, dtype=torch.long),
        )
