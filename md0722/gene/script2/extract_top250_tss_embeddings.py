#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import pysam


DEFAULT_GENE_LIST = Path("/mnt/zzbnew/peixunban/yancaiting/WorkSpace_GenOmics/result/CIMA_100k_ElasticNet_cv/raw/genelist_Elasticnet_cv_top1000.txt")
DEFAULT_EXPRESSION_MATRIX = Path("/mnt/genos100-new/peixunban/yecheng/data/CIMA/Monocyte_matrix_log2_TPM_annot.tsv.gz")
DEFAULT_CHROMOSOME_ROOT = Path("/mnt/a100-nas-new/peixunban/tanxinjiang/13.SNPbag.pre_exp/model_training/embeddings_gaussian_sigma15.0")
DEFAULT_SAMPLE_FILE = Path("/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/101samples.name.txt")
DEFAULT_VCF_ROOT = Path("/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding/101vcf")
DEFAULT_OUTPUT_ROOT = Path("/mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top250_TSS_1Mb")


@dataclass(frozen=True)
class GeneWindow:
    rank: int
    gene_id: str
    chrom: str
    tss: int
    flank: int = 500_000

    @property
    def start(self) -> int:
        return max(0, int(self.tss) - int(self.flank))

    @property
    def end(self) -> int:
        return int(self.tss) + int(self.flank)

    def contains(self, position: int) -> bool:
        return self.start <= int(position) < self.end


def normalize_chrom(chrom: str) -> str:
    value = str(chrom)
    return value if value.startswith("chr") else f"chr{value}"


def read_top_genes(path: Path, count: int = 250) -> list[str]:
    genes = [line.split()[0] for line in Path(path).read_text().splitlines() if line.strip()]
    selected = genes[: int(count)]
    if len(selected) != int(count):
        raise ValueError(f"Expected {count} nonempty genes, found {len(selected)}")
    if len(set(selected)) != len(selected):
        raise ValueError("Top gene list contains duplicate gene IDs")
    return selected


def read_gene_windows(matrix_path: Path, genes: Sequence[str]) -> list[GeneWindow]:
    requested = set(genes)
    matches: dict[str, list[tuple[str, int]]] = {gene: [] for gene in genes}
    with gzip.open(matrix_path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene_id", "chr", "tss"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expression matrix must contain {sorted(required)}")
        for row in reader:
            gene = row["gene_id"]
            if gene in requested:
                matches[gene].append((normalize_chrom(row["chr"]), int(row["tss"])))
    invalid = [gene for gene in genes if len(matches[gene]) != 1]
    if invalid:
        details = ", ".join(f"{gene}={len(matches[gene])}" for gene in invalid)
        raise ValueError(f"Requested genes must occur exactly once in expression matrix: {details}")
    return [GeneWindow(rank=i, gene_id=gene, chrom=matches[gene][0][0], tss=matches[gene][0][1]) for i, gene in enumerate(genes)]


def canonical_variant_key(entry: Mapping) -> str:
    return f"{normalize_chrom(entry['chrom'])}_{int(entry['pos'])}_{entry['ref']}_{entry['alt']}"


def select_window_variants(variants: Mapping[str, Mapping], window: GeneWindow) -> list[tuple[str, Mapping]]:
    selected = []
    for stored_key, entry in variants.items():
        if normalize_chrom(entry["chrom"]) == window.chrom and window.contains(int(entry["pos"])):
            key = canonical_variant_key(entry)
            if str(stored_key) != key:
                raise ValueError(f"Variant key mismatch: stored={stored_key!r}, canonical={key!r}")
            selected.append((key, entry))
    selected.sort(key=lambda item: (int(item[1]["pos"]), item[0]))
    return selected


def diploid_hidden_state(entry: Mapping, alleles: Sequence[str | None]) -> torch.Tensor:
    if len(alleles) != 2 or any(allele is None for allele in alleles):
        raise ValueError("Genotype must contain two called alleles")
    ref, alt = str(entry["ref"]), str(entry["alt"])
    vectors = []
    for allele in alleles:
        if allele == ref:
            vectors.append(torch.as_tensor(entry["emb_ref"], dtype=torch.float32))
        elif allele == alt:
            vectors.append(torch.as_tensor(entry["emb_alt"], dtype=torch.float32))
        else:
            raise ValueError(f"unsupported allele {allele!r} for {ref}>{alt}")
    result = (vectors[0] + vectors[1]) * 0.5
    if result.ndim != 1 or result.shape[0] != 1024:
        raise ValueError(f"Expected 1024-dimensional embedding, got {tuple(result.shape)}")
    return result


def build_sample_payload(
    gene_id: str,
    selected_variants: Sequence[tuple[str, Mapping]],
    genotypes: Mapping[str, Sequence[str | None]],
) -> dict:
    keys = [key for key, _ in selected_variants]
    missing = [key for key in keys if key not in genotypes]
    if missing:
        raise ValueError(f"Missing genotypes for {len(missing)} variants; first={missing[0]}")
    hidden = torch.stack([diploid_hidden_state(entry, genotypes[key]) for key, entry in selected_variants]).to(torch.float16)
    positions = torch.tensor([int(entry["pos"]) for _, entry in selected_variants], dtype=torch.int64)
    return {"hidden_states": hidden, "positions": positions, "variant_keys": keys, "gene_id": str(gene_id)}


def validate_payload(payload: Mapping, gene_id: str, positions: Sequence[int], variant_keys: Sequence[str]) -> None:
    expected_keys = {"hidden_states", "positions", "variant_keys", "gene_id"}
    if set(payload) != expected_keys:
        raise ValueError(f"Payload keys must equal {sorted(expected_keys)}")
    hidden = payload["hidden_states"]
    stored_positions = payload["positions"]
    if not isinstance(hidden, torch.Tensor) or hidden.dtype != torch.float16 or tuple(hidden.shape) != (len(variant_keys), 1024):
        raise ValueError("hidden_states must be float16 [L, 1024]")
    if not isinstance(stored_positions, torch.Tensor) or stored_positions.dtype != torch.int64:
        raise ValueError("positions must be int64 tensor")
    if stored_positions.tolist() != list(positions):
        raise ValueError("positions do not match expected values")
    if list(payload["variant_keys"]) != list(variant_keys):
        raise ValueError("variant_keys do not match expected values")
    if payload["gene_id"] != gene_id:
        raise ValueError("gene_id does not match expected value")


def query_sample_genotypes(
    vcf_path: Path,
    sample_name: str,
    chrom: str,
    start: int,
    end: int,
    selected_variants: Sequence[tuple[str, Mapping]],
) -> dict[str, tuple[str, str]]:
    selected = {key: entry for key, entry in selected_variants}
    result = {key: (str(entry["ref"]), str(entry["ref"])) for key, entry in selected.items()}
    by_locus: dict[tuple[int, str], list[tuple[str, str]]] = {}
    for key, entry in selected.items():
        by_locus.setdefault((int(entry["pos"]), str(entry["ref"])), []).append((key, str(entry["alt"])))
    with pysam.VariantFile(str(vcf_path)) as vcf:
        if sample_name not in vcf.header.samples:
            if len(vcf.header.samples) == 1:
                sample_name = next(iter(vcf.header.samples))
            else:
                raise KeyError(f"Sample {sample_name!r} not found in {vcf_path}")
        for record in vcf.fetch(chrom, max(0, int(start)), int(end)):
            locus = (int(record.pos), str(record.ref))
            targets = by_locus.get(locus)
            if not targets:
                continue
            gt = record.samples[sample_name].get("GT")
            if gt is None or len(gt) != 2 or any(value is None for value in gt):
                raise ValueError(f"Missing diploid GT for {sample_name} at {chrom}:{record.pos}")
            allele_strings = tuple(record.alleles[index] if 0 <= index < len(record.alleles) else None for index in gt)
            if any(value is None for value in allele_strings):
                raise ValueError(f"Unsupported GT allele index for {sample_name} at {chrom}:{record.pos}")
            for key, target_alt in targets:
                ref = str(record.ref)
                result[key] = tuple(target_alt if allele == target_alt else ref for allele in allele_strings)
    return result


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_samples(sample_file: Path, vcf_root: Path, limit: int | None = None) -> list[tuple[str, str, Path]]:
    samples = []
    with Path(sample_file).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed sample line: {line!r}")
            sample = fields[1]
            matches = sorted(Path(vcf_root).glob(f"CIMA-{sample}_CIMA-{sample}.vcf.gz"))
            if len(matches) != 1 or not Path(str(matches[0]) + ".tbi").exists():
                raise FileNotFoundError(f"Expected indexed VCF for {sample}, found {len(matches)}")
            samples.append((sample, f"CIMA-{sample}_CIMA-{sample}", matches[0]))
    if limit is not None:
        samples = samples[:limit]
    return samples


def safe_gene_dir_name(window: GeneWindow) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", window.gene_id)
    return f"{window.rank + 1:03d}_{symbol}"


def chromosome_embedding_path(root: Path, chrom: str) -> Path:
    path = Path(root) / f"1KGP.{normalize_chrom(chrom)}.snps.embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_chromosome_embeddings(path: Path) -> Mapping[str, Mapping]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_chromosome(
    chrom: str,
    windows: Sequence[GeneWindow],
    samples: Sequence[tuple[str, str, Path]],
    chromosome_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> list[dict]:
    started = time.time()
    source_path = chromosome_embedding_path(chromosome_root, chrom)
    print(f"load {chrom}: {source_path}", flush=True)
    variants = load_chromosome_embeddings(source_path)
    selected_by_gene = {window.gene_id: select_window_variants(variants, window) for window in windows}
    del variants
    union: dict[str, Mapping] = {}
    for selected in selected_by_gene.values():
        for key, entry in selected:
            union[key] = entry
    union_selected = sorted(union.items(), key=lambda item: (int(item[1]["pos"]), item[0]))
    if not union_selected:
        raise ValueError(f"No chromosome embeddings selected for {chrom}")
    fetch_start = min(window.start for window in windows)
    fetch_end = max(window.end for window in windows)
    rows = []
    for sample, vcf_sample, vcf_path in samples:
        genotypes = query_sample_genotypes(vcf_path, vcf_sample, chrom, fetch_start, fetch_end, union_selected)
        for window in windows:
            selected = selected_by_gene[window.gene_id]
            destination = Path(output_root) / safe_gene_dir_name(window) / f"{vcf_sample}.vcf.pt"
            expected_positions = [int(entry["pos"]) for _, entry in selected]
            expected_keys = [key for key, _ in selected]
            if destination.exists() and not overwrite:
                existing = torch.load(destination, map_location="cpu", weights_only=False)
                validate_payload(existing, window.gene_id, expected_positions, expected_keys)
                continue
            payload = build_sample_payload(window.gene_id, selected, genotypes)
            validate_payload(payload, window.gene_id, expected_positions, expected_keys)
            atomic_torch_save(payload, destination)
    for window in windows:
        selected = selected_by_gene[window.gene_id]
        gene_dir = Path(output_root) / safe_gene_dir_name(window)
        meta = {
            "rank": window.rank + 1,
            "gene_id": window.gene_id,
            "chrom": window.chrom,
            "tss": window.tss,
            "start": window.start,
            "end": window.end,
            "n_snps": len(selected),
            "n_samples": len(samples),
            "chromosome_embedding": str(source_path),
            "same_position_alt_variants_retained": True,
        }
        gene_dir.mkdir(parents=True, exist_ok=True)
        temporary = gene_dir / f"meta.json.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(temporary, gene_dir / "meta.json")
        rows.append(meta)
    print(f"done {chrom}: genes={len(windows)} union_snps={len(union_selected)} samples={len(samples)} seconds={time.time()-started:.1f}", flush=True)
    return rows


def write_manifest(rows: Sequence[Mapping], destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "gene_id", "chrom", "tss", "start", "end", "n_snps", "n_samples", "chromosome_embedding", "same_position_alt_variants_retained"]
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-list", type=Path, default=DEFAULT_GENE_LIST)
    parser.add_argument("--expression-matrix", type=Path, default=DEFAULT_EXPRESSION_MATRIX)
    parser.add_argument("--chromosome-root", type=Path, default=DEFAULT_CHROMOSOME_ROOT)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--vcf-root", type=Path, default=DEFAULT_VCF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-genes", type=int, default=250)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    genes = read_top_genes(args.gene_list, args.n_genes)
    windows = read_gene_windows(args.expression_matrix, genes)
    samples = read_samples(args.sample_file, args.vcf_root, args.limit_samples)
    chromosomes = sorted({window.chrom for window in windows}, key=lambda value: int(value.removeprefix("chr")))
    for chrom in chromosomes:
        chromosome_embedding_path(args.chromosome_root, chrom)
    print(json.dumps({"genes": len(windows), "samples": len(samples), "chromosomes": len(chromosomes), "dry_run": args.dry_run}))
    if args.dry_run:
        return 0
    rows = []
    for chrom in chromosomes:
        chrom_windows = [window for window in windows if window.chrom == chrom]
        rows.extend(extract_chromosome(chrom, chrom_windows, samples, args.chromosome_root, args.output_root, args.overwrite))
    rows.sort(key=lambda row: int(row["rank"]))
    write_manifest(rows, args.output_root / "manifest.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
