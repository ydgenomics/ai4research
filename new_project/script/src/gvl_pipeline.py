# -*- coding: utf-8 -*-
"""数据管线：QC/norm → 窗口 BED → gvl.write → Dataset.open/subset。

前置事实：
  - 参考 FASTA: osa1_r7.asm.ch.fa（Chr1-12 有效；ChrUn/ChrSy 需排除）—— 已核实与 251 VCF 同 IRGSP-1.0 R7
  - GVL write 要求 VCF 为 left-aligned / biallelic / atomized → 必须先 bcftools norm
  - 未 phase：用 deterministic=True（GVL 确定性分配），不走 unphased_union
    （unphased_union 与 annotated 输出互斥，见 GVL 文档）—— C1/C2 落为 ploidy=0 链
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("variant_cpt.gvl_pipeline")


def _import_gvl():
    """import genvarloader（loguru 会把 root logger level 提到 WARNING，吞 INFO 日志）。

    用 loguru 前 root level=INFO；import 后变 WARNING → 恢复，保证 variant_cpt.* 日志可见。
    """
    import logging as _logging
    import genvarloader as gvl
    root = _logging.getLogger()
    # loguru 通过 root.handlers/level 修改，这里无条件复位 INFO（幂等）
    root.setLevel(_logging.INFO)
    return gvl


def env_setup(cfg: dict):
    os.environ.setdefault("GVL_NUM_THREADS", str(cfg.get("compute", {}).get("gvl_threads", 8)))


def _shutil_which(name: str):
    from shutil import which
    return which(name)


# ---------- preprocess ----------
def _bcftools_norm(cfg: dict) -> Path:
    d = cfg["data"]
    out_dir = Path(d["out_dir"]) / "norm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "251.norm.vcf.gz"
    if out.exists():
        logger.info(f"norm 产物已存在: {out}")
        return out
    # 1) 类型/长度过滤：SNP + indel<50bp（SV 丢弃）；测试可再限 region
    #    注：纯 SNP 输入(如 251.SNP.final.vcf.gz)header 无 SVLEN，动态跳过长度过滤
    tmp = out_dir / "251.filter.vcf.gz"
    has_svlen = False
    with os.popen(f"gzip -dc '{d['vcf']}' 2>/dev/null | head -1000 | grep -m1 '##INFO=<ID=SVLEN'" ) as f:
        has_svlen = bool(f.read().strip())
    cmd_filter = ["bcftools", "view", "-v", "snps,indels"]
    if has_svlen:
        cmd_filter += ["-i", "MAX(INFO/SVLEN)<=%d" % d["max_sv_len"]]
    if d.get("region"):                     # 如 Chr1:1000000-2000000
        cmd_filter += ["-r", d["region"]]
    cmd_filter += ["-Oz", "-o", str(tmp), d["vcf"]]
    logger.info("运行: " + " ".join(cmd_filter))
    subprocess.run(cmd_filter, check=True)
    # 2) MAF / HWE / 缺失率（plink2）→ 位点列表
    plink_prefix = out_dir / "251_qc"
    cmd_plink = ["plink2", "--vcf", str(tmp), "--maf", str(d["maf"]),
                 "--hwe", "1e-6", "--geno", str(d["max_missing"]),
                 "--write-snplist", "--make-just-pvar",
                 "--out", str(plink_prefix)]
    if _shutil_which("plink2"):
        logger.info("运行 plink2 QC: " + " ".join(cmd_plink))
        subprocess.run(cmd_plink, check=True)
        snplist = Path(str(plink_prefix) + ".snplist")
        tmp2 = out_dir / "251.qc.vcf.gz"
        subprocess.run(["bcftools", "view", "-T", str(snplist),
                        "-Oz", "-o", str(tmp2), str(tmp)], check=True)
        tmp = tmp2
    else:
        logger.warning("plink2 未找到，跳过 MAF/HWE/geno QC（仅做类型/长度过滤）")
    # 3) 归一化：left-align + 拆 multi-allelic（GVL 硬要求）
    subprocess.run(["bcftools", "norm", "-m", "-any", "-f", d["reference_fasta"],
                    "-Oz", "-o", str(out), str(tmp)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(out)], check=True)
    logger.info(f"norm 完成: {out}")
    return out


def _build_bed(cfg: dict, norm_vcf: Path) -> Path:
    """滑窗网格 BED（step=window.step, 长度=window.length, 仅 Chr1-12）。
    若 cfg.data.region 指定（如 Chr1:1000000-2000000）只生成该区间内窗口。"""
    win, step = cfg["window"]["length"], cfg["window"]["step"]
    chroms = cfg["window"]["chroms"]
    region = cfg["data"].get("region")        # 如 Chr1:1000000-2000000
    fai = Path(cfg["data"]["reference_fasta"] + ".fai")
    if not fai.exists():
        subprocess.run(["samtools", "faidx", cfg["data"]["reference_fasta"]], check=True)
    sizes = {}
    with open(fai) as f:
        for line in f:
            c, s, *_ = line.split()
            sizes[c] = int(s)
    # 解析 region → (re_chrom, re_lo, re_hi)
    re_chrom, re_lo, re_hi = None, 0, None
    if region:
        try:
            rc, rr = region.split(":", 1)
            re_chrom = rc
            if "-" in rr:
                re_lo_s, re_hi_s = rr.split("-", 1)
                re_lo, re_hi = int(re_lo_s), int(re_hi_s)
            else:
                re_lo, re_hi = int(rr), None
        except Exception as e:
            logger.warning(f"region 解析失败({e})，忽略 region 限制")
    rows = []
    for c in chroms:
        if re_chrom is not None and c != re_chrom:
            continue                       # region 只落在某条染色体上
        clen = sizes[c]
        hi = re_hi if re_hi is not None else clen
        lo_start = re_lo if (re_chrom == c and re_lo) else 0
        for start in range(lo_start, min(hi, clen - win + 1), step):
            rows.append((c, start, start + win))
    bed = pd.DataFrame(rows, columns=["chrom", "chromStart", "chromEnd"])
    out_dir = Path(cfg["data"]["out_dir"]) / "bed"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{region.replace(':', '_').replace('-', '_')}" if region else ""
    out = out_dir / f"grid_{win}_{step}{suffix}.bed"
    bed.to_csv(out, sep="\t", header=False, index=False)
    logger.info(f"BED 完成: {out} n_rows={len(bed)}")
    return out


def run_preprocess(cfg: dict):
    norm_vcf = _bcftools_norm(cfg)
    _build_bed(cfg, norm_vcf)


# ---------- 路径帮助 ----------
def _bed_path(cfg: dict) -> Path:
    w = cfg["window"]
    region = cfg["data"].get("region")
    suffix = f"_{region.replace(':', '_').replace('-', '_')}" if region else ""
    return Path(cfg["data"]["out_dir"]) / "bed" / f"grid_{w['length']}_{w['step']}{suffix}.bed"


def _gvl_path(cfg: dict) -> Path:
    return Path(cfg["data"]["out_dir"]) / "gvl" / f"gvl_{cfg['window']['length']}.gvl"


# ---------- write ----------
def run_write(cfg: dict):
    gvl = _import_gvl()
    out = _gvl_path(cfg)
    if out.exists():
        logger.info(f"gvl 已存在: {out}")
        return out
    bed = _bed_path(cfg)
    norm_vcf = Path(cfg["data"]["out_dir"]) / "norm" / "251.norm.vcf.gz"
    if not bed.exists() or not norm_vcf.exists():
        run_preprocess(cfg)
    # sample_subset：取前 N 个样本（测试用）
    samples = cfg["data"].get("samples")
    n_sub = cfg["data"].get("sample_subset")
    if n_sub:
        import cyvcf2
        v = cyvcf2.VCF(str(norm_vcf))
        samples = v.samples[: int(n_sub)]
        logger.info(f"sample_subset={n_sub} → 取前 {len(samples)} 个样本")
    gvl.write(
        path=str(out),
        bed=str(bed),
        variants=str(norm_vcf),
        samples=samples,
        max_jitter=0,                     # 固定网格，可复现 + 与下游对齐
        max_mem=str(cfg["compute"].get("gvl_max_mem", "8g")),
    )
    logger.info(f"gvl.write 完成: {out}")
    return out


# ---------- read ----------
def open_dataset(cfg: dict, chroms: list[str]):
    """返回 gvl_ds（annotated + with_len(L)，subset 到 chroms）。
    未 phase → deterministic=True（GVL 确定性单倍型分配）。"""
    gvl = _import_gvl()
    w, d = cfg["window"], cfg["data"]
    ds = gvl.Dataset.open(
        str(_gvl_path(cfg)), reference=d["reference_fasta"],
        jitter=0, deterministic=True,
    )
    ds = ds.with_seqs("annotated")         # (haps, var_idxs, ref_coords)
    ds = ds.with_len(w["length"])          # 固定长度 → ArrayDataset
    ridx = ds.regions["chrom"].is_in(chroms).to_numpy()
    ds = ds.subset_to(regions=ridx)
    # 样本数由 gvl.write 时的 samples 决定（已含 sample_subset 限制），无需重复 subset
    logger.info(f"open_dataset: chroms={len(chroms)} n_regions={ds.n_regions} "
                f"n_samples={ds.n_samples} ploidy={ds.ploidy}")
    return ds