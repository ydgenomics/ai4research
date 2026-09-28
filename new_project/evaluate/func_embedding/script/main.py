# -*- coding: utf-8 -*-
"""多模型基因区域 embedding 提取 —— 唯一入口。

用法:
    python main.py --run configs/run_example.yaml
    python main.py --run configs/run_example.yaml --models rice_1B_stage2_8k NTv3_650M_pre
    # 测试: 只取前 500 条基因 / 前 10%
    python main.py --run configs/run_example.yaml --max-genes 500
    python main.py --run configs/run_example.yaml --gene-frac 0.1
    # 多卡: 串行把不同模型依次放到不同卡, 或并行同时跑
    python main.py --run configs/run_example.yaml --gpus 0 1            # 多卡串行(默认)
    python main.py --run configs/run_example.yaml --gpus 0 1 --parallel # 多卡并行(每卡一进程)
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ogr_extract.adapter import get_adapter
from ogr_extract.embedder import embed_batch, validate_pooling
from ogr_extract.genome import (extract_sequences, load_genes,
                                load_genes_from_csv, normalize_chrom)
from ogr_extract.loader import load_model, model_device
from ogr_extract.output import is_done, save_run
from ogr_extract.packing import pack_batches, pack_batches_by_count, padding_ratio
from ogr_extract.registry import load_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_path(p: str, base: Path) -> Path:
    """相对路径以 run yaml 所在目录为基准解析。"""
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def parse_run_yaml(run_yaml: Path) -> dict:
    with open(run_yaml, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pooling_desc(pooling: str, pool_k: int, pool_trim: float,
                  l2_normalize: bool) -> str:
    """把池化配置拼成一行可读日志, 如 ``mean_last_k(k=64)+L2``。"""
    desc = pooling
    if pooling == "mean_last_k":
        desc += f"(k={pool_k})"
    elif pooling == "mean_trim":
        desc += f"(trim={pool_trim})"
    return desc + ("+L2" if l2_normalize else "")


def run_one_model(name: str, entry, run_cfg: dict, genes, seqs, run_dir: Path,
                  device: str) -> None:
    t0 = time.time()
    layer = entry.effective_layer(run_cfg.get("layer", -1))
    batch_size = entry.effective_batch_size(run_cfg.get("batch_size", 0))
    # 池化: run 级配置 + 模型级覆盖（causal 用 last / kmer6_cls 用 first 等）
    pooling = entry.effective_pooling(run_cfg.get("pooling", "mean"))
    pool_k = int(run_cfg.get("pool_k", 64))
    pool_trim = float(run_cfg.get("pool_trim", 0.25))
    l2_normalize = bool(run_cfg.get("l2_normalize", False))
    validate_pooling(pooling, pool_k, pool_trim)
    out_dir = run_dir / name

    # 影响 embedding 数值的配置: 与产物不一致时不跳过，避免沿用旧池化的结果
    expect = {"pooling": pooling, "pool_k": pool_k, "pool_trim": pool_trim,
              "l2_normalize": l2_normalize, "layer": layer}
    if run_cfg.get("resume", True) and is_done(out_dir, expect):
        logger.info(f"[{name}] 已存在 gene_embeddings.npy 且配置一致, 跳过(断点续跑)")
        return

    # 加载模型 + 适配器
    model, tokenizer = load_model(entry, device)
    adapter = get_adapter(entry)
    dev = model_device(model)          # 可靠设备 (device_map 场景)

    # 预处理序列（每模型独立规则：k-mer 补齐、大小写、上下文截断）
    prep_seqs, truncated_flags = [], []
    for s in seqs:
        s = adapter.prepare_seq(s, run_cfg.get("max_len") or None)
        s, is_trunc = adapter.truncate_to_context(s)
        prep_seqs.append(s)
        truncated_flags.append(is_trunc)
    lengths = np.array([len(s) for s in prep_seqs], dtype=np.int64)

    # 打包：batch_size>0 用固定条数; 否则 max_tokens 动态打包
    if batch_size > 0:
        batches = pack_batches_by_count(lengths, batch_size)
        pack_desc = f"固定 batch_size={batch_size}"
    else:
        batches = pack_batches(lengths, entry.max_tokens)
        pack_desc = f"max_tokens={entry.max_tokens}"
    logger.info(
        f"[{name}] {len(prep_seqs)} 条 -> {len(batches)} 批 ({pack_desc}) | "
        f"最长 {lengths.max():,} bp | 平均 {lengths.mean():,.0f} bp | "
        f"padding 率 {padding_ratio(lengths, batches):.1%} | "
        f"截断 {sum(truncated_flags)} 条 | 池化 {_pooling_desc(pooling, pool_k, pool_trim, l2_normalize)}"
    )

    # 逐批前向 + 池化
    embs = np.empty((len(prep_seqs), entry.hidden_size), dtype=np.float32)
    tokenizer_kwargs = adapter.tokenize_kwargs()
    for bi, batch in enumerate(batches):
        enc = tokenizer(
            [prep_seqs[i] for i in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        )
        ids = enc["input_ids"].to(dev)
        if "attention_mask" in enc:
            mask = enc["attention_mask"].to(dev)
        else:
            # 模型 tokenizer 不返回 attention_mask(NTv3 等): 用 pad_token_id 构造
            pad_id = tokenizer.pad_token_id
            mask = (ids != pad_id).long() if pad_id is not None else torch.ones_like(ids)
        # NTv3 等 U-Net 卷积 encoder: 长度需为 pad_to_multiple 的倍数(否则残差连接
        # 对不齐, 如 1216bp -> 下采样 2^7 后 36 vs 37)。右补齐, mask 同步置 0(不污染池化)。
        pad_m = getattr(entry, "pad_to_multiple", None)
        if pad_m and ids.shape[1] % pad_m != 0:
            pad_len = pad_m - ids.shape[1] % pad_m
            pad_id_use = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=pad_id_use)
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        embs[batch] = embed_batch(ids, mask, model, layer,
                                  pooling=pooling, pool_k=pool_k,
                                  pool_trim=pool_trim, l2_normalize=l2_normalize,
                                  use_attention_mask=getattr(entry, "attention_mask", True))
        del enc, ids, mask
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if (bi + 1) % 20 == 0 or bi + 1 == len(batches):
            logger.info(f"[{name}] 批 {bi + 1}/{len(batches)}")

    # meta 信息：基因坐标 + 该模型看到的真实长度/截断
    meta = genes[["id", "chr", "start", "end", "strand"]].copy()
    meta["seq_len"] = [len(s) for s in prep_seqs]
    meta["truncated"] = truncated_flags

    info = {
        "model_name": name,
        "model_path": entry.path,
        "arch": entry.arch,
        "tokenizer": entry.tokenizer,
        "max_context": entry.max_context,
        "layer": layer,
        "pooling": pooling,
        "pool_k": pool_k,
        "pool_trim": pool_trim,
        "l2_normalize": l2_normalize,
        "device": str(dev),
        "hidden_size": entry.hidden_size,
        "num_layers": entry.num_layers,
        "batch_size": batch_size,
        "max_tokens": entry.max_tokens,
        "n_genes": len(genes),
        "n_batches": len(batches),
        "n_truncated": sum(truncated_flags),
        "elapsed_s": round(time.time() - t0, 1),
    }
    save_run(embs, meta, info, out_dir)
    logger.info(f"[{name}] 完成 {time.time() - t0:.1f}s")


# ---------------- 多卡并行（每 worker 进程独占一张卡, 顺序处理分配到的模型） ----------------
def _worker_parallel(gpu_id: int, tasks: list, run_cfg: dict, genes, seqs, run_dir: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"[GPU {gpu_id}] worker 启动, 承担 {len(tasks)} 个模型")
    for name, entry in tasks:
        try:
            run_one_model(name, entry, run_cfg, genes, seqs, run_dir, device="cuda")
        except Exception as e:
            logger.error(f"[{name}] (GPU {gpu_id}) 失败: {e}", exc_info=True)


def run_parallel(selected: dict, run_cfg: dict, genes, seqs, run_dir: Path, gpus: list[int]) -> None:
    """多卡并行: 每个 GPU 一个进程, 模型按序轮询分配。"""
    ctx = mp.get_context("fork")          # fork: 共享基因/序列内存(COW), 不重复 pickle
    n_workers = min(len(gpus), len(selected))
    per_worker: list[list] = [[] for _ in range(n_workers)]
    for i, (name, entry) in enumerate(selected.items()):
        per_worker[i % n_workers].append((name, entry))

    procs = []
    for wid, tasks in enumerate(per_worker):
        if not tasks:
            continue
        p = ctx.Process(target=_worker_parallel,
                        args=(gpus[wid], tasks, run_cfg, genes, seqs, run_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    logger.info("多卡并行运行结束")


def run_serial(selected: dict, run_cfg: dict, genes, seqs, run_dir: Path, gpus: list[int]) -> None:
    """多卡串行: 同一进程依次加载模型, 第 i 个模型放 gpus[i % len(gpus)]。"""
    for i, (name, entry) in enumerate(selected.items()):
        if entry.device:
            device = entry.device          # 模型级 device 优先
        elif gpus:
            device = f"cuda:{gpus[i % len(gpus)]}"
        else:
            device = run_cfg.get("device", "cuda")
        try:
            run_one_model(name, entry, run_cfg, genes, seqs, run_dir, device=device)
        except Exception as e:
            logger.error(f"[{name}] 失败: {e}", exc_info=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description="多模型基因区域 embedding 提取")
    ap.add_argument("--run", type=str, default="configs/run_example.yaml", help="run yaml 路径")
    ap.add_argument("--registry", type=str, default=None,
                    help="模型注册表 yaml 路径(覆盖 run yaml 的 models_registry)")
    ap.add_argument("--models", nargs="*", default=None, help="只跑指定模型(空格分隔, 覆盖 run yaml)")
    ap.add_argument("--max-genes", type=int, default=None, help="只取 GFF 前 N 条基因(测试用)")
    ap.add_argument("--gene-frac", type=float, default=None, help="只取 GFF 前 fraction(如 0.1=前10%%)")
    ap.add_argument("--csv", type=str, default=None,
                    help="面板 CSV 路径(如 all_7class.csv); 提供后只提取 CSV 里的基因, 不再读 gff")
    ap.add_argument("--id-col", type=str, default="MSU",
                    help="CSV 中代表基因 ID 的列名(默认 MSU, 即 LOC_Os..; 空值时回退 gene_id/RAPdb)")
    ap.add_argument("--csv-coord-gff", type=str, default=None,
                    help="CSV 坐标参考 GFF(IRGSP-1.0 注释, ID=Os..); 缺省用 run yaml 的 gff")
    ap.add_argument("--csv-source-gff", type=str, default=None,
                    help="MSU 版本 GFF(osa1_r7.all_models.gff3, ID=LOC_..); 提供时坐标以它为准")
    ap.add_argument("--gpus", nargs="*", type=int, default=None, help="使用的 GPU id 列表, 如 0 1")
    ap.add_argument("--parallel", action="store_true", help="多卡并行(每卡一进程); 默认多卡串行")
    ap.add_argument("--serial", action="store_true", help="强制单卡串行(忽略 gpus)")
    args = ap.parse_args()

    run_yaml = Path(args.run)
    base_dir = run_yaml.parent
    run_cfg = parse_run_yaml(run_yaml).get("run") or {}

    # 注册表路径：--registry > run yaml 的 models_registry > 默认(run yaml 同目录 models.yaml)
    if args.registry:
        registry_path = resolve_path(args.registry, base_dir)
    else:
        registry_path = resolve_path(
            run_cfg.get("models_registry", "models.yaml"), base_dir)
    registry = load_registry(registry_path)

    models = args.models or run_cfg.get("models")
    if models:
        missing = [m for m in models if m not in registry]
        if missing:
            raise ValueError(f"未注册的模型: {missing} (可选 {list(registry)})")
        selected = {m: registry[m] for m in models}
    else:
        selected = registry
    logger.info(f"本次运行模型: {list(selected)}")

    # 提前校验池化配置（run 级 + 模型级覆盖），免得起完进程才报错
    run_pooling = run_cfg.get("pooling", "mean")
    pool_k = int(run_cfg.get("pool_k", 64))
    pool_trim = float(run_cfg.get("pool_trim", 0.25))
    validate_pooling(run_pooling, pool_k, pool_trim)
    pooling_by_model = {}
    for m, e in selected.items():
        p = e.effective_pooling(run_pooling)
        if p != run_pooling:
            validate_pooling(p, pool_k, pool_trim)      # 模型级覆盖的名字也校验
        pooling_by_model[m] = p
    logger.info(f"池化方式: {pooling_by_model}")

    # 基因数据只加载一次，所有模型共用（保证顺序一致 = 跨模型 index 对齐）
    fasta = resolve_path(run_cfg["fasta"], base_dir)
    csv_path = args.csv or run_cfg.get("csv")
    if csv_path:
        # 面板 CSV 模式: 只提取 CSV 里列出的基因(id 用 MSU 列, 空值回退 gene_id)
        csv_path = resolve_path(csv_path, base_dir)
        id_col = args.id_col or run_cfg.get("id_col", "MSU")
        coord_gff = resolve_path(args.csv_coord_gff or run_cfg.get("csv_coord_gff",
                                run_cfg.get("gff")), base_dir)
        src_gff = args.csv_source_gff or run_cfg.get("csv_source_gff")
        source_gff = resolve_path(src_gff, base_dir) if src_gff else None
        genes = load_genes_from_csv(str(csv_path), id_col=id_col,
                                    coord_gff=str(coord_gff) if coord_gff.exists() else None,
                                    source_gff=str(source_gff) if source_gff else None)
        genes = normalize_chrom(genes, str(fasta))
        logger.info(f"面板 CSV 模式: {csv_path} (id 列: {id_col})")
    else:
        gff = resolve_path(run_cfg["gff"], base_dir)
        genes = load_genes(str(gff), feature=run_cfg.get("feature", "gene"))
        genes = normalize_chrom(genes, str(fasta))
    seqs = extract_sequences(genes, str(fasta), run_cfg.get("flank", 0), run_cfg.get("max_len", 0))
    genes["seq_len"] = [len(s) for s in seqs]
    logger.info(f"基因序列提取完成: {len(seqs)} 条")

    # 可选子集（前 N 条 / 前 frac）—— 冒烟测试
    n_total = len(seqs)
    n_sub = n_total
    if args.max_genes is not None:
        n_sub = min(n_sub, args.max_genes)
    elif run_cfg.get("max_genes"):
        n_sub = min(n_sub, int(run_cfg["max_genes"]))
    if args.gene_frac is not None:
        n_sub = min(n_sub, int(n_total * args.gene_frac))
    elif run_cfg.get("gene_frac"):
        n_sub = min(n_sub, int(n_total * run_cfg["gene_frac"]))
    if n_sub < n_total:
        genes = genes.iloc[:n_sub].reset_index(drop=True)
        seqs = seqs[:n_sub]
        logger.info(f"仅跑前 {n_sub} 条基因({n_sub / n_total:.1%}) 用于测试")

    run_dir = resolve_path(run_cfg.get("output_dir", "./output"), base_dir)

    # 多卡决策
    gpus = args.gpus or run_cfg.get("gpus")
    if args.serial or not gpus or len(gpus) < 1:
        # 单卡(或未配 gpus): 串行, device 取自 run_cfg
        run_serial(selected, run_cfg, genes, seqs, run_dir, gpus=[])
    elif len(gpus) > 1 and (args.parallel or run_cfg.get("parallel", False)) and len(selected) > 1:
        run_parallel(selected, run_cfg, genes, seqs, run_dir, gpus)
    else:
        run_serial(selected, run_cfg, genes, seqs, run_dir, gpus)


if __name__ == "__main__":
    main()