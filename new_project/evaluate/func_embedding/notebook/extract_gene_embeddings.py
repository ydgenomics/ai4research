# -*- coding: utf-8 -*-
"""从 GFF3 + FASTA 提取基因区域 embedding（OGR 水稻单碱基模型）。

解决四个经典问题：
1. padding=True     -> 同 batch 内 token 张量等长（"张量必须等长"）
2. attention_mask   -> 注意力对 pad 位置屏蔽，padding 不参与计算（"padding 污染结果"）
3. mask-aware 池化  -> mean = Σ(hs·mask)/Σ(mask)，pad 位置不拉低均值（"长度不公平"）
4. 排序+动态打包    -> 长度降序 + 每批总 token 限额，padding 率最低、不 OOM（"显存浪费"）

用法：
    python extract_gene_embeddings.py --max-tokens 65536 --flank 0
输出（output/ 目录）：
    gene_embeddings.npy   [N, hidden=1024] float16，与 gene_names.txt 逐行对应
    gene_names.txt        gene_id（与 GFF 顺序一致）
    meta.tsv              gene_id chr start end strand seq_len token_len
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------- 配置 ----------------------------
MODEL = "/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/rice_1B_stage2_8k_hf"
REF   = Path(MODEL).parent / "osa1_r7.asm.ch.fa"
GFF   = Path(MODEL).parent / "osa1_r7.all_models.gff3"
OUT   = Path(__file__).parent / "output"

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# ---------------------------- 模型 ----------------------------
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    logger.info(f"模型加载完成, hidden={model.config.hidden_size}, device={model.device}")
    return tok, model

# ---------------------------- 基因序列 ----------------------------
def load_genes():
    """读 GFF3 的 gene 特征; attributes 提取 ID。"""
    df = pd.read_csv(GFF, sep="\t", comment="#", header=None,
                     usecols=[0, 2, 3, 4, 6, 8],
                     names=["chr", "feature", "start", "end", "strand", "attrs"])
    genes = df[df.feature == "gene"].copy()
    genes["id"] = genes.attrs.str.extract(r"ID=([^;]+)", expand=False)
    genes = genes.drop(columns=["feature", "attrs"])[["id", "chr", "start", "end", "strand"]]
    return genes

def fetch_seq(fa: Fasta, chrom: str, start: int, end: int, strand: str, flank: int) -> str:
    """取含侧翼的基因序列（pyfaidx 切片为 0-based 半开区间），负链反互补。"""
    lo, hi = max(start - flank, 1), end + flank          # 侧翼自然跟随正/负链
    s = str(fa[chrom][lo - 1:hi])
    return s if strand == "+" else s.translate(COMP)[::-1]

def rc(s: str) -> str:
    return s.translate(COMP)[::-1]

# ---------------------------- 打包 ----------------------------
def pack_batches(lengths: np.ndarray, max_tokens: int) -> list[np.ndarray]:
    """贪心装包：长度降序, 每批总 token <= max_tokens。返回 batch 内的原始下标。"""
    order = np.argsort(lengths)[::-1]
    batches, cur, used = [], [], 0
    for i in order:
        if cur and used + lengths[i] > max_tokens:
            batches.append(np.array(cur)); cur, used = [], 0
        cur.append(int(i)); used += int(lengths[i])
    if cur:
        batches.append(np.array(cur))
    return batches

# ---------------------------- 前向 + 池化 ----------------------------
@torch.no_grad()
def embed_batch(ids, mask, model, layer=-1):
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states[layer]                 # [B, T, H]
    m = mask.unsqueeze(-1).to(hs.dtype)           # [B, T, 1]
    emb = (hs * m).sum(1) / m.sum(1).clamp_min(1) # mask-aware mean, 防除零
    return emb.float().cpu().numpy()

# ---------------------------- 主流程 ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank", type=int, default=0, help="基因两侧侧翼 bp")
    ap.add_argument("--max-tokens", type=int, default=65536, help="每批总 token 限额")
    ap.add_argument("--max-len", type=int, default=0, help="单条序列上限, 0=不截断")
    ap.add_argument("--layer", type=int, default=-1, help="取第几层隐藏状态")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load_model()
    fa = Fasta(str(REF))
    genes = load_genes()
    logger.info(f"基因数: {len(genes)}")

    seqs, metas = [], []
    for _, r in genes.iterrows():
        s = fetch_seq(fa, r.chr, r.start, r.end, r.strand, a.flank)
        if a.max_len and len(s) > a.max_len:
            s = s[:a.max_len]
        seqs.append(s)
        metas.append((r.id, r.chr, r.start, r.end, r.strand, len(s)))
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)

    batches = pack_batches(lengths, a.max_tokens)
    pad_rate = 1 - lengths.sum() / sum(len(seqs[i]) for b in batches for i in b)
    logger.info(f"打包完成: {len(seqs)} 条 -> {len(batches)} 批, "
                f"最大序列 {lengths.max():,} bp, 平均 {lengths.mean():,.0f} bp, "
                f"整体 padding 率 {pad_rate:.1%}")

    embs = np.empty((len(seqs), model.config.hidden_size), dtype=np.float16)
    for bi, batch in enumerate(batches):
        enc = tok([seqs[i] for i in batch], padding=True, truncation=True,
                  add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        embs[batch] = embed_batch(ids, mask, model, a.layer)
        if (bi + 1) % 20 == 0 or bi + 1 == len(batches):
            mem = f", 峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if torch.cuda.is_available() else ""
            logger.info(f"批 {bi + 1}/{len(batches)} 完成{mem}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 输出（顺序与 GFF 一致）
    df = pd.DataFrame(metas, columns=["gene_id", "chr", "start", "end", "strand", "seq_len"])
    df["token_len"] = df.seq_len  # 单碱基 tokenizer: 1 bp = 1 token
    df.to_csv(OUT / "meta.tsv", sep="\t", index=False)
    with open(OUT / "gene_names.txt", "w") as f:
        f.write("\n".join(df.gene_id) + "\n")
    np.save(OUT / "gene_embeddings.npy", embs)
    logger.info(f"完成 -> {OUT}/gene_embeddings.npy {embs.shape} ({embs.dtype})")

if __name__ == "__main__":
    main()