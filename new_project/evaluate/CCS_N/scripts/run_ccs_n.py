"""CCS-N 主脚本: 评测 DNA 模型对变异 (SNV) 的感知能力。

流程:
1. 从 GVL 数据集读 reference (ref) 和 haplotypes (alt) 序列对
   - 每个区间对应一个真实 SNV (窗口 BED 中 name 列含 {chrom}:{pos}:{ref}>{alt})
   - haplotypes 应用了目标样本在该位点的真实变异 (GVL 只对携带变异的区间生效)
2. 构建随机对照: 同位置替换为随机碱基 (≠ref), 保持 ref≠alt
3. 模型前向: ref/alt/rand 序列 -> 最后一层 hidden [T, H]
4. 逐 token 距离 (欧氏 + 余弦), 以变异 token 为中心累加 CCS-N (N=0/1/2/4/8/16)
5. 汇总: 每个模型每个 N 的 CCS-N 分布, 与随机对照的区分度 (AUC, ratio)

用法:
    python run_ccs_n.py --model-dir <dir> [--model-dir <dir2> ...] \
        --data-dir ./data --out-dir ./output \
        --batch-size 16 --device cuda:0

注意:
- 只评测"真实携带变异"的区间 (GVL 中该样本 haplotypes != reference 的窗口),
  参考纯合位点的 haplotypes == reference, 无法构造 alt 序列, 自动跳过。
- 随机对照每次运行固定 seed, 保证可复现。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import ccs_n_scores, load_model, token_distances, variant_awareness_scores  # noqa: E402


def _parse_region_name(name: str):
    """解析 BED name 列: {chrom}:{pos}:{ref}>{alt} 或 {prefix}:{chrom}:{pos}:{ref}>{alt}。"""
    parts = name.split(":")
    # 找到形如 ref>alt 的部分, 前面的部分是 chrom, 再前面是 pos
    ref_alt = None
    for i, p in enumerate(parts):
        if ">" in p:
            ref_alt = p
            break
    if ref_alt is None:
        raise ValueError(f"无法解析 name: {name}")
    pos = int(parts[i - 1])
    chrom = parts[i - 2]
    ref_s, alt_s = ref_alt.split(">")[:2]
    return chrom, pos, ref_s, alt_s


def read_gvl_pairs(gvl_dir: Path, ref_fasta: str, sample: str, max_regions: int | None):
    """读 GVL reference/haplotypes 序列对。

    返回: (regions, ref_seqs, alt_seqs)
    - regions: [B] 每窗口信息 (chrom, pos, ref, alt) 来自 BED name
    - ref_seqs: [B] 参考序列 str (无变异)
    - alt_seqs: [B] 单倍体序列 str (含该样本变异, 纯合/杂合取 h0)
    """
    import polars as pl
    import genvarloader as gvl

    # 窗口信息 (来自 input_regions.arrow 的 name + chromStart 列)
    reg = pl.read_ipc(f"{gvl_dir}/input_regions.arrow").select(["name", "chromStart"])
    names = reg["name"].to_list()
    chrom_starts = reg["chromStart"].to_list()

    ds = gvl.Dataset.open(str(gvl_dir), reference=ref_fasta)
    ds_ref = ds.with_seqs("reference")
    ds_alt = ds.with_seqs("haplotypes")

    s_idx = ds.samples.index(sample)
    regions, ref_seqs, alt_seqs, var_idxs = [], [], [], []
    for r in range(ds.n_regions):
        if max_regions is not None and len(regions) >= max_regions:
            break
        name = names[r]
        chrom, pos, ref_s, alt_s = _parse_region_name(name)
        # 变异在窗口内的下标 = VCF POS(1-based) - 1 - chromStart(0-based)
        var_idx = pos - 1 - chrom_starts[r]
        ref = ds_ref[r, s_idx].to_numpy()
        alt = ds_alt[r, s_idx].to_numpy()
        ref_str = b"".join(ref[0]).decode()
        alt_str = b"".join(alt[0]).decode()
        if alt_str == ref_str:
            continue  # 该样本此位点参考纯合, 无 alt 序列
        regions.append((chrom, pos, ref_s, alt_s, chrom_starts[r]))
        ref_seqs.append(ref_str)
        alt_seqs.append(alt_str)
        var_idxs.append(var_idx)
    return regions, ref_seqs, alt_seqs, np.array(var_idxs, dtype=np.int64)


def make_random_controls(ref_seqs, regions, var_idxs, rng: np.random.Generator):
    """随机对照: 同位置替换为随机碱基 (≠ref), 保持 ref≠alt。

    var_idxs: [B] 变异在窗口内的 token 下标 (坐标计算, 可靠)。
    返回: rand_seqs [B] str
    """
    bases = np.array(list("ACGT"))
    rand_seqs = []
    for i, seq in enumerate(ref_seqs):
        chrom, pos, ref_b, alt_b, chrom_start = regions[i]
        var_i = var_idxs[i]
        seq_list = list(seq)
        choices = bases[bases != ref_b]
        new_b = str(rng.choice(choices))
        seq_list[var_i] = new_b
        rand_seqs.append("".join(seq_list))
    return rand_seqs


def _find_variant_idx(region) -> int:
    """变异在窗口内的 token 下标: VCF POS(1-based) - 1 - chromStart(0-based)。"""
    chrom, pos, ref_b, alt_b, chrom_start = region
    return pos - 1 - chrom_start


def run_model(model_dir: str, device: str, batch_size: int, ref_seqs, alt_seqs, rand_seqs, var_idxs,
              n_list=(0, 1, 2, 4, 8, 16), tokenizer_type: str = "single_base",
              model_class: str = "causal_lm", save_per_site_dir: str | None = None,
              save_distance_profiles: bool = False):
    """对 ref/alt/rand 三组序列前向, 计算逐 token 距离和 CCS-N。

    tokenizer_type: 'single_base' 用 var_idxs 坐标映射 (精确);
                    'kmer_diff' 用 ref/alt token id 差分定位 (自动, 适用 k-mer/合并tokenizer)。
    model_class:   'causal_lm' 用 AutoModelForCausalLM; 'auto' 用 AutoModel (ESM 双向等)。
    save_per_site_dir: 若提供, 保存 per-site 分数到该目录 (npz: real/rand 每个 metric 每个 N 的 [B] 数组)。
    save_distance_profiles: 若 True, 额外保存每个位点以变异 token 为中心的逐 token 距离剖面
                            (euclidean_real/euclidean_rand: [B, 2R+1], R=16)。
    """
    from utils import embed_sequences, find_variant_token

    tok, model = load_model(model_dir, device, model_class=model_class)
    N_list = tuple(n_list)
    R_MAX = 128  # 剖面半宽 (与最大 N 对齐, 扩展到上下 128)
    # 累积 CCS-N: {metric: {N: array}}
    agg = {m: {n: [] for n in N_list} for m in ["euclidean", "cosine"]}
    prof = {"euclidean_real": [], "euclidean_rand": []} if save_distance_profiles else None
    n_total = len(ref_seqs)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        b_ref, b_alt, b_rand = ref_seqs[start:end], alt_seqs[start:end], rand_seqs[start:end]
        b_var = var_idxs[start:end]

        # 分别前向
        ref_h, ref_ids, ref_mask = embed_sequences(tok, model, b_ref, device)
        alt_h, alt_ids, alt_mask = embed_sequences(tok, model, b_alt, device)
        rand_h, rand_ids, rand_mask = embed_sequences(tok, model, b_rand, device)

        # 变异 token 定位
        if tokenizer_type == "kmer_diff":
            b_var = find_variant_token(ref_ids, alt_ids, b_var)
            assert (b_var >= 0).all(), "存在未定位到变异的序列"

        # 逐 token 距离 (忽略 padding)
        d_real = token_distances(ref_h, alt_h)
        d_rand = token_distances(ref_h, rand_h)

        if prof is not None:
            # 以变异 token 为中心切剖面 [B, 2R+1] (边界处裁剪, 不足处忽略)
            for key, d in (("euclidean_real", d_real["euclidean"]),
                           ("euclidean_rand", d_rand["euclidean"])):
                rows = []
                for i in range(d.shape[0]):
                    v = b_var[i]
                    lo, hi = max(0, v - R_MAX), min(d.shape[1], v + R_MAX + 1)
                    row = np.full(2 * R_MAX + 1, np.nan)
                    row[R_MAX - (v - lo): R_MAX + (hi - v)] = d[i, lo:hi]
                    rows.append(row)
                prof[key].append(np.stack(rows))

        for metric in ["euclidean", "cosine"]:
            ccs_real = ccs_n_scores(d_real[metric], b_var, N_list)
            ccs_rand = ccs_n_scores(d_rand[metric], b_var, N_list)
            for n in N_list:
                agg[metric][n].append((ccs_real[n], ccs_rand[n]))
        if (start // batch_size) % 20 == 0:
            print(f"  [run] {model_dir.split('/')[-1]} batch {start}-{end}/{n_total}")

    # 合并
    out = {}
    for metric in ["euclidean", "cosine"]:
        for n in N_list:
            real = np.concatenate([x[0] for x in agg[metric][n]])
            rand = np.concatenate([x[1] for x in agg[metric][n]])
            out[(metric, n)] = {"real": real, "rand": rand}

    # 保存 per-site 分数
    if save_per_site_dir is not None:
        sp = Path(save_per_site_dir)
        sp.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sp / "ccs_n_per_site.npz",
            **{f"real_{m}_N{n}": out[(m, n)]["real"] for m in ["euclidean", "cosine"] for n in N_list},
            **{f"rand_{m}_N{n}": out[(m, n)]["rand"] for m in ["euclidean", "cosine"] for n in N_list},
        )
        print(f"  [run] per-site 分数已保存: {sp / 'ccs_n_per_site.npz'}")
        if prof is not None:
            np.savez_compressed(
                sp / "distance_profiles.npz",
                **{k: np.concatenate(v) for k, v in prof.items()},
                var_idx=var_idxs,
            )
            print(f"  [run] 距离剖面已保存: {sp / 'distance_profiles.npz'}")

    # 释放模型 + GPU 显存 (避免多模型累积 OOM)
    import gc
    import torch as _torch

    del model, tok
    gc.collect()
    if device.startswith("cuda"):
        _torch.cuda.empty_cache()
    return out


def main(args=None):
    """CCS-N 评测主流程。args: 命令行解析的 Namespace 或 run_eval.py 注入的 Namespace。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", action="append", required=True, help="模型目录, 可多次指定")
    ap.add_argument("--data-dir", default=str(Path(__file__).parent.parent / "data"), help="GVL 数据目录")
    ap.add_argument("--gvl-dir", default=None, help="GVL 数据集目录 (默认 <data-dir>/gvl)")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "output"), help="结果输出目录")
    ap.add_argument("--ref", required=True, help="参考基因组 FASTA")
    ap.add_argument("--sample", default="NH001", help="目标样本")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-regions", type=int, default=None, help="最多评测区间数 (冒烟测试)")
    ap.add_argument("--seed", type=int, default=42, help="随机对照 seed")
    ap.add_argument("--n-list", nargs="*", type=int, default=None, help="CCS-N 累加半径列表 (默认 0 1 2 4 8 16)")
    ap.add_argument("--save-per-site", action="store_true", default=False,
                    help="保存 per-site CCS-N 分数到 out_dir/per_site/<model>/ (供热图分析)")
    ap.add_argument("--save-distance-profiles", action="store_true", default=False,
                    help="额外保存以变异 token 为中心的逐 token 距离剖面 (需 --save-per-site)")
    # run_eval.py 注入 Namespace 时直接用; 命令行方式用 parse_args
    if args is None or not isinstance(args, argparse.Namespace):
        args = ap.parse_args(args)
    # 兼容 run_eval.py 注入的 Namespace (无 n_list 属性时用默认)
    if not hasattr(args, "n_list") or args.n_list is None:
        args.n_list = (0, 1, 2, 4, 8, 16)

    data_dir = Path(args.data_dir)
    gvl_dir = Path(args.gvl_dir) if args.gvl_dir else data_dir / "gvl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读 GVL 序列对
    print(f"[run] 读取 GVL 序列对: {gvl_dir}")
    regions, ref_seqs, alt_seqs, var_idxs = read_gvl_pairs(gvl_dir, args.ref, args.sample, args.max_regions)
    print(f"[run] 真实变异区间: {len(ref_seqs)}")
    if not ref_seqs:
        print("[run] 无携带变异的区间, 退出")
        sys.exit(1)

    # 2. 随机对照
    rng = np.random.default_rng(args.seed)
    rand_seqs = make_random_controls(ref_seqs, regions, var_idxs, rng)
    print(f"[run] 随机对照序列: {len(rand_seqs)}")

    # 4. 每模型运行
    all_results = {}
    for md in args.model_dir:
        # 支持 yaml 里 models: [{path: ..., tokenizer_type: ..., model_class: ...}] 或纯字符串路径
        ttype = "single_base"
        mclass = "causal_lm"
        if isinstance(md, dict):
            ttype = md.get("tokenizer_type", "single_base")
            mclass = md.get("model_class", "causal_lm")
            md = md["path"]
        name = Path(md).name
        print(f"[run] ==== 模型 {name} (tokenizer_type={ttype}, model_class={mclass}) ====")
        per_site_dir = None
        if getattr(args, "save_per_site", False):
            per_site_dir = out_dir / "per_site" / name
        res = run_model(md, args.device, args.batch_size, ref_seqs, alt_seqs, rand_seqs,
                        var_idxs, args.n_list, tokenizer_type=ttype, model_class=mclass,
                        save_per_site_dir=per_site_dir,
                        save_distance_profiles=getattr(args, "save_distance_profiles", False))
        all_results[name] = res

    # 5. 汇总输出
    summary = {"sample": args.sample, "n_regions": len(ref_seqs), "n_list": list(args.n_list), "models": {}}
    for name, res in all_results.items():
        summary["models"][name] = {}
        for (metric, n), v in res.items():
            scores = variant_awareness_scores(v["real"], v["rand"])
            # AUC 同时报原始值和 1-auc (mean_real<mean_rand 时 1-auc 为区分强度)
            scores["auc_flip"] = 1.0 - scores["auc"]
            summary["models"][name][f"{metric}_N{n}"] = scores
            print(f"  {name} {metric} N={n}: mean_real={scores['mean_real']:.4f} "
                  f"mean_rand={scores['mean_rand']:.4f} ratio={scores['ratio']:.2f} "
                  f"auc={scores['auc']:.3f} 1-auc={scores['auc_flip']:.3f}")

    out_json = out_dir / "ccs_n_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[run] 汇总已写入 {out_json}")


if __name__ == "__main__":
    main()
