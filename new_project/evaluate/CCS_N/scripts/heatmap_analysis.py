"""CCS-N 热图 + 变异位点 × 模型 得分可视化。

数据来源 (run_ccs_n.py --save-per-site 生成):
    <out_dir>/per_site/<model>/ccs_n_per_site.npz
    键: real_euclidean_N{n}, real_cosine_N{n}, rand_euclidean_N{n}, rand_cosine_N{n} ([B] per-site)
    <out_dir>/per_site/<model>/distance_profiles.npz
    键: euclidean_real, euclidean_rand ([B, 2R+1]), var_idx

输出图:
1. heatmap_model_site.png  — 模型 × 位点 per-site CCS-N (euclidean N=16), 位点按平均得分排序
2. heatmap_model_site_auc.png — 模型 × 位点 per-site AUC (每个位点 real vs rand 分布近似为得分差)
3. heatmap_distance_profile.png — 模型 × token 偏移的平均距离剖面 (real / rand)
4. heatmap_corr.png — 模型间 per-site 得分相关性 (N=16)

用法:
    python scripts/heatmap_analysis.py output/hom_1000_multi7 [--site-meta variant_stats.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_per_site(out_dir: Path, n_list=(0, 1, 2, 4, 8, 16), metric="euclidean", n=16):
    """加载所有模型的 per-site 分数, 返回 (model_names, real [M,B], rand [M,B])。"""
    ps_dir = out_dir / "per_site"
    models, reals, rands = [], [], []
    for d in sorted(ps_dir.iterdir()):
        if not d.is_dir():
            continue
        npz = d / "ccs_n_per_site.npz"
        if not npz.exists():
            print(f"[warn] 缺少 {npz}, 跳过 {d.name}")
            continue
        z = np.load(npz)
        models.append(d.name)
        reals.append(z[f"real_{metric}_N{n}"])
        rands.append(z[f"rand_{metric}_N{n}"])
    if not models:
        raise FileNotFoundError(f"per_site 目录无数据: {ps_dir}")
    return models, np.stack(reals), np.stack(rands)


def load_profiles(out_dir: Path):
    """加载所有模型的平均距离剖面。返回 (model_names, real [M,2R+1], rand [M,2R+1])。"""
    ps_dir = out_dir / "per_site"
    models, reals, rands = [], [], []
    for d in sorted(ps_dir.iterdir()):
        if not d.is_dir():
            continue
        npz = d / "distance_profiles.npz"
        if not npz.exists():
            print(f"[warn] 缺少距离剖面 {npz}, 跳过 {d.name}")
            continue
        z = np.load(npz)
        models.append(d.name)
        reals.append(np.nanmean(z["euclidean_real"], axis=0))
        rands.append(np.nanmean(z["euclidean_rand"], axis=0))
    if not models:
        raise FileNotFoundError(f"距离剖面无数据: {ps_dir}")
    return models, np.stack(reals), np.stack(rands)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="评测输出目录 (含 per_site/)")
    ap.add_argument("--site-meta", default=None, help="variant_stats.json (用于排序/标注位点属性)")
    ap.add_argument("--n", type=int, default=16, help="CCS-N 半径 (默认16)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_list = (0, 1, 2, 4, 8, 16)
    N = args.n

    # ---------- 1. 模型 × 位点 per-site CCS-N (euclidean N=16) ----------
    print("[heat] 加载 per-site 分数 ...")
    models, real, rand = load_per_site(out_dir, n_list, "euclidean", N)
    M, B = real.shape
    print(f"[heat] {M} 模型 × {B} 位点")

    # 位点排序: 按所有模型平均 CCS-N 升序 (从左到右难度递增 → 得分从低到高)
    order = np.argsort(real.mean(axis=0))
    X = real[:, order]

    fig, ax = plt.subplots(figsize=(max(10, B / 100 * 2 + 2), 6))
    vmax = np.nanpercentile(X, 95)
    vmin = np.nanpercentile(X, 5)
    im = ax.imshow(X, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(M))
    ax.set_yticklabels([m[:28] for m in models])
    ax.set_xticks([])
    ax.set_xlabel(f"sites sorted by mean CCS-N (n={B})")
    ax.set_title(f"Per-site CCS-N (euclidean, N={N}) — real variant effect")
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("CCS-N")
    fig.tight_layout()
    p1 = out_dir / "heatmap_model_site.png"
    fig.savefig(p1, dpi=150)
    print(f"[heat] 已保存 {p1}")
    plt.close(fig)

    # ---------- 2. 模型 × 位点 per-site 区分度 (real - rand) ----------
    D = real - rand  # [M, B] per-site 区分度
    order2 = np.argsort(D.mean(axis=0))
    X2 = D[:, order2]
    fig, ax = plt.subplots(figsize=(max(10, B / 100 * 2 + 2), 6))
    lim = np.nanpercentile(np.abs(X2), 95)
    im = ax.imshow(X2, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_yticks(range(M))
    ax.set_yticklabels([m[:28] for m in models])
    ax.set_xticks([])
    ax.set_xlabel(f"sites sorted by mean (real-rand) (n={B})")
    ax.set_title(f"Per-site variant discrimination (real − random), euclidean N={N}")
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("CCS-N real − rand")
    fig.tight_layout()
    p2 = out_dir / "heatmap_model_site_disc.png"
    fig.savefig(p2, dpi=150)
    print(f"[heat] 已保存 {p2}")
    plt.close(fig)

    # ---------- 3. 模型 × token 偏移 平均距离剖面 ----------
    # 方案 B (rand 归一化): 每行除以该模型自己的 rand 剖面, 消除模型间表征范数差异。
    # real/rand > 1 → 真实变异在该 token 位置的表征变化超出随机替换 (区分信号);
    # real/rand ≈ 1 → 无区分; < 1 → 真实变异响应反而低于随机基线。
    # 注意: rand 剖面本身含模型自身的背景响应形状, 归一化后各模型可比。
    prof_dir = out_dir / "per_site"
    if any((d / "distance_profiles.npz").exists() for d in prof_dir.iterdir() if d.is_dir()):
        print("[heat] 加载距离剖面 ...")
        pmodels, preal, prand = load_profiles(out_dir)
        R = (preal.shape[1] - 1) // 2
        offs = np.arange(-R, R + 1)
        # real/rand 比值剖面 (方案 B)
        # 注意: 因果模型上游 rand 剖面≈0 (单向注意力看不到下游变异) → 比值爆炸。
        # 处理: 按模型加尺度匹配的 epsilon (rand 中位数 × 1e-3), 用 log2 压缩动态范围,
        #       并裁剪到 log2(0.25)~log2(4) = [-2, +2], 使上下对称且中心信号不被边缘淹没。
        pratio = np.full_like(preal, np.nan)
        for i in range(len(pmodels)):
            row_eps = max(np.nanmedian(prand[i]) * 1e-3, 1e-12)
            pratio[i] = np.log2((preal[i] + row_eps) / (prand[i] + row_eps))
        pratio = np.clip(pratio, -2.0, 2.0)
        pratio = np.nan_to_num(pratio, nan=0.0)

        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        # 上图: log2(real/rand) 比值 (核心图, 模型间可比)
        ax = axes[0]
        im = ax.imshow(pratio, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-2.0, vmax=2.0))
        ax.set_yticks(range(len(pmodels)))
        ax.set_yticklabels([m[:28] for m in pmodels])
        ax.set_xticks(range(0, len(offs), 4))
        ax.set_xticklabels(offs[::4])
        ax.set_xlabel("token offset from variant")
        ax.set_title("Mean log2(real/rand) distance per token (euclidean) — comparable across models")
        cb = fig.colorbar(im, ax=ax, fraction=0.025)
        cb.set_label("log2(real / rand distance ratio)")
        # 下图: 绝对距离剖面 (原始尺度, 展示各模型表征范数差异)
        ax = axes[1]
        mat = preal
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(pmodels)))
        ax.set_yticklabels([m[:28] for m in pmodels])
        ax.set_xticks(range(0, len(offs), 4))
        ax.set_xticklabels(offs[::4])
        ax.set_xlabel("token offset from variant")
        ax.set_title("Mean absolute distance profile — ref vs alt (real, euclidean, raw scale)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        p3 = out_dir / "heatmap_distance_profile.png"
        fig.savefig(p3, dpi=150)
        print(f"[heat] 已保存 {p3}")
        plt.close(fig)

    # ---------- 4. 模型间 per-site 得分相关性 (N=16) ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    # 用 real 得分的秩相关 (Spearman 近似, 对分布敏感度低)
    from scipy.stats import spearmanr
    corr = np.ones((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            c = spearmanr(real[i], real[j]).statistic
            corr[i, j] = corr[j, i] = c
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    ax.set_xticklabels([m[:14] for m in models], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([m[:14] for m in models], fontsize=7)
    for i in range(M):
        for j in range(M):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(f"Model agreement — per-site CCS-N Spearman corr (N={N})")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    p4 = out_dir / "heatmap_model_corr.png"
    fig.savefig(p4, dpi=150)
    print(f"[heat] 已保存 {p4}")
    plt.close(fig)

    # ---------- 5. 按位点属性分组的 per-site 得分 (Ti/Tv, AF 分箱) ----------
    # 位点属性: 从 BED name + variant_stats.json 关联 (顺序与 GVL 一致)
    summary = {"per_site_shape": [M, B]}
    bed_path = out_dir.parent.parent / "data" / "carrier_hom.bed"
    # 加载 AF (从 variant_stats 所在目录的原始数据重建; 若无则跳过)
    af_vals = None
    if args.site_meta and Path(args.site_meta).exists():
        # 从 BED + VCF 重新解析 AF
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent))
            from variant_stats import load_vcf_info, parse_bed
            _bed = out_dir.parent.parent / "data" / "carrier_hom.bed"
            _vcf = out_dir.parent.parent / "data" / "251.SNP.bsnp.vcf.gz"
            if _bed.exists() and _vcf.exists():
                _regions = parse_bed(str(_bed))
                _info = load_vcf_info(str(_vcf), regions=_regions)
                af_vals = np.array([_info.get((r[0], r[4]), (None, None, None))[2]
                                    for r in _regions], dtype=float)
                af_vals = np.where(np.isnan(af_vals), np.nanmedian(af_vals), af_vals)
                print(f"[heat] AF 已加载 (n={len(af_vals)})")
        except Exception as e:
            print(f"[warn] AF 加载失败: {e}")

    if bed_path.exists():
        # 读 BED name 解析 ref>alt → Ti/Tv 标签
        labels = []
        with open(bed_path) as f:
            for line in f:
                name = line.strip().split("\t")[3]
                ref_alt = next(x for x in name.split(":") if ">" in x)
                r, a = ref_alt.split(">")[:2]
                labels.append("Ti" if {r, a} in ({"A", "G"}, {"C", "T"}) else "Tv")
        labels = np.array(labels)

        # 5a. 每个模型在 Ti/Tv 位点上的 mean CCS-N (分组条形图)
        fig, ax = plt.subplots(figsize=(10, 5))
        ti_mask, tv_mask = labels == "Ti", labels == "Tv"
        x = np.arange(M)
        w = 0.38
        ti_mean = real[:, ti_mask].mean(axis=1)
        tv_mean = real[:, tv_mask].mean(axis=1)
        # 相对比例 (除以自身 rand 中位数, 归一化到可比尺度)
        scale = np.median(rand, axis=1)
        ax.bar(x - w / 2, ti_mean / scale, w, label=f"Ti (n={ti_mask.sum()})", color="#1f77b4")
        ax.bar(x + w / 2, tv_mean / scale, w, label=f"Tv (n={tv_mask.sum()})", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([m[:22] for m in models], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("mean CCS-N / median rand CCS-N")
        ax.set_title(f"Variant effect by substitution type (euclidean N={N})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p5 = out_dir / "heatmap_by_titv.png"
        fig.savefig(p5, dpi=150)
        print(f"[heat] 已保存 {p5}")
        plt.close(fig)
        summary["n_ti"] = int(ti_mask.sum())
        summary["n_tv"] = int(tv_mask.sum())

        # 5b. AF 分箱 × 模型 mean CCS-N (热图)
        if af_vals is not None:
            af_bins = [("low (<0.1)", af_vals < 0.1),
                       ("mid (0.1-0.5)", (af_vals >= 0.1) & (af_vals < 0.5)),
                       ("high (>=0.5)", af_vals >= 0.5)]
            bin_names, bin_masks = zip(*af_bins)
            mat = np.stack([real[:, m].mean(axis=1) / scale for _, m in af_bins], axis=1)  # [M, 3]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_yticks(range(M))
            ax.set_yticklabels([m[:24] for m in models], fontsize=7)
            ax.set_xticks(range(len(bin_names)))
            ax.set_xticklabels([f"{b}\n(n={m.sum()})" for b, m in af_bins], fontsize=8)
            for i in range(M):
                for j in range(len(bin_names)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
            ax.set_title(f"Variant effect by allele frequency (N={N})")
            fig.colorbar(im, ax=ax, fraction=0.04)
            fig.tight_layout()
            p6 = out_dir / "heatmap_by_af.png"
            fig.savefig(p6, dpi=150)
            print(f"[heat] 已保存 {p6}")
            plt.close(fig)
            summary["af_bins"] = {b: int(m.sum()) for b, m in af_bins}

    # 6. 各模型 top/bottom 位点重合度 (哪些位点所有模型都难/都易)
    rank_mean = np.argsort(real.mean(axis=0))  # 位点按所有模型平均得分排序
    top10 = rank_mean[-10:]   # 最难 (最高 CCS-N = 变异效应最大)
    bot10 = rank_mean[:10]    # 最易 (最低 CCS-N)
    summary["hardest_sites_idx"] = top10.tolist()
    summary["easiest_sites_idx"] = bot10.tolist()
    with open(out_dir / "heatmap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[heat] summary: {out_dir / 'heatmap_summary.json'}")

    print("[heat] 全部完成 ✅")


if __name__ == "__main__":
    main()
