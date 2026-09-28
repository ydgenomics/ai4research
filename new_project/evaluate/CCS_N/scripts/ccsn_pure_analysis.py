"""纯 CCS-N 感知分析 (无 rand): 比较不同模型对真实变异的感知能力。

只用 ref vs alt 的真实 CCS-N 分数 (per_site/<model>/ccs_n_per_site.npz 的 real_* key),
完全不用 rand。核心思想:
- 绝对 CCS-N 分数受模型表征范数影响 (NTv3 7904 vs PlantCAD2 56), 不可直接跨模型比
- 用「模型内归一化」消除尺度差异: 每模型除以自身的 median (fold-over-median)
  → 得到"该模型对哪些位点感知强/弱"的相对模式, 跨模型可比

输出:
1. heatmap_ccsn_pure.png      — 模型 × 位点 归一化 CCS-N (N=16), 位点按所有模型均值排序
2. ccsn_ranking.png           — 各模型感知排名 (mean / top10% / CV) 条形图
3. heatmap_ccsn_Ncurve.png    — 各模型 CCS-N 随 N 的累积曲线 (归一化后, 展示上下文增益)
4. ccsn_pure_summary.json     — 排名表 + 归一化系数

用法:
    python scripts/ccsn_pure_analysis.py output/hom_1000_multi7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_LIST = (0, 1, 2, 4, 8, 16)


def load_real_scores(out_dir: Path, metric="euclidean"):
    """加载所有模型的 per-site real CCS-N 分数。返回 (names, {N: [M,B]})。"""
    ps_dir = out_dir / "per_site"
    names = sorted(d.name for d in ps_dir.iterdir() if d.is_dir())
    scores = {n: [] for n in N_LIST}
    for d in names:
        z = np.load(ps_dir / d / "ccs_n_per_site.npz")
        for n in N_LIST:
            scores[n].append(z[f"real_{metric}_N{n}"])
    return names, {n: np.stack(scores[n]) for n in N_LIST}  # [M, B]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--n", type=int, default=16, help="主排名用的 N")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, scores = load_real_scores(out_dir, args.metric)
    M = len(names)
    B = scores[0].shape[1]
    N = args.n
    print(f"[ccsn] {M} 模型 × {B} 位点, metric={args.metric}")

    # ---------- 归一化: 用每模型自己的变异中心感知 s0 (N=0) 作为基线 ----------
    # 核心无 rand 指标: 位点级上下文增益 ratio_site = N16/N0 (per-site 比率)
    # 意义: 变异中心感知为 1, 上下文累积倍数 = 模型把变异信号扩散到上下文的能力。
    # 该比率不受模型表征范数影响 (分子分母同尺度), 可跨模型直接比较。
    scale = np.median(scores[N], axis=1)              # [M] 每模型 N 中位数 (仅用于热图)
    norm = {n: scores[n] / scale[:, None] for n in N_LIST}   # fold-over-median (仅热图用)
    s0 = scores[0]                                    # [M, B] 变异中心感知
    nN = scores[N]                                    # [M, B]
    ratio_site = nN / s0                              # 位点级上下文增益 [M, B]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_site = np.nan_to_num(ratio_site, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- 1. 排名表 (按 ctx_gain 中位数降序) ----------
    summary = {"metric": args.metric, "n": N, "n_models": M, "n_sites": B, "models": {}}
    ranking = []
    for i, name in enumerate(names):
        med = float(np.median(ratio_site[i]))
        mean = float(ratio_site[i].mean())
        top10 = float(np.percentile(norm[N][i], 90))
        cv = float(norm[N][i].std() / norm[N][i].mean())
        ranking.append((name, med, mean, top10, cv))
        summary["models"][name] = {"ctx_gain_median": round(med, 3),
                                   "ctx_gain_mean": round(mean, 3),
                                   "top10": round(top10, 3), "cv": round(cv, 3)}
    ranking.sort(key=lambda x: -x[1])  # 按 ctx_gain 中位数降序
    print(f"\n===== 纯 CCS-N 感知排名 (无 rand, 上下文增益 N{N}/N0, {args.metric}) =====")
    print(f"{'排名':<4s} {'模型':<26s} {'ctx增益中位':>10s} {'ctx增益均值':>10s} {'top10%':>7s} {'CV':>6s}")
    for r, (name, med, mean, top10, cv) in enumerate(ranking, 1):
        print(f"{r:<4d} {name[:24]:<26s} {med:>10.1f} {mean:>10.1f} {top10:>7.2f} {cv:>6.2f}")
    summary["ranking"] = [{"rank": r, "model": n, "ctx_gain_median": round(m, 3),
                           "ctx_gain_mean": round(mm, 3), "top10": round(t, 3), "cv": round(c, 3)}
                          for r, (n, m, mm, t, c) in enumerate(ranking, 1)]

    # ---------- 2. 热图: 模型 × 位点 归一化 CCS-N ----------
    order = np.argsort(norm[N].mean(axis=0))          # 位点按所有模型平均感知排序
    X = norm[N][:, order]
    fig, ax = plt.subplots(figsize=(max(10, B / 100 * 2 + 2), 6))
    im = ax.imshow(X, aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(X, 95))
    ax.set_yticks(range(M))
    ax.set_yticklabels([m[:28] for m in names])
    ax.set_xticks([])
    ax.set_xlabel(f"sites sorted by mean normalized CCS-N (n={B})")
    ax.set_title(f"Pure CCS-N perception (no random control) — fold-over-median, "
                 f"{args.metric} N={N}")
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("CCS-N / model median")
    fig.tight_layout()
    p1 = out_dir / "heatmap_ccsn_pure.png"
    fig.savefig(p1, dpi=150)
    print(f"[ccsn] 已保存 {p1}")
    plt.close(fig)

    # ---------- 3. 排名条形图 ----------
    names_r = [r[0] for r in ranking]   # 模型名
    gains = [r[1] for r in ranking]     # ctx_gain 中位数
    tops = [r[3] for r in ranking]      # top10%
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, M))
    ax.barh(range(M), gains, color=colors)
    ax.set_yticks(range(M))
    ax.set_yticklabels([n[:26] for n in names_r], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"median context gain (CCS-N N={N} / N=0)")
    ax.set_title(f"Context accumulation ability (no random control)")
    ax = axes[1]
    ax.barh(range(M), tops, color=colors)
    ax.set_yticks(range(M))
    ax.set_yticklabels([n[:26] for n in names_r], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("top-10% percentile of normalized CCS-N")
    ax.set_title("Perception of strongest variants")
    fig.tight_layout()
    p2 = out_dir / "ccsn_ranking.png"
    fig.savefig(p2, dpi=150)
    print(f"[ccsn] 已保存 {p2}")
    plt.close(fig)

    # ---------- 4. N 累积曲线 (以 s0 为基线, 展示上下文增益) ----------
    # 每模型: mean(CCS-N_N / CCS-N_0) 随 N 的变化 → 上下文累积倍数曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        curve = [(scores[n][i] / s0[i]).mean() for n in N_LIST]
        ax.plot(N_LIST, curve, marker="o", label=name[:26])
    ax.set_xlabel("N (accumulation radius)")
    ax.set_ylabel("mean context gain (CCS-N_N / CCS-N_0)")
    ax.set_title(f"Context accumulation curves (no random control, {args.metric})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = out_dir / "ccsn_Ncurve.png"
    fig.savefig(p3, dpi=150)
    print(f"[ccsn] 已保存 {p3}")
    plt.close(fig)

    with open(out_dir / "ccsn_pure_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ccsn] summary: {out_dir / 'ccsn_pure_summary.json'}")
    print("[ccsn] 全部完成 ✅")


if __name__ == "__main__":
    main()
