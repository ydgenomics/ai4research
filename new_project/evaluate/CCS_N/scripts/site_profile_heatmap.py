"""位点 × 偏移 格子热图: 显示每个变异位点的感知能力 (变异在中心, 左右各 N)。

数据来源: per_site/<model>/distance_profiles.npz (euclidean_real [B, 2R+1], 中心为变异 token)

每个模型输出一张图:
- 行 = 1000 个变异位点 (按变异中心感知强度排序, 中心最强在顶部)
- 列 = token 偏移 (-16 .. +16, 0 = 变异中心)
- 颜色 = 该位点在该偏移处的 ref-vs-alt 表征距离 (欧氏)

归一化 (关键):
- 绝对距离跨位点/跨模型尺度差异巨大, 直接画会全被大值主导
- 按行归一化: 每位点除以自身 N 偏移内距离的 95% 分位数 → [0,1] 相对感知模式
  (每个位点自己最亮的位置 = 该位点感知最强的区域)
- 行间排序: 按中心 (offset=0) 归一化距离降序 → 中心感知强的位点在上方

无 rand: 只用 euclidean_real (真实变异 ref vs alt)。

用法:
    python scripts/site_profile_heatmap.py output/hom_1000_multi7 [--model agront_1b ...]
    不带 --model 则输出所有模型。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--model", action="append", default=None, help="只处理指定模型 (可多次); 默认全部")
    ap.add_argument("--max-sites", type=int, default=None, help="最多位点数 (默认全部 1000)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ps_dir = out_dir / "per_site"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = args.model or sorted(d.name for d in ps_dir.iterdir() if d.is_dir())
    for name in models:
        npz = ps_dir / name / "distance_profiles.npz"
        if not npz.exists():
            print(f"[warn] 缺少 {npz}, 跳过")
            continue
        z = np.load(npz)
        r = z["euclidean_real"]  # [B, 2R+1]
        B, W = r.shape
        R = (W - 1) // 2
        if args.max_sites:
            r = r[: args.max_sites]
            B = r.shape[0]
        offs = np.arange(-R, R + 1)

        # ---- 按行归一化: 除以该位点自身距离的 95% 分位数 ----
        p95 = np.nanpercentile(r, 95, axis=1, keepdims=True)
        p95 = np.where(p95 < 1e-9, 1e-9, p95)      # 防除零
        rn = r / p95                               # [B, 2R+1] 相对感知模式

        # ---- 行排序: 按中心归一化距离降序 (中心感知强 → 顶部) ----
        center = rn[:, R]
        order = np.argsort(center)[::-1]
        rn = rn[order]

        # ---- 绘图 ----
        fig, ax = plt.subplots(figsize=(14, 10))
        vmax = 1.0
        im = ax.imshow(rn, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.axvline(R, color="cyan", lw=1.0, ls="--", alpha=0.7)   # 变异中心列
        ax.set_yticks([])
        ax.set_xticks(range(0, W, 2))
        ax.set_xticklabels(offs[::2])
        ax.set_xlabel("token offset from variant (0 = variant center)")
        ax.set_ylabel(f"sites sorted by center perception (n={B})")
        ax.set_title(f"Per-site perception profile — {name[:40]} (euclidean, row-normalized)")
        cb = fig.colorbar(im, ax=ax, fraction=0.02)
        cb.set_label("distance / site-95th-pct (relative perception)")
        fig.tight_layout()
        png = out_dir / f"site_profile_{name[:30]}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"[site] 已保存 {png}")


if __name__ == "__main__":
    main()
