"""CCS-N 多模型结果汇总: 读取 ccs_n_summary.json, 生成统计表 + 对比图。

用法: python scripts/analyze_results.py output/hom_1000_multi5/ccs_n_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_table(summary: dict) -> dict:
    """按 (metric, N) 聚合各模型 AUC, 返回 {metric: {N: {model: auc}}}。"""
    n_list = summary.get("n_list", [0, 1, 2, 4, 8, 16])
    models = summary["models"]
    tables = {"euclidean": {}, "cosine": {}}
    for metric in tables:
        for n in n_list:
            tables[metric][n] = {
                name: m.get(f"{metric}_N{n}", {}).get("auc")
                for name, m in models.items()
            }
    return tables


def print_summary(path: str):
    summary = load_summary(path)
    print(f"\n===== CCS-N 多模型评测汇总: {Path(path).parent.name} =====")
    print(f"样本: {summary['sample']}  位点数: {summary['n_regions']}")
    tables = build_table(summary)
    for metric in ("euclidean", "cosine"):
        print(f"\n--- {metric} AUC ---")
        header = "N".ljust(6) + "".join(f"{name[:22]:>24}" for name in summary["models"])
        print(header)
        for n, row in tables[metric].items():
            line = f"N={str(n):<3}" + "".join(
                f"{('--' if v is None else f'{v:.3f}'):>24}" for v in row.values()
            )
            print(line)

    # 模型排序 (按 N=16 euclidean AUC 降序)
    print("\n--- 排名 (欧氏距离 N=16 AUC, 降序) ---")
    ranking = sorted(
        summary["models"].items(),
        key=lambda kv: kv[1].get("euclidean_N16", {}).get("auc", 0),
        reverse=True,
    )
    for i, (name, m) in enumerate(ranking, 1):
        auc16 = m.get("euclidean_N16", {}).get("auc")
        auc8 = m.get("euclidean_N8", {}).get("auc")
        auc1 = m.get("euclidean_N1", {}).get("auc")
        print(f"  {i}. {name:40s} AUC(N=1)={auc1:.3f} AUC(N=8)={auc8:.3f} AUC(N=16)={auc16:.3f}")


def plot_summary(path: str, out_png: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = load_summary(path)
    n_list = summary.get("n_list", [0, 1, 2, 4, 8, 16])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("euclidean", "cosine")):
        for name, m in summary["models"].items():
            aucs = [m.get(f"{metric}_N{n}", {}).get("auc", np.nan) for n in n_list]
            ax.plot(n_list, aucs, marker="o", label=name[:30])
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_title(f"CCS-N {metric} AUC (real vs random)")
        ax.set_xlabel("N (accumulation radius)")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_png is None:
        out_png = str(Path(path).parent / "ccs_n_compare.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] 已保存 {out_png}")


if __name__ == "__main__":
    p = sys.argv[1]
    print_summary(p)
    plot_summary(p, sys.argv[2] if len(sys.argv) > 2 else None)
