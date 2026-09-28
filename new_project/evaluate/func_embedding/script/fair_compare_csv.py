#!/usr/bin/env python3
"""
fair_compare_csv.py — 跨模型「基因功能类别表征能力」公平对比（标签来自 all_7class.csv）

与 fair_compare.py 的唯一区别：**标签来源**
  fair_compare.py     : 标签 = 区间类型（CDS/Intron/Intergenic/Repeat/UTR），从 gene_id 前缀解析
  fair_compare_csv.py : 标签 = **基因功能类别**（Protein kinase / NAC TF / Cytochrome P450 ...），
                        从 all_7class.csv 用 MSU ID join 到 meta.tsv 的 gene_id

其余管线**完全一致**，以便两份结果可直接对比：
  统一 bp 长度分箱 → 箱内减均值 + 行 L2（去混杂）→ PCA + 白化
  → 维度敏感性曲线（n_comps 8..256）→ 逻辑回归探针 → OOF 准确率
  → global（跨箱训练）与 LM（长度匹配、箱内训练）双口径
  → 配对 McNemar 精确检验 + 配对 bootstrap CI + Holm 校正

为什么仍要做长度分箱？
  功能类别与基因长度强烈耦合（NAC TF 中位 2.1kb、Protein kinase 4.2kb）。
  若不分箱，模型可以靠"长度"猜功能类别 → 必须用长度匹配口径排除这条捷径。

  本数据集的两点特殊处理（与 fair_compare.py 不同）：

  ① **长度窗口**：功能类的长度分布虽大幅重叠，但等频箱两端会有类别缺样本
     （不设窗口时 5 箱只有 2 箱合格、覆盖仅 40%）。默认限定
     log10(bp) ∈ [3.1, 3.85]（bp 1259–7079，7 类共同重叠区）→ 4 箱全部合格，
     覆盖 438/512 = 85.5%。用 --len-window 0 0 可关闭。

  ② **宏平均召回**：类别严重不均衡（Protein kinase 159/438 = 36%），
     多数类基线就有 0.363 准确率 → 准确率会被多数类主导。
     故同时报告 macro recall（各类召回等权），并以「仅长度基线」为地板：
     本数据 lm 长度基线 acc=0.379 / macro=0.208（随机 macro=0.143）。

用法
----
  /root/miniconda3/envs/vllm/bin/python fair_compare_csv.py
  ... --bins 4 --len-window 3.1 3.85 --ks 8 16 32 64 128 256 --k-ref 128 --folds 5
  ... --quick                     # 冒烟：2 模型 / 1 个 k / 3 折
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
CLASSES = ["Protein kinase", "RING/FYVE/PHD", "TPR", "Cytochrome P450",
           "MFS transporter", "NAC TF", "bZIP TF"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 标签：从 CSV 读功能类别（Description 含逗号 → 必须用 csv 模块，不能用 pandas 默认解析）
# --------------------------------------------------------------------------- #
def load_function_labels(csv_path: Path) -> dict[str, str]:
    """返回 {MSU_ID: 功能类别}。"""
    lut: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        i_lab, i_msu = header.index("label"), header.index("MSU")
        for line in r:
            if len(line) <= max(i_lab, i_msu):
                continue
            msu = line[i_msu].strip().upper()
            lab = line[i_lab].strip()
            if not msu or msu in ("NONE", "NA", ""):
                continue
            lut.setdefault(msu, lab)          # 重复 MSU 取第一条
    return lut


# --------------------------------------------------------------------------- #
# 数据载入
# --------------------------------------------------------------------------- #
class ModelData:
    __slots__ = ("name", "dir", "X", "bp", "y", "gene_ids", "info", "dim")

    def __init__(self, name, d, X, bp, y, gene_ids, info):
        self.name = name
        self.dir = d
        self.X = X
        self.bp = bp
        self.y = y
        self.gene_ids = gene_ids
        self.info = info
        self.dim = X.shape[1]


def discover_model_dirs(out_root: Path, only: list[str] | None) -> list[Path]:
    dirs = []
    for p in sorted(out_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "gene_embeddings.npy").exists() and (p / "meta.tsv").exists():
            if only and p.name not in only:
                continue
            dirs.append(p)
    return dirs


def load_model(d: Path, lut: dict[str, str]) -> ModelData:
    X = np.load(d / "gene_embeddings.npy").astype(np.float64)
    meta = pd.read_csv(d / "meta.tsv", sep="\t")
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        log(f"  !! {d.name}: {n_bad} 行含非有限值，已置零")
        X = np.nan_to_num(X)
    # gene_id 即 MSU ID（如 LOC_Os01g06280）
    msu = meta["gene_id"].astype(str).str.strip().str.upper()
    y = msu.map(lut).fillna("__UNLABELED__").values
    bp = (meta["end"].values - meta["start"].values + 1).astype(np.int64)
    info = {}
    mi = d / "model_info.json"
    if mi.exists():
        info = json.loads(mi.read_text())
    return ModelData(d.name, d, X, bp, y, meta["gene_id"].values, info)


# --------------------------------------------------------------------------- #
# 分箱 / 去混杂 / PCA（与 fair_compare.py 完全一致）
# --------------------------------------------------------------------------- #
def make_bins(bp: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """按 log10(bp) 分位数切 n_bins 个等频箱，返回 (bin_idx, edges)。"""
    l = np.log10(bp)
    edges = np.unique(np.quantile(l, np.linspace(0, 1, n_bins + 1)))
    bin_idx = np.clip(np.digitize(l, edges[1:-1]), 0, len(edges) - 2)
    return bin_idx, edges


def valid_bins(bin_idx: np.ndarray, y: np.ndarray, min_per_class: int = 5,
               min_size: int = 60, base: np.ndarray | None = None
               ) -> tuple[np.ndarray, pd.DataFrame]:
    """筛掉「样本过少 / 类内样本过稀」的箱（全局统一筛，所有模型共享）。

    base: 先验合格掩码（如「有标签 且 落在长度窗口内」）。
          返回的 keep 已包含 base，即 keep ⊆ base。
    """
    if base is None:
        base = np.ones(len(y), dtype=bool)
    rows, keep = [], base.copy()
    for b in np.unique(bin_idx):
        m = base & (bin_idx == b)
        cnt = pd.Series(y[m]).value_counts()
        if len(cnt) == 0:
            rows.append(dict(bin=int(b), n=0, n_class=0, min_class_n=0, keep=False))
            continue
        ok = (m.sum() >= min_size) and (len(cnt) >= 2) and (cnt.min() >= min_per_class)
        rows.append(dict(bin=int(b), n=int(m.sum()), n_class=int(len(cnt)),
                         min_class_n=int(cnt.min()), keep=bool(ok)))
        if not ok:
            keep[m] = False
    return keep, pd.DataFrame(rows)


def deconfound(X: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
    """① 箱内逐维减均值 → ② 行 L2 归一化。"""
    Xd = X.copy()
    for b in np.unique(bin_idx):
        m = bin_idx == b
        Xd[m] -= X[m].mean(0, keepdims=True)
    Xd /= (np.linalg.norm(Xd, axis=1, keepdims=True) + 1e-8)
    return Xd


def eff_rank(A: np.ndarray) -> tuple[float, float, int]:
    """(participation ratio 有效秩, top-1 方差占比, 90% 方差所需维数)。"""
    A = A - A.mean(0, keepdims=True)
    G = A @ A.T if A.shape[0] <= A.shape[1] else A.T @ A
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    if ev.sum() <= 0:
        return float("nan"), float("nan"), -1
    p = ev / ev.sum()
    return float(ev.sum() ** 2 / np.sum(ev ** 2)), float(p[0]), int(np.searchsorted(np.cumsum(p), 0.90) + 1)


def pca_white(Xd: np.ndarray, k_max: int, seed: int):
    """PCA → 逐列 z-score（白化）。"""
    k_max = int(min(k_max, Xd.shape[1], Xd.shape[0] - 1))
    p = PCA(n_components=k_max, svd_solver="randomized", random_state=seed)
    S = p.fit_transform(Xd)
    S = (S - S.mean(0)) / (S.std(0, ddof=1) + 1e-12)
    return S, np.cumsum(p.explained_variance_ratio_)


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def make_folds(y: np.ndarray, bk: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """按 (箱, 类) 分层；用**轮转分配**保证极小的 strata 也能均衡落各折。"""
    rng = np.random.default_rng(seed)
    strat = bk * 1000 + pd.factorize(y)[0]
    folds = np.full(len(y), -1, dtype=int)
    for s in np.unique(strat):
        idx = np.where(strat == s)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_splits
    return folds


def oof_predict(F: np.ndarray, y: np.ndarray, folds: np.ndarray, n_splits: int,
                within_bin: np.ndarray | None = None) -> np.ndarray:
    """OOF 预测（-1 = 未预测）。within_bin=None → 全局口径；否则 → 长度匹配口径。"""
    n = len(y)
    pred = np.full(n, -1, dtype=object)
    groups = np.unique(within_bin) if within_bin is not None else [None]
    for g in groups:
        gm = np.ones(n, dtype=bool) if g is None else (within_bin == g)
        for f in range(n_splits):
            te = gm & (folds == f)
            tr = gm & (folds != f)
            if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(F[tr], y[tr])
            pred[te] = clf.predict(F[te])
    return pred


def acc_of(pred: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    m = pred != -1
    corr = np.zeros(len(y), dtype=bool)
    corr[m] = pred[m] == y[m]
    return float(corr[m].mean()) if m.any() else float("nan"), corr


def macro_recall(pred: np.ndarray, y: np.ndarray) -> float:
    """宏平均召回（每个类内召回率的算术平均）。

    本数据类别严重不均衡（Protein kinase 占 36%），准确率会被多数类主导，
    必须同时报宏召回。
    """
    rs = []
    for c in np.unique(y):
        m = (y == c) & (pred != -1)
        if m.sum() == 0:
            continue
        rs.append(float((pred[m] == c).mean()))
    return float(np.mean(rs)) if rs else float("nan")


def per_class_recall(pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in sorted(set(y)):
        m = (y == c) & (pred != -1)
        if m.sum() == 0:
            continue
        rows.append(dict(label=c, n=int(m.sum()),
                         recall=float((pred[m] == y[m]).mean())))
    return pd.DataFrame(rows)


def paired_tests(corrs: dict[str, np.ndarray], names: list[str], metric: str,
                 n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """配对 McNemar 精确检验 + 配对 bootstrap 95% CI + Holm 校正。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = corrs[a], corrs[b]
            a_only = int((ca & ~cb).sum())
            b_only = int((~ca & cb).sum())
            n_disc = a_only + b_only
            p = binomtest(a_only, n_disc, 0.5).pvalue if n_disc > 0 else 1.0
            idx = rng.integers(0, len(ca), size=(n_boot, len(ca)))
            d = ca[idx].mean(1) - cb[idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(metric=metric, model_a=a, model_b=b,
                             acc_a=ca.mean(), acc_b=cb.mean(), delta=ca.mean() - cb.mean(),
                             ci_lo=lo, ci_hi=hi, a_only=a_only, b_only=b_only,
                             p_mcnemar=p))
    df = pd.DataFrame(rows)
    order = np.argsort(df["p_mcnemar"].values)
    m = len(df)
    padj = np.empty(m)
    running = 0.0
    for rank, oi in enumerate(order):
        val = min(1.0, (m - rank) * df["p_mcnemar"].values[oi])
        running = max(running, val)
        padj[oi] = running
    df["p_holm"] = padj
    df["significant"] = df["p_holm"] < 0.05
    return df.sort_values("delta", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="跨模型基因功能类别表征能力公平对比")
    ap.add_argument("--out-root", default=str(here / "output" / "csv_7class_Os7"))
    ap.add_argument("--out-dir", default=str(here / "fair_compare_csv_out"))
    ap.add_argument("--labels", default="/mnt/rice/default/Workspace/yangdong/ai4research/"
                                        "DATA/rice/gene_sets/panel/csv/all_7class.csv")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--bins", type=int, default=4, help="统一 bp 长度分位箱数（主口径）")
    ap.add_argument("--bins-extra", type=int, default=0,
                    help="更严格分箱数（0=关闭；功能类别与长度强耦合，建议靠窗口而非加箱）")
    ap.add_argument("--len-window", nargs=2, type=float, default=[3.1, 3.85],
                    metavar=("LO", "HI"),
                    help="限制在 log10(bp) ∈ [LO, HI] 的**共同长度重叠区**内评测。\n"
                         "功能类别与基因长度强耦合（TF 短、受体激酶长），不加窗口时\n"
                         "等频箱两端会有类别缺样本 → 覆盖率崩到 20-50%%。设为 0 0 关闭。")
    ap.add_argument("--ks", nargs="*", type=int, default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--k-ref", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-per-class", type=int, default=3, help="每箱每类最少样本")
    ap.add_argument("--min-size", type=int, default=30, help="每箱最少样本")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    if a.quick:
        a.ks, a.folds, a.k_ref, a.n_boot = [32], 3, 32, 200
        if not a.models:
            a.models = ["Botanic0_L", "NTv3_650M_pre"]

    out_root, out_dir = Path(a.out_root), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(a.ks) | {a.k_ref})
    k_max = max(ks)

    lut = load_function_labels(Path(a.labels))
    log(f"功能标签表: {len(lut)} 条 unique MSU，{len(set(lut.values()))} 类")

    dirs = discover_model_dirs(out_root, a.models)
    if not dirs:
        log(f"在 {out_root} 下没找到任何模型产物"); return 1
    log(f"发现 {len(dirs)} 个模型: {[d.name for d in dirs]}")

    # ---------------- 载入 + 对齐校验 ----------------
    models: list[ModelData] = []
    ref_gids = ref_bp = ref_lbl = None
    for d in dirs:
        t0 = time.time()
        md = load_model(d, lut)
        n_unlab = int((md.y == "__UNLABELED__").sum())
        if ref_gids is None:
            ref_gids, ref_bp, ref_lbl = md.gene_ids, md.bp, md.y
            log_bp = np.log10(ref_bp.astype(float))
            # 先验合格掩码：有标签 且 落在共同长度重叠窗口内
            elig = ref_lbl != "__UNLABELED__"
            if a.len_window and (a.len_window[1] > a.len_window[0] > 0):
                win = (log_bp >= a.len_window[0]) & (log_bp <= a.len_window[1])
                log(f"长度窗口 log10(bp) ∈ [{a.len_window[0]}, {a.len_window[1]}] "
                    f"= bp [{10**a.len_window[0]:.0f}, {10**a.len_window[1]:.0f}] "
                    f"→ 命中 {int(win.sum())}/{len(win)}")
            else:
                win = np.ones(len(ref_bp), dtype=bool)
            elig = elig & win
            # 分位边界只在窗口内算 → 箱在重叠区内等频
            edges = np.unique(np.quantile(log_bp[elig], np.linspace(0, 1, a.bins + 1)))
            g_bk = np.clip(np.digitize(log_bp, edges[1:-1]), 0, len(edges) - 2).astype(int)
            g_edges = edges
            g_keep, bin_tab = valid_bins(g_bk, ref_lbl, a.min_per_class, a.min_size,
                                         base=elig)
        else:
            assert np.array_equal(md.gene_ids, ref_gids), f"{d.name} 行序与基准不一致！"
            assert np.array_equal(md.bp, ref_bp), f"{d.name} 的 bp 长度与基准不一致！"
        models.append(md)
        log(f"  载入 {md.name:<28} dim={md.dim:<5} 无标签={n_unlab} ({time.time()-t0:.1f}s)")

    y = ref_lbl
    keep = g_keep
    log(f"统一 bp 分箱: {a.bins} 箱（窗口内等频）, 有效箱 {int(bin_tab['keep'].sum())}/{len(bin_tab)}, "
        f"有效样本 {int(keep.sum())}/{len(y)} ({100*keep.mean():.1f}%)")
    log("每箱构成:\n" + bin_tab.to_string(index=False))
    if keep.sum() < 100:
        log("!! 有效样本过少，建议放宽 --len-window / 减小 --bins / 放宽 --min-per-class")

    bk = g_bk.copy()
    folds = make_folds(y[keep], bk[keep], a.folds, a.seed)
    yk = y[keep]
    log(f"CV 折分配完成（箱×类 分层轮转, {a.folds} 折）；类别构成: "
        f"{pd.Series(yk).value_counts().to_dict()}")

    be = int(a.bins_extra) if a.bins_extra and a.bins_extra > a.bins else 0
    if be:
        edges2 = np.unique(np.quantile(log_bp[elig], np.linspace(0, 1, be + 1)))
        g_bk2 = np.clip(np.digitize(log_bp, edges2[1:-1]), 0, len(edges2) - 2).astype(int)
        g_keep2, bin_tab2 = valid_bins(g_bk2, ref_lbl, a.min_per_class, a.min_size,
                                       base=elig)
        folds2 = make_folds(y[g_keep2], g_bk2[g_keep2], a.folds, a.seed)
        log(f"严格对照分箱 {be} 箱 → 有效箱 {int(bin_tab2['keep'].sum())}/{len(bin_tab2)}，"
            f"有效样本 {int(g_keep2.sum())} ({100*g_keep2.mean():.1f}%)")

    # ---------------- 逐模型处理 ----------------
    curve_rows, audit_rows, per_bin_rows, per_class_rows = [], [], [], []
    corr_global, corr_lm = {}, {}
    summary = {}

    for md in models:
        t0 = time.time()
        Xd = deconfound(md.X, bk)
        er_raw, _, _ = eff_rank(md.X)
        er_dec, _, d90_dec = eff_rank(Xd[keep])
        S, cumvar = pca_white(Xd, k_max, a.seed)
        Sk = S[keep]
        top30 = float(cumvar[min(30, len(cumvar)) - 1] * 100)
        audit_rows.append(dict(
            model=md.name, dim=md.dim, tokenizer=md.info.get("tokenizer"),
            arch=md.info.get("arch"), max_context=md.info.get("max_context"),
            n_truncated=md.info.get("n_truncated"), pooling=md.info.get("pooling"),
            eff_rank_raw=round(er_raw, 2), eff_rank_decon=round(er_dec, 2),
            d90_decon=d90_dec, top30_var_pct=round(top30, 1),
            n_comp_used=int(min(k_max, md.dim, len(yk) - 1))))
        log(f"  {md.name}: eff_rank raw={er_raw:.1f} decon={er_dec:.1f}, top30保留={top30:.1f}%")

        row = dict(model=md.name)
        for k in ks:
            Fk = Sk[:, :k]
            pg = oof_predict(Fk, yk, folds, a.folds)
            pl = oof_predict(Fk, yk, folds, a.folds, within_bin=bk[keep])
            ag, cg = acc_of(pg, yk)
            al, cl = acc_of(pl, yk)
            alm, agm = macro_recall(pl, yk), macro_recall(pg, yk)
            curve_rows.append(dict(model=md.name, n_comps=k, global_acc=ag, lm_acc=al,
                                   lm_macro=alm, global_macro=agm,
                                   lm_coverage=float((pl != -1).mean())))
            if k == a.k_ref:
                # 配对检验要求各模型覆盖的样本集完全一致
                assert (pl != -1).all(), \
                    f"{md.name}: 箱内 CV 有 {int((pl == -1).sum())} 行未被预测"
                corr_global[md.name] = cg
                corr_lm[md.name] = cl
                # 每箱 LM 准确率
                for b in np.unique(bk[keep]):
                    m = bk[keep] == b
                    per_bin_rows.append(dict(
                        model=md.name, bin=int(b),
                        bp_lo=float(10 ** g_edges[b]), bp_hi=float(10 ** g_edges[b + 1]),
                        n=int(m.sum()),
                        lm_acc=float((pl[m] == yk[m]).mean()) if (pl[m] != -1).any() else np.nan))
                # 每类召回率（global 与 LM 双口径）
                rr = per_class_recall(pl, yk).rename(columns={"recall": "lm_recall"})
                rg = per_class_recall(pg, yk).rename(columns={"recall": "global_recall"})
                pc = rr.merge(rg[["label", "global_recall"]], on="label", how="outer")
                pc["model"] = md.name
                per_class_rows.append(pc)
            row[f"lm@k{k}"] = al
            row[f"global@k{k}"] = ag
            row[f"lm_macro@k{k}"] = alm
            row[f"global_macro@k{k}"] = agm
            log(f"    k={k:<4} global={ag:.4f}  LM={al:.4f}  "
                f"LM_macro={alm:.4f}  ({time.time()-t0:.0f}s)")

        # 长度基线（log bp + 二次项）
        Lb = np.log10(md.bp[keep]).reshape(-1, 1)
        Fb = np.c_[Lb, Lb ** 2]
        pgb = oof_predict(Fb, yk, folds, a.folds)
        plb = oof_predict(Fb, yk, folds, a.folds, within_bin=bk[keep])
        row["lm@length_only"], _ = acc_of(plb, yk)
        row["global@length_only"], _ = acc_of(pgb, yk)
        row["lm_macro@length_only"] = macro_recall(plb, yk)
        row["global_macro@length_only"] = macro_recall(pgb, yk)
        row.update(dict(eff_rank=round(er_dec, 2), top30_var_pct=round(top30, 1),
                        n_bins=int(bin_tab["keep"].sum()),
                        coverage_pct=round(100 * keep.mean(), 1),
                        len_window=f"{a.len_window[0]}-{a.len_window[1]}",
                        bp_window=f"{10**a.len_window[0]:.0f}-{10**a.len_window[1]:.0f}"))

        if be:
            pl2 = oof_predict(S[g_keep2][:, :a.k_ref], y[g_keep2], folds2, a.folds,
                              within_bin=g_bk2[g_keep2])
            ok2 = pl2 != -1
            row[f"lm@bins{be}@k{a.k_ref}"] = float((pl2[ok2] == y[g_keep2][ok2]).mean())
            row[f"bins{be}_n_bins"] = int(bin_tab2["keep"].sum())
            row[f"bins{be}_coverage_pct"] = round(100 * g_keep2.mean(), 1)
            log(f"    严格对照 {be} 箱: LM={row[f'lm@bins{be}@k{a.k_ref}']:.4f} "
                f"过筛箱={row[f'bins{be}_n_bins']} 覆盖={row[f'bins{be}_coverage_pct']}%")

        summary[md.name] = row
        log(f"  完成 {md.name}（{time.time()-t0:.0f}s）")

    # ---------------- 汇总表 ----------------
    curve = pd.DataFrame(curve_rows)
    sdf = pd.DataFrame(list(summary.values()))
    front = ["model", "eff_rank", "top30_var_pct", "n_bins", "coverage_pct",
             f"lm@k{a.k_ref}", f"global@k{a.k_ref}",
             f"lm_macro@k{a.k_ref}", f"global_macro@k{a.k_ref}",
             "lm@length_only", "global@length_only",
             "lm_macro@length_only", "global_macro@length_only"]
    if be:
        front.insert(6, f"lm@bins{be}@k{a.k_ref}")
    sdf = sdf[[c for c in front if c in sdf.columns] +
              [c for c in sdf.columns if c not in front]]
    sdf = sdf.sort_values(f"lm@k{a.k_ref}", ascending=False).reset_index(drop=True)

    curve.to_csv(out_dir / "dim_curve.csv", index=False)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "fairness_audit.csv", index=False)
    pd.DataFrame(per_bin_rows).to_csv(out_dir / "per_bin_lm_acc.csv", index=False)
    if per_class_rows:
        pd.concat(per_class_rows, ignore_index=True).to_csv(
            out_dir / "per_class_recall.csv", index=False)

    # ---------------- 配对检验 ----------------
    names = list(corr_global.keys())
    pt_g = paired_tests({k: corr_global[k] for k in names}, names,
                        f"global@{a.k_ref}", a.n_boot, a.seed)
    pt_l = paired_tests({k: corr_lm[k] for k in names}, names,
                        f"lm@{a.k_ref}", a.n_boot, a.seed)
    pt = pd.concat([pt_g, pt_l], ignore_index=True)
    pt.to_csv(out_dir / "paired_tests.csv", index=False)

    # ---------------- 控制台输出 ----------------
    n_cls = len(pd.unique(yk))
    print("\n" + "=" * 112)
    print(f"跨模型【基因功能类别】表征能力对比（统一 bp 分箱 {a.bins} 箱 | {n_cls} 类 | "
          f"n={int(keep.sum())} | k_ref={a.k_ref} | {a.folds} 折 | 随机={1/n_cls:.3f}）")
    print(f"长度窗口 log10(bp) ∈ [{a.len_window[0]}, {a.len_window[1]}] "
          f"= bp [{10**a.len_window[0]:.0f}, {10**a.len_window[1]:.0f}]")
    print(f"多数类基线 = {pd.Series(yk).value_counts().iloc[0]/len(yk):.3f}"
          "（类别严重不均衡 → 准确率会被多数类主导，务必同看 macro_recall）")
    print("=" * 112)
    show = sdf.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4)
    print(show.to_string(index=False))

    print(f"\n【长度匹配口径 @k={a.k_ref} 配对 McNemar（Holm 校正后）】")
    print(pt_l[["model_a", "model_b", "acc_a", "acc_b", "delta", "ci_lo", "ci_hi",
                "p_holm", "significant"]].round(4).to_string(index=False))

    if per_class_rows:
        pcs = pd.concat(per_class_rows, ignore_index=True)
        print(f"\n【各功能类别 LM 召回率 @k={a.k_ref}】")
        print(pcs.pivot_table(index="label", columns="model",
                              values="lm_recall").round(3).to_string())
        print(f"\n【宏平均召回（各类召回等权）@k={a.k_ref}｜按长度匹配排序】")
        mac = pcs.groupby("model")["lm_recall"].mean().sort_values(ascending=False)
        base = float(sdf["lm_macro@length_only"].iloc[0])
        disp2 = pd.DataFrame({
            "lm_macro": mac.round(4),
            "global_macro": pcs.groupby("model")["global_recall"].mean().round(4),
            "excess_over_length_only": (mac - base).round(4),
        }).loc[mac.index]
        print(disp2.to_string())
        print(f"（仅长度基线 macro_recall = {base:.3f}）")

    if not a.no_fig:
        plot_all(curve, sdf, pt_l, pd.DataFrame(audit_rows), a, out_dir)

    log(f"全部产物已写入 {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_all(curve, sdf, pt_lm, audit, a, out_dir: Path) -> None:
    names = list(sdf["model"])
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
    cmap = dict(zip(names, colors))
    n_cls = 7
    chance = 1 / n_cls

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    ax = axes[0, 0]
    for n in names:
        d = curve[curve["model"] == n].sort_values("n_comps")
        ax.plot(d["n_comps"], d["lm_acc"], "o-", color=cmap[n], label=n, lw=2, ms=6)
    lo = float(sdf["lm@length_only"].mean())
    ax.axhline(lo, ls="--", c="gray", lw=1.4, label=f"length-only baseline ({lo:.3f})")
    ax.axhline(chance, ls=":", c="k", lw=1.2, label=f"chance ({chance:.3f})")
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(curve["n_comps"].unique()))
    ax.set_xticklabels(sorted(curve["n_comps"].unique()))
    ax.set_xlabel("n_comps (PCA dims fed to probe)"); ax.set_ylabel("Accuracy")
    ax.set_title("(A) Length-matched accuracy vs. embedding width\n"
                 "(7 gene-function classes, same bp bins & folds for all models)")
    ax.legend(fontsize=8, frameon=False, ncol=2); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for n in names:
        d = curve[curve["model"] == n].sort_values("n_comps")
        ax.plot(d["n_comps"], d["global_acc"], "s-", color=cmap[n], label=n, lw=2, ms=6)
    ax.axhline(chance, ls=":", c="k", lw=1.2, label=f"chance ({chance:.3f})")
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(curve["n_comps"].unique()))
    ax.set_xticklabels(sorted(curve["n_comps"].unique()))
    ax.set_xlabel("n_comps"); ax.set_ylabel("Accuracy")
    ax.set_title("(B) Global (all-lengths) accuracy vs. embedding width")
    ax.legend(fontsize=8, frameon=False, ncol=2); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    piv = curve.pivot(index="model", columns="n_comps", values="lm_acc").loc[names]
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=chance, vmax=0.85, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    ax.set_xlabel("n_comps"); ax.set_title("(C) Length-matched accuracy heatmap")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    M = pd.DataFrame(np.nan, index=names, columns=names)
    P = pd.DataFrame(1.0, index=names, columns=names)
    for _, r in pt_lm.iterrows():
        M.loc[r["model_a"], r["model_b"]] = r["delta"]
        M.loc[r["model_b"], r["model_a"]] = -r["delta"]
        P.loc[r["model_a"], r["model_b"]] = r["p_holm"]
        P.loc[r["model_b"], r["model_a"]] = r["p_holm"]
        M.loc[r["model_a"], r["model_a"]] = 0.0
    v = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"(D) Paired delta in length-matched acc @k={a.k_ref}\n"
                 "(row - column; * = Holm-adjusted p < 0.05)")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or np.isnan(M.values[i, j]):
                continue
            star = "*" if P.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{M.values[i, j]:+.3f}{star}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Fair cross-model comparison: representation of GENE FUNCTION classes\n"
                 f"unified {a.bins} bp-length bins | 7 functional classes | "
                 f"{a.folds}-fold paired CV | seed={a.seed}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fair_compare_csv_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 审计图
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ad = audit.set_index("model").loc[names]
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.2, ad["eff_rank_raw"], 0.38, label="raw", color="#94A3B8")
    ax.bar(x + 0.2, ad["eff_rank_decon"], 0.38, label="after de-confounding", color="#2563EB")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Effective rank (participation ratio, log)")
    ax.set_title("(A) Nominal dim vs. effective rank")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    ax.bar(x, ad["top30_var_pct"], color="#E11D48")
    ax.axhline(100, ls=":", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Variance retained by top-30 PCs [%]")
    ax.set_title("(B) Truncation risk of fixed low n_comps")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ax.bar(x, sdf.set_index("model").loc[names, "coverage_pct"], color="#10B981")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Sample coverage [%]"); ax.set_ylim(0, 105)
    ax.set_title("(C) Coverage of length-matched eval\n(identical for all models by design)")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "fairness_audit_csv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
