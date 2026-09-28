#!/usr/bin/env python3
"""
fair_compare.py — 跨模型"序列功能类别表征能力"的**公平**对比

解决原 visual.ipynb 对比里的 3 个不公平点：

  1. 分箱口径不统一（每个模型用自己的 seq_len / 自己的过筛箱 → LM 准确率算在不同子集上）
     → 本脚本统一用 **bp 长度 = end - start + 1**（所有模型完全相同的 GFF 区间），
       用**全局固定的分位边界**切 N 个箱，所有模型共享同一套箱与同一套 CV 折。

  2. PCA(30) 固定截断对不同容量模型利弊相反
     → 本脚本输出**维度敏感性曲线**（n_comps = 8/16/32/64/128/256），
       同时报告每个模型的有效秩 (participation ratio) 与 top-k 方差保留率。

  3. 只有点估计、没有显著性
     → 所有模型跑在**完全相同的行 + 完全相同的 CV 折**上，
       用配对 McNemar 精确检验 + 配对 bootstrap 置信区间给出 p 值与 Holm 校正。

指标定义
--------
  global acc : 5 折 OOF，训练集含所有长度箱（常规口径）
  LM acc     : **长度匹配**口径 —— 对每个长度箱 b、每折 f，
               只用「同箱 b 内」的样本训练并预测 b∩f。
               长度信息被完全固定，模型无法靠长度作弊。

用法
----
  conda activate vllm
  python fair_compare.py                       # 全部模型，默认 out-root/configs 同目录约定
  python fair_compare.py --quick               # 冒烟：2 个模型 / 1 个 k / 3 折
  python fair_compare.py --bins 5 --ks 8 16 32 64 128 256 --k-ref 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
CLASSES = ["CDS", "Intron", "Intergenic", "Repeat", "3_UTR", "5_UTR"]  # ncRNA 样本量过小，排除


def to_label(gene_id: str) -> str:
    """按最长前缀匹配取功能类别。

    注意：不能简单用 split('_')[0]，否则 5_UTR -> '5'、3_UTR -> '3'。
    """
    for pre in ("5_UTR", "3_UTR", "CDS", "Intron", "Intergenic", "Repeat", "ncRNA"):
        if gene_id.startswith(pre):
            return pre
    return "OTHER"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def load_model(d: Path) -> ModelData:
    X = np.load(d / "gene_embeddings.npy").astype(np.float64)
    meta = pd.read_csv(d / "meta.tsv", sep="\t")
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        log(f"  !! {d.name}: {n_bad} 行含非有限值，已置零")
        X = np.nan_to_num(X)
    meta["label"] = meta["gene_id"].map(to_label)
    bp = (meta["end"].values - meta["start"].values + 1).astype(np.int64)
    info = {}
    mi = d / "model_info.json"
    if mi.exists():
        info = json.loads(mi.read_text())
    return ModelData(d.name, d, X, bp, meta["label"].values, meta["gene_id"].values, info)


# --------------------------------------------------------------------------- #
# 分箱 / 去混杂 / PCA
# --------------------------------------------------------------------------- #
def make_bins(bp: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """按 log10(bp) 的分位数切 n_bins 个**等频**箱，返回 (bin_idx, edges)。"""
    l = np.log10(bp)
    edges = np.quantile(l, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)                      # 去掉重复边界
    bin_idx = np.clip(np.digitize(l, edges[1:-1]), 0, len(edges) - 2)
    return bin_idx, edges


def legacy_bins(seq_len: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """复现原 notebook 的分箱逻辑（per-model seq_len + 50 分位箱），用于 before/after 对照。"""
    l = np.log10(seq_len.astype(float))
    edges = np.unique(np.quantile(l, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.digitize(l, edges[1:-1]), 0, len(edges) - 2)


def valid_bins(bin_idx: np.ndarray, y: np.ndarray, min_per_class: int = 5,
               min_size: int = 60) -> tuple[np.ndarray, pd.DataFrame]:
    """筛掉「样本过少 / 类内样本过稀」的箱（**全局统一筛**，所有模型共享）。"""
    rows, keep = [], np.ones(len(y), dtype=bool)
    for b in np.unique(bin_idx):
        m = bin_idx == b
        cnt = pd.Series(y[m]).value_counts()
        ok = (m.sum() >= min_size) and (len(cnt) >= 2) and (cnt.min() >= min_per_class)
        rows.append(dict(bin=int(b), n=int(m.sum()), n_class=int(len(cnt)),
                         min_class_n=int(cnt.min()), keep=bool(ok)))
        if not ok:
            keep[m] = False
    return keep, pd.DataFrame(rows)


def deconfound(X: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
    """① 箱内逐维减均值（去任意单调/非线性长度效应）→ ② 行 L2 归一化（去幅度泄漏）。"""
    Xd = X.copy()
    for b in np.unique(bin_idx):
        m = bin_idx == b
        Xd[m] -= X[m].mean(0, keepdims=True)
    Xd /= (np.linalg.norm(Xd, axis=1, keepdims=True) + 1e-8)
    return Xd


def eff_rank(A: np.ndarray) -> tuple[float, float, int]:
    """(participation ratio 有效秩, top-1 方差占比, 达标 90% 方差所需维数)。

    用 Gram 矩阵 (n×n) 的特征分解代替 (n×p) 的全 SVD —— n=3000 时快 10 倍以上，
    非零特征值与 SVD 的奇异值平方完全等价。
    """
    A = A - A.mean(0, keepdims=True)
    G = A @ A.T if A.shape[0] <= A.shape[1] else A.T @ A
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    if ev.sum() <= 0:
        return float("nan"), float("nan"), -1
    p = ev / ev.sum()
    return float(ev.sum() ** 2 / np.sum(ev ** 2)), float(p[0]), int(np.searchsorted(np.cumsum(p), 0.90) + 1)


def pca_white(Xd: np.ndarray, k_max: int, seed: int):
    """PCA → z-score（白化）每列。返回 (白化后的 PC 得分, 累计方差比)。"""
    k_max = int(min(k_max, Xd.shape[1], Xd.shape[0] - 1))
    p = PCA(n_components=k_max, svd_solver="randomized", random_state=seed)
    S = p.fit_transform(Xd)
    S = (S - S.mean(0)) / (S.std(0, ddof=1) + 1e-12)
    return S, np.cumsum(p.explained_variance_ratio_)


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def make_folds(y: np.ndarray, bk: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """按 (箱, 类) 联合分层 → 每个箱内部每类均衡分到各折（保证箱内 CV 可行）。"""
    strat = bk * 100 + pd.factorize(y)[0]
    folds = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), strat)):
        folds[te] = f
    return folds


def oof_predict(F: np.ndarray, y: np.ndarray, folds: np.ndarray, n_splits: int,
                within_bin: np.ndarray | None = None) -> np.ndarray:
    """返回 OOF 预测（-1 表示未预测）。

    within_bin 为 None → 全局口径（训练集 = 同折外所有样本）；
    否则 → 长度匹配口径（训练集 = **同一箱**内、折外样本）。
    """
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
            clf = LogisticRegression(max_iter=1000)
            clf.fit(F[tr], y[tr])
            pred[te] = clf.predict(F[te])
    return pred


def acc_of(pred: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    m = pred != -1
    corr = np.zeros(len(y), dtype=bool)
    corr[m] = pred[m] == y[m]
    return float(corr[m].mean()) if m.any() else float("nan"), corr


# --------------------------------------------------------------------------- #
# 配对检验
# --------------------------------------------------------------------------- #
def paired_tests(corrs: dict[str, np.ndarray], names: list[str], metric: str,
                 n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """配对 McNemar 精确检验 + 配对 bootstrap 95% CI。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = corrs[a], corrs[b]
            both = ca & cb
            b_only = int((~ca & cb).sum())      # b 对 a 错
            a_only = int((ca & ~cb).sum())      # a 对 b 错
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
    # Holm 校正（同一指标内多重比较）
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
    ap = argparse.ArgumentParser(description="跨模型表征能力公平对比")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "output" / "IRGSP_0915"),
                    help="各模型产物根目录（含 {model}/gene_embeddings.npy）")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "fair_compare_out"))
    ap.add_argument("--models", nargs="*", default=None, help="只跑指定模型（默认全部）")
    ap.add_argument("--bins", type=int, default=5, help="统一 bp 长度分位箱数（主口径）")
    ap.add_argument("--bins-extra", type=int, default=10,
                    help="更严格分箱（更窄的箱 = 更强的长度控制）；只在 k_ref 上评估，0=关闭")
    ap.add_argument("--ks", nargs="*", type=int, default=[8, 16, 32, 64, 128, 256],
                    help="维度敏感性曲线的 n_comps 取值")
    ap.add_argument("--k-ref", type=int, default=128, help="出主表/配对检验所用的 n_comps")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--legacy", action="store_true", default=True,
                    help="同时复现原 notebook 的 per-model 分箱口径做 before/after 对照")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--quick", action="store_true", help="冒烟：2 模型 / 1 个 k / 3 折")
    a = ap.parse_args()

    if a.quick:
        a.ks, a.folds, a.k_ref, a.n_boot = [32], 3, 32, 200
        if not a.models:
            a.models = ["Botanic0_L", "NTv3_650M_pre"]

    out_root, out_dir = Path(a.out_root), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(a.ks) | {a.k_ref})
    k_max = max(ks)

    dirs = discover_model_dirs(out_root, a.models)
    if not dirs:
        log(f"在 {out_root} 下没找到任何模型产物"); return 1
    log(f"发现 {len(dirs)} 个模型: {[d.name for d in dirs]}")

    # ---------------- 载入 + 对齐校验 ----------------
    models: list[ModelData] = []
    ref_gids = ref_bp = ref_lbl = None
    for d in dirs:
        t0 = time.time()
        md = load_model(d)
        if ref_gids is None:
            ref_gids, ref_bp, ref_lbl = md.gene_ids, md.bp, md.y
            # 全局有效样本筛选只做一次（与模型无关）
            g_bk, g_edges = make_bins(ref_bp, a.bins)
            g_keep, bin_tab = valid_bins(g_bk, ref_lbl)
        else:
            assert np.array_equal(md.gene_ids, ref_gids), f"{d.name} 行序与基准不一致！"
            assert np.array_equal(md.bp, ref_bp), f"{d.name} 的 bp 长度与基准不一致！"
        models.append(md)
        log(f"  载入 {md.name:<28} dim={md.dim:<5} ({time.time()-t0:.1f}s)")

    y = ref_lbl
    keep = g_keep
    log(f"统一 bp 分箱: {a.bins} 箱, 有效箱 {int(bin_tab['keep'].sum())}/{len(bin_tab)}, "
        f"有效样本 {int(keep.sum())}/{len(y)} ({100*keep.mean():.1f}%)")
    log("每箱构成:\n" + bin_tab.to_string(index=False))

    bk = g_bk.copy()
    folds = make_folds(y[keep], bk[keep], a.folds, a.seed)
    yk = y[keep]
    log(f"CV 折分配完成（按 箱×类 分层, {a.folds} 折）")

    # 严格性对照：更窄的统一 bp 箱（同样是全局统一筛 + 全局统一折，只是箱更多更窄）
    be = int(a.bins_extra) if a.bins_extra and a.bins_extra > a.bins else 0
    if be:
        g_bk2, g_edges2 = make_bins(ref_bp, be)
        g_keep2, bin_tab2 = valid_bins(g_bk2, ref_lbl)
        folds2 = make_folds(y[g_keep2], g_bk2[g_keep2], a.folds, a.seed)
        log(f"严格对照分箱 {be} 箱 → 有效箱 {int(bin_tab2['keep'].sum())}/{len(bin_tab2)}，"
            f"有效样本 {int(g_keep2.sum())} ({100*g_keep2.mean():.1f}%)")
        print(bin_tab2.to_string(index=False), flush=True)

    # ---------------- 逐模型处理 ----------------
    curve_rows, audit_rows, per_bin_rows = [], [], []
    corr_global, corr_lm = {}, {}
    summary = {}

    for md in models:
        t0 = time.time()
        F_all = md.X
        Xd = deconfound(F_all, bk)
        er_raw, _, _ = eff_rank(F_all)
        # 有效秩记录「分类器实际看到的」：去混杂后 + 有效样本
        er_dec, _, d90_dec = eff_rank(Xd[keep])
        S, cumvar = pca_white(Xd, k_max, a.seed)
        Sk = S[keep]
        top30 = float(cumvar[min(30, len(cumvar)) - 1] * 100)
        audit_rows.append(dict(
            model=md.name, dim=md.dim, tokenizer=md.info.get("tokenizer"),
            arch=md.info.get("arch"), max_context=md.info.get("max_context"),
            n_truncated=md.info.get("n_truncated"),
            eff_rank_raw=round(er_raw, 2), eff_rank_decon=round(er_dec, 2),
            d90_decon=d90_dec, top30_var_pct=round(top30, 1),
            n_comp_used=int(min(k_max, md.dim, len(y) - 1))))
        log(f"  {md.name}: eff_rank raw={er_raw:.1f} decon={er_dec:.1f}, "
            f"top30保留={top30:.1f}%")

        row = dict(model=md.name)
        for k in ks:
            Fk = Sk[:, :k]
            pg = oof_predict(Fk, yk, folds, a.folds)
            pl = oof_predict(Fk, yk, folds, a.folds, within_bin=bk[keep])
            ag, cg = acc_of(pg, yk)
            al, cl = acc_of(pl, yk)
            curve_rows.append(dict(model=md.name, n_comps=k, global_acc=ag, lm_acc=al,
                                   lm_coverage=float((pl != -1).mean())))
            if k == a.k_ref:
                # 存「是否预测正确」的正确性向量供配对检验使用；
                # 必须确认箱内 CV 覆盖了每一行，否则各模型配对的样本集不一致。
                assert (pl != -1).all(), \
                    f"{md.name}: 箱内 CV 有 {int((pl == -1).sum())} 行未被预测"
                corr_global[md.name] = cg
                corr_lm[md.name] = cl
                # 每箱单独准确率（长度匹配口径）
                for b in np.unique(bk[keep]):
                    m = bk[keep] == b
                    per_bin_rows.append(dict(
                        model=md.name, bin=int(b),
                        bp_lo=float(10 ** g_edges[b]), bp_hi=float(10 ** g_edges[b + 1]),
                        n=int(m.sum()),
                        lm_acc=float((pl[m] == yk[m]).mean()) if (pl[m] != -1).any() else np.nan))
            row[f"lm@k{k}"] = al
            row[f"global@k{k}"] = ag
            log(f"    k={k:<4} global={ag:.4f}  LM={al:.4f}  ({time.time()-t0:.0f}s)")

        # 长度基线（仅 log bp + 二次项）
        Lb = np.log10(md.bp[keep]).reshape(-1, 1)
        Fb = np.c_[Lb, Lb ** 2]
        pgb = oof_predict(Fb, yk, folds, a.folds)
        plb = oof_predict(Fb, yk, folds, a.folds, within_bin=bk[keep])
        row["lm@length_only"], _ = acc_of(plb, yk)
        row["global@length_only"], _ = acc_of(pgb, yk)
        row.update(dict(eff_rank=round(er_dec, 2), top30_var_pct=round(top30, 1),
                        n_bins=int(bin_tab["keep"].sum()),
                        coverage_pct=round(100 * keep.mean(), 1)))

        # ---- 严格对照：更窄的箱（只算 k_ref） ----
        if be:
            pl2 = oof_predict(S[g_keep2][:, :a.k_ref], y[g_keep2], folds2, a.folds,
                              within_bin=g_bk2[g_keep2])
            ok2 = pl2 != -1
            row[f"lm@bins{be}@k{a.k_ref}"] = float((pl2[ok2] == y[g_keep2][ok2]).mean())
            row[f"bins{be}_n_bins"] = int(bin_tab2["keep"].sum())
            row[f"bins{be}_coverage_pct"] = round(100 * g_keep2.mean(), 1)
            log(f"    严格对照 {be} 箱: LM={row[f'lm@bins{be}@k{a.k_ref}']:.4f} "
                f"过筛箱={row[f'bins{be}_n_bins']} 覆盖={row[f'bins{be}_coverage_pct']}%")

        # ---- legacy（原 notebook 口径）对照，只算 k_ref ----
        if a.legacy:
            try:
                seq_len = pd.read_csv(md.dir / "meta.tsv", sep="\t")["seq_len"].values
                lb = legacy_bins(seq_len)
                lkeep, ltab = valid_bins(lb, y)
                lfold = make_folds(y[lkeep], lb[lkeep], a.folds, a.seed)
                Fl = S[lkeep][:, :a.k_ref]
                pll = oof_predict(Fl, y[lkeep], lfold, a.folds, within_bin=lb[lkeep])
                lm_legacy, _ = acc_of(pll, y[lkeep])
                row["lm_legacy@k%d" % a.k_ref] = lm_legacy
                row["legacy_n_bins"] = int(ltab["keep"].sum())
                row["legacy_coverage_pct"] = round(100 * lkeep.mean(), 1)
                log(f"    legacy 口径: LM={lm_legacy:.4f} 过筛箱={int(ltab['keep'].sum())} "
                    f"覆盖={100*lkeep.mean():.1f}%")
            except Exception as e:      # legacy 仅为对照，失败不影响主结果
                log(f"    legacy 对照失败（忽略）: {e}")
        summary[md.name] = row
        log(f"  完成 {md.name}（{time.time()-t0:.0f}s）")

    # ---------------- 汇总表 ----------------
    curve = pd.DataFrame(curve_rows)
    sdf = pd.DataFrame(list(summary.values()))
    front = ["model", "eff_rank", "top30_var_pct", "n_bins", "coverage_pct",
             f"lm@k{a.k_ref}", f"global@k{a.k_ref}", "lm@length_only", "global@length_only"]
    if be:
        front.insert(6, f"lm@bins{be}@k{a.k_ref}")   # 严格分箱口径紧邻主指标
    sdf = sdf[[c for c in front if c in sdf.columns] +
              [c for c in sdf.columns if c not in front]]
    sdf = sdf.sort_values(f"lm@k{a.k_ref}", ascending=False).reset_index(drop=True)

    curve.to_csv(out_dir / "dim_curve.csv", index=False)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "fairness_audit.csv", index=False)
    pd.DataFrame(per_bin_rows).to_csv(out_dir / "per_bin_lm_acc.csv", index=False)

    # ---------------- 配对检验 ----------------
    names = list(corr_global.keys())
    pt_g = paired_tests({k: corr_global[k] for k in names}, names, f"global@{a.k_ref}",
                        a.n_boot, a.seed)
    pt_l = paired_tests({k: corr_lm[k] for k in names}, names, f"lm@{a.k_ref}",
                        a.n_boot, a.seed)
    pt = pd.concat([pt_g, pt_l], ignore_index=True)
    pt.to_csv(out_dir / "paired_tests.csv", index=False)

    # ---------------- 控制台输出 ----------------
    print("\n" + "=" * 110)
    print(f"跨模型公平对比（统一 bp 分箱 {a.bins} 箱 | 6 类 | n={int(keep.sum())} | "
          f"k_ref={a.k_ref} | {a.folds} 折 | 随机猜测={1/len(CLASSES):.3f}）")
    print("=" * 110)
    show = sdf.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4)
    print(show.to_string(index=False))

    print(f"\n【长度匹配口径 @k={a.k_ref} 的配对 McNemar（已 Holm 校正）】")
    disp = pt_l[["model_a", "model_b", "acc_a", "acc_b", "delta", "ci_lo", "ci_hi",
                 "p_holm", "significant"]].round(4)
    print(disp.to_string(index=False))

    # ---------------- 图 ----------------
    if not a.no_fig:
        plot_all(curve, sdf, pt_l, pd.DataFrame(audit_rows), pt_g, a, out_dir)

    log(f"全部产物已写入 {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_all(curve, sdf, pt_lm, audit, pt_g, a, out_dir: Path) -> None:
    names = list(sdf["model"])
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
    cmap = dict(zip(names, colors))
    chance = 1 / len(CLASSES)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # A: 长度匹配准确率 vs n_comps
    ax = axes[0, 0]
    for n in names:
        d = curve[curve["model"] == n].sort_values("n_comps")
        ax.plot(d["n_comps"], d["lm_acc"], "o-", color=cmap[n], label=n, lw=2, ms=6)
    lo = float(sdf["lm@length_only"].mean())
    ax.axhline(lo, ls="--", c="gray", lw=1.4, label=f"length-only baseline ({lo:.3f})")
    ax.axhline(chance, ls=":", c="k", lw=1.2, label=f"chance ({chance:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(curve["n_comps"].unique()))
    ax.set_xticklabels(sorted(curve["n_comps"].unique()))
    ax.set_xlabel("n_comps (PCA dims fed to probe)")
    ax.set_ylabel("Accuracy")
    ax.set_title("(A) Length-matched accuracy vs. embedding width\n"
                 "(same bp bins & folds for all models)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25)

    # B: 全局准确率 vs n_comps
    ax = axes[0, 1]
    for n in names:
        d = curve[curve["model"] == n].sort_values("n_comps")
        ax.plot(d["n_comps"], d["global_acc"], "s-", color=cmap[n], label=n, lw=2, ms=6)
    ax.axhline(chance, ls=":", c="k", lw=1.2, label=f"chance ({chance:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(curve["n_comps"].unique()))
    ax.set_xticklabels(sorted(curve["n_comps"].unique()))
    ax.set_xlabel("n_comps")
    ax.set_ylabel("Accuracy")
    ax.set_title("(B) Global (all-lengths) accuracy vs. embedding width")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25)

    # C: heatmap models × k
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

    # D: 配对差异矩阵
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
                 "(row − column; * = Holm-adjusted p < 0.05)")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or np.isnan(M.values[i, j]):
                continue
            star = "*" if P.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{M.values[i, j]:+.3f}{star}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Fair cross-model comparison of sequence-feature-type representation\n"
                 f"unified {a.bins} bp-length bins | 6 classes | {a.folds}-fold paired CV "
                 f"| seed={a.seed}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "fair_compare_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 公平性审计图
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ad = audit.set_index("model").loc[names]
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, ad["eff_rank_raw"], 0.38, label="raw", color="#94A3B8")
    ax.bar(x + 0.2, ad["eff_rank_decon"], 0.38, label="after de-confounding", color="#2563EB")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Effective rank (participation ratio, log)")
    ax.set_title("(A) Nominal dim (1024-3072) is a myth:\neffective rank is 1 - 55")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    ax.bar(x, ad["top30_var_pct"], color="#E11D48")
    ax.axhline(100, ls=":", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Variance retained by top-30 PCs [%]")
    ax.set_title("(B) The old fixed PCA(30) discarded ~50% of\n"
                 "variance for the richest models only")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    if "legacy_coverage_pct" in sdf.columns:
        lc = sdf.set_index("model").loc[names, "legacy_coverage_pct"]
        ax.bar(x - 0.2, lc, 0.38, label="legacy (per-model bins)", color="#F59E0B")
        ax.bar(x + 0.2, sdf.set_index("model").loc[names, "coverage_pct"], 0.38,
               label="unified bp bins", color="#10B981")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("Sample coverage of length-matched eval [%]")
        ax.set_title("(C) Legacy per-model bins covered only 8-16%\n"
                     "of samples -> accuracies were not comparable")
        ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fairness_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
