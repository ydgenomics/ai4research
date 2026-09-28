"""绘制按功能区分类 (intergenic, cds, intron) 的多模型对比格子热图。

格子每 6 个 token 一个。
数据来源: per_site/<model>/distance_profiles.npz (euclidean_real [1000, 33])
分类来源: osa1_r7.all_models.gff3

热图结构:
- 纵轴: 模型 × 功能区 (共 7 模型 × 3 区域 = 21 行)
- 横轴: token 偏移分箱 (每 6 个 token 一个格子)
- 颜色: 归一化后的平均表征距离 (无 rand)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_bed(bed_path: str):
    regions = []
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom, start, end, name = parts[0], int(parts[1]), int(parts[2]), parts[3]
            p = name.split(":")
            ref_alt = next(x for x in p if ">" in x)
            pos = int(p[p.index(ref_alt) - 1])
            chrom_n = p[p.index(ref_alt) - 2]
            ref, alt = ref_alt.split(">")[:2]
            regions.append((chrom, pos, ref, alt))
    return regions


def classify_sites(bed_path: str, gff_path: str):
    sites = parse_bed(bed_path)
    from collections import defaultdict
    genes = defaultdict(list)
    exons = defaultdict(list)
    cds = defaultdict(list)
    
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, source, ftype, start, end = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
            if ftype == "gene":
                genes[chrom].append((start, end))
            elif ftype == "exon":
                exons[chrom].append((start, end))
            elif ftype == "CDS":
                cds[chrom].append((start, end))

    categories = []
    for chrom, pos, ref, alt in sites:
        # 1. CDS
        in_cds = False
        for s, e in cds[chrom]:
            if s <= pos <= e:
                in_cds = True
                break
        if in_cds:
            categories.append("cds")
            continue
            
        # 2. Intron (在 gene 内但不在 exon 内)
        in_gene = False
        for s, e in genes[chrom]:
            if s <= pos <= e:
                in_gene = True
                break
        if in_gene:
            in_exon = False
            for s, e in exons[chrom]:
                if s <= pos <= e:
                    in_exon = True
                    break
            if not in_exon:
                categories.append("intron")
            else:
                # 在 exon 内但不在 CDS 内，通常是 UTR，这里也归入 intron 或单独处理。用户要求分三类，我们归入 intron
                categories.append("intron")
        else:
            categories.append("intergenic")
            
    return categories


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="output/hom_1000_multi7")
    ap.add_argument("--bed", default="data/carrier_hom.bed")
    ap.add_argument("--gff", default="/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.all_models.gff3")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ps_dir = out_dir / "per_site"
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. 分类位点
    categories = classify_sites(args.bed, args.gff)
    cat_indices = {
        "cds": [i for i, c in enumerate(categories) if c == "cds"],
        "intron": [i for i, c in enumerate(categories) if c == "intron"],
        "intergenic": [i for i, c in enumerate(categories) if c == "intergenic"],
    }
    
    # 2. 确定模型列表 (排除 AgriGenome_4n8a 和 final)
    exclude_models = {"AgriGenome_4n8a_1.2b_8k_pt_32k_cpt_stage1_iter14000_hf", "final"}
    models = sorted(d.name for d in ps_dir.iterdir() if d.is_dir() and d.name not in exclude_models)
    
    # 3. 定义分箱 (257 个 token: -128 到 +128)
    # 索引 128 是变异中心 (0)
    # 变异中心 [0] 单独作为一个格子，不与周围混淆。
    # 左右两侧各按 6 个 token 进行分箱。
    # 128 往左: 128 - 1 = 127。127 // 6 = 21 个完整的 6-token 箱子，还剩 127 % 6 = 1 个 token。
    # 我们可以从中心向外精确分箱：
    # 往左：
    # [128, 128] -> [0] (index 128)
    # [122, 127] -> [-6, -1] (index 122 to 127)
    # [116, 121] -> [-12, -7]
    # ...
    # 往右：
    # [129, 134] -> [+1, +6] (index 129 to 134)
    # [135, 140] -> [+7, +12]
    # ...
    # 为了简单且对称，我们直接从 0 到 256 索引进行分箱：
    # 128 对应 [0]
    # 128 往左有 128 个 token (索引 0 到 127)
    # 128 往右有 128 个 token (索引 129 到 256)
    # 我们可以定义：
    # 128 往左分箱：
    # 127 对应 -1，122 对应 -6 -> [122, 127]
    # 121 对应 -7，116 对应 -12 -> [116, 121]
    # ...
    # 索引 0 到 1 对应 [-128, -127] (2个token)
    # 索引 2 到 7 对应 [-126, -121] (6个token)
    # ...
    # 128 往右分箱：
    # 129 对应 +1，134 对应 +6 -> [129, 134]
    # 135 对应 +7，140 对应 +12 -> [135, 140]
    # ...
    # 索引 249 到 254 对应 [+121, [+126] (6个token)
    # 索引 255 到 256 对应 [+127, +128] (2个token)
    
    left_bins = []
    # 从 127 开始往左，每 6 个一组
    curr = 127
    while curr >= 5:
        left_bins.append((curr - 5, curr, f"[-{128 - (curr - 5)}, -{128 - curr}]"))
        curr -= 6
    if curr >= 0:
        left_bins.append((0, curr, f"[-128, -{128 - curr}]"))
    left_bins.reverse() # 恢复从左到右的顺序
    
    center_bin = [(128, 128, "[0]")]
    
    right_bins = []
    # 从 129 开始往右，每 6 个一组
    curr = 129
    while curr <= 251:
        right_bins.append((curr, curr + 5, f"[+{curr - 128}, +{curr + 5 - 128}]"))
        curr += 6
    if curr <= 256:
        right_bins.append((curr, 256, f"[+{curr - 128}, +128]"))
        
    bins = left_bins + center_bin + right_bins
    
    # 4. 收集数据并计算平均值
    # 结构: [模型数 * 3, len(bins)]
    heatmap_data = []
    row_labels = []
    
    # 为了让模型之间可比，我们需要对每个模型的绝对距离进行归一化。
    # 论文中通常是按模型进行归一化，或者全局归一化。
    # 这里我们对每个模型，除以该模型在所有位点、所有偏移上的 95% 分位数，使其尺度在 [0, 1] 左右。
    for name in models:
        npz = ps_dir / name / "distance_profiles.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        r = z["euclidean_real"]  # [1000, 257]
        
        # 模型级归一化 (除以该模型全局 95% 分位数)
        p95 = np.nanpercentile(r, 95)
        r_norm = r / p95
        
        for cat in ["cds", "intron", "intergenic"]:
            idx = cat_indices[cat]
            if not idx:
                continue
            sub_r = r_norm[idx]  # [N_cat, 257]
            
            # 对每个 bin 计算平均值
            row_vals = []
            for start_idx, end_idx, label in bins:
                val = np.nanmean(sub_r[:, start_idx : end_idx + 1])
                row_vals.append(val)
            heatmap_data.append(row_vals)
            # 缩短模型显示名称
            short_name = name
            if "AgriGenome" in name:
                short_name = "AgriGenome"
            elif "rice_1B_stage2" in name:
                short_name = "rice_1B_stage2"
            elif "PlantCAD2" in name:
                short_name = "PlantCAD2-L"
            row_labels.append(f"{short_name} — {cat}")
            
    heatmap_data = np.array(heatmap_data)
    
    # 5. 绘图
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(heatmap_data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1.0)
    
    # 格子里面不要数字 (已去掉 ax.text 标注)
            
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(range(len(bins)))
    # 标签太多时，可以稀疏显示或者旋转
    ax.set_xticklabels([b[2] for b in bins], fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("Token offset from variant (binned by 6 tokens)", fontsize=12)
    ax.set_title("Variant Perception Profile by Functional Region (128 tokens, 6-token bins)", fontsize=14, pad=20)
    
    # 添加分割线，区分不同模型
    for i in range(1, len(models)):
        ax.axhline(i * 3 - 0.5, color="gray", lw=1.0, ls="--")
        
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("Normalized Euclidean Distance", fontsize=12)
    
    fig.tight_layout()
    png = out_dir / "heatmap_by_functional_region_128.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[heatmap] 已保存 {png}")


if __name__ == "__main__":
    main()
