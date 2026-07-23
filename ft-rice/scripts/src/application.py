import os
import sys
import json
import time
import argparse
import random
import glob
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import pyfaidx
import dotenv
from pathlib import Path
import re


def process_one_mutation(predictor, args, df_riceNavi, example_name, mut_fasta_path,
                         output_csv, output_dir, viewer=None):
    """处理单个突变文件，预测并记录结果"""
    # 获取该示例的信息
    row = df_riceNavi[df_riceNavi['Example'] == example_name]
    if row.empty:
        print(f"Warning: Example '{example_name}' not found in riceNavi.txt, skipping {mut_fasta_path}")
        return False

    gene_ID = row['MSU_ID'].values[0]
    chrom = str(row['Chr'].values[0])
    gene_start = int(row['start'].values[0])
    gene_end = int(row['end'].values[0])
    strand = row['strand'].values[0]
    window_start = int(row['window_start'].values[0]) - 1
    window_end = int(row['window_end'].values[0]) - 1
    length_diff = int(row['length_diff'].values[0])
    regulation = str(row['regulation'].values[0])

    print(f"\n{'='*60}")
    print(f"Processing {example_name}: {gene_ID} on {chrom}:{gene_start}-{gene_end}")
    print(f"{'='*60}")

    # 预测参考序列
    print("\n--- Predicting reference sequence ---")
    ref_predict = predictor.predict(chrom=chrom, start=window_start, end=window_end,
                                    biosample_names=args.biosample_names)

    # 预测突变序列
    print("\n--- Predicting mutant sequence ---")
    mut_predict = predictor.predict2(chrom=chrom, start=window_start, end=window_end,
                                     seq=mut_fasta_path, biosample_names=args.biosample_names)

    # 计算基因区域表达量（柱状图用）
    gene_relative_start = gene_start - window_start - 1
    gene_relative_end = gene_end - window_start
    ref_plus = ref_predict['values']['total_RNA-seq_+']['NIP_Panicle1'].float().cpu().numpy().flatten()
    ref_minus = ref_predict['values']['total_RNA-seq_-']['NIP_Panicle1'].float().cpu().numpy().flatten()
    mut_plus = mut_predict['values']['total_RNA-seq_+']['NIP_Panicle1'].float().cpu().numpy().flatten()
    mut_minus = mut_predict['values']['total_RNA-seq_-']['NIP_Panicle1'].float().cpu().numpy().flatten()

    expression_ref_plus = ref_plus[gene_relative_start:gene_relative_end].sum()
    expression_ref_minus = ref_minus[gene_relative_start:gene_relative_end].sum()
    expression_mut_plus = mut_plus[gene_relative_start:gene_relative_end + length_diff].sum()
    expression_mut_minus = mut_minus[gene_relative_start:gene_relative_end + length_diff].sum()

    plus_change = ((expression_mut_plus - expression_ref_plus) / expression_ref_plus) * 100 if expression_ref_plus != 0 else 0
    minus_change = ((expression_mut_minus - expression_ref_minus) / expression_ref_minus) * 100 if expression_ref_minus != 0 else 0

    # 打印统计
    print(f"Reference: + {expression_ref_plus:.2f}, - {expression_ref_minus:.2f}")
    print(f"Mutant:    + {expression_mut_plus:.2f}, - {expression_mut_minus:.2f}")
    print(f"Change:    + {plus_change:+.2f}%, - {minus_change:+.2f}%")

    # 保存柱状图（如果启用）
    if args.save_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35
        bars1 = ax.bar(x - width/2, [expression_ref_plus, expression_ref_minus], width, label='Wild-type',
                       color='#4A90E2', alpha=0.85, edgecolor='#2E5C8A', linewidth=1.5)
        bars2 = ax.bar(x + width/2, [expression_mut_plus, expression_mut_minus], width, label='Mutant',
                       color='#FF6B6B', alpha=0.85, edgecolor='#C92A2A', linewidth=1.5)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Expression Area', fontsize=12, fontweight='bold')
        ax.set_title(f'Gene {gene_ID} Expression Comparison ({example_name})',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(['Plus strand (+)', 'Minus strand (-)'], fontsize=11)
        ax.set_ylim(0, max(expression_ref_plus, expression_ref_minus, expression_mut_plus, expression_mut_minus) * 1.15)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='gray')
        ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        stats_text = f'+ strand: {plus_change:+.2f}%\n- strand: {minus_change:+.2f}%'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9, edgecolor='#CCCCCC', linewidth=1.5))
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{example_name}_expression.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"✅ Bar chart saved to: {plot_path}")

    # 保存轨道对比图（如果启用）
    # 保存轨道对比图（如果启用）
    # 保存轨道对比图（如果启用）
    if args.plot_tracks and viewer is not None:
        print("\n--- Generating track comparison plot ---")
        try:
            # 计算基因区域最大值（仍使用原始基因区间，与显示窗口无关）
            gene_relative_start = gene_start - window_start - 1
            gene_relative_end = gene_end - window_start
            ref_plus = ref_predict['values']['total_RNA-seq_+']['NIP_Panicle1'].float().cpu().numpy().flatten()
            ref_minus = ref_predict['values']['total_RNA-seq_-']['NIP_Panicle1'].float().cpu().numpy().flatten()
            mut_plus = mut_predict['values']['total_RNA-seq_+']['NIP_Panicle1'].float().cpu().numpy().flatten()
            mut_minus = mut_predict['values']['total_RNA-seq_-']['NIP_Panicle1'].float().cpu().numpy().flatten()

            gene_region_ref = slice(gene_relative_start, gene_relative_end)
            gene_region_mut = slice(gene_relative_start, gene_relative_end + length_diff)

            max_ref_plus = ref_plus[gene_region_ref].max() if len(ref_plus[gene_region_ref]) > 0 else 0
            max_ref_minus = ref_minus[gene_region_ref].max() if len(ref_minus[gene_region_ref]) > 0 else 0
            max_mut_plus = mut_plus[gene_region_mut].max() if len(mut_plus[gene_region_mut]) > 0 else 0
            max_mut_minus = mut_minus[gene_region_mut].max() if len(mut_minus[gene_region_mut]) > 0 else 0
            y_max = max(max_ref_plus, max_ref_minus, max_mut_plus, max_mut_minus) * 1.05

            # 确定显示窗口
            if args.track_window_pad is not None:
                display_start = max(0, gene_start - args.track_window_pad)
                display_end = gene_end + args.track_window_pad
            else:
                display_start = window_start
                display_end = window_end

            fig, axes = viewer.plot3(
                ref_predict, mut_predict,
                smoothing_sigma=args.smoothing_sigma,
                window_start=display_start,
                window_end=display_end,
                exclude_legend_labels=["NIP_Panicle1"]
            )

            # 设置 y 轴范围
            if len(axes) >= 3:
                axes[1].set_ylim(0, y_max)
                axes[2].set_ylim(0, y_max)
            else:
                for ax in axes[1:]:
                    if ax.get_ylabel() and ('RNA-seq' in ax.get_ylabel()):
                        ax.set_ylim(0, y_max)

            track_plot_path = os.path.join(output_dir, f"{example_name}_tracks.png")
            fig.savefig(track_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Track comparison plot saved to: {track_plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate track plot: {e}")

    # 写入 CSV
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a') as f:
        if not file_exists:
            header = '\t'.join([
                'model_name', 'example_name', 'gene_ID', 'chr', 'gene_start', 'gene_end', 'strand',
                'ref_plus', 'mut_plus', 'change_plus',
                'ref_minus', 'mut_minus', 'change_minus', 'regulation'
            ])
            f.write(header + '\n')
        result_line = '\t'.join([
            args.model_name, example_name, gene_ID, chrom, str(gene_start), str(gene_end), strand,
            f"{expression_ref_plus:.2f}", f"{expression_mut_plus:.2f}", f"{plus_change:.2f}%",
            f"{expression_ref_minus:.2f}", f"{expression_mut_minus:.2f}", f"{minus_change:.2f}%", regulation
        ])
        f.write(result_line + '\n')
    print(f"✅ Results appended to {output_csv}")

    return True