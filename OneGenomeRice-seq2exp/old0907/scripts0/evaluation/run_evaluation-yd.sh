#!/usr/bin/env bash
set -euo pipefail
#-------------------------------------for train data evaluation-------------------------------------#
# 配置
chr=${1:?用法: $0 <chr>}

CHROM_SIZES="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P7-new.chrom.sizes"
EXPR_COL="predicted_expression"
OUT_DIR="/mnt/rice/default/Workspace/yangdong/demo/output/202608051219/outputs/P7"
METRICS_SCRIPT="/mnt/rice/default/Workspace/yangdong/demo/scripts0/evaluation/calc_metrics_for_batch_bw2.py"
SEG_EVAL_SCRIPT="/mnt/rice/default/Workspace/yangdong/demo/scripts0/evaluation/csv_seg_eval.py"
FASTA="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P7-new.fasta"

# 遍历目录下每个预测 CSV，以输入文件名为统一前缀「广播」生成全部输出
shopt -s nullglob
csvs=("$OUT_DIR"/*.csv)
if [ ${#csvs[@]} -eq 0 ]; then
    echo "未找到 CSV 文件：$OUT_DIR"
    exit 0
fi

for csv in "${csvs[@]}"; do
    base="${csv%.csv}"   # 输出前缀 = 输入 CSV 去掉扩展名

    # 1) CSV -> npy/bw（生成 $base.bw.npy 预测、$base.bw_true.npy 真实值）
    echo "Convert: $csv -> $base.bw"
    python scripts/evaluation/csv2bw2.py \
        --csv "$csv" \
        --output "$base.bw" \
        --chrom_sizes "$CHROM_SIZES" \
        --expression_col "$EXPR_COL"

    # 2) 碱基级指标（依赖上一步生成的预测 npy）
    pred_npy="$base.bw.npy"
    if [ -f "$pred_npy" ]; then
        stats="${base}_track-level_stats.txt"
        raw_npy="$base.bw_true.npy"
        echo "Metrics: $pred_npy vs $raw_npy -> $stats"
        python "$METRICS_SCRIPT" \
            --pred_files "$pred_npy" \
            --raw_files "$raw_npy" \
            --output "$stats" \
            --fasta "$FASTA" \
            --chrom "$chr"
    else
        echo "跳过 metrics（未找到预测 npy）: $pred_npy"
    fi

    # 3) 片段级评估（skip_bigwig 仅相关性分析，输出 $base.npy / $base_true.npy）
    python "$SEG_EVAL_SCRIPT" \
        --csv "$csv" \
        --output "$base" --skip_bigwig \
        --chrom_sizes "$CHROM_SIZES" \
        --expression_col "$EXPR_COL"
done