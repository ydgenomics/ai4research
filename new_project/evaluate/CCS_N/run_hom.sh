#!/bin/bash
# CCS-N 正式评测: 采样1000纯合位点 -> GVL写入 -> 双模型评测
# 用法: bash run_hom.sh [--max 1000] [--sample NH001]
set -e
cd "$(dirname "$0")"

PY=/root/miniconda3/envs/vllm/bin/python
BASE=/mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N
REF=/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.asm.ch.fa
DATA=$BASE/data
VCF=$DATA/251.SNP.bsnp.vcf.gz
SAMPLE=${SAMPLE:-NH001}
MAX=${MAX:-1000}
OUT=$BASE/output/hom_${MAX}

mkdir -p $OUT

# 1. 采样纯合位点 (若 BED 已存在则跳过)
BED=$DATA/carrier_hom.bed
if [ ! -f "$BED" ]; then
    echo "=== 1/3 采样纯合位点 ($MAX) ==="
    $PY $BASE/scripts/sample_carriers.py \
        --vcf $VCF --sample $SAMPLE --genotype hom \
        --out $BED --max-variants $MAX --seed 42
fi

# 2. GVL 写入 (若已存在则跳过)
GVL=$DATA/gvl_hom
if [ ! -d "$GVL" ]; then
    echo "=== 2/3 GVL 写入 ==="
    $PY -c "
import sys; sys.path.insert(0, '$BASE/scripts')
from prepare_data import write_gvl_dataset
write_gvl_dataset('$BED', '$VCF', '$REF', '$GVL', ['$SAMPLE'])
"
fi

# 3. 双模型评测
echo "=== 3/3 CCS-N 评测 ==="
$PY $BASE/scripts/run_ccs_n.py \
    --model-dir /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/rice_1B_stage2_8k_hf \
    --model-dir /mnt/rice/default/Workspace/yangdong/ai4research/new_project/script/output/variant_cpt_32k_10samp/final \
    --data-dir $DATA --gvl-dir $GVL --out-dir $OUT \
    --ref $REF --sample $SAMPLE --batch-size 16 --device cuda:1 \
    2>&1 | tee $OUT/run.log

echo "=== 完成: $OUT/ccs_n_summary.json ==="
