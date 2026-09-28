#!/bin/bash
# CCS-N 冒烟测试: 1样本(NH001) 5000 SNP
set -e
cd "$(dirname "$0")"

PY=/root/miniconda3/envs/vllm/bin/python
BASE=/mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N
REF=/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.asm.ch.fa
DATA=$BASE/data
VCF=$DATA/251.SNP.bsnp.vcf.gz

# 1. 生成 5000 SNP 窗口 BED
$PY -c "
import sys; sys.path.insert(0, '$BASE/scripts')
from prepare_data import build_windows_bed
build_windows_bed('$VCF', '$DATA/smoke_windows.bed', 256, 5000)
"

# 2. GVL 写入 (NH001)
$PY -c "
import sys; sys.path.insert(0, '$BASE/scripts')
from prepare_data import write_gvl_dataset
import shutil
shutil.rmtree('$DATA/gvl_smoke', ignore_errors=True)
write_gvl_dataset('$DATA/smoke_windows.bed', '$VCF', '$REF', '$DATA/gvl_smoke', ['NH001'])
"

# 3. 运行 CCS-N (base + ft)
$PY $BASE/scripts/run_ccs_n.py \
    --model-dir /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/rice_1B_stage2_8k_hf \
    --model-dir /mnt/rice/default/Workspace/yangdong/ai4research/new_project/script/output/variant_cpt_32k_10samp/final \
    --data-dir $DATA \
    --gvl-dir $DATA/gvl_smoke \
    --out-dir $BASE/output/smoke \
    --ref $REF \
    --sample NH001 \
    --batch-size 16 \
    --device cuda:1

echo "冒烟测试完成"
