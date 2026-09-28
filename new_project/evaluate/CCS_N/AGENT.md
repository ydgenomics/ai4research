# CCS-N Variant Perception Evaluation Agent

This agent is designed to evaluate DNA language models' variant perception capabilities using the Contextual Consensus Score (CCS-N) metric across different genomic functional regions (CDS, Intron, Intergenic).

## Workflow Overview

1. **Data Preparation**:
   - Extract genomic regions centered at homozygous carrier variants of sample `NH001` with a window size of 2048 bp.
   - Generate reference (`ref`), haplotype (`alt`), and random control (`rand`) sequences.
   - Cache the dataset using `genvarloader` (GVL) at `data/gvl_hom`.

2. **Model Evaluation**:
   - Run forward passes for 7 models to extract hidden states.
   - Compute Euclidean and Cosine distance profiles centered at the variant token with a radius of $R_{\text{max}} = 128$ (total 257 tokens).
   - For k-mer models, use a robust target-variant token localization strategy to avoid centering on background variants.
   - Save per-site CCS-N scores and distance profiles.

3. **Visualization**:
   - Classify the 1000 variant sites into CDS, Intron, and Intergenic regions using the GFF3 annotation.
   - Normalize distance profiles by dividing by each model's 95th percentile distance.
   - Group token offsets into 6-token bins (keeping the center `[0]` independent).
   - Plot a comprehensive heatmap comparing all models across functional regions.

## Execution Commands

### Step 1: Run CCS-N Evaluation Pipeline
```bash
/root/miniconda3/envs/vllm/bin/python /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/scripts/run_ccs_n.py \
    --model-dir /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/rice_1B_stage2_8k_hf \
    --model-dir /mnt/rice/default/Workspace/yangdong/ai4research/new_project/script/output/variant_cpt_32k_10samp/final \
    --model-dir /mnt/rice/default/Workspace/Rice-Genome/model/RiceModel2/AgriGenome_4n8a_1.2b_8k_pt_32k_cpt_stage1_iter14000_hf \
    --model-dir /mnt/rice/default/Workspace/Rice-Genome/model/agront_1b \
    --model-dir /mnt/rice/default/Workspace/Rice-Genome/model/Botanic0-L \
    --model-dir /mnt/rice/default/Workspace/Rice-Genome/model/PlantCAD2-Large-l48-d1536 \
    --model-dir /mnt/rice/default/Workspace/Rice-Genome/model/NTv3_650M_pre \
    --ref /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.asm.ch.fa \
    --data-dir /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/data \
    --gvl-dir /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/data/gvl_hom \
    --out-dir /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/output/hom_1000_multi7 \
    --save-per-site \
    --save-distance-profiles \
    --device cuda:1
```

### Step 2: Generate Heatmap by Functional Region
```bash
/root/miniconda3/envs/vllm/bin/python /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/scripts/heatmap_by_functional_region.py \
    /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/output/hom_1000_multi7 \
    --bed /mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/data/carrier_hom.bed \
    --gff /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.all_models.gff3
```
