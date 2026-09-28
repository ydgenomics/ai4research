#source .env

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_COMM_OVERLAP=0
export NCCL_P2P_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=1

export CUDA_VISIBLE_DEVICES=0,1

DISTRIBUTED_ARGS=(
    --nnodes 1
    --nproc_per_node 2
    --node_rank 0
    --master_addr localhost
    --master_port 29501
)

# #  1 GPU
# export CUDA_VISIBLE_DEVICES=0

# DISTRIBUTED_ARGS=(
#     --nnodes 1
#     --nproc_per_node 1
#     --node_rank 0
#     --master_addr localhost
#     --master_port 29501
# )

torchrun ${DISTRIBUTED_ARGS[@]} predict.py \
  --model_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
  --tokenizer_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
  --ckpt_path /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/output/202609081121/checkpoint-380/model.safetensors \
  --sequence_split_test data/indices/test_leaf_ck_leaf_salt_NH014_multitrack/sequence_split_train.csv \
  --index_stat_json data/indices/test_leaf_ck_leaf_salt_NH014_multitrack/index_stat.json \
  --bigWig_labels_meta data/indices/test_leaf_ck_leaf_salt_NH014_multitrack/bigWig_labels_meta.csv \
  --max_predict_samples 10000 \
  --output_base_dir output/202609081121/outputs/NH014 \
  --test_chromosomes Chr5 \
  --batch_size 3 \
  --num_workers 8 \
  --use_flash_attn 

torchrun ${DISTRIBUTED_ARGS[@]} predict.py \
  --model_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
  --tokenizer_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
  --ckpt_path /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/output/202609081121/checkpoint-380/model.safetensors \
  --sequence_split_test data/indices/valid_leaf_ck_leaf_salt_NH013_multitrack/sequence_split_train.csv \
  --index_stat_json data/indices/valid_leaf_ck_leaf_salt_NH013_multitrack/index_stat.json \
  --bigWig_labels_meta data/indices/valid_leaf_ck_leaf_salt_NH013_multitrack/bigWig_labels_meta.csv \
  --max_predict_samples 10000 \
  --output_base_dir output/202609081121/outputs/NH013 \
  --test_chromosomes Chr5 \
  --batch_size 3 \
  --num_workers 8 \
  --use_flash_attn

train_species=(NH001 NH002 NH003 NH005 NH006 NH007 NH008 NH009)
for species in "${train_species[@]}"; do
  torchrun ${DISTRIBUTED_ARGS[@]} predict.py \
    --model_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
    --tokenizer_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
    --ckpt_path /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/output/202609081121/checkpoint-380/model.safetensors \
    --sequence_split_test data/indices/train_leaf_ck_leaf_salt_${species}_multitrack/sequence_split_train.csv \
    --index_stat_json data/indices/train_leaf_ck_leaf_salt_${species}_multitrack/index_stat.json \
    --bigWig_labels_meta data/indices/train_leaf_ck_leaf_salt_${species}_multitrack/bigWig_labels_meta.csv \
    --max_predict_samples 10000 \
    --output_base_dir output/202609081121/outputs/${species} \
    --test_chromosomes Chr5 \
    --batch_size 3 \
    --num_workers 8 \
    --use_flash_attn 
done