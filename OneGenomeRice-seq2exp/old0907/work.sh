bash /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/data_prepare.sh \
"NH001,NH002,NH003,NH005,NH006,NH007,NH008,NH009" "NH013" "NH014" "leaf_ck,leaf_salt" 0 "Chr5" \
/mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/data

python scripts/filter_sequence_regions.py --indices-dir data/indices --region 90 100

cd /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/
bash run_train.sh "mx"

# cd /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/
# bash run_train.sh "no"

conda activate vllm
bash /mnt/rice/default/Workspace/yangdong/demo/run_predict.sh


/mnt/rice/default/Workspace/xuxiaolong/mamba/envs/model_env/bin/python \
/mnt/rice/default/Workspace/yangdong/ai4research/ft-rice/scripts/evaluation3/run_evaluation.py \
--config /mnt/rice/default/Workspace/yangdong/ai4research/OneGenomeRice-seq2exp/old0907/config.yaml

conda activate vllm
bash /mnt/rice/default/Workspace/yangdong/demo/scripts0/evaluation/run_evaluation-yd.sh "Chr09"