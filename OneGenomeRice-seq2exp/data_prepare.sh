# 选择要使用的个体，使用的组织/处理
# 个体sample，组织/处理biosample
# bw文件命名规则：${biosample}_${sample}_1.bw (不分链特异性)
# genome文件命名规则：sample.fa and sample.gff; sample.vcf
# 基因组文件在一开始就要对齐好，所有的fa文件和gff注释文件，统一用chr01这种


bw_dir=""
genome_dir=""
out_dir=""

train_sample=("P1" "P4" "P6") # "train_" +
valid_sample=("P7") # "valid_" +
test_sample=("P11") # "test_" +
tissue=("CSQ" "YG")

# 构建成类似 train_P1_CSQ train_P1_YG
# 将已有数组接入关联数组
declare -A samples

samples["train"]="${train_sample[*]}"
samples["valid"]="${valid_sample[*]}"
samples["test"]="${test_sample[*]}"

for group in "${!samples[@]}"; do
    for sample in ${samples[$group]}; do
        for t in "${tissue[@]}"; do
            dir_name="${group}_${sample}_${t}"
            echo "${dir_name}"
            mkdir -p "${out_dir}/${dir_name}"
        done
    done
done

# 构建meta.csv
# /mnt/rice/default/Workspace/yangdong/gene_expression_prediction/data/indices/test_YMY_YG_CSQ_Z_MFZ_P5_multitrack/bigWig_labels_meta.csv

# 构建index.json


# 构建window.csv
# get_window_csv.py

# generate index index_stat.json and sequence_split_train.csv file
for species_file in ${train_species_file[@]};do
    species_file_dir=${dir}/ref/${species_file}-new.fasta
    species=$species_file
    python ${outdir}/scripts/data_preprocess/sequence_split_and_meta_extract2.py \
        --genome_fasta $species_file_dir \
        --chromosomes Chr01 Chr02 Chr03 Chr04 Chr05 Chr06 Chr07 Chr08 Chr09 Chr10 Chr11 Chr12 \
        --window_size 32768 \
        --overlap 16384 \
        --meta_csv ref/${tissues}_${species}.csv \
        --assay_titles "total RNA-seq" \
        --biosample_names "rice" \
        --output_base_dir ${outdir}/data/indices/test_${tissues}_${species}_multitrack \
        --processed_bw_dir ${outdir}/data/processed/renorm_bigwig_output
done