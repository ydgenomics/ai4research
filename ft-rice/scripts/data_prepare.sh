# 选择要使用的个体，使用的组织/处理
# 个体sample，组织/处理biosample
# bw文件命名规则：${biosample}_${sample}_1.bw (不分链特异性)
# genome文件命名规则：sample.fa and sample.gff; sample.vcf


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
        for t in "${tissue[@]}; do
            dir_name="${group}_${sample}_${t}"
            echo "${dir_name}"
            mkdir -p "${out_dir}/${dir_name}"
        done
    done
done

# 构建meta.csv


# 构建index.json


# 构建window.csv
get_window_csv.py
