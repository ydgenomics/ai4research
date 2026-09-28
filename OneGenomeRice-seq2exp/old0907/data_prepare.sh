#!/bin/bash
# 260527

# set -euo pipefail

# ==============================================================================
# 帮助信息函数
# ==============================================================================
usage() {
    echo "Usage: $0 [train_species] [valid_species] [test_species] [tissue] [changeMinus] [chromosomes]"
    echo ""
    echo "参数说明 (外部传入多个值时请用逗号 ',' 连接):"
    echo "  1. train_species : 训练集物种 (默认: P1,P4,P6)"
    echo "  2. valid_species : 验证集物种 (默认: P7)"
    echo "  3. test_species  : 测试集物种 (默认: P11)"
    echo "  4. tissue        : 组织类型   (默认: CSQ,YG)"
    echo "  5. changeMinus   : 调整参数   (默认: 0)"
    echo "  6. chromosomes   : 染色体列表 (默认: Chr01到Chr12)"
    echo ""
    echo "示例:"
    echo "  $0 \"P1,P2\" \"P3\" \"P4\" \"CSQ\" 1 \"Chr01,Chr02\""
    echo "  使用默认值调用: $0"
    exit 1
}

# 如果输入 -h 或 --help，显示帮助信息
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# ==============================================================================
# 1. 接收外部位置参数，并提供默认值 (使用 逗号 分隔的字符串形式)
# ==============================================================================
raw_train_species="${1:-P1,P2,P3,P4,P5,P6,P7,P10,P11,P12,P13,P14}"
raw_valid_species="${2:-P9}"
raw_test_species="${3:-P9}"
raw_tissue="${4:-CSQ,Z}"
changeMinus="${5:-0}"
raw_chromosomes="${6:-Chr06}"
output_dir="${7:-./data}"

# ==============================================================================
# 2. 将逗号分隔的字符串反解为 Bash 真实数组
# ==============================================================================
# 核心逻辑：将 IFS（内部字段分隔符）临时设为逗号，通过 read -a 读入数组
IFS=',' read -r -a train_species_file <<< "$raw_train_species"
IFS=',' read -r -a valid_species_file <<< "$raw_valid_species"
IFS=',' read -r -a test_species_file  <<< "$raw_test_species"
IFS=',' read -r -a tissue             <<< "$raw_tissue"
IFS=',' read -r -a CHROMOSOMES        <<< "$raw_chromosomes"

# ==============================================================================
# 3. 验证与打印（验证环境配置是否成功还原）
# ==============================================================================
echo "==================== 运行参数配置 ===================="
echo "Train Species Array: ${train_species_file[*]} (长度: ${#train_species_file[@]})"
echo "Valid Species Array: ${valid_species_file[*]} (长度: ${#valid_species_file[@]})"
echo "Test Species Array:  ${test_species_file[*]}  (长度: ${#test_species_file[@]})"
echo "Tissue Array:        ${tissue[*]}        (长度: ${#tissue[@]})"
echo "Change Minus:        ${changeMinus}"
echo "Chromosomes Count:   ${#CHROMOSOMES[@]} 条染色体"
echo "======================================================"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/mnt/rice/default/Workspace/yangdong/ai4research/DATA/shang_data/"
OUTDIR="${SCRIPT_DIR}/data"
WORK_REF_DIR="${OUTDIR}/ref"
BW_OUTPUT_DIR="${OUTDIR}/processed/renorm_bigwig_output"
INDICES_DIR="${OUTDIR}/indices"

CSV_GENERATOR="${SCRIPT_DIR}/scripts/csv.generator.py"
SEQUENCE_SPLIT_SCRIPT="${SCRIPT_DIR}/scripts/sequence_split_and_meta_extract2.py"
CHROM_FEATURE_SCRIPT="${SCRIPT_DIR}/scripts/build_chromosome_features.py"

BUILD_CHROMOSOME_FEATURES="${BUILD_CHROMOSOME_FEATURES:-1}"
CHROM_FEATURE_MAX_WINDOWS_PER_CHROM="${CHROM_FEATURE_MAX_WINDOWS_PER_CHROM:-128}"
CHROM_FEATURE_MAX_VALUES_PER_TRACK_CHROM="${CHROM_FEATURE_MAX_VALUES_PER_TRACK_CHROM:-200000}"


mkdir -p "${OUTDIR}" "${WORK_REF_DIR}" "${BW_OUTPUT_DIR}" "${INDICES_DIR}"

if [[ ! -f "${CSV_GENERATOR}" ]]; then
    echo "Missing csv.generator.py: ${CSV_GENERATOR}" >&2
    exit 1
fi

if [[ ! -f "${SEQUENCE_SPLIT_SCRIPT}" ]]; then
    echo "Missing sequence_split_and_meta_extract2.py: ${SEQUENCE_SPLIT_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${CHROM_FEATURE_SCRIPT}" ]]; then
    echo "Missing build_chromosome_features.py: ${CHROM_FEATURE_SCRIPT}" >&2
    exit 1
fi

bwlist_file="${WORK_REF_DIR}/bwList.txt"
rm -f "${bwlist_file}"

append_bw_paths() {
    local species
    local atissue
    for species in "$@"; do
        for atissue in "${tissue[@]}"; do
            local bw_path="${DATA_DIR}/bw/${atissue}/${species}.bw"
            if [[ ! -f "${bw_path}" ]]; then
                echo "Missing BigWig file: ${bw_path}" >&2
                exit 1
            fi
            printf '%s\n' "${bw_path}" >> "${bwlist_file}"
        done
    done
}

append_bw_paths "${train_species_file[@]}"
append_bw_paths "${valid_species_file[@]}"
append_bw_paths "${test_species_file[@]}"

if [[ "${changeMinus}" -eq 1 ]]; then
    shopt -s nullglob
    for bw_file in "${DATA_DIR}"/*minus*.bw; do
        file_name="$(basename "${bw_file}")"
        name="${file_name%.*}"
        species="${name%%_*}"
        "${DATA_DIR}/trans_minus2plus.sh" "${bw_file}" "${DATA_DIR}/${name}2.bw" "${DATA_DIR}/ref/${species}"*.chrom.sizes
    done
    shopt -u nullglob
fi

# while IFS= read -r bw_file; do
#     cp "${bw_file}" "${BW_OUTPUT_DIR}/"
# done < "${bwlist_file}"
while IFS= read -r bw_file; do
    # 跳过空行
    [[ -z "${bw_file}" ]] && continue

    # 1. 获取文件名带后缀: NH001.bw
    filename=$(basename "${bw_file}")
    
    # 2. 获取文件名不带后缀: NH001
    filename_no_ext="${filename%.bw}"
    
    # 3. 获取上级目录名: leaf_ck
    parent_dir=$(basename "$(dirname "${bw_file}")")
    
    # 4. 拼接新文件名: leaf_ck_NH001_1.bw
    new_filename="${parent_dir}_${filename_no_ext}_1.bw"
    
    # 5. 执行复制
    cp "${bw_file}" "${BW_OUTPUT_DIR}/${new_filename}"
done < "${bwlist_file}"

tissues=""
for atissue in "${tissue[@]}"; do
    tissues="${tissues}_${atissue}"
done
tissues="${tissues:1}"

rm -f "${bwlist_file}"

generate_meta_csv() {
    local group_name="$1"
    shift
    local species_list=("$@")
    local bwlists=()
    local atissue
    local species

    for atissue in "${tissue[@]}"; do
        local list_file="${WORK_REF_DIR}/bwList_${group_name}_${atissue}.txt"
        rm -f "${list_file}"
        for species in "${species_list[@]}"; do
            local bw_path="${DATA_DIR}/bw/${atissue}/${species}.bw"
            printf '%s\n' "${bw_path}" >> "${list_file}"
        done
        bwlists+=("${list_file}")
    done

	python "${CSV_GENERATOR}" \
	    --tissues "${tissue[@]}" \
	    --species_range "${species_list[@]}" \
	    --bwlist "${bwlists[@]}" \
	    --output_dir "${WORK_REF_DIR}"

	for species in "${species_list[@]}"; do
	    local meta_csv="${WORK_REF_DIR}/${tissues}_${species}.csv"
	    python - "${meta_csv}" <<'PY'
import sys
import pandas as pd

meta_csv = sys.argv[1]
df = pd.read_csv(meta_csv)

required = {"target_file_name", "organism", "biosample_name"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"{meta_csv}: missing columns {sorted(missing)}")

# The model interprets biosample_name as the per-head output channel.
# CSQ/YG must therefore be distinct biosamples, not both collapsed to "rice".
df["biosample_name"] = df["organism"].astype(str)
df.to_csv(meta_csv, index=False)

print(f"Updated biosample_name from organism in {meta_csv}: {df['biosample_name'].tolist()}")
PY
	done

	rm -f "${bwlists[@]}"
}

generate_meta_csv train "${train_species_file[@]}"
generate_meta_csv valid "${valid_species_file[@]}"
generate_meta_csv test "${test_species_file[@]}"

generate_indices() {
    local split_name="$1"
    shift
    local species_list=("$@")
    local species

    for species in "${species_list[@]}"; do
        local species_fasta="${DATA_DIR}/genome/${species}.fasta"
        local species_meta="${WORK_REF_DIR}/${tissues}_${species}.csv"
        local output_dir="${INDICES_DIR}/${split_name}_${tissues}_${species}_multitrack"

        if [[ ! -f "${species_fasta}" ]]; then
            echo "Missing FASTA file: ${species_fasta}" >&2
            exit 1
        fi

        if [[ ! -f "${species_meta}" ]]; then
            echo "Missing metadata CSV: ${species_meta}" >&2
            exit 1
        fi

	        python "${SEQUENCE_SPLIT_SCRIPT}" \
            --genome_fasta "${species_fasta}" \
            --chromosomes "${CHROMOSOMES[@]}" \
	            --window_size 32768 \
	            --overlap 16384 \
	            --meta_csv "${species_meta}" \
	            --assay_titles "total RNA-seq" \
	            --biosample_names "${tissue[@]}" \
	            --output_base_dir "${output_dir}" \
	            --processed_bw_dir "${BW_OUTPUT_DIR}"
	    done
}

generate_indices train "${train_species_file[@]}"
generate_indices valid "${valid_species_file[@]}"
generate_indices test "${test_species_file[@]}"

build_chromosome_features() {
    if [[ "${BUILD_CHROMOSOME_FEATURES}" != "1" && "${BUILD_CHROMOSOME_FEATURES}" != "true" && "${BUILD_CHROMOSOME_FEATURES}" != "TRUE" ]]; then
        echo "Skipping chromosome feature JSON because BUILD_CHROMOSOME_FEATURES=${BUILD_CHROMOSOME_FEATURES}"
        return
    fi

    local sequence_splits=()
    local index_stats=()
    local species

    for species in "${train_species_file[@]}"; do
        local train_dir="${INDICES_DIR}/train_${tissues}_${species}_multitrack"
        local sequence_split="${train_dir}/sequence_split_train.csv"
        local index_stat="${train_dir}/index_stat.json"

        if [[ ! -f "${sequence_split}" ]]; then
            echo "Missing training sequence split for chromosome features: ${sequence_split}" >&2
            exit 1
        fi
        if [[ ! -f "${index_stat}" ]]; then
            echo "Missing training index_stat for chromosome features: ${index_stat}" >&2
            exit 1
        fi

        sequence_splits+=("${sequence_split}")
        index_stats+=("${index_stat}")
    done

    local chrom_feature_json="${INDICES_DIR}/chromosome_features_train_${train_species_file[*]}.json"
    chrom_feature_json="${chrom_feature_json// /_}"

    python "${CHROM_FEATURE_SCRIPT}" \
        --sequence_splits "${sequence_splits[@]}" \
        --index_stats "${index_stats[@]}" \
        --chromosomes "${CHROMOSOMES[@]}" \
        --max_windows_per_chrom "${CHROM_FEATURE_MAX_WINDOWS_PER_CHROM}" \
        --max_values_per_track_chrom "${CHROM_FEATURE_MAX_VALUES_PER_TRACK_CHROM}" \
        --output "${chrom_feature_json}"

    echo "Chromosome feature JSON: ${chrom_feature_json}"
}

build_chromosome_features

echo "data_prepare.sh finished successfully."
