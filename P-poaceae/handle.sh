#!/usr/bin/env bash
set -euo pipefail

# Usage: ./clip_pipeline.sh <fasta.fa> <lowcomp_ratio_threshold> [MIN_KEEP]
# Example: ./clip_pipeline.sh genome.fa 0.5 200

FASTA="$1"
THRESHOLD="${2:-0.5}"   # 阈值，比如 0.5 表示小写+低复杂度比例超过50%就去掉
MIN_KEEP="${3:-8192}"

# 输出文件夹
CLIP_DIR="clipped"
MIDDLE_DIR="middle"
mkdir -p "$CLIP_DIR" "$MIDDLE_DIR"

# derive base names
FASTA_BASE=$(basename "$FASTA")
PREFIX="${FASTA_BASE%.*}"

# 0. Run sdust to generate low-complexity bed
echo "Running sdust on $FASTA ..."
/mnt/zzb_new/peixunban/tanxinjiang/tools/sdust/sdust -w64 -t20 \
  "$FASTA" > "${FASTA}.sdust.bed"

SDUST_BED="${FASTA}.sdust.bed"

# 1. build .fai and genome file
if [ ! -s "${FASTA}.fai" ]; then
  samtools faidx "$FASTA"
fi
awk '{print $1"\t"$2}' "${FASTA}.fai" > "${MIDDLE_DIR}/${PREFIX}.genome"

# 2. make sliding windows: size=1024, step=512 (50% overlap)
WINDOW_SIZE=1024
STEP=512
WINDOWS_BED="${MIDDLE_DIR}/${PREFIX}.w${WINDOW_SIZE}_s${STEP}.bed"
bedtools makewindows -g "${MIDDLE_DIR}/${PREFIX}.genome" -w ${WINDOW_SIZE} -s ${STEP} > "${WINDOWS_BED}"

# 3. compute overlap between windows and sdust bed
TMP_OVERLAP="${MIDDLE_DIR}/${PREFIX}.windows.overlap.tmp"
bedtools intersect -a "${WINDOWS_BED}" -b "${SDUST_BED}" -wo > "${TMP_OVERLAP}" || true

LOWCOMP_WINDOWS="${MIDDLE_DIR}/${PREFIX}.windows_lowcomp.bed"
if [ -s "${TMP_OVERLAP}" ]; then
  awk -v ws=${WINDOW_SIZE} '
    BEGIN{ OFS="\t" }
    {
      key = $1 FS $2 FS $3
      overlap[key] += $NF
    }
    END{
      for (k in overlap){
        split(k, a, FS)
        if (overlap[k] > ws*0.5) {
          print a[1], a[2], a[3]
        }
      }
    }
  ' "${TMP_OVERLAP}" | sort -k1,1 -k2,2n > "${LOWCOMP_WINDOWS}"
else
  : > "${LOWCOMP_WINDOWS}"
fi

# 4. merge adjacent/overlapping low-complexity windows
MERGED_MASK="${MIDDLE_DIR}/${PREFIX}.merged_mask.bed"
if [ -s "${LOWCOMP_WINDOWS}" ]; then
  bedtools sort -i "${LOWCOMP_WINDOWS}" | bedtools merge -i - > "${MERGED_MASK}"
else
  : > "${MERGED_MASK}"
fi

# 5. clip fasta: 如果窗口内小写+低复杂度比例超过阈值，就去掉该片段
CLIPPED_FASTA="${CLIP_DIR}/${PREFIX}.clipped.fa"
python3 - <<PY
import sys
from pathlib import Path

fasta_file = Path("$FASTA")
mask_bed = Path("$MERGED_MASK")
out_fa = Path("$CLIPPED_FASTA")
threshold = float($THRESHOLD)
min_keep = int($MIN_KEEP)

# 读取 mask bed
mask_regions = {}
if mask_bed.exists() and mask_bed.stat().st_size > 0:
    with mask_bed.open() as f:
        for line in f:
            chrom, s, e = line.strip().split()[:3]
            s, e = int(s), int(e)
            mask_regions.setdefault(chrom, []).append((s,e))

def is_lowcomp_or_lowercase(chrom, start, end, seq):
    # 统计序列长度
    l = len(seq)
    if l == 0:
        return True
    # mask覆盖长度
    masked_len = 0
    for s,e in mask_regions.get(chrom, []):
        # 计算重叠
        overlap_s = max(start, s)
        overlap_e = min(end, e)
        if overlap_e > overlap_s:
            masked_len += (overlap_e - overlap_s)
    # 小写碱基数量
    lower_len = sum(1 for c in seq if c.islower())
    frac = (masked_len + lower_len)/l
    return frac >= threshold

total_bp = 0
clipped_bp = 0

with fasta_file.open() as fin, out_fa.open("w") as fout:
    header = None
    seq_parts = []
    for line in fin:
        line = line.rstrip()
        if line.startswith(">"):
            if header:
                chrom = header
                seq = "".join(seq_parts)
                total_bp += len(seq)
                if not is_lowcomp_or_lowercase(chrom, 0, len(seq), seq) and len(seq) >= min_keep:
                    fout.write(f">{header}\n")
                    for i in range(0, len(seq), 60):
                        fout.write(seq[i:i+60] + "\n")
                else:
                    clipped_bp += len(seq)
            header = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)
    # last record
    if header:
        chrom = header
        seq = "".join(seq_parts)
        total_bp += len(seq)
        if not is_lowcomp_or_lowercase(chrom, 0, len(seq), seq) and len(seq) >= min_keep:
            fout.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fout.write(seq[i:i+60] + "\n")
        else:
            clipped_bp += len(seq)

percent = (clipped_bp/total_bp*100) if total_bp>0 else 0
print(f"Total bp: {total_bp}", file=sys.stderr)
print(f"Clipped bp: {clipped_bp}", file=sys.stderr)
print(f"Percentage clipped: {percent:.2f}%", file=sys.stderr)
PY

echo "Clipped fasta saved to ${CLIPPED_FASTA}"
echo "Intermediate files stored in ${MIDDLE_DIR}"