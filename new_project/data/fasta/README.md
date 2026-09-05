```bash
BCF=/mnt/rice/default/Workspace/xuxiaolong/mamba/envs/tools/bin/bcftools
FASTA="/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/osa1_r7.asm.ch.fa"
VCF="/mnt/rice/data/Lianguang_shang/251.SNP.vcf.gz"
$BCF view -h $VCF | grep -i "reference"
# 仅仅校验前 10 万行位点（几秒钟出结果）
$BCF view $VCF | head -n 100000 | $BCF norm --check-ref w -f $FASTA - -o /dev/null
# 添加 -P/--threads 8 (根据你服务器的 CPU 核心数调整，如 8 或 16)
$BCF norm --threads 8 --check-ref w -f $FASTA $VCF -o /dev/null
```