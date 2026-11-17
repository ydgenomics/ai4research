```bash
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/rice_mut && \
mkdir -p logs && \
bash scripts/pregen_bigwigs.sh \
  --chrom Chr1 Chr2 Chr3 Chr4 Chr5 Chr6 Chr7 Chr8 Chr9 Chr10 Chr11 Chr12 \
  > logs/pregen_2gpu_chr1-12.out 2>&1 &
```