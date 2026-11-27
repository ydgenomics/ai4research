```bash
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/OGR_Prediction && \
mkdir -p logs && \
bash scripts/pregen_bigwigs.sh \
  --chrom Chr1 Chr2 Chr3 Chr4 Chr5 Chr6 Chr7 Chr8 Chr9 Chr10 Chr11 Chr12 \
  > logs/pregen_2gpu_chr1-12.out 2>&1 &
```

> 批量预生成预测 bigWig（供 IGV 浏览 / 离线查看）；结果写入 `.env` 的 `BACKEND_PREDICTION_CACHE`。该目录仅在 **web 模式**（`start_all.sh web`/`monitor-web`，`OGR_ENABLE_STARTUP_CLEAN=1`）启动后端时会被清空；纯 API 模式（`start`/`monitor`）不清理。
> `--chrom` 使用 FASTA 实际染色体名（osa1_r7 为 `Chr1`…`Chr12`）。