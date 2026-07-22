优化json文件，以后每个样本只包含两个东西
分析一下模型主要在哪里画了很多时间，优化计算效率，subset窗口的时间，窗口文件token化的时间，然后才是真正的训练的时间

- inputs
  - genome_fasta
  - genome_gff
  - chromosomes
  - window_size
  - overlap
  - assay_names: ["RNA-seq"]
  - biosample_names: ["CSQ", "YG"]
  - 
- outputs
- counts
- created_at