Optimized tool selection

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb)

这个目录很大，包含 **1000 个基因子目录** + 1 个清单文件。让我进一步查看里面的结构。 

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb/001_RETN)

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb/100_ITGB3BP)

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb/500_FBXO2)

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb/manifest.tsv)

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb/manifest.tsv)

Read [](file:///mnt/rice/default/Workspace/xuxiaolong/human/SNPembedding_top1000_common_TSS_1Mb/001_RETN/meta.json)

## 目录概述

### 总览

| 项目 | 数量 |
|------|------|
| 基因子目录 | **1000 个**（`001_RETN` ~ `1000_UMODL1`） |
| 清单文件 | manifest.tsv（1 个） |

---

### 每个基因子目录内部结构

每个子目录（如 `001_RETN/`）包含：

| 文件 | 数量 | 说明 |
|------|------|------|
| meta.json | **1 个** | 基因元信息（见下方字段说明） |
| `CIMA-Hxxx_CIMA-Hxxx.vcf.pt` | **~101 个** | 每个个体一个 SNP embedding 文件 |

**meta.json 字段：**
```json
{
  "rank": 1,                          // 排名
  "gene_id": "RETN",                  // 基因 symbol
  "chrom": "chr19",                   // 染色体
  "tss": 7669049,                     // TSS 位置
  "start": 7169049,                   // 窗口起始（TSS - 500kb）
  "end": 8169049,                     // 窗口结束（TSS + 500kb）
  "n_embedding_snps": 26091,          // 1KGP 参考集 SNP embedding 数量
  "n_snps": 2583,                     // CIMA 队列实际覆盖的 SNP 数量
  "n_samples": 101,                   // 样本数
  "status": "complete",               // 状态（complete / skipped_no_snps）
  "chromosome_embedding": "..." ,     // 染色体级 SNP embedding 文件路径
  "cohort_vcf": "...",                // CIMA 群体 VCF 路径
  "same_position_alt_variants_retained": true
}
```

**`.vcf.pt` 文件：**
- 命名格式：`CIMA-{个体ID}_CIMA-{个体ID}.vcf.pt`
- 覆盖 **101 个 CIMA 个体**（H005~H324）
- 每个文件是该个体在该基因 TSS ±500kb 窗口内所有 SNP 的 **hidden state embedding**（torch tensor，推测形状为 `[n_snps, hidden_dim]`，`hidden_dim` 通常为 1024）

---

### manifest.tsv 结构

包含所有 1000 个基因的完整索引，共 **13 列**：

| 列名 | 说明 |
|------|------|
| `rank` | 排序编号 |
| `gene_id` | 基因 symbol |
| `chrom` | 染色体 |
| `tss` | TSS 位置 |
| `start` / `end` | ±500kb 窗口范围 |
| `n_embedding_snps` | 1KGP 参考集的 SNP embedding 数量 |
| `n_snps` | CIMA 队列实际覆盖的 SNP 数量 |
| `n_samples` | 个体数量 |
| `status` | 处理状态（大部分 `complete`，少数如 TMEM176B 为 `skipped_no_snps`） |
| `chromosome_embedding` | 对应染色体的参考 embedding 文件路径 |
| `cohort_vcf` | CIMA 共识 VCF 路径 |
| `same_position_alt_variants_retained` | 是否保留同位置替代变异 |

---

### 关键特征

1. **窗口设计**：每个基因取 **TSS ± 500kb**（1Mb 窗口）范围内的 SNP
2. **数据来源**：使用 Genos 预训练模型从染色体级 embedding 中提取对应位置的 SNP hidden states
3. **个体覆盖**：101 个 CIMA 个体，使用共识 VCF（`maf01con.only_snp.vcf.gz`，MAF ≥ 0.01 的 SNP）
4. **大多数基因状态为 `complete`**，少数（如 TMEM176A/B）因窗口内无 SNP 被跳过（`skipped_no_snps`，目录仍然存在但为空）