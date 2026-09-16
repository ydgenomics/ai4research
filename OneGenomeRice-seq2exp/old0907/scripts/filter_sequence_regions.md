# filter_sequence_regions.py 使用说明

按**染色体区域（百分比）**过滤滑动窗口，缩小训练集规模，并同步更新
`sequence_split_train.csv` / `bigWig_labels_meta.csv` / `index_stat.json`。

- 脚本位置：`scripts/filter_sequence_regions.py`
- 文档位置：`scripts/filter_sequence_regions.md`
- 依赖：`pandas`、`numpy`；仅当使用 `--recompute-track-stats` 时需要 `pyBigWig`

---

## 1. 解决的问题

`data_prepare.sh` 会为每个样本（如 `NH001`）生成一个 multitrack 目录：

```
data/indices/train_leaf_ck_leaf_salt_NH001_multitrack/
├── sequence_split_train.csv   # 染色体滑动窗口列表 (chromosome, start, end)
├── bigWig_labels_meta.csv     # track 级元数据 (target_file_name, nonzero_mean, ...)
└── index_stat.json            # 数据集统计 (counts.num_samples 等)
```

其中 `sequence_split_train.csv` 覆盖了整条染色体。当希望**只保留染色体上某个
区域**（例如只训练后半段 50%~100%）来缩小训练集时，需要同时改三个文件：

| 文件 | 改动 |
| --- | --- |
| `sequence_split_train.csv` | 仅保留区域内的窗口 |
| `index_stat.json` | 更新 `counts.num_samples`、`counts.num_samples_by_chromosome`，并新增 `filtered` 追溯字段 |
| `bigWig_labels_meta.csv` | 可选：用保留区域重算每条 track 的 `nonzero_mean`（`--recompute-track-stats`） |

> `bigWig_labels_meta.csv` 本身是 track 级元数据，与窗口位置无关，**仅过滤窗口
> 时无需修改**。只有当希望"track 归一化均值"与新的区域分布一致时才重算。

---

## 2. 快速开始

在 `old0907` 目录下执行（默认处理 `data/indices` 下所有 multitrack 目录，区域 50%~100%）：

> `--indices-dir` **既可传父目录** `data/indices`（脚本自动递归发现其中所有 multitrack
> 子目录），**也可传具体的 multitrack 目录或 glob**，如 `"data/indices/train_*_multitrack"`。

```bash
# 先预览，不写盘
python scripts/filter_sequence_regions.py --indices-dir data/indices --region 90 100 --dry-run

# 确认无误后原地覆盖（建议加备份）
python scripts/filter_sequence_regions.py --indices-dir data/indices \
    --region 50 100 --backup-suffix .bak
```

---

## 3. 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--indices-dir` | `data/indices` | multitrack 目录路径或 glob；也可直接传父目录 `data/indices`（自动递归发现子目录），可传多个 |
| `--max-depth` | `2` | 传父目录时，向下递归查找 multitrack 子目录的最大深度 |
| `--region LO HI` | `50 100` | 保留区域占染色体长度的百分比区间 `[LO, HI]`，需满足 `0 ≤ LO ≤ HI ≤ 100` |
| `--keep-mode` | `center` | `center`：窗口中心落在区域内的保留；`overlap`：窗口与区域有交集即保留 |
| `--dry-run` | 关 | 只打印统计，不写任何文件 |
| `--backup-suffix` | 空 | 覆盖前把三个索引文件备份为 `*.后缀`，例如 `.bak` |
| `--recompute-track-stats` | 关 | 用保留区域重算 `nonzero_mean` 并回写 meta 与 json（需 `pyBigWig`） |

### 示例

```bash
# 1) 仅训练目录，保留每条染色体后半段，带备份
python scripts/filter_sequence_regions.py \
    --indices-dir "data/indices/train_*_multitrack" \
    --region 50 100 --backup-suffix .bak

# 2) 预览 10%~40% 区域（不写盘）
python scripts/filter_sequence_regions.py --indices-dir data/indices \
    --region 10 40 --dry-run

# 3) 过滤全部目录，并用保留区域重算 track 均值
python scripts/filter_sequence_regions.py --indices-dir data/indices \
    --region 50 100 --recompute-track-stats
```

---

## 4. 过滤规则说明

- `sequence_split_train.csv` 坐标为 **0-based 半开区间**（`[start, end)`），
  窗口 `32768` bp、步长 `16384` bp。
- 染色体长度**无需读取 FASTA**：滑动窗口的最后一个窗口 `end` 必贴合染色体末端，
  因此取每条染色体 `max(end)` 即为其全长（精确值）。
- `center` 模式用窗口中心 `(start+end)/2` 判断是否落在 `[LO%, HI%] × chrom_len` 内。
- 过滤后 `index_stat.json` 会新增：

```json
"filtered": {
  "region_percent": [50.0, 100.0],
  "keep_mode": "center",
  "num_windows_before": 1830,
  "num_windows_after": 915,
  "filtered_at": "2026-09-08T10:00:00"
}
```

用于追溯数据版本。

---

## 5. 注意事项

1. **先备份再覆盖**：默认是原地覆盖。建议第一次运行加 `--backup-suffix .bak`，
   并先用 `--dry-run` 确认保留窗口数。
2. **染色体特征 JSON 不自动重建**：`chromosome_features_train_*.json` 是按
   整条染色体统计的（GC 含量、信号分位数等）。若训练流程依赖它且希望与过滤后的
   区域分布一致，请用过滤后的 `sequence_split_train.csv` 重新运行
   `scripts/build_chromosome_features.py`。
3. **与训练脚本无缝衔接**：`train.py` / `src/dataset.py` 完全从上述 CSV/JSON
   读取样本与统计，因此过滤后直接复用现有的 `run_train.sh` 即可。
4. 若过滤某条染色体后窗口为空，该目录会整体跳过且不写文件，避免产生空数据集。