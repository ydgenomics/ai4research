# rice_reg 全基因组 bigWig 预生成(策略 B)

把 **固定 (参考基因组 FASTA + 内置 ATAC bigWig)** 组合的全基因组滑窗预测提前算好,
最终只产出 **两个全基因组 bigWig(RNA-seq_+ / RNA-seq_-)**,查询时无需再跑 GPU。

相比 `rice_mut` 的预生成(**每个滑窗一个 `.bw`**,窗口数 ~2.6 万,再逐个读回合并),本方案:

- **不写任何窗口级 `.bw`**;窗口重叠的平均在**内存**中完成(逐碱基 `sum/count`)。
- 断点粒度从"窗口"提升到 **染色体**:每完成一条染色体写 1 个紧凑 `.npz`/链。
- 最终合并只读 ~24 个染色体 `.npz` → 拼成 2 个全基因组文件。

## 产物布局

```
rice_reg/cache/pregen/
├── manifest.json                        # fasta/atac/ckpt sha256、window/hop、窗口清单
├── MH63RS3__SAM2_MH63_1_plus.bw         # ★ 最终:正链
├── MH63RS3__SAM2_MH63_1_minus.bw        # ★ 最终:负链
└── parts/
    └── MH63RS3__SAM2_MH63_1/            # 染色体级中间(合并后 --cleanup 可删)
        ├── Chr1_plus.npz
        ├── Chr1_minus.npz
        └── ...
```

文件名为 `<Genome>__<Atac>_<strand>.bw`,同一基因组不同 ATAC 互不覆盖。

## 使用

模型/基因组/ATAC 路径从 `rice_reg/.env` 读取(与后端一致)。

```bash
# 跑 MH63RS3 全基因组(默认 workers = 检测到的 GPU 数)
bash scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1

# 指定 2 卡
bash scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1 --workers 2

# 只跑一条染色体冒烟
bash scripts/pregen_bigwigs.sh --genome MH63RS3 --atac SAM2_MH63_1 --chrom Chr1

# 断点续跑(保留已完成染色体)
python scripts/pregen_bigwigs.py --genome MH63RS3 --atac SAM2_MH63_1 --resume

# 只合并已生成的染色体 parts
python scripts/pregen_bigwigs.py --genome MH63RS3 --atac SAM2_MH63_1 --merge-only

# 合并成功后删除中间 parts
python scripts/pregen_bigwigs.py --genome MH63RS3 --atac SAM2_MH63_1 --cleanup
```

两个基因组都推(前台串行,便于观察):

```bash
for g in "MH63RS3 SAM2_MH63_1" "NIP SAM2_NIP_1"; do
  set -- $g
  bash scripts/pregen_bigwigs.sh --genome "$1" --atac "$2"
done
```

```bash
# pane 1 — GPU 0
export CUDA_VISIBLE_DEVICES=0,1
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/rice_reg
/root/miniconda3/envs/vllm/bin/python scripts/pregen_bigwigs.py \
--genome MH63 --atac SAM2_MH63_1 --workers 2 --gpus 2 > logs/pregen_MH63.log 2>&1

# pane 2 — GPU 1
export CUDA_VISIBLE_DEVICES=1
cd /mnt/rice/default/Workspace/yangdong/ai4research/rice_server/rice_reg
/root/miniconda3/envs/vllm/bin/python scripts/pregen_bigwigs.py --genome NIP --atac SAM2_NIP_1 > logs/pregen_NIP.log 2>&1
```

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--genome` / `--atac` | env 中唯一时自动 | 内置组合 id(对应 `GENOME_<id>_FASTA` / `ATAC_PATH_<id>`) |
| `--hop` | `target_len//2`(=16339) | 滑窗步长,50% 重叠 |
| `--chrom` | 全部 | 只处理指定染色体(.fai 真实名,如 `Chr1`) |
| `--workers`/`--gpus` | 检测到的 GPU 数 | 多进程,每 worker 独立 predictor/GPU,按染色体 round-robin |
| `--chunk-windows` | 256 | 每次 `predict_batch` 的窗口数(内存/吞吐权衡) |
| `--resume` | off | 跳过已有 parts 的染色体 |
| `--merge-only` | off | 只合并现有 parts |
| `--no-merge` | off | 只推理到 parts,不合并 |
| `--cleanup` | off | 合并成功后删除 parts 目录 |

窗口长度固定取模型 `TARGET_LEN`(`.env`=32678),与 `run_prediction_core` 口径一致;
与 rice_mut(32768)不同,勿照抄。

## 与在线单窗口结果一致性

预生成与 `/predict/rice-reg` 使用同一 `RiceRegPredictor`、同一
`FIXED_TRACK_MEAN_PLUS/MINUS`(2.83/2.85)反归一化,重叠区取均值。
冒烟建议:先 `--chrom Chr1 --no-merge`,再 `--merge-only`,然后随机取几个位点
与单窗口 API 返回比对(diff 应 ≈ 0)。

## 注意:ATAC 染色体命名

脚本按 **FASTA `.fai` 的真实染色体名** 读取 ATAC。若 ATAC bigWig 的染色体名与
FASTA 不一致(如 `chr1` vs `Chr1`,或 MH63RS2 vs MH63RS3 命名),启动时会打印告警
并列出两边名称,整条染色体会预测不出。此时请先做一份"染色体名对齐"的 ATAC 副本
再运行(不要覆盖原始 ATAC)。

## 已知限制 / 后续

- 目前**只做生成脚本**;API 层"预生成命中免推理"查询尚未接入(后续工作)。
- 水稻 12 条染色体约 400 Mb,1 bp 分辨率两链 float32 的 parts 合计约 1–2 GB;
  合并后两个全基因组 `.bw` 体积与信号稀疏度相关(参考 rice_mut 已有先例)。
