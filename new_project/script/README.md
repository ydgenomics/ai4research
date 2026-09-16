# 水稻变异感知 CPT 微调（GenVarLoader 版）

基于 OGR(HF) 续训 + 变异聚焦 loss_mask（方案一）。懒加载、不落个性化 fasta。

## 设计要点
- **数据通路**：`251.SNP.final.vcf.gz`（573 万 SNP × 251 样本，未 phase）
  → bcftools 过滤/norm（left-align + atomize，GVL 硬要求）
  → gvl.write（懒加载 `.gvl`，省 >2000× 存储）
  → Dataset.open(reference, jitter=0, **deterministic=True**)（未 phase 保真，C1/C2）
- **窗口**：32k 主（16k 步长 50% overlap 网格）；64k 消融（yaml 切换，需重跑 write）
- **loss_mask**（方案一核心）：
  - 变异位点 v 及周边聚焦窗 `[v-100, v+25]` → 3.0
  - motif ±6-15bp → 1.5
  - 背景 → 0.02；padding（ref_coords==-1）→ 0
  - 多变异重叠取 max；NTP shift 在 model 内与 mask 同步右移
- **采样**（`window.sampling_mode`，配置驱动）：
  - `density`     原行为：w=n_variants^0.5，cap p99；无变异窗权重=0 不参与采样
  - `two_stage`   两段采样（推荐正式训练）：
    ① **覆盖段** —— 每窗均匀抽 `coverage_k` 个代表样本（确定性 seed），
       窗口外循环排序 → 预算内前 N 个最大化**全基因组位置覆盖**（异等位形态冗余消除：
       251 样本同窗约 95% 碱基相同，K=8 代表样本即可等形态覆盖同时省样本量）
    ② **聚焦段** —— 剩余预算（`cover_frac` 比例后）按 density 加权抽高变异窗
       比例：覆盖段 ≤ budget×cover_frac（至少 1 轮全窗覆盖），剩余 → density 聚焦
- **分离信号日志**（每步）：`variant_ce`（变异区 CE）/ `background_ce`（背景区 CE）/ `variant_frac`
  - variant_ce 下降 = 学到变异；background_ce 保持 = 无灾难性遗忘（双指标同看）
- **训练**：OGR 基模权重加载，低 LR(1e-5) 续训，保留 MoE aux-loss（C4）；
  LoRA 可选（`train.lora` config 控制，见下）
- **染色体隔离**：train Chr1-7 / val Chr8-9 / test Chr10-12（ChrUn/ChrSy 已排除）

## 快速开始：统一入口（配置驱动）

> **只有一个启动脚本 `run.sh`**，所有配置（窗口/样本/region/gpu/超参…）都在 yaml 里。
> `run.sh <config.yaml> [all|data|model] [smoke]`，smoke=冒烟（不存最终权重，用于定 lr）。
> 数据预处理（QC/norm/BED/gvl.write）**完全不需要 GPU**，`data` 模式可在 CPU 节点/空闲时段先跑；
> 训练阶段只读已生成的 `.gvl`，不再触碰 VCF/BCF，节约 GPU 独占时间。

### 8k 小数据端到端验证
```bash
bash run.sh configs/test_8k.yaml all smoke   # 全流程: 数据预处理(CPU) + 冒烟(GPU, 不存权重)
bash run.sh configs/test_8k.yaml data        # 只跑数据预处理（不需 GPU）
bash run.sh configs/test_8k.yaml model smoke # 只跑模型冒烟（需 GPU）
```

### 正式 32k 训练（两阶段）
```bash
# 阶段一（CPU，可放后台，几小时）
bash run.sh configs/train_32k.yaml data
# 阶段二（GPU）
bash run.sh configs/train_32k.yaml model smoke   # 冒烟定 lr（C5，10-20 步看 loss 量级）
bash run.sh configs/train_32k.yaml model         # 正式训练
```

> `model` 模式启动前会校验 gvl 产物存在，若未跑 `data` 会直接提示先跑。

### 装依赖
```bash
pip install -r requirements.txt
# 系统层：bcftools / samtools / tabix（已核实 tools 环境有）；plink2 需补装（当前降级跳过 QC）
```

## 目录结构
```
script/
├── run.sh                # ★ 唯一启动入口: run.sh <config> [all|data|model] [smoke]
├── main.py               # 主调用（--stage: data_preprocess/train/smoke）
├── configs/
│   ├── train_32k.yaml    # 默认 32k 配置
│   ├── train_64k.yaml    # 64k 消融
│   └── test_8k.yaml      # 8k 小数据测试（region+sample_subset 限制）
├── gvl_pipeline.py       # QC/norm → BED → gvl.write → open/subset
├── window_sampling.py    # 网格索引对 + 采样权重
├── variant_dataset.py    # 懒加载 Dataset + collate（byte→token + loss_mask）
├── model_head.py         # VariantAwareCausalLM（Σ(w·CE)/Σw 加权）
├── trainer_utils.py      # Trainer 子类 + run_training + run_smoke
├── verify_gvl.py         # gvl 产物验证（元信息/loss_mask/变异覆盖）
└── requirements.txt
```

## 关键约束 / 注意事项
1. **变更窗口大小需重跑 data_preprocess**：gvl 产物路径按 `gvl_{length}.gvl` 分目录，新长度需重新 `bash run.sh <config> data`。
2. **未 phase 的实现**：`deterministic=True` + collate 取 ploidy=0 链
   （GVL 的 `unphased_union` 与 `with_seqs("annotated")` 互斥，故不用）。
   若后续做了 phasing（SHAPEIT5），再切 double-haplotype 输入。
3. **GVL DataLoader**：`num_workers=0/1`（GVL 内部多线程，勿用多进程 worker）。
4. **VCF 预处理是硬前置**：GVL write 要求 left-aligned / biallelic / atomized，
   由 `_bcftools_norm` 保证（自动降级为跳过 plink2 若无该命令）。
5. **坐标一致性**：`251.SNP.final.vcf.gz` 与参考 fasta 同为 IRGSP-1.0 R7
   （`##reference=Os-7.fa`），CHROM=Chr1-12 直接对齐，无需 liftover（已核实）。

## 监督指标
- `variant_frac`：loss_mask>0.1 的 token 占比（应远小于 1，聚焦有效）
- `variant_ce` vs `background_ce`：变异区 CE 应下降更快（等位区分度）；
  背景区 CE 保持平稳 = 无灾难性遗忘（每步日志自动打印）
- MoE aux-loss 稳定（C4 沿用）
- 冒烟后按 loss 量级校准 lr（C5）：32k 如 loss 偏大 → lr 下调

## LoRA 可选微调
`config train.lora` 控制（配置驱动，纯代码不变）：
```yaml
train:
  lora: false                          # false=全参微调（默认）；true=LoRA 冻结基模
  # 或用 dict 精细配置：
  # lora: {enabled: true, r: 8, alpha: 16, dropout: 0.05, target_modules: [q_proj, k_proj, v_proj, o_proj]}
```
- LoRA 模式只训适配器（默认 ~0.05% 参数，显存省 ~90%），适合快速消融 lr/采样策略
- 保存时自动 `merge_and_unload()` 回原始权重 → 产物仍是**完整 OGR 基模格式**，可直接加载/推理
- 依赖：`pip install peft`（vllm 环境已装 0.18.0）

## 两段采样配置速查
```yaml
window:
  sampling_mode: two_stage    # density | two_stage
  coverage_k: 8               # 覆盖段每窗代表样本数（≤样本数）
  cover_frac: 0.6             # 覆盖段占预算比例（≥ 每窗 1 次时自动顶满全窗覆盖）
```