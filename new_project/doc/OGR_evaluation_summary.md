# OGR（OneGenome-Rice）测评总结

## 一、模型简介

OGR 是一个水稻基因组基础模型（1.25B 参数，MoE 架构：8 experts、top-2 路由、12 层、GQA、RoPE base=50M 支持最长 1Mb 上下文），在 422 个水稻基因组上预训练。评测目的是检验其 DNA embedding 在下游任务上的表征能力，并与同类模型对比。

## 二、测评了哪些内容（6 大类共 26 个任务）

来自 `dataset/datasets_info.yaml`，任务按序列长度/生物学尺度划分：

| 类别 | 数据集 | 任务类型 |
|---|---|---|
| **AgroNT 基准**（5） | `chromatin_access_MH63/ZS97_agront`、`poly_a_japonica/indica_agront`、`bulk_gene_exp_agront` | 染色质可及性、poly(A) 位点、基因表达（回归） |
| **短序列**（6） | `chromatin_access_512`、`Epigenetic_Marks_Prediction-Histone`、`smallRNA`、`splice_sites_labels`、`RiceVar_512`、`enhancer_strength` | 染色质、组蛋白修饰、small RNA、剪接位点、变异检测、增强子（强度回归） |
| **长序列**（5） | `enhancer_regions`、`lncRNA`、`rice_nlr`、`RiceVar_6k`、`RiceVar_8k` | 增强子区域、lncRNA、NLR 基因、长上下文变异检测 |
| **单核苷酸**（2） | `CDS_multilabel`、`bulk_single_base_Callus` | CDS 多标签、单碱基基因表达（高分辨率） |
| **选择清除区识别**（5） | `sweep_region_psr/sor/8k/32k/100k` | 扫描区间长度从 6kb 到 100kb |
| **品种分类**（3） | `varieties_classification_8k/32k/128k` | 品种识别，上下文从 8kb 到 128kb |

数据规模示例：`smallRNA` 约 19 万条，`varieties_classification_8k` 约 30 万条；多数按 8:1:1 划分 train/test/eval（部分只有 train/test）。

## 三、如何测评（两阶段流水线：embedding 提取 → 下游探测任务）

评测框架代码在 `evaluation/benchmark_code/`，流程为：

### 阶段 1：Embedding 提取
1. 输入：模型 + 每个数据集的 `train/test/eval.jsonl`（每行 `{"seq": "ATCG...", "label": ...}`）
2. 通过模型前向传播拿到每层的 `hidden_states`，用 attention mask 做**平均池化**得到每条序列的向量（`[B, H]`，H=1024）
3. 按指定层（`layer_to_eval`，如第 12 层）分别保存为 `.pt` 文件；长序列会降低 batch size，超长序列 batch=1
4. **二倍体/家系数据**（`seq_for_item`>1）：多序列按 `pooled_embeddings_cat_dim` 策略合并——1 表示拼接（2×1024→2048）、0 表示均值池化

### 阶段 2：下游分类器训练与评估
- 默认分类器 **MLP**（3 隐层，维度 H/2→H/4→H/8 截断至 128，dropout 0.2，Adam，默认 100 epochs）；也可选 **XGBoost** / **Random Forest**
- 损失函数：分类→CrossEntropy，回归→HuberLoss，多标签→BCEWithLogits
- 用 train 集训练、eval 集验证、test 集上给出最终指标
- **评测指标**：
  - 分类：`accuracy`、`roc_auc`（二分类用正类概率；多分类用 macro OVR）、`precision`、`recall`、`f1`、`mcc`（多标签任务取各标签 MCC 均值）
  - 回归：`mse`、`mae`、`r2`、`pearsonr`、`spearmanr`（多目标取 macro 平均）

### 阶段 3：报告生成（`analysis_reprot.py`）
- `best_{指标}.tsv`：每个数据集挑最优层，报告该层全部指标
- `best_{多指标}.tsv`：多指标模式下各自挑最优值并记录其层号
- `last_layer.tsv`：最后一层的结果
- `line_plot_{指标}.png`：**逐层性能曲线**（x=层数，y=指标），标注最优层

### 任务调度（`schedule.py`）
- 加权调度：embedding 任务权重大（4）、评测任务权重小（1），每张 GPU 有容量上限（如 5）
- embedding 任务优先、分配到负载最低的 GPU；embedding 完成后自动级联触发评测任务；支持续跑（已存在的 `.pt`/结果文件自动跳过）

## 四、主要结论（来自 README「Performance Evaluation」）

- OGR 在 **26 个任务中 16 个排名第一或第二**
- **优势任务**：染色质可及性、组蛋白修饰、small RNA 预测、增强子强度、**选择清除区识别（8kb–100kb 长上下文优势明显）**、**品种分类（随序列长度增加优势扩大，体现群体结构/演化模式捕捉能力）**
- **短板任务**：剪接位点识别、短序列变异检测（RiceVar）、poly(A) 位点与基因表达预测（AgroNT）、**单核苷酸分辨率预测**——反映了模型在精细调控建模和核苷酸级建模上的局限

## 五、与当前工作的关联

该评测框架是通用的"**DNA 基础模型 embed + 线性探测（probing）**"范式：不直接微调模型，而是冻结 embedding 训小分类头，公平对比不同模型的表征质量。这与你 `gene_expression_prediction` 中基于 OGR 做基因表达预测（Case 3 App）不同——后者是**全参数微调 + U-Net 回归头**做单碱基分辨率 RNA-seq 预测，属于 OGR 的应用场景而非 benchmark 本身。