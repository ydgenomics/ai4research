# 变异模型文献收集（利用群体变异信息的基因组模型）

> 主题：如何把**群体的变异信息**利用起来——从数据、预训练目标、下游评测三个层面。
> 整理日期：2026-08-28。范围：以"变异感知预训练"为核心，辐射序列→功能监督模型、VEP、长上下文方法学、植物/群体侧。
> 与本项目的关系：本清单服务于"变异感知 NTP 微调 OGR 基模"（A2×M2 方案）的文献支撑与对标；条目标注了与 `variant_aware_pretraining_plan.md` / `variant_ntp_finetuning_impl.md` 的具体接口。

---

## 0. 一句话结论

变异感知预训练目前几乎全部集中在**人类**（UKB 500K、1KG/HGDP、150K WGS），**作物/植物侧尚无对等工作**——GPN 是唯一例外且是小模型。"OGR 续训 NTP + 变异感知 loss_mask + 251 材料群体变异"在稻属/作物领域**没有直接竞争对手**。文献三路独立证据（GPN、BMFM-DNA、UKBioBERT）共同指向：**"群体变异暴露"比"参考序列+变异作为扰动"学得更好**。

---

## 一、变异感知预训练（核心对标）

### 1. UKBioBERT / UKBioFormer / UKBioZoi ⭐精读
- **出处**：Liu T, Zhang X, Lin J, Pinello L, Ying R, Zhao H. *Pre-training Genomic Language Model with Variants for Better Modeling Functional Genomics*. bioRxiv, doi:10.1101/2025.02.26.640468 (v3).
- **核心机制**：500K UKB 个体**单倍型伪序列**预训练 BERT（MLM）；**变异感知掩码**——变异位点 + 6–15bp motif 区域 mask 权重 2–3×，强制模型从上下文学习变异效应；冻背后接 Enformer/Borzoi 头成 UKBioFormer/UKBioZoi。
- **与本项目接口**：本项目 loss_mask 加权（3.0/1.5/0.1）直接受其启发；但其 MLM + 从头训 vs 本项目 NTP + 续训 OGR，是策略对比的主参照（详见 `variant.md`）。
- **必读理由**：变异选择 QC（MAF/HWE/缺失率、Indel<50bp、N 填充过滤、染色体隔离防 LD 泄露）是可直接复用的数据管线蓝本。

### 2. GPN（Genomic Pre-trained Network）⭐精读
- **出处**：Benegas G, Batra SS, Song YS. *DNA language models are powerful predictors of genome-wide variant effects*. PNAS 120(44):e2311218120, 2023. 教程站：gpn.live
- **核心机制**：用**群体单倍型序列**做碱基级 autoregressive 预训练（25 植物 / 190 人类），无需参考基因组，直接从序列学出偏好等位基因→变异效应（delta log-likelihood 打分）。
- **与本项目接口**：**植物界标杆**。它证明"以单倍型而非参考序列训练"能学会变异效应——为本项目 A2（变异位点为中心采样）提供最强的直接先例；其 delta-LL 可作为微调后"变异感知能力"的候选评测指标。
- **必读理由**：植物侧唯一成体系的变异感知 LM，方法学（滑动窗口内变异上下文、物种分组训练、突变可解释性）都是本项目直接对标对象。

### 3. BMFM-DNA-SNP ⭐精读
- **出处**：Li H, Dey S, Kwon BC, et al. *A SNP-aware DNA foundation model to capture variant effects*. arXiv:2507.05265, 2025.
- **核心机制**：ModernBERT 预训练两个模型：BMFM-DNA-REF（参考基因组序列+反向互补）vs BMFM-DNA-SNP（用新型表示方案**编码序列变异**）；对照实验证明整合变异提升所有微调任务。
- **与本项目接口**：明确验证"编码变异进入预训练→下游任务全面提升"；与 2×2 矩阵（格①原始 OGR 冻结 vs ③微调后冻结）的对照逻辑同构。
- **必读理由**：它是"变异表示方案（reference vs variant-encoded）"最干净的消融实验设计参考。

### 4. Nucleotide Transformer v2 / NT-1000G（泛读）
- **出处**：Dalla-Torre H, et al. *Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics*. Nature Methods 22, 2025. v2 系列持续放出。
- **核心机制**：v1 用 3,202 人类基因组（含 ~3.4M 变异）；后续版本纳入 1000 Genomes 等**群体变异序列**做 masked 预训练；多物种（16 物种）版本 NT-Multi。
- **与本项目接口**：业界事实标准的多物种 DNA LM；其"多参考/群体感知"版本是变异感知路径的工业实现参考。
- **必读理由**：NT 的评测协议（~26 任务 benchmark，含 18 个 BEND 任务子集）是本项目微调后评测基准的模板来源。

### 5. Phenformer（泛读）
- **出处**：Träuble F, Stuart L, Georgiou A, … Marks D, Schwab P. *Multi-megabase scale genome interpretation with genetic language models*. arXiv:2501.07737, 2025.
- **核心机制**：**150K 个体全基因组 WGS**、88Mb 上下文遗传 LM（多尺度），不接监督头直接生成疾病机制假设；eQTL 富集验证机制合理性。
- **与本项目接口**：个体级建模需**超长上下文 + 群体序列**的最强佐证——为本项目"1Mb 长上下文"方向背书；其 RoPE 超长外推工程可参考。
- **必读理由**：回答了"为什么变异建模需要从 4kb 走向 Mb 级上下文"的科学问题。

### 6. Evo / Evo 2 ⭐精读
- **出处**：Nguyen E, et al. *Sequence modeling and design from molecular to genome scale with Evo*. Science 386(6723):ead0016, 2024. Brixi G, et al. *... with Evo 2*. Science 390:eado7596, 2025.
- **核心机制**：基因组尺度 decoder LM（Evo 2: 7B 参数 / 8.3T 碱基训练），MMF（multi-modal mixture）训练目标；变异效应预测为核心评测之一（RIME 基准）。
- **与本项目接口**：与 OGR 同源于演化压力假设（NTP 学序列约束）；其 **VEP 评测协议**可直接借鉴到本项目 26 任务 benchmark。
- **必读理由**：基因组尺度 NTP 的最强工程与评测范本；Evo 2 开源权重+评测代码，可做跨物种对照基线。

---

## 二、序列→功能监督模型（下游头，常被变异感知 backbone 挂载）

### 7. DeepSEA（泛读）
- **出处**：Zhou J, Troyanskaya OG. *Predicting effects of noncoding variants with deep learning-based sequence model*. Nature Methods 12:931–934, 2015.
- **要点**：非编码变异功能性预测开山作；变异效应 = 突变前后输出 log 比值（in-silico mutagenesis 的起源）。

### 8. ExPecto（泛读）
- **出处**：Zhou J, et al. *Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk*. Nature Genetics 50:336–343, 2019.
- **要点**：序列→表达 + 染色质 + 疾病风险；**eQTL 变异 0.94–0.99 预测相关**；"序列差异→功能差异"投影方法学是项目评测②③的模板。

### 9. Enformer（精读）
- **出处**：Avsec Ž, et al. *Effective gene expression prediction from sequence by integrating long-range interactions*. Nature Methods 18:1196–1203, 2021.
- **要点**：200kb 上下文转录调控预测；UKBioBERT 的下游头；变异效应 / eQTL fine-mapping（Sawyer et al. 2024）反复验证；本项目 GenOmics 头（Conv1D UNet）的结构对照。

### 10. Sei（泛读）
- **出处**：Chen KM, et al. *A sequence-based global map of regulatory activity*. Nature Genetics 54:946–953, 2022.
- **要点**：~40 个调控活性类的全调控基因组地图，变异→调控活性差异评分。

### 11. Borzoi（泛读）
- **出处**：Calico. *Borzoi: Sequence models for predicting gene expression and regulation* (bioRxiv 2023; Calico 2024 正式版).
- **要点**：Enformer 的改进（信噪比建模、可变窗长 ~524kb 上下文）；UKBioZoi 的下游头。

---

## 三、变异效应预测（VEP）框架与基准

### 12. AlphaMissense（精读，读校准思路）
- **出处**：Cheng J, et al. *Accurate proteome-wide missense variant effect prediction with AlphaMissense*. Science 381(6664):eadg7492, 2023.
- **要点**：蛋白侧变异效应；**预训练语言模型 + 群体等位频率校准**的完整范式——"结构约束+频率先验"如何与序列模型结合，值得移植到非编码侧设计。

### 13. PrimateAI-3D（泛读）
- **出处**：Gao H, et al. *PrimateAI-3D: ... missense variant pathogenicity*. Nature 2023.
- **要点**：灵长类群体（primate cohort）作为变异来源训练——"群体变异作为监督信号"的另一范式。

### 14. 综述：LM × 变异效应预测（泛读，扫盲）
- **出处**：Hegde M, Nebel JC, Rahman F. *Language modelling techniques for analysing the impact of human genetic variation*. arXiv:2503.10655, 2025.
- **要点**：近十年 LM 用于变异效应预测的架构梳理（Transformer 及替代），识别趋势与空白；写 related work 可直接引用。

---

## 四、长上下文 / 可解释性方法学（与下游元件识别相关）

### 15. HyenaDNA（精读，工程先例）
- **出处**：Nguyen E, et al. *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*. NeurIPS 2023 (Spotlight).
- **要点**：1Mb 上下文、单碱基 token、次二次复杂度；本项目"1Mb 长上下文缺失"问题的既有解法样本。

### 16. The Dark Regulome（精读，做 ISM 前必读）
- **出处**：Baranwal C, Baranwal A, Tandon LN. *The Dark Regulome: Disentangling Predictability from Regulation in Genomic Foundation Models*. arXiv:2606.06834, 2026.
- **要点**：对 ISM 打分做**残差化 + 置换检验**，拆开"序列可预测性"与"调控信号"；跨架构（Caduceus-Ph / HyenaDNA / Enformer）分解证实"预测性层"与"调控输出层"几乎零重叠。
- **与本项目接口**：本项目计划用 ISM 做下游元件识别——**必须先读这篇**，避免把语言模型的可预测性当成调控证据（假阳性控制）。

### 17. GTA（泛读）
- **出处**：Honig E, Zhan H, Wu YN, Zhang ZF. *Long-range gene expression prediction with token alignment of large language model*. arXiv:2410.01858, 2024.
- **要点**：LLM token 对齐的基因表达预测（Geuvadis 上 Spearman 0.65，+10% vs Enformer）；用注意力定位远端调控区——与本项目"注意力做远距离元件"思路一致。

### 18. dnaGrinder（泛读，算力受限时的对标基线）
- **出处**：Zhao Q, Zhang C, Zhang W. *dnaGrinder: a lightweight and high-capacity genomic foundation model*. arXiv:2409.15697, 2024.
- **要点**：轻量长上下文基因组 FM（>17k token 微调、单卡 >140k token 推理）；若卡时预算紧张可作对标基线。

---

## 五、植物 / 群体侧（与数据强相关）

### 19. AgroNT（泛读）
- **出处**：Mendel C, et al. *AgroNT: ... agricultural genomes & metagenomes*. Nature Plants, 2024/2025.
- **要点**：农业泛基因组 + 生态数据基础模型；作物侧最接近的参照物，但**未做变异感知**——空白印证。

### 20. 商连光 251 材料 / OGR 422 泛基因组（数据地基，待核实引用）
- **出处（待核实）**：251 份亚洲栽培稻 + 结构变异图谱（商连光团队, Nature Genetics 系, ~2022）。OGR 预训练基于 422 泛基因组（257 栽培 + 165 野生）。
- **待办**：核对 `/mnt/rice/data/Lianguang_shang/` 目录 README 确认对应论文（用于正式引用与 Material 部分写法）。

### 21. Genome-models 相关其他候选（建议补充检索）
- **NTv2 植物版本**（NT-Multi 覆盖 16 物种含植物）
- **RiceSIR / 水稻调控元件模型**类（如有公开可补）
- **植物 pangenome LM**：如 *Grapevine pangenome LM*（2024）、*Maize pangenome*（Hufford 2021，数据侧）——泛基因组数据如何进模型的方法学参考。

---

## 六、与本项目决策的映射表

| 本项目决策 | 文献支撑 | 出处 |
|-----------|---------|------|
| A2：变异位点为中心采样 | 单倍型训练优于参考序列 | GPN, BMFM-DNA, UKBioBERT |
| M2：loss_mask 加权（3.0/1.5/0.1） | 变异感知掩码 2–3× 加权 | UKBioBERT |
| 续训 NTP 而非从头 MLM | 演化压力假设 + 站点可复用 | OGR/Evo 先例 |
| 变异 QC（MAF/HWE/缺失率、Indel<50bp、N 过滤、染色体隔离） | 数据管线蓝本 | UKBioBERT §Methods |
| 2×2 矩阵 ①vs③ 关键对照 | REF vs SNP 表示消融 | BMFM-DNA-SNP |
| 26 任务微调后评测 benchmark | 评测协议模板 | NT v1/v2, Evo 2 |
| 1Mb 长上下文方向 | 超长上下文必要性 | Phenformer, HyenaDNA |
| ISM 元件识别 | 注意控制可预测性假阳性 | Dark Regulome |
| delta-LL 变异打分候选指标 | 变异效应评测 | GPN |

---

## 七、检索备注（后续可扩展方向）

1. **重点补检索**：植物 pangenome × LM（RiceSIR、pangenomeLM-classes）、NTv2 具体版本页面（biocore 官方 HF）、Evo 2 的 RIME 基准细节。
2. **优先级建议**：精读 6 篇（带 ⭐ 或标注精读）→ 先读 UKBioBERT 方法 + GPN + Dark Regulome（影响数据管线与评测设计），再读 Evo 2 + NTv2 评测协议，Phenformer 看超长上下文工程。
3. **写论文时的 related work 骨架**：变异感知预训练（§一）→ 序列功能监督（§二）→ VEP 基准（§三）→ 长上下文/可解释（§四）→ 植物空白声明（§五）。