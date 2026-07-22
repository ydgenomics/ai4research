*_prediction.csv本身带有predict和true expression的值
✅ 模型能很好地区分哪个基因高表达、哪个低表达（排序 rank 正确）
⚠️ 但对单个基因内部每个碱基的具体预测值还不够精确（局部波动大）

## TODO
- delta_pcc的计算公式，保证所有都有值 去除"所有训练品种平均表达模式"后，模型预测残差与真实残差的相关性
  - delta_pcc是不是先要求同一基因在不同品种个体真值的均值，然后对预测值减均值，真实值减均值，然后计算pcc？
    | 模式 | $\mu_{\text{ref}}$ | 含义 |
    |------|-------------------|------|
    | `delta_pcc_zero` | $0$ | 等价于归一化后直接算 PCC。检测"基因开关"一致性 |
    | `delta_pcc_feature_ref` ⭐ | **该基因在训练集中所有品种的真实表达均值** | 去除了基因的"基础表达水平"，只看品种特异的偏离 |
    | `delta_pcc_global_ref` | 当前样本全体基因的均值 | 零均值化后的 PCC |
- 基因水平的表达是怎么计算的，是否包含intron？
  - 包含 intron。基因水平的表达值是对基因整个区间（从起始密码子到终止密码子）内所有碱基位置的预测/真实表达取均值
  - 是否修改为只计算exon区域的基因表达？
    - 生物学意义上，只计算 exon 区域更好。RNA-seq 的生物学基础是成熟 mRNA——intron 已被剪接体切除。标准 RNA-seq 定量（TPM/FPKM/RPKM）只计数外显子区域的 reads，intron reads 被视为背景噪音或未成熟 pre-mRNA
- nozero_pcc计算的时候是要求预测值和真实值都不为0才纳入计算吗？
  - 对，要求预测值和真实值都不为 0（> 0，非 != 0）

我的理解是不同品种求各自品种所有基因表达的均值，作为各自品种的scale因子。ref = true_mean/真值的scale。之前预测值在归一化使用的是预测值的所有基因表达的均值，现在想统一都使用真值的的scale。我的理解对吗？
建议保持当前方案不变——pred 和 true 各自归一化，因为 delta_pcc 关注的只是"相对偏差模式"，而非"绝对尺度匹配"。


## think
RNA预测任务，基于1B的foundation model，1）单组织多品种训练，使用4个品种 Huanghuazhan(P1), IAC25(P4), Wuyungeng(P6), Zhongzao35(P11) 的 CSQ (leaf tissue data collected 2 to 3 days before heading date) 全染色体进行全量微调，以Xiushui134(P7) 作为测试集，说明其跨品种的能力；2）双组织多品种训练，使用3个品种 Huanghuazhan(P1), IAC25(P4), Wuyungeng(P6) 的 CSQ (leaf tissue data collected 2 to 3 days before heading date) 和 YG (root tissue data collected 38 days after sowing) 进行全量微调，非全染色体，以P7作为测试集，说明其跨染色体的能力

如何评价模型的能力？
seq2expression，这种数值型回归任务，我们使用pcc, log1p_pcc, nozero_pcc, and r2来表示
- pcc反映整体水平
- log1p_pcc去除极值，能反映小值
- nozero_pcc回应编辑，去除背景大量零值
- r2反映尺度的准确性

品种-组织-染色体
从三个水平来说明
- track水平/染色体水平
- exon水平和gene水平
- 对于不同表达水平的gene做分箱展示，因为背景噪声，低表达基因预测确实很低

补充指标的表格类似下面这样，文章画的图
|varities|tissue|split|chromosome|resolution|pcc|log1p_pcc|nozero_pcc|r2|delta_pcc|
|-|-|-|-|-|-|-|-|-|-|
|P1|CSQ|train|Chr01|bp||||||
|P1|CSQ|train|Chr01|exon||||||
|P1|CSQ|train|Chr01|gene||||||
|P1|CSQ|train|Chr01|gene-low||||||
|P1|CSQ|train|Chr01|gene-medium||||||
|P1|CSQ|train|Chr01|gene-high||||||

|varities|tissue|split|resolution|pcc|log1p_pcc|nozero_pcc|r2|delta_pcc|
|-|-|-|-|-|-|-|-|-|
|P1|CSQ|train|bp||||||
|P1|CSQ|train|exon||||||
|P1|CSQ|train|gene||||||
|P1|CSQ|train|gene-low||||||
|P1|CSQ|train|gene-medium||||||
|P1|CSQ|train|gene-high||||||

为了说明跨品种预测的能力，而非学到相似的轨迹，我们补充了delta_pcc相关的指标


## ref
颜色 #4874CB 蓝色 #ef822f 橘黄色

