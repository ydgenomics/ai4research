首先明确数据准备后的格式：
- 品种/个体的fasta文件(optional: gff)
- 品种/个体对应的RNA表达bw文件

对数据早期处理，要先下载fasta和gff文件。RNA表达的bw文件是下载的原始测序的fa文件，然后通过比对到参考，然后定量得到bw文件。考虑到测序深度和批次应该要先对数据做一些标准化处理，例如TPM和CPM

> TPM和CPM标准化是什么？（200字以内回答）
> 在RNA-seq数据分析中，CPM和TPM是两种对原始基因读段计数（read count）进行标准化的方法，目的是消除不同样本间测序深度等技术差异带来的影响，使基因表达量可比。
> *   **CPM (Counts Per Million)**：只校正**测序深度**，将基因的read count除以样本总reads数再乘以1,000,000。其计算简单，适用于差异表达分析或长度差异小的数据（如small RNA）。
> *   **TPM (Transcripts Per Million)**：同时校正**基因长度和测序深度**。它先除以基因长度，再基于长度校正后的值计算每百万份数。由于每个样本的TPM总和恒定，更便于样本间直接比较，是目前学术界更推荐的指标。

对于模型而言
- dataloader
- 模型架构的forward, loss, backward，desc_gradient
- 模型的训练和推理，推理就是没有了反向这一步
- 早期训练的时候要保留各种实验日志，swanlab，记录loss和各种指标，例如回归的Pearson
- 模型的评测，真实的bw和推理得到的bw计算各种维度的指标

输入
- bw文件，fa文件和拆分好窗口的csv文件，这俩用于后面的dataloader构建每个样本，然后json文件记录一些元信息