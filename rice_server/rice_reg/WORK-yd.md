1.提供中英文两版
- 取消选择End，默认是32k窗口，具体是32768
- 布局修改为
Genome	Chromosome Start
ATAC-seq File (下拉选择内置文件或点击上传)
先内联网页一起看一下前端布局
ATAC-seq File的bw文件一定要与选择的基因组版本一致



制作MH63_1对应的预测bw文件，如果用户只是对测试数据例如MH63划定区间可视化，完全不调用模型，直接用模型预测好的数据做展示。测试csv里面的预测的结果和模型预测的结果是否一致，拿几个窗口做一下测试
基于下面几个文件，将模型预测的表达聚合为基因组对应的bw文件，注意overlapping值的处理，保留代码到rice_server/rice_reg/scripts
/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_reg/genome/MH63.fa
/mnt/rice/default/Workspace/yangdong/gene_expression_prediction/answer/metrices/ATAC_RNA-seq_SAM2/rename_chr_csv/reg/SAM2_MH63_1.checkpoint-5868800/RNA_plus_predictions.csv
/mnt/rice/default/Workspace/yangdong/gene_expression_prediction/answer/metrices/ATAC_RNA-seq_SAM2/rename_chr_csv/reg/SAM2_MH63_1.checkpoint-5868800/RNA_minus_predictions.csv

head -n 1 /mnt/rice/default/Workspace/yangdong/gene_expression_prediction/answer/metrices/ATAC_RNA-seq_SAM2/rename_chr_csv/reg/SAM2_MH63_1.checkpoint-5868800/RNA_plus_predictions.csv
chromosome,start,end,sequence,true_expression,predicted_expression
