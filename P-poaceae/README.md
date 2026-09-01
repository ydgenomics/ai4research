描述性统计
数据质量（二/三代测序，组装和注释的情况）
倍性、染色体数量/组装的条数、注释类别、基因数量、
基因组大小、重复序列的情况、字符类别（区分大小写）、N或其它字符的分析

```shell
curl -X POST "https://dcsapi.dcs.cloud/api/aigress/openai/jobs" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
    -d '{
    "model": "Oriongeno",
    "species_name": "Achnatherum splendens",
    "genome": "/mnt/rice/data/Poaceae_Fan_longjiang/genome/GWHBHQB00000000.genome.fasta.gz",
    "output": "/mnt/rice/default/Workspace/yangdong/ai4research/DATA/Achnatherum_splendens.gtf",
    "gpu_count": 1,
    "batch_size": 2,
    "length": 512000,
    "flank": 12000,
    "output_gene": true,
    "output_repeat": false,
    "assembly_mode": "auto",
    "overwrite": false,
    "callback_url": "https://example.org/oriongeno/callback"
  }'


curl -X POST "https://dcsapi.dcs.cloud/api/aigress/openai/jobs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{
    "model": "Oriongeno",
    "species_name": "Festuca petraea",
    "genome": "/mnt/rice/data/Poaceae_Fan_longjiang/genome/GCA_051419415.1_ASM5141941v1_genomic.fna.gz",
    "output": "/mnt/rice/default/Workspace/yangdong/ai4research/DATA/Festuca_petraea.gtf",
    "gpu_count": 1,
    "batch_size": 2,
    "length": 512000,
    "flank": 12000,
    "output_gene": true,
    "output_repeat": false,
    "assembly_mode": "auto",
    "overwrite": false,
    "callback_url": "https://example.org/oriongeno/callback"
  }'
moedl:部署模型名字
species_name:输入序列的物种名
genome:序列文件地址
output:输出文件地址
gpu_count:调用gpu数量
length:默认
flank:侧翼长度
output_gene:是否输出gtf
output_repeat:是否输出repeat注释
```

```shell
调用：
############云平台：
curl -X POST "https://cloud.stomics.tech/api/aigress/openai/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{
    "model": "Oriongeno",
    "genome": "/oriongeno/tests/Fungi/GCA_030579395.1_ASM3057939v1_genomic.fna.gz",
    "output": "/output/fungi10.gtf",
    "checkpoint": "/oriongeno/checkpoints/Fungi",
    "batch_size": 4,
    "dcs_storage_path": "/test_data"
  }'
查询状态：
curl -X POST https://cloud.stomics.tech/api/aigress/openai/predict/status \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
     -d '{
       "model": "Oriongeno",
       "task_id": "20260806-075755-dc41c471"
     }'
查询运行日志：
curl -X POST https://cloud.stomics.tech/api/aigress/openai/predict/log \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
     -d '{
       "model": "Oriongeno",
       "task_id": "20260820-073602-4ccb2b57",
       "lines": 100
     }'

curl -X POST https://cloud.stomics.tech/api/aigress/openai/predict/health \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
     -d '{
       "model": "Oriongeno"
     }'

curl -X POST https://cloud.stomics.tech/api/aigress/openai/predict/health \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
     -d '{
       "model": "Oriongeno"
     }'


curl -X POST "https://cloud.stomics.tech/api/aigress/openai/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
  -d '{
    "model": "Oriongeno",
    "genome": "/oriongeno/test_Poaceae/GCA_051419415.1_ASM5141941v1_genomic.fna.gz",
    "output": "/output/Festuca_petraea.gtf",
    "checkpoint": "/oriongeno/checkpoints/Viridiplantae",
    "batch_size": 1,
    "dcs_storage_path": "/test_data"
  }'

curl -X POST https://cloud.stomics.tech/api/aigress/openai/predict/log \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" \
     -d '{
       "model": "Oriongeno",
       "task_id": "20260901-074844-76345793",
       "lines": 100
     }'

curl -X POST https://cloud.stomics.tech/api/aigress/openai/predict/log \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-qe1sUERFblU2KU0RahyZ2qtxzuz7hPUCQfI4ssm28YRm11ev" \
     -d '{
       "model": "Oriongeno",
       "task_id": "20260820-073602-4ccb2b57",
       "lines": 100
     }'
```