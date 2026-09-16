# 水稻模型 OGR 及其应用模型 API（极简 shell 版）

# 统一鉴权: Authorization: Bearer $KEY
# 统一入口: https://www.dcs.cloud/api/aigress/openai/OGR/{rice_ogr,rice_mut,rice_reg,rice_intro}[/{mode}]
# 返回统一: {"usage":{"prompt_tokens","completion_tokens"},"status","message","result"}
# 计费: 1bp=1token; prompt_tokens=输入碱基数×系数, completion_tokens=输出元素数×系数(系数见.env,默认1)

# ===== 部署架构 =====
# 浏览器(Gradio :8000/:7000) --HTTP--> 后端(:8001/:7001 网页版)
#   rice_mut: DNA→多组学表达(参考 vs SNV) | rice_reg: DNA+ATAC→RNA-seq | rice_OGR: embedding/碱基预测(Sanic :8000)
# DCS网关(dcs_gateway/ :9000) --model_sub路由--> 四个dcs_adapter进程
# 外部唯一入口: POST /api/aigress/openai/OGR/{model_sub}[/{mode}]
# 端口: rice_mut 8000/8001 | rice_reg 7000/7001 | rice_intro Gradio 5000/后端 5001 | rice_OGR Sanic 8000/DCS适配6001 | 网关9000
# 一键启动(推荐): cd dcs_gateway && bash start_all.sh   # 先 conda activate vllm, 按.env拉起四后端+网关
# 说明: OneGenomeRice(MoE,Decoder-only,1bp=1token,长文本1M) 衍生下游: ①Mutation 序列→表达(可SNV对比) ②Reg DNA+ATAC→RNA ③Intro 粳/籼血缘渗入分析

# sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx
# sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx
# ===== 1. rice_ogr: embedding 提取 / 下游碱基预测 =====
# 必填: sequence(DNA序列,大小写均可)  model_name(1B_8k|1B_32k)
# 可选: mode(dna_embedding默认|predict;带predict_length自动进predict)
#       pooling_method(mean默认|max|last|none;none输出逐位[1,L,1024])
#       predict_length(1-1000,默认10)
# 长度上限: 1B_32k 最大32678bp, 1B_8k 最大32768bp, 超出截断
# 注意: 请求体里 model=服务名; 实际模型名放 model_name

# embedding(默认mode) 提取整条序列 1024 维向量
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/dna_embedding \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "model":"OGR",
    "model_name":"1B_8k",
    "sequence":"ACGTTGCATGCAACGT",
    "pooling_method":"mean"
}'

# predict 基于前16bp续写10bp(predicted_bases 可再回灌做自回归续写)
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_ogr/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "model":"OGR",
    "model_name":"1B_8k",
    "mode":"predict",
    "sequence":"ACGTTGCATGCAACGT",
    "predict_length":10
}'


# ===== 2. rice_mut: 参考序列表达预测 / SNV(单碱基突变对比) =====
# 必填: chromosome(chr01-12)  start(1-based inclusive,窗口32768中心对齐)
# 可选: genome(默认osa1_r7)  end(窗口终点,缺省按规则补齐)
#       output_format(full默认逐碱基数组|mean每轨标量均值|downsample降采样)
# SNV必填: snv_index(1-based绝对坐标,须落在窗内)  snv_base(A|C|G|T)

# predict 参考序列表达(mean格式)
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "model":"OGR",
    "genome":"osa1_r7",
    "chromosome":"chr09",
    "start":20716774,
    "end":20749541,
    "output_format":"mean"
}'

# snv 单碱基突变对比(chr09:20731844 C→T)
curl -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_mut/snv \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "model":"OGR",
    "genome":"osa1_r7",
    "chromosome":"chr09",
    "start":20716774,
    "snv_index":20731844,
    "snv_base":"T",
    "output_format":"mean"
}'

# 返回: ref_values=参考表达, mut_values=突变后表达

# ===== 3. rice_reg: DNA+ATAC→RNA-seq(输出区分正负链) =====
# 必填: genome(MH63RS3|NIP)  chromosome(chr01-12)  start(1-based inclusive,窗口固定TARGET_LEN=32678)
# ATAC二选一: atac_source(SAM2_MH63_1↔MH63RS3 | SAM2_NIP_1↔NIP)
# 可选: end(窗口终点)  output_format(full默认|mean|downsample)
# 返回: values 分 RNA-seq_+(正链)/RNA-seq_-(负链) 两通道

curl -s -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_reg/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "model":"OGR",
    "genome":"MH63RS3",
    "chromosome":"chr01",
    "start":20716774,
    "atac_source":"SAM2_MH63_1",
    "output_format":"mean"
}'


# ===== 4. rice_intro: 粳/籼血缘渗入分析(256k 滑动窗口聚合) =====
# 参数全可选:
#   genome(默认第一个 GENOME_*_FASTA, 如 YF47) chromosome(默认 Chr01; FASTA 实际命名, 不归一化)
#   start(1-based inclusive) end(1-based inclusive)
# 窗口选择语义(start/end):
#   空/空    → 整条染色体(受后端 MAX_NUMBER_256W 限制: 空=全部窗口, N=前N个256k窗)
#   start/空 → 1个覆盖 start 的最大 256k 窗口(忽略 MAX_NUMBER_256W)
#   start/end → 与该区间覆盖度最大的 ≤MAX_NUMBER_256W 个 256k 窗口
# 返回: windows[] 每个 256k 窗口的 topk_mean_jap / topk_mean_ind + group(Ind|Jap|uncertain)
#       分组: Jap = topk_mean_jap ≥ THRESHOLD_JAP(0.5552) 且 topk_mean_ind < THRESHOLD_IND(0.5347)
#             Ind = topk_mean_jap < 0.5552 且 topk_mean_ind ≥ 0.5347; 否则 uncertain
# 计费: prompt_tokens=推理总碱基数(片段数×8000), completion_tokens=返回窗口数

# 窗口模式(推荐): 指定区间
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_intro/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{
    "genome": "YF47",
    "chromosome": "Chr01",
    "start": 100001,
    "end": 356001
  }'

# 整条模式: start/end 都留空(窗口数受 MAX_NUMBER_256W 限制; 当前部署 .env=4)
curl -s -X POST https://www.dcs.cloud/api/aigress/openai/OGR/rice_intro/predict \
  -H "Authorization: Bearer sk-zkXF-2J2-qwcSMgGh5KGPlZGw1HyTROJv70o2bJ5Uch5H5fx" -H "Content-Type: application/json" \
  -d '{"genome": "YF47", "chromosome": "Chr01"}'


# ===== 5. 测试 =====

# 一键启动(本地联调)
cd dcs_gateway && bash start_all.sh                                  # 先 conda activate vllm → 按.env拉起四后端+网关
curl -s http://127.0.0.1:9000/health | python3 -m json.tool          # 聚合健康检查(四服务全绿)

# 本机网关冒烟(URL路径路由): 复用上面 4 组命令, 仅把 BASE 换成网关地址
BASE=http://127.0.0.1:9000/api/aigress/openai/OGR

# DCS 平台联调: 同一入口 + 路径段区分子服务
#   /rice_mut/{predict,snv} | /rice_reg/predict | /rice_intro/predict | /rice_ogr/{dna_embedding,predict}
#   请求头带 Authorization: Bearer ${dcs_api_key}

# 常见错误
# 400: 参数缺漏/格式错(检查 mode/必填/chr01-12/1-based; model_sub 只能 rice_mut|rice_reg|rice_intro|rice_ogr)
# 401: 鉴权失败(检查 Authorization: Bearer <key> / X-API-Key: <key>)
# 404: 基因组/染色体不存在(检查 genome/chromosome 拼写与 FASTA 实际命名)
# 503: 模型初始化失败(/health 查 init_error)
# 5xx: 模型未就绪/推理异常(/health 查 init_error; 看日志 /tmp/rice_dcs/*.log)
# 502: 后端不可达(检查后端进程/端口是否存活)

# ===== 6. 部署心得 =====

# 地址不要含 '-' (DCS平台入口/服务命名规范)
# 用 tosutil 传模型/基因组到云平台对象存储
# 基因组 fa 和 fa.fai: 一定要先传 fa, 再传 fai (索引依赖主文件)
# 一键部署: dcs_gateway/start_all.sh 自动 conda activate vllm 并按 .env 指定解释器起四后端+网关(日志 /tmp/rice_dcs/*.log)
#   跨机部署: 改 dcs_gateway/.env 的 RICE_*_HOST / RICE_*_PORT / RICE_*_PYTHON
# 容器内模型路径: 环境变量注入覆盖, 如 MODEL_1B_8k_PATH=/AI_models/rice_mut/rice_1B_stage2_8k_hf(rice_OGR注册表); rice_mut/rice_reg 同理覆盖checkpoint
# 端口: rice_OGR DCS适配层用6001(与rice_mut的8001错开); 同机部署勿用 rice_OGR/.env 默认的 BACKEND_PORT=8001(会冲突)
#       rice_intro DCS适配层默认监听 5001(平台注入 PORT 优先); 与网页版 rice_intro 后端 5001 在不同机器
# 多卡: rice_ogr --device cuda:0,cuda:1 --device_map auto; rice_mut/reg 用 CUDA_VISIBLE_DEVICES 控制
# 安全: rice_mut/reg 网页后端把根目录挂 /static-files 静态服务(IGV加载bigWig用), 暴露公网前必须白名单目录或加鉴权
# 缓存: 预测结果写 bigWig(cache/predictions), 后台线程按TTL自动清理; Docker 不含 cache/ logs/ *.pid