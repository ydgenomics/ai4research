# Rice-Server — 水稻基因组学模型服务仓库

基于 **Genos DNA 大模型**衍生的 1B 水稻基模，提供 **四个水稻基因组学服务**（共享 `source/` 模型/基因组数据），
通过 **DCS 平台统一网关** 以 OpenAI 风格对外提供 API。

> **API 调用（唯一完整文档）**：[dcs_gateway/API.md](dcs_gateway/API.md)（四合一：统一入口 + 本机/DCS 测试代码 + 各服务参数表 + 错误排查 + fs 探查接口）
> **调用速查（含网页端）**：[OGR_Applications_Call_Guide_private.md](OGR_Applications_Call_Guide_private.md)
> **网关实现**：[dcs_gateway/README.md](dcs_gateway/README.md)

---

## 1. 四个服务一览

| 服务 | 能力 | 基础模型 | 输入 | 输出 | DCS 子路径 |
|---|---|---|---|---|---|
| **OGR-Prediction** | DNA → 多组学表达预测（**参考 vs SNV 变异对比**） | `rice_1B_stage2_8k_hf` | DNA 序列 | assay×biosample 多维表达 + bigWig | `Prediction` / `SNV` / `RegisterGenome` |
| **OGR-Regulation** | DNA + **ATAC** → RNA-seq 表达预测 | `rice_1B_32k_hf` | DNA + ATAC bigWig | RNA-seq ± 两通道 + bigWig | `Genome` / `Chromosome` / `Prediction` / `RegisterGenome` |
| **OGR-Introgression** | 粳/籼血缘渗入分析 | LoRA + 头 | DNA 区域 | 渗入窗口分组（Jap/Ind/uncertain） | `Prediction`（兼容 `predict`）/ `Genome` / `Chromosome` / `RegisterGenome` |
| **OGR-Base**（**缺省**） | **embedding 提取 / 下游碱基预测** | 注册表（固定 `1B_8k`） | DNA 序列 | 1024 维向量 / 预测碱基 | `Embedding` / `NTP` |

**统一入口（DCS 平台只配一个地址）**：`POST .../api/aigress/openai/OGR/{model_sub}[/{subpath}]`
（URL 路径路由，推荐；或 body `model_sub` 字段路由，兼容旧式）。网关 `dcs_gateway/` 单端口（默认 4000）收口四服务，**不加载模型**。

**公共约定**：
- **返回结构**：`{"usage": {prompt_tokens, completion_tokens}, "token_cnt", "status", "message", "result"}`
- **计费**：水稻基模单碱基编码，**1 bp = 1 token**；`token_cnt = prompt_tokens + completion_tokens`
- **鉴权**：`Authorization: Bearer <key>` 或 `X-API-Key: <key>`（`/health` 免鉴权）
- **自定义文件**：OGR-Prediction / Regulation / Introgression 支持 `RegisterGenome` 传**本地文件路径**注册自定义基因组（FASTA/GFF，不做文件传输）

---

## 2. 目录结构

```
rice_server/
├── README.md                # 本文档（索引导航）
├── OGR_Applications_Call_Guide_private.md  # 调用速查（网页端 + DCS API 端）
│
├── OGR_Prediction/          # 服务一：变异对比表达预测（网页 :8000 / API 后端 :8001，同一进程）
├── OGR_Regulation/          # 服务二：ATAC 条件表达预测（网页 :7000 / API 后端 :7001，同一进程）
├── OGR_Introgression/       # 服务三：粳/籼血缘渗入分析（网页 :5000 / API 后端 :5001，同一进程）
├── OGR_Base/                # 服务四：embedding / 碱基预测（纯 API，DCS 适配 :6001）
│
├── dcs_gateway/            # ★ 统一网关：单端口收口四服务的 DCS API
│   ├── app.py               #   FastAPI 轻量反代（model_sub/子路径路由，不加载模型）
│   ├── API.md               #   ★ 四合一 API 调用文档（本机 + DCS 测试代码）
│   ├── .env                 #   后端地址 / 端口 / 服务开关 / fs 探查开关
│   ├── start_all.sh         #   一键拉起：四后端 + 网关
│   └── run_gateway.sh       #   网关启动脚本
│
├── docker/                  # 交付镜像（org_web / org_web-jupyter / push 脚本）
│
└── source/                  # ★ 模型 + 基因组数据（交付必带）
    ├── OGR_Prediction/      #   rice_1B_stage2_8k_hf + csq-5_model.safetensors + osa1_r7（fa/gff3/fai/gvlfa）+ index_stat.json
    ├── OGR_Regulation/      #   rice_1B_32k_hf + three_sam2.model.safetensors + genome（MH63/NIP）+ ATAC bigWig
    └── OGR_Introgression/   #   渗入分析基因组（genome/）+ model.safetensors
```

> 各服务内部结构一致：`backend/`（FastAPI 后端，**网页端与 DCS 子路径同一进程**）、`frontend/`（Gradio 前端 + IGV.js）、
> `tools/startup_self_check.sh`（启动自检）、`test/`（API 冒烟测试）、`cache/`（运行时缓存）、`logs/`。
> OGR-Prediction/Regulation/Introgression 已删除独立 `dcs_adapter.py`（并入各自 `main.py`）；仅 OGR-Base 仍为独立 `dcs_adapter.py` 适配层（子路径 `OGR-Base/Embedding`、`OGR-Base/NTP`）。

---

## 3. 启动

### 3.1 一键拉起（推荐）

```bash
cd dcs_gateway && bash start_all.sh        # 全部服务（四后端 + 网关）
bash start_all.sh ogr_prediction ogr_regulation   # 只启动指定后端 + 网关
```

脚本先 `conda activate vllm`，按优先级解析各后端 Python 解释器与端口（`.env` 的 `RICE_*_PYTHON`/`RICE_*_PORT` 最高），
等待各后端 `/health` 就绪（最长 180s）后前台拉起网关（默认 4000）。日志在 `/tmp/rice_dcs/*.log`。

### 3.2 手动启动

```bash
# 各后端统一服务（网页端 + DCS 子路径同一进程，加载模型）
cd ../OGR_Prediction/backend && /root/miniconda3/envs/vllm/bin/python rice_mutation/main.py   # :8001
cd ../OGR_Regulation/backend && /root/miniconda3/envs/vllm/bin/python rice_reg/main.py          # :7001
cd ../OGR_Introgression/backend && /root/miniconda3/envs/vllm/bin/python rice_introgression/main.py  # :5001
cd ../OGR_Base && BACKEND_PORT=6001 /root/miniconda3/envs/vllm/bin/python dcs_adapter.py  # :6001（默认 8001，必须覆盖为网关期望的 6001）

# 网关
cd dcs_gateway && bash run_gateway.sh      # 加载 .env，监听 GATEWAY_PORT（默认 4000；DCS 平台注入 PORT 优先）
```

> 网关 `start_all.sh` 启动时统一以 `BACKEND_PORT=<端口>` 环境变量覆盖各后端端口（OGR_Base 由 `RICE_OGR_PORT` 决定，默认 6001）。

> 各服务完整说明（网页版启动、配置 .env、模型路径）见各自 `README.md`。

---

## 4. 验证

```bash
curl -s http://127.0.0.1:4000/health | python3 -m json.tool   # 网关聚合健康检查（免鉴权）
# → {"status":"ok|degraded","services":{"ogr_prediction":{...},"ogr_regulation":{...},...}}
```

完整调用示例（本机 + DCS 两套 curl）见 [dcs_gateway/API.md](dcs_gateway/API.md)。

---

## 5. 部署注意事项

1. **安全**：OGR-Prediction / Regulation 后端将根文件系统挂载为 `/static-files` 静态服务（IGV 加载 bigWig），**暴露公网前必须改白名单目录或加鉴权**。
2. **数据必带**：`source/`（模型 + 基因组）无法从网络重取，打包交付必须包含；`cache/`、`logs/`、`*.pid` 为运行时产物，打包前删除。
3. **路径耦合**：`.env` 均为开发机绝对路径，部署机必须逐项修改（各服务 `.env.example` 有说明）。
4. **版本锁定**：`requirements.txt` 用 `>=`，交付前建议 `pip freeze > requirements-lock.txt`。
5. **fs 探查接口**（调试用）：`dcs_gateway/.env` 的 `EXPLORE_ENABLED=true` 可开放 `POST .../OGR/fs/{list,stat,find,cat}` 远程查看部署机文件（详见 [API.md §4.6](dcs_gateway/API.md)）；**排查完请关闭**。
