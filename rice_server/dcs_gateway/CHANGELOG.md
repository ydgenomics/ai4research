# CHANGELOG — dcs_gateway 统一网关

## [0.4.2] - 2026-11-26

### 变更：OGR-Introgression 网页端与 API 端统一走同一 DCS 子路径（前端 = 填好参数的 curl）

对齐 OGR-Prediction 架构：网页端不再用内部端点 `/analyze`，而是直接调用
`POST …/OGR-Introgression/Prediction`（camelCase + 1-based，与 curl 同方言、同返回信封），
**Plotly 渲染数据完全来自该 API 返回值**。

**后端（`OGR_Introgression/backend/rice_introgression/api.py`）**：
- `Prediction` 子路径在 `run_introgression` 后追加 `inject_genome_context(payload)`，
  并附带 `elapsed_seconds` / `success`；`_predict_payload_to_result` 补全绘图字段：
  `chromosomes` / `chromosome_lengths`（12 条染色体车道骨架）、`regions[].chromosome`、
  `segments`、`message`、`cached`、`elapsed_seconds`。
- 网页内部端点 `/analyze`、`/analyze-genome` 保留但不再是网页预测入口。

**前端（`OGR_Introgression/frontend/app.py`）**：
- `_predict()` 改调 `…/Prediction` 并读信封 `result`（不再 POST `/analyze`）；
- 辅助查询仍走内部端点（`/genomes`、`/progress`），自定义基因组注册走 DCS `RegisterGenome`。

**文档**：README / API.md / backend/AGENT.md / start.sh 注释同步；返回结构示例补
`chromosomes`/`chromosome_lengths`/`regions.chromosome`/`cached`/`elapsed_seconds`。

**验证**：`py_compile` 通过；`_predict_payload_to_result` 字段完整性断言通过（渲染字段全部就位）。

---

## [0.4.1] - 2026-11-26

### 文档：全仓文档与代码对齐（无网关代码变更）

- **顶层 README.md**：四服务子路径表补齐（Prediction 增 `RegisterGenome`，并注明 `Genome`/`Chromosome` 仅后端直连/网页端；Introgression 改 `Prediction`（兼容 `predict`）并补 `Genome`/`Chromosome`）；
  `source/` 目录内容与各服务 `.env` 端口更新；手动启动命令改为统一后端入口，并注明 `start_all.sh` 以 `BACKEND_PORT` 覆盖各后端端口。
- **dcs_gateway/README.md**：路由目标由「各后端 `dcs_adapter` 进程」更正为「各后端统一服务进程」；
  手动启动命令改为 `rice_mutation/main.py` / `rice_reg/main.py` / `rice_introgression/main.py`（仅 OGR-Base 仍为 `dcs_adapter.py`）；
  补充 `start_all.sh` 的 `web` / `monitor` / `monitor-web` / `stop` 动作与 `setsid` 后台启动说明；停止方式改为精确命令行匹配。
- **dcs_gateway/API.md**：OGR-Introgression 子路径统一为 `Prediction`（兼容旧 `predict`）并补齐 `Genome`/`Chromosome`；
  删除已废弃的「整条染色体模式」示例，明确 **`Start` 必填**（缺 400）与 start 锚定窗口语义；
  OGR-Base 明确 **无 `model_name`/`mode`**（模型固定 `1B_8k`，功能由子路径决定）；参数命名说明补充 OGR-Base 小写键与 Introgression `Start` 必填。
- **OGR_Base**：`README.md` / `API.md` / `ref_genos.md` 全面重写——适配层与 Sanic 原生服务均已改为
  `OGR-Base/{Embedding,NTP}` 子路径、取消 `mode`/`model_name`、模型固定 `1B_8k`；模型路径更正为
  `source/OGR_Prediction/rice_1B_stage2_8k_hf`；`MODEL_1B_8k_MAX_LEN` 更正为 `131072`（128k）；
  补充容错预加载（`_INIT_ERROR` → 503）、端口 6001 约定与 `BACKEND_PORT=8001` 冲突提示。
- **OGR_Regulation/README.md**：目录结构（`rice_reg/cache_service.py` 等）、`CACHE_MODE` 说明、接口一览
  改为 DCS 子路径、前端 API Key 必填与网页端 `_call_dcs()` 说明。
- **OGR_Introgression/README.md**：目录根名 `rice_intro` → `OGR_Introgression`，补 `progress.py` / `scripts/`。
- **OGR_Applications_Call_Guide_private.md**：Introgression 更正为 **start 锚定**（`Start` 必填、取消网格匹配与整条模式、
  坐标 1-based inclusive）；Prediction/Regulation 自定义基因组在 `CACHE_MODE=false` 下为**路径文本框**（零拷贝注册）；
  OGR-Base 长度上限更正为 `131072`。
- **OGR_Prediction**：`test/README.md` 更新端点命名并标注 `test_dcs_api.sh` / `test_dcs_adapter.py` 为 legacy
  （独立 `dcs_adapter.py` 已删除）；`scripts/README.md` 路径更正为 `OGR_Prediction`。

**验证**：仅文档变更；已核对 `rice_mutation/api.py`、`rice_reg/api.py`、`rice_introgression/api.py`、
`OGR_Base/dcs_adapter.py`、`OGR_Base/dna_embedding.py`、`dcs_gateway/app.py` 与 `start_all.sh` 的实际路由/端口/启动方式。

---

## [0.4.0] - 2026-11-25

### 新增：文件系统探查接口（排查部署机文件/数据结构用）

部署机不在本地、无法 SSH，新增 4 个网关本地**只读**接口用于远程排查文件位置与目录结构
（网关实现，不转发后端、不碰模型进程）：

- `POST .../OGR/fs/list`（等价 `ls -lh /path`）：列目录，`{path}`，默认限 200 条
- `POST .../OGR/fs/stat`（等价 `ll /path`）：单文件/目录详情（type/size/mode/mtime/symlink target）
- `POST .../OGR/fs/find`（等价 `find /path -name PATTERN`）：递归按文件名模式搜索，
  `{path, pattern, max_depth, limit}`，默认深度 4 / 200 条（上限 20 / 1000）
- `POST .../OGR/fs/cat`（等价 `head -c 64K /path`）：读文件内容，`{path, tail, limit}`，默认/上限 64KB，UTF-8 容错解码

**设计**：
- **只读**：仅 list/stat/find/cat，无任何写、删、改、上传操作。
- **不限路径**：排查阶段可访问部署机任意绝对路径（`realpath` 解析）；用后请立即关闭。
- **开关控制**：`.env` 新增 `EXPLORE_ENABLED`（默认 `false`，生产保持关闭）；
  开启后若配置了 `DCS_API_KEY` 则强制鉴权（`Authorization: Bearer <key>` / `X-API-Key: <key>`）。
- **限流限长**：`EXPLORE_MAX_FILE_BYTES`（cat，默认 65536）/ `EXPLORE_MAX_LIST_ITEMS`（list，默认 200）/ `EXPLORE_MAX_FIND_DEPTH`（find，默认 4）。
- 返回沿用四服务统一结构（`usage`/`token_cnt`/`status`/`message`/`result`，计费为 0）；
  错误方言一致：404（不存在）/ 403（无权限或未启用）/ 400（参数错）/ 500（IO 异常）。

**文档**：API.md 新增 §4.6 探查接口；README.md 目录结构/行为约定补充说明。

**验证**：`py_compile` 通过；路由为具体路径（`/api/aigress/openai/OGR/fs/*`），优先于 catch-all 转发路由。

---

## [0.3.1] - 2026-11-17

### 修复：响应头透传的压缩错位隐患（0.2.0 遗留）

审阅 0.2.0 连接复用/超时修复时发现响应头透传存在潜在缺陷：

- **问题**：httpx 默认自动解压 gzip/deflate/br 响应（`resp.content` 已是解码后
  字节），但 `resp.headers` 仍保留 `content-encoding` 与压缩前的
  `content-length`。0.2.0 改为「除 hop-by-hop 外原样透传」后，一旦任一后端
  启用压缩（当前四个后端均未启用 GZipMiddleware，未触发），客户端会收到
  「已解压 body + gzip 头 + 错误长度」→ 二次解压失败。
- **改动**：
  - 响应侧剔除 `content-encoding` / `content-length`：由 Starlette 按实际
    （已解压）body 重算长度，保证客户端拿到未压缩的正确 JSON；
    其余头（Set-Cookie/X-Backend 等）继续原样透传。
  - 请求侧不透传 `accept-encoding`：避免把客户端的压缩偏好转发给后端，
    从源头杜绝后端响应被压缩。
- **验证**：`py_compile` 通过；本地 gzip 假后端端到端验证——直接请求返回
  `content-encoding: gzip` + 压缩 body，经网关返回解压后的 JSON +
  正确 `content-length: 32`，其余头正常透传。

**注**：风险等级为中（当前后端未启用压缩，线上未受影响），属防御性修复。

---

## [0.3.0] - 2026-11-16

### 变更：适配四个应用 API 改版（子路径路由 + 新命名）

四个后端 dcs_adapter 已改版：**取消 body `mode` 分发，全部改为纯子路径路由**，
服务与参数统一重命名为 OGR 系列（大写驼峰参数）。网关同步更新：

- **服务改名**（外部 `model_sub`）：
  - `rice_mut` → `OGR-Prediction`（变异对比表达预测，:8001）
  - `rice_reg` → `OGR-Regulation`（ATAC 条件表达预测，:7001）
  - `rice_intro` → `OGR-Introgression`（粳/籼血缘渗入分析，:5001）
  - `rice_ogr` → `OGR-Base`（embedding / 碱基预测，:6001，缺省）
- **子路径映射**（URL 第二段 → 后端子路径，大小写/连字符不敏感）：
  - `OGR-Prediction`：`prediction`→`/Prediction`、`snv`→`/SNV`
  - `OGR-Regulation`：`genome`→`/Genome`、`chromosome`→`/Chromosome`、`prediction`→`/Prediction`
  - `OGR-Introgression`：`predict`→`/predict`
  - `OGR-Base`：`embedding`→`/Embedding`、`ntp`→`/NTP`
- **转发逻辑**：四个后端均为纯子路径路由，网关**不再注入 body `mode`**，只做路径重写；
  `model_sub` 剥离后其余字段（`Genome`/`Chromosome`/`Start`/`sequence`/…）原样透传。
- **路由匹配**：`_parse_path` 对 model_sub 做去连字符归一（`ogrprediction`→`ogr_prediction` 等），
  兼容 `OGR-Prediction` / `OGRPrediction` / `ogr_prediction` 写法。
- **start_all.sh / .env**：服务开关 `ENABLED_SERVICES` 与目录映射更新
  （`ogr_prediction`→`OGR_Prediction/backend`、`ogr_regulation`→`OGR_Regulation/backend`、
  `ogr_introgression`→`OGR_Introgression/backend`、`ogr_base`→`OGR_Base`）；
  `.env` 变量名保留 `RICE_*_` 前缀（代码引用不变），默认 `GATEWAY_PORT=4000`。
- **文档**：README.md / API.md 全部更新为新服务名与子路径调用方式。

**验证**：`py_compile` 通过；URL 路径路由（`/OGR/OGR-Prediction/Prediction` 等）、
body 路由（`model_sub`）、聚合 `/health` 均按新路由表工作。

---

## [0.2.0] - 2026-10-14

### 修复：统一网关在 DCS 平台下随机报错（503）问题

**背景**：`model=OGR`（统一网关部署）实测并发时 30-60% 请求失败（503 `upstream connect error ... connection failure`），而 `model=OGR2`（直接部署）100% 成功。根因在网关 `app.py` 自身。

**改动**：

- **连接复用**：`http.client`（每请求新建 TCP 连接）→ `httpx.AsyncClient` 全局 keep-alive 连接池。避免瞬时新建连接数冲高撞上平台对后端的连接配额（本地并发 20 压测仅保持 2 条连接，20/20 全 200）。
- **健康检查异步化**：聚合 `/health` 由同步串行探活（最多冻结事件循环 20s）→ `asyncio.gather` 并发异步探活（超时 2s），探活不再阻塞转发请求。
- **精细超时语义**：`timeout=180` 一刀切 → `httpx.Timeout(connect=5s, read=600s, ...)`；连接失败→502，读超时→504（区分"不可达"与"慢"）。
- **响应头透传**：仅透传 content-type → 除 hop-by-hop 外原样透传（保留 Content-Encoding 等）。
- 移除 `asyncio.to_thread` 线程池转发（默认 32 线程上限，并发排队瓶颈）。

**依赖变化**：新增 `httpx>=0.28`（环境内已有，无需额外安装）。

**验证**：`py_compile` + 路由单测通过；本地假后端端到端冒烟（URL 路径路由 / body 路由 / 聚合健康）通过；并发 20 压测 20/20 成功。

### 部署注意

- 监听端口：`PORT`（平台注入）> `GATEWAY_PORT` > 9000。部署时如外部监听 4000，需设置 `.env` 的 `GATEWAY_PORT=4000`（或平台注入 `PORT=4000`）。
- 启动：`bash start_all.sh rice_ogr rice_mut`（仅跑指定后端 + 网关）。
