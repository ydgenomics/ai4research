# CHANGELOG — dcs_gateway 统一网关

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
