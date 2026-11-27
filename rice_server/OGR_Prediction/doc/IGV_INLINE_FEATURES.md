# OGR-Prediction 可视化改造方案：基于 API 返回值内联 IGV，摆脱 bigWig 落盘

> 文档日期：2026-11-26
> 状态：**方案评审（未实施）**
> 目标：预测流程不再生成/依赖 `.bw` 文件，IGV 直接消费 API（curl）返回值里的数值数组进行渲染。

---

## 1. 背景与现状：bw 文件是怎么产生、为什么必须产生

### 1.1 产生位置

后端 `backend/rice_mutation/prediction_service.py`：

- `save_prediction_bigwigs()` → `_array_to_bigwig()`：每次预测完成后，把 `{assay: {biosample: np.ndarray[L]}}` 的数值数组（L = 32768，即 `MAX_SEQ_LEN`）用 `pyBigWig` 逐轨道写成一个 `.bw` 文件。
- 输出目录跟随缓存模式：
  - `CACHE_MODE=true` → `BACKEND_PREDICTION_CACHE`（持久，TTL 1800s）
  - `CACHE_MODE=false`（当前部署）→ `BACKEND_PREDICTION_TMP=/tmp/rice_prediction_tmp`（临时，TTL 900s，后台线程每 120s 清理）

### 1.2 为什么必须产生

IGV.js 的 wig 轨道需要 **URL 数据源**。当前链路：

```
预测完成
  → 写 .bw 到 /tmp
  → igv_payload.tracks[].url = /backend/static-files/<绝对路径>/xxx.bw
  → 前端 _igv_html 把 payload 原样塞进 iframe srcdoc
  → 浏览器经前端 /backend/* 反向代理拉取 .bw
  → IGV 渲染
```

关键事实：**API 返回值里其实已经带了完整的 `values`（Full 数组，32768 长度）**，但前端 IGV 渲染没有直接用，而是选择了"再拉一个 bw 文件"这条路径。

### 1.3 当前数据流（单次预测）

| 环节 | 行为 |
|---|---|
| 模型推理 | 输出 `{assay: {biosample: np.ndarray[32768]}}` |
| 写盘 | 每个 (assay, biosample) 写一个 `.bw`（当前单轨道场景 1 个，SNV 双轨 2 个） |
| API 返回 | `result.values`（Full 数组）+ `result.igv_payload`（bw URL + locus + tracks） |
| IGV 渲染 | 浏览器再发 Range 请求经反代拉 bw |

---

## 2. 目标方案：直接基于 curl 返回值传给 IGV，不依赖 bw

### 2.1 核心思路

返回的 `result.values` 已包含每个轨道的完整浮点数组。IGV.js 的 wig 轨道**原生支持 `features` 内联数组**：

```js
{
  name: "Leaf RNA-seq",
  type: "wig",
  format: "wig",          // ← 关键：不要 "bigwig"
  color: "#1f77b4",
  autoscale: true,
  // url 字段直接省略
  features: [
    {chr: "Chr9", start: 20716772, end: 20716773, value: 0.234},
    // ...（每 bin 一条）
  ]
}
```

### 2.2 可行性依据（已在本地静态 IGV 库中核实）

对 `frontend/static/igv.min.js`（本地静态 IGV.js）的代码确认：

1. **内联数组优先**：轨道配置带 `features` 时：
   ```js
   e.features) this.featureSource = new Rc(e.features, t)
   ```
   直接用内联数组做 feature source，**不碰 url**。

2. **格式判断**：`"bigwig" / "bw" / "wig" / "bedgraph" / "tdf" → type "wig"`；只有 `format === "bigwig"` 时才构造 BigWig 读取器（`new uc(...)`）。所以把 `format` 设为 `"wig"` 并去掉 `url`，**不会触发任何 bw 下载**。

3. **已做过独立验证**：`/tmp/igvtest/test_wigfmt.html` 用 5000 条 `{chr,start,end,value}` 内联 features（`format:"wig"`）实测，IGV 正常渲染（标题 `IGV_WIGFMT_OK`）。

### 2.3 改造落点（未实施，仅规划）

| 文件 | 改动 |
|---|---|
| `backend/rice_mutation/prediction_service.py` | `save_prediction_bigwigs()` 不再写 `.bw`，改为返回每轨道内联 features（或降采样 bins）；`_array_to_bigwig` 可保留供"bw 模式"开关使用 |
| `backend/rice_mutation/igv_payload.py` | `_track()` 从 `url` 模式改为 `features` 模式（`format:"wig"`，省略 url） |
| `backend/rice_mutation/api.py` | `result.values` 本就带着，直接复用；payload 结构升级 |
| `frontend/app.py` | `_igv_html` 无需改 URL 重写逻辑，payload 原样 JSON 进 iframe（当前已兼容） |
| `.env` | 新增开关 `IGV_INLINE_FEATURES=true/false`（见 §4 建议） |

---

## 3. 收益

- **预测完完全不再落盘任何 bw**：`/tmp/rice_prediction_tmp` 及 TTL 清理线程可移除。
- 单次预测少一次磁盘写 + 少一轮"IGV 再拉 bw"的网络往返。
- SNV 场景少写 2 个文件。
- DCS 网关 / 纯 curl 调用天然获得"可视化数据即返回值"。
- 与"前端直接消费 API 返回内容"的整体改造方向一致（参见仓库记忆 `rice_server-arch.md` 的 API 返回 URL 改造）。

---

## 4. 风险与使用损失（重点）

### 4.1 响应体变大（最主要）

- Full 模式下 `values` 已是 32768 长度数组；若再逐 bin 展开成 `features`（每条 4 字段 JSON），单轨道约 32768 条 → 序列化后约 **1.5–2 MB JSON**。
- 多轨道、SNV 双轨（ref+mut）会到 **3–6 MB**。
- 网页端（Gradio iframe srcdoc 内嵌 JSON）影响不大；但**纯 API 调用方 / DCS 网关链路**有传输与解析开销，若平台对响应体有大小限制需评估。
- **缓解**：bins 降采样到 4096（每 8bp 一条，渲染无肉眼差别），单轨道 JSON 可压到 ~200KB 内；`Format=Mean` 场景不带 features。

### 4.2 IGV 渲染性能

- 32768 条 features 全部内联给 IGV，初次渲染 DOM/Canvas 遍历量大于 bw 的 zoom-level 多分辨率读取。
- 本地静态 IGV 1.2.x 可扛，但缩放/平移时 iframe 内全量数组常驻内存。
- **缓解**：同样用降采样（4096 bins）。

### 4.3 IGV 原生"数据导出/另存"能力降级

- IGV 轨道右键导出 bigWig、以及其他依赖真实文件 URL 的数据导出功能，在内联 features 轨道上不可用（没有 URL）。
- 对当前网页端**无影响**：Save PNG 是自绘 canvas；右键 Bar/Line 切换用的是 `graphType` 属性，均不依赖 URL。
- 若未来有"让用户下载这条 bw"的需求，需保留 bw 模式开关。

### 4.4 缓存命中路径（CACHE_MODE=true 时）

- 内容寻址缓存里存的 `igv_payload` 目前是 bw URL。
- features 模式不再落盘后，缓存命中时 features 必须从 `_REF_CACHE` 数组重新生成，或 payload 结构升级。
- 当前部署是 `CACHE_MODE=false`，影响不大，但两种模式都要兼容。

### 4.5 坐标语义需精确对齐

- 现在 `_array_to_bigwig` 用 `start + i*step` 的 float 步长分段。
- 改 features 后 bin 边界要么同样用 `step=(end-start)/n` 推导，要么用整数 bin（每 bin `start + i*(len/n)`）。
- **IGV 的 locus 是 1-based 含端点**：bin 的 start/end 若按 0-based half-open 写，IGV 上会偏移 1bp。32768 窗口内看不出，但 SNV 精确比对（SNV 位点是否在窗口内）等场景必须保证一致。

### 4.6 首次 iframe 体积

- `_igv_html` 把整个 JSON 塞进 `srcdoc` iframe；features 数组会让 srcdoc 增大几 MB，页面初次渲染 HTML 转义/解析有一次性开销（可接受）。

---

## 5. 结论与建议

方案技术上完全成立，核心权衡是 **"用带宽/内存换掉磁盘文件"**。

对于当前"网页端可视化 + 短 TTL 临时 bw"的使用场景，风险主要是响应体大小与 IGV 内联渲染性能，均可用降采样（4096 bins）把单轨道 JSON 压到 ~200KB 内解决；真正的使用损失只有"IGV 轨道级导出 / 依赖 URL 的能力"一项，而当前网页 UI 没有用到。

**建议实施方式**：按 **"降采样 features + 保留可选 bw 开关"** 改造：

```
.env:
IGV_INLINE_FEATURES=true      # 默认零 bw：payload 带 features，不写盘
IGV_INLINE_FEATURES=false     # 回到现有 bw URL 模式（兼容/导出场景）
IGV_INLINE_BINS=4096          # 降采样 bin 数（默认 4096，可调 1024/2048/4096/32768）
```

- 默认 `true`：零 bw、短 TTL 清理线程下线。
- 需要 IGV 轨道导出或纯 API 下游要 bw 时，一键切 `false` 回到现链路。

---

## 附录 A：相关文件索引

| 文件 | 说明 |
|---|---|
| `backend/rice_mutation/prediction_service.py` | `save_prediction_bigwigs` / `_array_to_bigwig` / `get_prediction_cache_dir` / `get_bigwig_cleanup_settings` |
| `backend/rice_mutation/igv_payload.py` | `build_prediction_payload` / `_track` / `_path_to_url` |
| `backend/rice_mutation/api.py` | `/predict` `/predict/snv` 统一 envelope，`result.values` 源头 |
| `frontend/app.py` | `_igv_html`（srcdoc iframe）、`_predict_reference` / `_predict_snv` |
| `frontend/static/igv.min.js` | 本地静态 IGV.js（已核实 features 内联支持） |
| `.env` | `CACHE_MODE` / `BACKEND_PREDICTION_TMP` / `TMP_BIGWIG_*` |
