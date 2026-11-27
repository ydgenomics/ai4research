"""OGR-Introgression 前端 —— Gradio + Plotly(iframe) 交互式渗入浏览器。

风格与 rice_reg（OGR-Reg）保持一致：
- 橙色主题 + 顶部标题栏（左）+ EN / 中文 语言切换 pills（右）
- 卡片式参数面板在上，可视化全宽在下（无左右布局）
- 基因组提交：下拉含「📤 Custom Genome」哨兵，选中后输入本机 FASTA 路径
  （轻量模式：只传路径，RegisterGenome 零拷贝懒注册，不传文件）
- 中英文切换（I18N）；API Key 必填（页面输入框，请求头鉴权）

可视化：全基因组三 track 图 —— 所有染色体车道（Chr12 顶 → Chr01 底），每条
染色体 3 个紧密 track（Jap 蓝 / Ind 橘 / uncertain 绿，块内无行距），未推理区
gray 底色（Plotly iframe 渲染）；鼠标悬浮区域显示位置/类别细节；推理进度通过
/progress 端点轮询（每 2s）。
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 用户 2026-11 指定：Jap 蓝 / Ind 橘 / uncertain 绿 / 未推理 gray
GROUP_COLORS = {"Jap": "#4874cb", "Ind": "#dc6b2d", "uncertain": "#4caf50"}
# 未推理（无 region 区域）底色：gray
UNINFERENCED_COLOR = "#bdbdbd"

DEFAULT_LANG = "en"
CUSTOM_GENOME_VALUE = "__custom__"


# ---------------------------------------------------------------------------
#  i18n — UI 文案（默认中文；右上角 EN/中文 切换）
# ---------------------------------------------------------------------------
I18N = {
    "en": {
        "title": "OGR-Introgression",
        "subtitle": "Introgression analysis (Jap / Ind) for a region on one chromosome. Inference runs only on the requested region; the result is shown as three per-chromosome tracks (Jap / Ind / uncertain).",
        "genome": "Genome",
        "chromosome": "Chromosome",
        "start": "Start (required, 1-based)",
        "end": "End (optional; empty = start+256k)",
        "start_placeholder": "e.g. 100000",
        "end_placeholder": "e.g. 356000",
        "predict": "🚀 Predict",
        "custom_genome_option": "📤 Custom Genome",
        "upload_genome_fasta": "Genome FASTA path (.fa/.fasta/.fna)",
        "upload_fasta_ph": "/abs/path/to/xxx.fa",
        "upload_status_idle": "Select “Custom Genome” and enter the local FASTA path to register a new genome (lightweight mode: path-only, no file upload).",
        "upload_status_fasta_ok": "Genome “{gid}” registered ({n} chromosomes).",
        "upload_status_err_fasta": "FASTA registration failed: {msg}",
        "placeholder": "Select a genome & chromosome, enter Start, then click Predict to see the three-track result.",
        "err_no_genome": "Please select a genome.",
        "err_no_chromosome": "Please select a chromosome.",
        "err_no_start": "Please enter a Start (1-based) — it is required.",
        "err_need_api_key": "API Key is required — please enter your DCS API key.",
        "api_key": "API Key (required)",
        "api_key_ph": "sk-... (required — enter your DCS API key)",
        "progress_run": "Running: {pct:.0f}% (batch {done}/{total} · {elapsed:.0f}s) · {chrom}",
        "done": "{cached}Done ({elapsed}s · {n_win} windows · {n_seg} segments)",
        "predict_failed": "Prediction failed: {msg}",
        "empty_result": "No window to infer: {msg}",
    },
    "zh": {
        "title": "OGR-Introgression",
        "subtitle": "渗入分析 —— 粳/籼。仅对一条染色体的请求区域推理，结果按 Jap / Ind / uncertain 三条 track 展示。",
        "genome": "基因组",
        "chromosome": "染色体",
        "start": "Start（必填，1-based）",
        "end": "End（可选，留空=start+256k）",
        "start_placeholder": "如 100000",
        "end_placeholder": "如 356000",
        "predict": "🚀 预测",
        "custom_genome_option": "📤 自定义基因组",
        "upload_genome_fasta": "基因组 FASTA 路径 (.fa/.fasta/.fna)",
        "upload_fasta_ph": "/abs/path/to/xxx.fa",
        "upload_status_idle": "选择「自定义基因组」并输入本机 FASTA 路径以注册新基因组（轻量模式：只传路径，不传文件）。",
        "upload_status_fasta_ok": "基因组「{gid}」已注册（{n} 条染色体）。",
        "upload_status_err_fasta": "FASTA 注册失败：{msg}",
        "placeholder": "选择基因组与染色体、填写 Start 后点击预测，查看三 track 结果。",
        "err_no_genome": "请选择基因组。",
        "err_no_chromosome": "请选择染色体。",
        "err_no_start": "请填写 Start（1-based）——该项必填。",
        "err_need_api_key": "API Key 必填——请输入你的 DCS API Key。",
        "api_key": "API Key（必填）",
        "api_key_ph": "sk-...（必填——请输入你的 DCS API Key）",
        "progress_run": "推理中：{pct:.0f}%（批次 {done}/{total} · {elapsed:.0f}s） · {chrom}",
        "done": "{cached}完成（{elapsed}s · {n_win} 窗口 · {n_seg} 片段）",
        "predict_failed": "预测失败：{msg}",
        "empty_result": "无窗口可推理：{msg}",
    },
}


def _t(lang: str | None) -> dict:
    return I18N.get(lang or DEFAULT_LANG, I18N[DEFAULT_LANG])


# ---------------------------------------------------------------------------
#  样式（与 rice_reg 保持一致：橙色调、header bar、卡片面板）
# ---------------------------------------------------------------------------
_CSS = """footer {display:none !important}
html, body {overflow-x:hidden !important;}
.gradio-container {max-width:100% !important; overflow-x:hidden !important;}
body {font-family: Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif !important;}
main.app {max-width:none !important; width:100% !important;}

/* ── Header bar: title left, EN/中文 pills right ── */
.app-header-bar {display:flex !important; align-items:center !important; justify-content:space-between !important; gap:16px; margin-bottom:4px; flex-wrap:wrap;}
.app-header-title {margin:0 !important; min-width:0 !important;}
.app-header-title h1 {margin:0 !important; font-size:1.5rem !important; line-height:1.3 !important;}
.app-header-actions {display:flex !important; justify-content:flex-end !important; align-items:center !important; gap:8px; flex-wrap:nowrap;}
.lang-toggle-btn {min-width:88px; min-height:36px; padding:0 18px; border-radius:999px !important; font-weight:600; box-shadow:none !important;}

.intro-copy {margin-top:2px;}
.intro-copy p {margin-top:0; color:#6B7280 !important;}

/* ── Card-style panel (white rounded card) ── */
.card-panel {background:#fff !important; border:1px solid #e5e7eb !important; border-radius:12px !important; padding:14px 16px !important; box-shadow:0 1px 2px rgba(16,24,40,.04) !important;}

/* Genome row splits into equal columns: Genome | FASTA path */
.layout-row-2sub {flex-wrap:nowrap !important;}
.layout-row-2sub > * {min-width:0 !important;}
#genome-upload {min-height:52px !important;}

#predict-btn {min-height:36px !important; height:36px !important; width:100% !important; margin-top:4px !important;}

@media (max-width: 900px) {
    .app-header-title h1 {font-size:1.25rem !important;}
    .lang-toggle-btn {min-width:72px; min-height:32px; padding:0 12px;}
}
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="{plotly_url}"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: #fafafa; }}
  #chart {{ width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<div id="chart"></div>
<script>
{js_payload}
Plotly.newPlot('chart', FIGURE.data, FIGURE.layout, FIGURE.config);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
#  后端通信（同步 httpx——Gradio 事件处理器运行在工作线程，避免 asyncio 嵌套）
# ---------------------------------------------------------------------------
# 3600s：最大窗口推理（5488 片段）可能超过 10 分钟，600s 默认超时不够
_HTTP = httpx.Client(timeout=httpx.Timeout(3600.0, connect=30.0))


def _auth_headers(api_key: str = None) -> dict:
    """根据 API Key 构造鉴权头（空 key 不带头）。

    只使用页面输入框提供的 key；前端故意不读 .env 的 DCS_API_KEY——
    那是后端校验侧的 key，读了会让输入框失去意义。
    """
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _backend_get(path: str, api_key: str = None):
    r = _HTTP.get(
        f"{config.get_backend_base_url()}{path}",
        headers=_auth_headers(api_key),
    )
    r.raise_for_status()
    return r.json()


def _backend_post(path: str, payload: dict, api_key: str = None):
    r = _HTTP.post(
        f"{config.get_backend_base_url()}{path}",
        json=payload,
        headers=_auth_headers(api_key),
    )
    r.raise_for_status()
    return r.json()


def _backend_upload_fasta(filename: str, local_path: str, api_key: str = None):
    # 轻量模式（CACHE_MODE=false）下同时带 path 表单字段，后端原地注册零拷贝；
    # 全量模式忽略 path（或后端未配置 path 分支时自动回退字节上传）。
    with open(local_path, "rb") as f:
        r = _HTTP.post(
            f"{config.get_backend_base_url()}/uploadFasta",
            files={"file": (filename, f)},
            data={"path": local_path},
            headers=_auth_headers(api_key),
        )
    r.raise_for_status()
    return r.json()


def _backend_register_genome(fasta_path: str, api_key: str = None) -> dict:
    """经 DCS RegisterGenome 子路径注册本机 FASTA（传路径，零拷贝懒注册）。

    CACHE_MODE=false（轻量模式）下前端/后端同机，用户粘贴绝对路径即可；
    后端注册该路径（managed=False，不复制文件），返回 {genome, chromosomes}。
    """
    r = _HTTP.post(
        f"{config.get_backend_base_url()}/api/aigress/openai/OGR-Introgression/RegisterGenome",
        json={"fasta_path": fasta_path},
        headers=_auth_headers(api_key),
    )
    r.raise_for_status()
    data = r.json()
    status = data.get("status", 200)
    if status != 200:
        raise RuntimeError(data.get("message", "RegisterGenome failed"))
    return data.get("result") or {}


def _uploaded_file_path(file) -> Optional[str]:
    """从 Gradio 上传文件对象提取稳定本地路径（防御各种中间态）。"""
    if file is None:
        return None
    if isinstance(file, tuple):
        for item in file:
            p = _uploaded_file_path(item)
            if p:
                return p
        return None
    if hasattr(file, "path"):
        return file.path
    if isinstance(file, dict):
        return file.get("name") or file.get("path")
    if isinstance(file, str):
        return file
    return getattr(file, "name", None)


def _fmt_upload_status(t: dict, key: str, gid: str = "", n: int = 0, msg: str = "") -> str:
    """构建上传状态 HTML 行（与 rice_reg 同款彩色状态）。"""
    ok = lambda s: f"<span style='color:#15803d;'>{s}</span>"
    err = lambda s: f"<span style='color:#dc2626;'>{s}</span>"
    info = lambda s: f"<span style='color:#6b7280;'>{s}</span>"
    if key == "idle":
        return info(t["upload_status_idle"])
    if key == "fasta_ok":
        return ok(t["upload_status_fasta_ok"].format(gid=gid, n=n))
    if key == "err_fasta":
        return err(t["upload_status_err_fasta"].format(msg=msg))
    return ""


def _backend_genome_ids() -> list[str]:
    """后端注册的基因组 id 列表（内置 + 已上传）。"""
    try:
        data = _backend_get("/genomes")
        return [g for g in (data.get("genomes") or [])]
    except Exception:
        return list(config.get_genome_configs().keys())


def _genome_options_all(lang: str | None = None) -> list:
    """内置 + 已上传（后端 /genomes 动态）基因组 + 自定义基因组哨兵。"""
    t = _t(lang)
    choices = [(g, g) for g in _backend_genome_ids()]
    choices.append((t["custom_genome_option"], CUSTOM_GENOME_VALUE))
    return choices


def _default_genome_id() -> Optional[str]:
    ids = _backend_genome_ids()
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
#  Plotly figure 构建（JSON 序列化即可，无需 plotly python 包）
# ---------------------------------------------------------------------------
_GROUP_LABEL = {"Ind": "Ind", "Jap": "Jap", "uncertain": "uncertain"}


def _fmt_bp(v) -> str:
    """bp -> 千分位字符串；None 返回 '-'。"""
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "-"


def _fmt_score(v):
    """topk 均值 -> 保留 4 位小数；None 返回 '-'。"""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "-"


def _region_hover_text(r: dict, group: str) -> str:
    """区域 hover 文本（恒英文）：类别 + 染色体 + 区间（bp 与 Mb）+ 统计。"""
    chrom = r.get("chromosome") or r.get("chr_id") or "-"
    start = r.get("start")
    end = r.get("end")
    if start is None or end is None:
        span_mb = "-"
    else:
        span_mb = f"{float(end - start) / 1e6:.3f}"
    lines = [
        f"<b>{_GROUP_LABEL.get(group, group)}</b>",
        f"Chromosome: {chrom}",
        f"Region: {_fmt_bp(start)} – {_fmt_bp(end)} ({span_mb} Mb)",
    ]
    if r.get("n_windows") is not None:
        lines.append(f"Windows: {int(r['n_windows'])}")
    j = r.get("topk_mean_jap")
    i = r.get("topk_mean_ind")
    if j is not None or i is not None:
        lines.append(f"Top-k mean: Jap={_fmt_score(j)}, Ind={_fmt_score(i)}")
    return "<br>".join(lines)


def _build_genome_figures(payload: dict) -> dict:
    """全基因组三 track 图：所有染色体车道，每条染色体 3 个紧密 track。

    用户 2026-11 确认：恢复为所有染色体都画，每个染色体有 3 个 track：
    - x 轴 = 全基因组范围（Mb；各染色体共享同一坐标轴）
    - y 轴 = 染色体车道（Chr12 顶 → Chr01 底），每条染色体 = 3 个紧密 track
      （Jap 蓝 / Ind 橘 / uncertain 绿），**块内 track 无行距**，块间留少量间距
    - 每条 track：未推理区 gray 底色 + 该分组 region 色带
    - hover 显示类别、区间（bp + Mb）、n_windows、top-k 均值
    """
    chromosomes = payload.get("chromosomes", [])
    chrom_lengths = payload.get("chromosome_lengths", {})
    regions = payload.get("regions", {})
    params = payload.get("params", {})

    if not chromosomes:
        return {"data": [], "layout": {"title": {"text": "No data"}}}

    # 三 track 顺序（每条染色体块内，顶 → 底）：uncertain / Ind / Jap
    # 视觉上 Jap 蓝在上、Ind 橘在中、uncertain 绿在下，与图例一致
    track_order = ("Jap", "Ind", "uncertain")
    # 块内 track 中心相对偏移：Jap=+1/3, Ind=0, uncertain=-1/3
    # （块高 1.0 三等分，3 track 正好填满块高，间距=0）
    track_offsets = {"Jap": 1.0 / 3.0, "Ind": 0.0, "uncertain": -1.0 / 3.0}

    # 车道 y 位置：Chr12 顶 → Chr01 底（倒序）。
    # 染色体中心间距 = 1.5（原 3.0 的一半，用户要求间距减半）。
    ordered = list(reversed(chromosomes))
    block_stride = 1.5
    y_block = {
        chrom: block_stride * (len(chromosomes) - i) for i, chrom in enumerate(ordered)
    }
    # 每条 track 的中心 y
    y_track = {
        (chrom, g): y_block[chrom] + track_offsets[g]
        for chrom in chromosomes for g in track_order
    }
    y_ticks = [y_block[c] for c in reversed(chromosomes)]

    max_len_mb = max((chrom_lengths.get(c, 0) for c in chromosomes), default=0) / 1e6

    # y 轴范围：底部 = chromosomes[0]（Chr01, y 最小），顶部 = chromosomes[-1]
    # （Chr12, y 最大）；range=[min, max] 自下而上。
    y_min = y_block[chromosomes[0]] - 1.0 / 3.0 - 0.5
    y_max = y_block[chromosomes[-1]] + 1.0 / 3.0 + 0.5
    y_span = y_max - y_min

    # 图高（每染色体约 56px，紧凑）；线宽动态计算使 3 track 正好填满块高（无间距）
    layout_height = max(420, len(chromosomes) * 56)
    margin_t, margin_b = 20, 40
    plot_h = layout_height - margin_t - margin_b
    px_per_unit = plot_h / y_span if y_span > 0 else 10.0
    line_w = max(4.0, px_per_unit / 3.0)  # 每个 track 占块高的 1/3 → 紧密相接

    traces = []
    # 背景空白（撑满全宽并保证 x 范围）
    traces.append({
        "type": "scatter",
        "mode": "lines",
        "x": [0, max_len_mb],
        "y": [0.5, 0.5],
        "line": {"color": "rgba(0,0,0,0)"},
        "showlegend": False,
        "hoverinfo": "skip",
    })

    # 每条染色体：3 个紧密 track（无行距），先画 gray 未推理底色再画 region 色带
    for chrom in chromosomes:
        block_y = y_block[chrom]
        length_mb = chrom_lengths.get(chrom, 0) / 1e6
        for group in track_order:
            y = y_track[(chrom, group)]
            color = GROUP_COLORS[group]
            # gray 未推理底色（整条染色体范围）
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": [0, length_mb],
                "y": [y, y],
                "line": {"color": UNINFERENCED_COLOR, "width": line_w},
                "name": group,
                "legendgroup": group,
                "showlegend": False,
                "hoverinfo": "skip",
            })
            # 该分组 region 色带（绝对坐标）
            recs = regions.get(group, [])
            xs: list[float] = []
            ys: list[float] = []
            hover_texts: list[Optional[str]] = []
            for r in recs:
                chrom_r = r.get("chromosome") or r.get("chr_id")
                if chrom_r != chrom:
                    continue
                r0 = max(int(r.get("start", 0)), 0)
                r1 = min(int(r.get("end", length_mb * 1e6)), int(length_mb * 1e6))
                if r1 < r0:
                    continue
                r0_mb = r0 / 1e6
                r1_mb = r1 / 1e6
                tx = _region_hover_text(r, group)
                n = max(2, min(500, int((r1_mb - r0_mb) / 0.2) + 2))
                part_x = [r0_mb + (r1_mb - r0_mb) * i / (n - 1) for i in range(n)]
                xs.extend(part_x)
                xs.append(None)
                ys.extend([y] * n)
                ys.append(None)
                hover_texts.extend([tx] * n)
                hover_texts.append(None)
            if xs:
                traces.append({
                    "type": "scatter",
                    "mode": "lines",
                    "x": xs,
                    "y": ys,
                    "line": {"color": color, "width": line_w},
                    "name": group,
                    "legendgroup": group,
                    "showlegend": False,
                    "hoverinfo": "text",
                    "text": hover_texts,
                    "hovertemplate": "%{text}<extra></extra>",
                    "opacity": 0.95,
                })

    # 图例：每个分组一条（零长占位 trace，位于 Chr01 块）
    for group in track_order:
        color = GROUP_COLORS[group]
        y0 = y_track[(chromosomes[0], group)]
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": [0, 0],
            "y": [y0, y0],
            "line": {"color": color, "width": line_w},
            "name": group,
            "legendgroup": group,
            "showlegend": True,
            "hoverinfo": "skip",
            "opacity": 0.95,
        })
    # 未推理 gray 图例项
    y0 = y_block[chromosomes[0]] + 1.0 / 3.0
    traces.append({
        "type": "scatter",
        "mode": "lines",
        "x": [0, 0],
        "y": [y0, y0],
        "line": {"color": UNINFERENCED_COLOR, "width": line_w},
        "name": "uninferenced",
        "legendgroup": "uninferenced",
        "showlegend": True,
        "hoverinfo": "skip",
    })

    layout = {
        "xaxis": {
            "title": "Position (Mb)",
            "zeroline": False,
            "range": [0, max_len_mb],
        },
        "yaxis": {
            "tickvals": y_ticks,
            "ticktext": [str(c) for c in reversed(chromosomes)],
            "range": [y_min, y_max],
            "zeroline": False,
        },
        "height": layout_height,
        "margin": {"l": 70, "r": 24, "t": margin_t, "b": margin_b},
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "hovermode": "closest",
        "config": {"responsive": True, "displaylogo": False},
    }

    return {"data": traces, "layout": layout}


def _payload_to_html(payload: dict, lang: str | None = None, empty_msg: str = "") -> str:
    plotly_url = config.get_plotly_js_local_url() or "https://cdn.plot.ly/plotly-2.35.2.min.js"

    # 全基因组三 track 图（所有染色体车道，每染色体 3 个紧密 track）
    figs = _build_genome_figures(payload)
    figure = {"data": figs["data"], "layout": figs["layout"],
              "config": {"responsive": True, "displaylogo": False}}
    js_payload = "const FIGURE = " + json.dumps(figure) + ";\n"
    inner_html = _HTML_TEMPLATE.format(
        plotly_url=plotly_url,
        js_payload=js_payload,
    )
    # srcdoc 模式（rice_mut 同款）：gr.HTML 直接注入的 <script> 不会执行，
    # 必须包装成 iframe srcdoc 让浏览器独立解析，script 才能顺序加载执行。
    escaped = (
        inner_html.replace("&", "&amp;").replace('"', "&quot;")
        .replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")
    )
    height = figs["layout"].get("height", 540) + 30
    banner = ""
    if empty_msg:
        # 空结果提示条（骨架图上方，橙色警示）
        escaped_msg = empty_msg.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
        banner = (
            f'<div style="background:#fff7ed;border:1px solid #fdba74;color:#9a3412;'
            f'padding:8px 12px;border-radius:8px;margin-bottom:6px;font-size:13px;">'
            f'⚠️ {escaped_msg}</div>'
        )
    return (
        banner
        + '<iframe srcdoc="' + escaped
        + f'" style="width:100%;height:{height}px;border:none;border-radius:8px;'
        + 'background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.1);"></iframe>'
    )


# ---------------------------------------------------------------------------
#  Gradio UI（与 rice_reg 风格一致：header bar + 卡片 + 全宽可视化在下）
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(
        title="OGR-Introgression",
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.gray,
        ),
        css=_CSS,
    ) as demo:

        cur_lang = gr.State(DEFAULT_LANG)

        # ── Header bar：标题 + EN / 中文 pill 切换 ──
        with gr.Row(equal_height=True, elem_classes=["app-header-bar"]):
            header_md = gr.Markdown(
                f"# {I18N[DEFAULT_LANG]['title']}",
                elem_classes=["app-header-title"],
            )
            with gr.Column(scale=0, min_width=220):
                with gr.Row(elem_classes=["app-header-actions"]):
                    # 初始高亮与 DEFAULT_LANG 对齐，避免界面中文却高亮 English 的启动 bug
                    _en_on_start = DEFAULT_LANG == "en"
                    btn_lang_en = gr.Button(
                        "English",
                        variant="primary" if _en_on_start else "secondary",
                        elem_classes=["lang-toggle-btn"],
                    )
                    btn_lang_zh = gr.Button(
                        "中文",
                        variant="primary" if not _en_on_start else "secondary",
                        elem_classes=["lang-toggle-btn"],
                    )

        intro_md = gr.Markdown(
            f"*{I18N[DEFAULT_LANG]['subtitle']}*",
            elem_classes=["intro-copy"],
        )

        # ── 参数卡片（单列，在上）──
        with gr.Column(elem_classes=["card-panel"]):
            # Genome 行 — 选「自定义基因组」时拆分为 Genome | Upload FASTA
            with gr.Row(equal_height=True, elem_classes=["layout-row-2sub"]):
                with gr.Column(scale=3, min_width=0, elem_id="genome-col") as genome_col:
                    genome_dd = gr.Dropdown(
                        choices=_genome_options_all(),
                        value=_default_genome_id(),
                        label=I18N[DEFAULT_LANG]["genome"],
                        interactive=True,
                    )
                with gr.Column(visible=False, scale=1, min_width=0, elem_id="fasta-col") as fasta_col:
                    genome_fasta_upload = gr.Textbox(
                        value="",
                        label=I18N[DEFAULT_LANG]["upload_genome_fasta"],
                        placeholder=I18N[DEFAULT_LANG]["upload_fasta_ph"],
                        interactive=True,
                        elem_id="genome-upload",
                    )
            # 上传状态（仅自定义基因组时显示）
            upload_status_md = gr.Markdown(
                value=_fmt_upload_status(I18N[DEFAULT_LANG], "idle"),
                visible=False,
            )
            # 染色体 | Start | End
            with gr.Row(equal_height=True):
                chromosome_dd = gr.Dropdown(
                    choices=[], value=None,
                    label=I18N[DEFAULT_LANG]["chromosome"], interactive=True, scale=1,
                )
                start = gr.Textbox(
                    label=I18N[DEFAULT_LANG]["start"], value="",
                    placeholder=I18N[DEFAULT_LANG]["start_placeholder"], scale=1,
                )
                end = gr.Textbox(
                    label=I18N[DEFAULT_LANG]["end"], value="",
                    placeholder=I18N[DEFAULT_LANG]["end_placeholder"], scale=1,
                )
            # API Key（可选，鉴权请求头；空 = 后端决定是否放行）
            api_key_input = gr.Textbox(
                value="", label=I18N[DEFAULT_LANG]["api_key"],
                placeholder=I18N[DEFAULT_LANG]["api_key_ph"],
                interactive=True, show_label=True,
            )
            # 预测按钮 + 进度
            predict_btn = gr.Button(
                I18N[DEFAULT_LANG]["predict"], variant="primary", elem_id="predict-btn",
            )
            status_md = gr.Markdown("")

        # ── 可视化全宽（在下）──
        out_html = gr.HTML(
            value=f"<p style='color:#6B7280;'>{I18N[DEFAULT_LANG]['placeholder']}</p>"
        )

        # ── 事件 ──
        def _parse_pos(s):
            s = (s or "").strip().replace(",", "")
            return int(s) if s else None

        def _on_genome_change(genome, lang):
            """Genome 变化：自定义哨兵拆列显示上传区；内置则恢复全宽并刷新染色体。"""
            t = _t(lang)
            if genome == CUSTOM_GENOME_VALUE:
                return (
                    gr.update(scale=1),
                    gr.update(visible=True, scale=1),
                    gr.update(value=None),
                    gr.update(visible=True, value=_fmt_upload_status(t, "idle")),
                    gr.update(choices=[], value=None, interactive=False),
                )
            try:
                data = _backend_get(f"/genomes/{genome}/chromosomes")
                chrs = data.get("chromosomes", [])
            except Exception:
                chrs = []
            return (
                gr.update(scale=3),
                gr.update(visible=False),
                gr.update(value=None),
                gr.update(visible=False, value=""),
                gr.update(choices=chrs, value=chrs[0] if chrs else None, interactive=True),
            )

        genome_dd.change(
            _on_genome_change,
            inputs=[genome_dd, cur_lang],
            outputs=[genome_col, fasta_col, genome_fasta_upload, upload_status_md, chromosome_dd],
        )
        # 页面首次加载时也加载默认基因组的染色体（change 事件在值未变化时不触发）
        demo.load(
            _on_genome_change,
            inputs=[genome_dd, cur_lang],
            outputs=[genome_col, fasta_col, genome_fasta_upload, upload_status_md, chromosome_dd],
        )

        def _on_upload_genome(fasta_path, lang, api_key=""):
            """注册自定义基因组：输入本机 FASTA 路径（轻量模式，零拷贝懒注册）。

            与 OGR-Prediction 一致：前端/后端同机，用户粘贴绝对路径，
            后端经 /uploadFasta（path 字段）或 DCS RegisterGenome 子路径原地注册。
            """
            t = _t(lang)
            path_str = (fasta_path or "").strip()
            if not path_str or not os.path.exists(path_str):
                return gr.update(), gr.update(), gr.update()
            try:
                out = _backend_register_genome(path_str, api_key=api_key)
            except Exception as e:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(visible=True, value=_fmt_upload_status(t, "err_fasta", msg=str(e))),
                )
            gid = out["genome"]
            chrs = out.get("chromosomes") or []
            return (
                gr.update(choices=_genome_options_all(lang), value=gid),
                gr.update(choices=chrs, value=chrs[0] if chrs else None, interactive=True),
                gr.update(visible=True, value=_fmt_upload_status(t, "fasta_ok", gid=gid, n=len(chrs))),
            )

        genome_fasta_upload.change(
            _on_upload_genome,
            inputs=[genome_fasta_upload, cur_lang, api_key_input],
            outputs=[genome_dd, chromosome_dd, upload_status_md],
        )

        def _predict(genome, chromosome, start, end, lang, api_key=""):
            t = _t(lang)
            if not genome or genome == CUSTOM_GENOME_VALUE:
                yield "", t["err_no_genome"]
                return
            if not chromosome:
                yield "", t["err_no_chromosome"]
                return
            if _parse_pos(start) is None:
                yield "", t["err_no_start"]
                return
            if not (api_key or "").strip():
                yield "", t["err_need_api_key"]
                return
            # 与纯 API（curl）同一 DCS 子路径 / 同一方言：camelCase + 1-based。
            # save_intermediate=True：网页端需要磁盘缓存/重跑语义（curl 省略
            # 该键 → 后端缺省不保存中间文件、跳过磁盘缓存）。
            payload = {
                "Genome": genome,
                "Chromosome": chromosome,
                "Start": _parse_pos(start),
                "End": _parse_pos(end),
                "save_intermediate": True,
            }
            task: dict = {"done": False, "html": "", "msg": ""}

            def _run():
                try:
                    data = _backend_post(
                        "/api/aigress/openai/OGR-Introgression/Prediction",
                        payload,
                        api_key=api_key,
                    )
                    # 网页端与 curl 拿同一份返回值：信封 result 即绘图/统计数据
                    result = data.get("result") or {}
                    empty_msg = result.get("message", "") if not result.get("windows") else ""
                    task["html"] = _payload_to_html(result, lang=lang, empty_msg=empty_msg)
                    elapsed = result.get("elapsed_seconds", "?")
                    n_win = len(result.get("windows", []))
                    n_seg = len(result.get("segments", []))
                    cached = "cached · " if result.get("cached") else ""
                    if empty_msg:
                        task["msg"] = t["empty_result"].format(msg=empty_msg)
                    else:
                        task["msg"] = t["done"].format(cached=cached, elapsed=elapsed,
                                                       n_win=n_win, n_seg=n_seg)
                except Exception as e:
                    task["msg"] = t["predict_failed"].format(msg=e)
                finally:
                    task["done"] = True

            threading.Thread(target=_run, daemon=True).start()

            # 轮询进度（每 2s），直到预测完成或异常
            last_status = ""
            while not task["done"]:
                try:
                    prog = _backend_get("/progress")
                    st = prog.get("status", "idle")
                    if st == "running":
                        pct = prog.get("percent", 0.0) or 0.0
                        done = prog.get("done_batches", 0)
                        total = prog.get("total_batches", 0)
                        elapsed = prog.get("elapsed_seconds", 0.0) or 0.0
                        chrom = prog.get("chromosome", chromosome)
                        txt = t["progress_run"].format(pct=pct, done=done, total=total,
                                                       elapsed=elapsed, chrom=chrom)
                        if txt != last_status:
                            last_status = txt
                            yield "", txt
                    elif st in ("done", "error", "idle") and last_status:
                        # 任务结束但预测尚未返回：显示简短的收尾提示
                        pass
                except Exception:
                    pass
                time.sleep(2.0)

            yield task["html"], task["msg"]
        predict_btn.click(
            _predict,
            inputs=[genome_dd, chromosome_dd, start, end, cur_lang, api_key_input],
            outputs=[out_html, status_md],
        )

        # ── 语言切换 ──
        def _on_lang_toggle(lang, current_genome=None) -> tuple:
            lang = lang if lang in I18N else DEFAULT_LANG
            t = I18N[lang]
            choices = _genome_options_all(lang)
            choice_ids = [v for _, v in choices]
            genome_val = (
                current_genome if current_genome in choice_ids else _default_genome_id()
            )
            en = lang == "en"
            return (
                gr.update(value=f"# {t['title']}"),
                gr.update(value=f"*{t['subtitle']}*"),
                gr.update(choices=choices, value=genome_val, label=t["genome"]),
                gr.update(label=t["chromosome"]),
                gr.update(label=t["start"], placeholder=t["start_placeholder"]),
                gr.update(label=t["end"], placeholder=t["end_placeholder"]),
                gr.update(value=t["predict"]),
                gr.update(label=t["upload_genome_fasta"]),
                gr.update(label=t["api_key"], placeholder=t["api_key_ph"]),
                gr.update(variant="primary" if en else "secondary"),
                gr.update(variant="primary" if not en else "secondary"),
                gr.update(value=f"<p style='color:#6B7280;'>{t['placeholder']}</p>"),
                gr.update(value=""),
                gr.update(value=lang),
            )

        btn_lang_en.click(
            fn=lambda g: _on_lang_toggle("en", g),
            inputs=[genome_dd],
            outputs=[
                header_md, intro_md, genome_dd, chromosome_dd, start, end,
                predict_btn, genome_fasta_upload, api_key_input,
                btn_lang_en, btn_lang_zh,
                out_html, status_md, cur_lang,
            ],
            queue=False,
        )
        btn_lang_zh.click(
            fn=lambda g: _on_lang_toggle("zh", g),
            inputs=[genome_dd],
            outputs=[
                header_md, intro_md, genome_dd, chromosome_dd, start, end,
                predict_btn, genome_fasta_upload, api_key_input,
                btn_lang_en, btn_lang_zh,
                out_html, status_md, cur_lang,
            ],
            queue=False,
        )

    return demo


# ---------------------------------------------------------------------------
#  反向代理 + 启动
# ---------------------------------------------------------------------------
def _reverse_proxy(demo):
    """把 /backend/* 反向代理到后端（避免跨域）。"""
    from fastapi import Request
    from fastapi.responses import Response

    _SKIP_REQ = {"host", "content-length", "connection"}
    _SKIP_RESP = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    # 复用单个 AsyncClient（keep-alive 连接池）而不是每请求新建 —— IGV 会并发
    # 加载 bigWig / GFF / FASTA 多个资源并带 Range 头，每请求新建连接容易瞬时
    # 建连冲高 / 偶发 ConnectError，导致轨道（如 Genes 注释）加载失败不显示。
    _BACKEND_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))

    @demo.app.api_route("/backend/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def proxy_backend(path: str, request: Request):
        backend_path = f"/{path}" if path else "/"
        url = f"{config.get_backend_base_url().rstrip('/')}{backend_path}"
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQ
        }
        r = await _BACKEND_CLIENT.request(
            request.method,
            url,
            params=request.query_params,
            headers=headers,
            content=body,
        )
        resp_headers = {
            k: v for k, v in r.headers.items() if k.lower() not in _SKIP_RESP
        }
        return Response(content=r.content, status_code=r.status_code, headers=resp_headers)
    return demo


def main():
    demo = build_ui()
    demo.queue().launch(
        server_name=config.get_frontend_host(),
        server_port=config.get_frontend_port(),
        prevent_thread_lock=True,
        show_error=True,
    )
    # 注意：必须在 launch() 之后、block_thread() 之前注册路由——
    # Gradio 的 launch() 内部会用真实 FastAPI app 替换 demo.app，
    # launch 前注册的 api_route 会全部丢失。
    demo = _reverse_proxy(demo)
    demo.block_thread()


if __name__ == "__main__":
    main()