"""Rice-Mutation Frontend — Gradio UI for DNA → multi-omics expression prediction.

Features:
- Reference prediction: genome / chromosome / window -> predicted tracks (IGV)
- Mutation comparison: paste or upload a mutant sequence -> ref vs mut dual tracks
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
from pathlib import Path
from typing import Optional
from urllib import request, error

import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_file(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


# Load .env before importing config
_load_env_file(os.path.join(Path(BASE_DIR).parent, ".env"))
_load_env_file(os.path.join(BASE_DIR, ".env"))

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from frontend.config import (
    GENOME_CONFIGS,
    CHROMOSOME_OPTIONS,
    DEFAULT_GENOME,
    BACKEND_API_URL,
    IGV_CDN_URL,
)
from frontend.igv_payload import (
    build_default_prediction_reference,
    set_static_base_url,
)

STATIC_DIR_ABS = os.path.join(BASE_DIR, "static")
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8000"))
ALLOWED_FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".FA", ".FASTA")

# Configure IGV static file serving via backend HTTP
set_static_base_url(f"{BACKEND_API_URL}/static-files")

# Default window
DEFAULT_WINDOW = 32768

# Unified font stack — consistent Latin/CJK rendering across the page and iframes.
FONT_STACK = "Inter, 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"

# ---------------------------------------------------------------------------
#  i18n — UI copy.  A small EN/中文 toggle switch at the top switches language.
# ---------------------------------------------------------------------------
DEFAULT_LANG = "en"

I18N = {
    "en": {
        "title": "🧬 OGR-Mutation: DNA → Expression Prediction",
        "subtitle": "Predict RNA-seq expression directly from a DNA sequence and visualize it as IGV tracks.",
        "required": "### Required parameters",
        "genome": "Genome",
        "chromosome": "Chromosome (e.g. chr09)",
        "start": "Start (prediction window: 32 kb, e.g. 20716773)",
        "custom_genome_option": "📤 Custom Genome",
        "custom_genome_sec": "### 📤 Custom Genome",
        "upload_genome_fasta": "Upload Genome FASTA (.fa/.fasta/.fna)",
        "upload_gff": "Upload Annotation GFF (.gff/.gff3/.gtf)",
        "upload_status_idle": "Select “Custom Genome” and upload a FASTA to register a new genome.",
        "upload_status_fasta_ok": "Genome “{gid}” registered ({n} chromosomes). You can now attach a GFF.",
        "upload_status_gff_ok": "GFF attached to “{gid}”. A Genes track will appear in IGV.",
        "upload_status_gff_wait": "GFF ready — will attach automatically once the FASTA upload finishes.",
        "upload_status_both_ok": "Genome “{gid}” registered ({n} chromosomes). GFF auto-attached — a Genes track will appear in IGV.",
        "upload_err_gff_auto": "GFF auto-attach failed: {msg}. Please re-upload the GFF.",
        "upload_err_fasta": "FASTA upload failed: {msg}",
        "upload_err_gff": "GFF upload failed: {msg}",
        "upload_need_fasta": "Upload a FASTA first, then a GFF can be attached.",
        "predict": "🚀 Predict",
        "snv_section": "### Single-nucleotide variant (optional)",
        "snv_hint": "Fill in both fields to run an SNV comparison; leave them empty for a reference-only prediction.",
        "snv_pos": "SNV position (e.g. 20731844)",
        "snv_base": "SNV base (replace with, e.g. T)",
        "placeholder": "Select inputs and click Predict to see results.",
        "err_no_chromosome": "Please select a chromosome.",
        "err_no_snv_pos": "Please enter an SNV position (1-based bp, inside the window).",
        "err_bad_snv_base": "SNV base must be A/C/G/T/N.",
        "err_bad_snv_pair": "To run an SNV comparison please fill in both SNV position and SNV base.",
        "err_predict": "Prediction failed",
        "err_snv_predict": "SNV prediction failed",
        "snv_info": "SNV: {ref}@{pos:,}→{alt} (window {ws:,}–{we:,} bp). Drag/zoom the IGV below to compare expression across regions.",
        "bar_placeholder": "Run a prediction, then the mean expression of each track in the current IGV window (result1/result2) appears here.",
        "bar_err_http": "Bar plot data unavailable (HTTP {code}{detail}). Please re-run the prediction.",
        "bar_err_load": "Bar plot failed to load: {msg} Please re-run the prediction.",
        "bar_title": "Mean expression in window (result1 vs result2)",
        "bar_y": "Mean expression",
        "bar_r1": "result1",
        "bar_r2": "result2",
        "upload_click": "📁 Click to upload",
        "lang_label": "en / 中文",
        "lang_en": "en",
        "lang_zh": "中文",
    },
    "zh": {
        "title": "🧬 OGR-Mutation：DNA → 表达预测",
        "subtitle": "从 DNA 序列直接预测 RNA-seq 表达量，并以 IGV 轨道可视化。",
        "required": "### 必要参数",
        "genome": "基因组",
        "chromosome": "染色体（如 chr09）",
        "start": "起始位置（预测窗口：32 kb，如 20716773）",
        "custom_genome_option": "📤 自定义基因组",
        "custom_genome_sec": "### 📤 自定义基因组",
        "upload_genome_fasta": "上传基因组 FASTA (.fa/.fasta/.fna)",
        "upload_gff": "上传注释 GFF (.gff/.gff3/.gtf)",
        "upload_status_idle": "选择「自定义基因组」并上传 FASTA 以注册新基因组。",
        "upload_status_fasta_ok": "基因组「{gid}」已注册（{n} 条染色体）。现在可附加 GFF。",
        "upload_status_gff_ok": "GFF 已附加到「{gid}」。IGV 将显示 Genes 轨道。",
        "upload_status_gff_wait": "GFF 已就绪，等待 FASTA 上传完成后自动附加。",
        "upload_status_both_ok": "基因组「{gid}」已注册（{n} 条染色体）。GFF 已自动附加，IGV 将显示 Genes 轨道。",
        "upload_err_gff_auto": "GFF 自动附加失败：{msg}。请重新上传 GFF。",
        "upload_err_fasta": "FASTA 上传失败：{msg}",
        "upload_err_gff": "GFF 上传失败：{msg}",
        "upload_need_fasta": "请先上传 FASTA，之后才能附加 GFF。",
        "predict": "🚀 预测",
        "snv_section": "### 单核苷酸变异（可选）",
        "snv_hint": "填写这两个字段将运行 SNV 对比；留空则仅做参考预测。",
        "snv_pos": "SNV 位置（如 20731844）",
        "snv_base": "SNV 碱基（替换为，如 T）",
        "placeholder": "选择输入后点击「预测」查看结果。",
        "err_no_chromosome": "请选择染色体。",
        "err_no_snv_pos": "请输入 SNV 位置（1-based bp，须在窗口内）。",
        "err_bad_snv_base": "SNV 碱基必须为 A/C/G/T/N。",
        "err_bad_snv_pair": "要运行 SNV 对比，请同时填写 SNV 位置与 SNV 碱基。",
        "err_predict": "预测失败",
        "err_snv_predict": "SNV 预测失败",
        "snv_info": "SNV：{ref}@{pos:,}→{alt}（窗口 {ws:,}–{we:,} bp）。拖动/缩放下方 IGV 查看不同区域的表达量对比。",
        "bar_placeholder": "运行预测后，这里显示当前 IGV 窗口内各轨道平均表达量（result1/result2）。",
        "bar_err_http": "柱状图数据不可用（HTTP {code}{detail}）。请重新运行预测。",
        "bar_err_load": "柱状图加载失败：{msg}。请重新运行预测。",
        "bar_title": "窗口内平均表达量（result1 vs result2）",
        "bar_y": "平均表达量",
        "bar_r1": "result1（ref）",
        "bar_r2": "result2（mut）",
        "upload_click": "📁 点击上传",
        "lang_label": "en / 中文",
        "lang_en": "en",
        "lang_zh": "中文",
    },
}


# ---------------------------------------------------------------------------
#  Intro copy + track legend (Genos-Mutation style)
# ---------------------------------------------------------------------------
INTRO_MARKDOWN = {
    "en": (
        "Optionally run a single-nucleotide variant (SNV) comparison to see how one base change "
        "shifts expression (reference vs. mutated), or upload a custom genome FASTA + annotation "
        "GFF for other assemblies."
    ),
    "zh": (
        "可选单碱基突变（SNV）对比，查看单个碱基改变对表达的影响（参考 vs 突变）；"
        "也可上传自定义基因组 FASTA + 注释 GFF，用于其它组装。"
    ),
}

TRACK_LEGEND_MD = {
    "en": (
        "<div style='margin-top:12px; color:#666; font-size:0.95rem;'>"
        "<strong>Gray</strong>: reference expression level<br>"
        "<strong>Other colors</strong>: model-predicted expression levels after applying the mutation"
        "</div>"
    ),
    "zh": (
        "<div style='margin-top:12px; color:#666; font-size:0.95rem;'>"
        "<strong>灰色</strong>：参考表达量<br>"
        "<strong>其他颜色</strong>：施加突变后模型预测的表达量"
        "</div>"
    ),
}


def _get_intro_markdown(lang: str) -> str:
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    return f"*{t['subtitle']}*<br>" + INTRO_MARKDOWN[lang]


def _get_track_legend_md(lang: str) -> str:
    return TRACK_LEGEND_MD.get(lang, TRACK_LEGEND_MD[DEFAULT_LANG])


def _igv_panel_html(content: str) -> str:
    """Wrap an IGV iframe (or placeholder) in a white rounded card like
    Genos-Mutation's prediction panels."""
    return f"""
    <div style="margin-top:4px;">
        <div style="font-weight:600; margin-bottom:4px; color:#111827;">
            🧬 RNA-seq Prediction
        </div>
        <div style="background:#fff; border:1px solid #e5e7eb; border-radius:12px; overflow:hidden;">
            {content}
        </div>
    </div>"""


def _igv_placeholder_html(lang: str) -> str:
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    return _igv_panel_html(
        f"<div style='padding:28px 16px; color:#9CA3AF; text-align:center; font-size:0.95rem;'>{t['placeholder']}</div>"
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
# Sentinel "Custom Genome" entry in the genome dropdown (not a real genome id).
CUSTOM_GENOME_VALUE = "__custom__"


def _custom_genome_label(lang: str = DEFAULT_LANG) -> str:
    return I18N.get(lang, I18N[DEFAULT_LANG])["custom_genome_option"]


def _genome_options() -> list:
    return list(GENOME_CONFIGS.keys())


def _call_backend_api(endpoint: str, data: dict) -> dict:
    """Call the backend FastAPI endpoint and return the JSON response."""
    url = f"{BACKEND_API_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend error ({e.code}): {body}")
    except error.URLError as e:
        raise RuntimeError(f"Cannot reach backend at {url}: {e}")


def _upload_genome_to_backend(file_path: str) -> dict:
    """Upload a genome FASTA to the backend; returns the full JSON response
    (genome id, chromosomes, file path)."""
    import http.client

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    url = f"{BACKEND_API_URL}/uploadFasta"
    from urllib.parse import urlparse as _urlparse

    parsed = _urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    conn.request("POST", parsed.path, body=body, headers=headers)
    resp = conn.getresponse()
    resp_body = resp.read().decode("utf-8")
    if resp.status != 200:
        raise RuntimeError(f"Upload failed ({resp.status}): {resp_body}")
    result = json.loads(resp_body)
    return result


def _upload_gff_to_backend(file_path: str, genome_id: str) -> dict:
    """Attach an annotation GFF to an uploaded custom genome via /uploadGff."""
    import http.client

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="genome"\r\n\r\n'
        f"{genome_id}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    url = f"{BACKEND_API_URL}/uploadGff"
    from urllib.parse import urlparse as _urlparse

    parsed = _urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    conn.request("POST", parsed.path, body=body, headers=headers)
    resp = conn.getresponse()
    resp_body = resp.read().decode("utf-8")
    if resp.status != 200:
        raise RuntimeError(f"GFF upload failed ({resp.status}): {resp_body}")
    return json.loads(resp_body)


def _uploaded_file_path(file) -> Optional[str]:
    """Extract a stable local path from a Gradio-uploaded file object."""
    if file is None:
        return None
    if hasattr(file, "path"):
        return file.path
    if isinstance(file, dict):
        return file.get("name") or file.get("path")
    if isinstance(file, str):
        return file
    return getattr(file, "name", None)


def _fmt_upload_status(t: dict, key: str, gid: str = "", n: int = 0, msg: str = "") -> str:
    """Build a coloured HTML status line for the custom-genome upload area."""
    ok = lambda s: f"<span style='color:#15803d;'>{s}</span>"
    err = lambda s: f"<span style='color:#dc2626;'>{s}</span>"
    info = lambda s: f"<span style='color:#6b7280;'>{s}</span>"
    if key == "idle":
        return info(t["upload_status_idle"])
    if key == "fasta_ok":
        return ok(t["upload_status_fasta_ok"].format(gid=gid, n=n))
    if key == "gff_ok":
        return ok(t["upload_status_gff_ok"].format(gid=gid))
    if key == "gff_wait":
        return info(t["upload_status_gff_wait"])
    if key == "both_ok":
        return ok(t["upload_status_both_ok"].format(gid=gid, n=n))
    if key == "fasta_ok_gff_err":
        return (
            ok(t["upload_status_fasta_ok"].format(gid=gid, n=n))
            + "<br>"
            + err(t["upload_err_gff_auto"].format(msg=msg))
        )
    if key == "err_fasta":
        return err(t["upload_err_fasta"].format(msg=msg))
    if key == "err_gff":
        return err(t["upload_err_gff"].format(msg=msg))
    if key == "need_fasta":
        return err(t["upload_need_fasta"])
    return ""


# ---------------------------------------------------------------------------
#  IGV HTML snippet
# ---------------------------------------------------------------------------
def _igv_html(
    genome: str,
    locus: str,
    igv_payload: Optional[dict] = None,
    prediction_id: str = "",
    kind: str = "ref",
    win_start: int = 0,
    win_len: int = 0,
) -> str:
    """Generate an HTML snippet that renders IGV.js inside an iframe srcdoc.

    Also bridges the IGV viewport to the parent page's bar-plot renderer so
    the bar chart follows the currently visible region (skips redraw when the
    viewport does not overlap the prediction window).
    """
    # Prefer the backend-provided reference (supports uploaded custom genomes);
    # fall back to the built-in reference builder for local configs.
    ref = None
    if igv_payload and igv_payload.get("reference"):
        ref = igv_payload["reference"]
    else:
        ref = build_default_prediction_reference(genome, GENOME_CONFIGS)
    if ref is None:
        return "<p style='color:red;'>Unknown genome. Check .env configuration.</p>"

    ref_json = json.dumps(ref)
    tracks_json = json.dumps(igv_payload.get("tracks", [])) if igv_payload else "[]"
    locus_str = igv_payload.get("locus", locus) if igv_payload else locus

    pred_id_json = json.dumps(prediction_id or "")
    win_start_int = int(win_start or 0)
    win_end_int = win_start_int + int(win_len or 0)

    igv_js_url = f"{BACKEND_API_URL}/static-files{STATIC_DIR_ABS}/igv.min.js"

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="{igv_js_url}"></script>
  <style>
    body {{ margin: 0; padding: 0; font-family: {FONT_STACK}; }}
    #igv-container {{ width: 100%; height: 520px; }}
  </style>
</head>
<body>
  <div id="igv-container"></div>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var ref = {ref_json};
        var tracks = {tracks_json};
        var locus = "{locus_str}";

        // Export the whole IGV view as PNG. IGV natively renders the view to
        // SVG (browser.toSVG) - we rasterize that SVG onto a canvas for PNG.
        function saveViewPNG(browser) {{
            try {{
                var svgStr = browser.toSVG();
                var blob = new Blob([svgStr], {{type: "image/svg+xml;charset=utf-8"}});
                var url = URL.createObjectURL(blob);
                var img = new Image();
                img.onload = function() {{
                    var canvas = document.createElement("canvas");
                    canvas.width = img.width;
                    canvas.height = img.height;
                    var ctx = canvas.getContext("2d");
                    ctx.fillStyle = "white";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                    URL.revokeObjectURL(url);
                    var a = document.createElement("a");
                    a.download = "igv_view.png";
                    a.href = canvas.toDataURL("image/png");
                    a.click();
                }};
                img.onerror = function() {{ alert("PNG export failed: cannot rasterize SVG"); }};
                img.src = url;
            }} catch (err) {{
                alert("PNG export failed: " + err.message);
            }}
        }}

        igv.createBrowser(document.getElementById("igv-container"), {{
            genome: ref,
            locus: locus,
            tracks: tracks,
            showNavigation: true,
            showRuler: true,
            genomeList: [],
            // Requirement 1: native IGV toolbar button to download PNG
            customButtons: [
                {{ label: "Save PNG", callback: function(b) {{ saveViewPNG(b); }} }}
            ],
        }}).then(function(browser) {{
            window.__igvBrowser = browser;

            // Requirement 3: follow the IGV viewport and ask the parent page's
            // bar-plot renderer to redraw (region means of result1/result2).
            // The redraw is skipped when the visible region does not overlap
            // the prediction window.
            var lastBarCall = 0;
            function parseLoci(s) {{
                var m = /^([^:]+):([0-9,]+)-([0-9,]+)$/.exec(s || "");
                if (!m) return null;
                return {{
                    chr: m[1],
                    start: parseInt(m[2].replace(/,/g, ""), 10),
                    end: parseInt(m[3].replace(/,/g, ""), 10),
                }};
            }}
            function emitViewport(loc) {{
                if (!loc) return;
                // IGV currentLoci() -> "chr:start-end" (1-based inclusive);
                // convert to 0-based half-open [v0, v1)
                var v0 = (loc.start || 0) - 1;
                var v1 = loc.end || 0;
                // No overlap with the prediction window -> do not redraw
                var ov0 = Math.max(v0, {win_start_int});
                var ov1 = Math.min(v1, {win_end_int});
                if (ov1 <= ov0) return;
                var now = Date.now();
                if (now - lastBarCall < 300) return;  // throttle pan/zoom
                lastBarCall = now;
                var fn = window.parent && window.parent.__rmBarRenderer;
                if (fn) {{
                    fn({{
                        predictionId: {pred_id_json},
                        regionStart: Math.round(ov0),
                        regionEnd: Math.round(ov1),
                    }});
                }}
            }}
            function fireBarFromBrowser() {{
                try {{
                    var s = browser.currentLoci();
                    if (s === lastLociStr) return;  // no change -> skip
                    lastLociStr = s;
                    var loc = parseLoci(s);
                    if (loc) emitViewport(loc);
                }} catch (e) {{}}
            }}
            browser.on("viewport", fireBarFromBrowser);
            // Polling fallback: some navigation paths (typing a locus, programmatic
            // goto) do not fire the "viewport" event; compare the locus string and
            // only redraw when it actually changes.
            var lastLociStr = null;
            setInterval(fireBarFromBrowser, 500);
            // Render the bar plot once with the initial viewport
            fireBarFromBrowser();

            // Requirement 2: right-click menu on wig tracks to switch Bar / Line
            var views = browser.trackViews || [];
            for (var i = 0; i < views.length; i++) {{
                var t = views[i].track;
                if (!t || !t.config || t.config.type !== "wig") continue;
                (function(track) {{
                    track.contextMenuItemList = function() {{
                        var cur = track.graphType || "bar";
                        function setType(gt) {{
                            track.graphType = gt;
                            if (track.config) track.config.graphType = gt;
                            var v = track.trackView || track.browser;
                            if (v && v.repaintViews) v.repaintViews();
                        }}
                        return [
                            {{ label: (cur === "line" ? "\\u2713 Line chart" : "Line chart"),
                               click: function() {{ setType("line"); }} }},
                            {{ label: (cur === "bar" ? "\\u2713 Bar chart" : "Bar chart"),
                               click: function() {{ setType("bar"); }} }},
                        ];
                    }};
                }})(t);
            }}
        }});
    }});
  </script>
</body>
</html>"""

    escaped = (
        inner_html.replace("&", "&amp;").replace('"', "&quot;")
        .replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")
    )
    iframe = f"""<iframe srcdoc="{escaped}" style="width:100%;height:540px;border:none;border-radius:8px;"></iframe>"""
    return _igv_panel_html(iframe)


def _error_html(msg: str) -> str:
    return f"<p style='color:red;'>{msg}</p>"


def _bar_plot_html(lang: str = DEFAULT_LANG) -> str:
    """Bar-plot panel (inside an iframe so its <script> is guaranteed to run).

    Loads Plotly and exposes ``window.renderBarPlot`` to the IGV iframe via the
    parent page (``window.parent.__rmBarRenderer``).  Each bar is one
    (assay, biosample); result1 (ref) and result2 (mut) are grouped side by
    side.  Without an SNV prediction only result1 bars are shown.
    """
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    plotly_js_url = f"{BACKEND_API_URL}/static-files{STATIC_DIR_ABS}/plotly.min.js"
    api_url_json = json.dumps(BACKEND_API_URL)

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="{plotly_js_url}"></script>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; font-family: {FONT_STACK}; }}
    #rm-barplot-wrap {{ width: 100%; height: 100%; position: relative; }}
    #rm-barplot {{ width: 100%; height: 100%; }}
    #rm-placeholder {{ color: #888; font-size: 12px; text-align: center; padding-top: 150px; position: absolute; top: 0; left: 0; right: 0; }}
  </style>
</head>
<body>
  <div id="rm-barplot-wrap">
    <div id="rm-barplot"></div>
    <div id="rm-placeholder">{t['bar_placeholder']}</div>
  </div>
  <script>
    var placeholderEl = document.getElementById('rm-placeholder');
    function showMsg(txt) {{
      if (!placeholderEl) return;
      placeholderEl.style.color = '#c0392b';
      placeholderEl.style.display = 'block';
      placeholderEl.textContent = txt;
    }}
    function renderBarPlot(p) {{
      if (!p || !p.predictionId) return;
      if (!window.Plotly) {{
        setTimeout(function() {{ renderBarPlot(p); }}, 400);
        return;
      }}
      fetch({api_url_json} + "/predict/bar", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{prediction_id: p.predictionId, region_start: p.regionStart, region_end: p.regionEnd}}),
      }})
      .then(function(r) {{
        if (!r.ok) return r.json().then(function(e) {{ return {{ httpError: r.status, detail: e && e.detail }}; }});
        return r.json();
      }})
      .then(function(d) {{
        if (d && d.httpError) {{
          showMsg({json.dumps(t['bar_err_http'])}.replace("{{code}}", d.httpError).replace("{{detail}}", d.detail ? ": " + d.detail : ""));
          return;
        }}
        if (!d || !d.success || !d.overlap || !d.values.length) return;
        var hasMut = d.values.some(function(v) {{ return v.result2 != null; }});
        var labels = d.values.map(function(v) {{ return v.biosample + " · " + v.assay; }});
        var traces = [{{
          x: labels,
          y: d.values.map(function(v) {{ return v.result1; }}),
          name: {json.dumps(t['bar_r1'])},
          type: "bar",
          marker: {{color: "#6b7280"}},
        }}];
        if (hasMut) {{
          traces.push({{
            x: labels,
            y: d.values.map(function(v) {{ return v.result2; }}),
            name: {json.dumps(t['bar_r2'])},
            type: "bar",
            marker: {{color: "#d62728"}},
          }});
        }}
        var layout = {{
          barmode: "group",
          title: {{text: {json.dumps(t['bar_title'])}, font: {{size: 13}}}},
          xaxis: {{tickangle: -35, tickfont: {{size: 10}}}},
          yaxis: {{title: {{text: {json.dumps(t['bar_y'])}, font: {{size: 11}}}}}},
          margin: {{l: 45, r: 15, t: 34, b: 110}},
          showlegend: true,
          legend: {{font: {{size: 10}}}},
        }};
        // Rebuild cleanly: purge any stale/corrupted Plotly instance first,
        // then render fresh. Hide the placeholder once the chart is drawn.
        try {{ Plotly.purge('rm-barplot'); }} catch (e) {{}}
        Plotly.newPlot('rm-barplot', traces, layout, {{displaylogo: false, responsive: true}})
          .then(function() {{
            if (placeholderEl) placeholderEl.style.display = 'none';
          }});
      }})
      .catch(function(err) {{
        console.error("bar plot error", err);
        showMsg({json.dumps(t['bar_err_load'])}.replace("{{msg}}", (err && err.message ? err.message : err)));
      }});
    }}
    window.renderBarPlot = renderBarPlot;
    // Expose to the IGV iframe (both iframes share the parent window object)
    if (window.parent) {{ window.parent.__rmBarRenderer = renderBarPlot; }}
  </script>
</body>
</html>"""

    escaped = (
        inner_html.replace("&", "&amp;").replace('"', "&quot;")
        .replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<iframe srcdoc="{escaped}" style="width:100%;height:100%;border:none;border-radius:8px;background:#fff;"></iframe>"""


# ---------------------------------------------------------------------------
#  Event handlers
# ---------------------------------------------------------------------------
def _on_lang_toggle(lang: str, current_genome=None):
    """Switch UI language via the top-right EN / 中文 toggle buttons."""
    lang = lang if lang in I18N else DEFAULT_LANG
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    choices = _genome_options_all(lang)
    choices_ids = [v for _, v in choices]
    # Only keep the current genome if it is still a valid choice (e.g. after a
    # frontend restart the uploaded-genome registry is re-synced from backend).
    genome_val = (
        current_genome if current_genome in choices_ids
        else (choices[0][1] if choices else None)
    )
    return (
        gr.update(value=f"# {t['title']}"),
        gr.update(value=_get_intro_markdown(lang)),
        gr.update(value=t["required"]),
        gr.update(choices=choices, value=genome_val, label=t["genome"]),
        gr.update(label=t["chromosome"]),
        gr.update(label=t["start"]),
        gr.update(value=t["snv_section"]),
        gr.update(value=t["snv_hint"]),
        gr.update(label=t["snv_pos"]),
        gr.update(label=t["snv_base"]),
        gr.update(value=t["predict"]),
        gr.update(label=t["upload_genome_fasta"]),
        gr.update(label=t["upload_gff"]),
        gr.update(value=_bar_plot_html(lang)),
        gr.update(value=_igv_placeholder_html(lang)),
        gr.update(value=_get_track_legend_md(lang)),
        gr.update(value=lang),
        gr.update(variant="primary" if lang == "en" else "secondary"),
        gr.update(variant="primary" if lang == "zh" else "secondary"),
    )


def _switch_to_english(current_genome):
    return _on_lang_toggle("en", current_genome)


def _switch_to_chinese(current_genome):
    return _on_lang_toggle("zh", current_genome)


def _on_predict(
    genome: str,
    chromosome: str,
    start: int,
    snv_index,
    snv_base,
    lang: str = DEFAULT_LANG,
) -> str:
    """Unified Predict button — runs an SNV comparison when the SNV inputs are
    filled in, otherwise a reference-only prediction."""
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    if not chromosome:
        return _error_html(t["err_no_chromosome"])

    has_pos = snv_index is not None and snv_index != ""
    has_base = snv_base is not None and str(snv_base).strip() != ""
    if has_pos or has_base:
        if not has_pos:
            return _error_html(t["err_bad_snv_pair"])
        if not has_base or str(snv_base).strip().upper() not in ("A", "C", "G", "T", "N"):
            return _error_html(t["err_bad_snv_base"])
        return _predict_snv(genome, chromosome, start, snv_index, snv_base, lang)
    return _predict_reference(genome, chromosome, start, lang)


def _predict_reference(
    genome: str,
    chromosome: str,
    start: int,
    lang: str = DEFAULT_LANG,
) -> str:
    """Reference prediction -> IGV HTML.  Inputs are 1-based coordinates.

    Window length is fixed to ``DEFAULT_WINDOW`` (32768 bp); end is auto-derived.
    """
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    if not chromosome:
        return _error_html(t["err_no_chromosome"])

    # Convert 1-based user input to a 0-based half-open window of fixed length.
    start_1 = int(start) if start else 1
    start_0 = max(0, start_1 - 1)
    end_0 = start_0 + DEFAULT_WINDOW
    req = {
        "genome": genome,
        "chromosome": chromosome,
        "start": start_0,
        "end": end_0,
    }

    try:
        result = _call_backend_api("/predict", req)
    except Exception as e:
        return _error_html(f"{t['err_predict']}: {e}")

    if not result.get("success"):
        return _error_html(f"{t['err_predict']}: {result.get('message', '')}")

    meta = result.get("metadata") or {}
    igv_payload = result.get("igv_payload", {})
    locus = igv_payload.get("locus", f"{chromosome}:{start_1:,}-{end_0:,}")
    return _igv_html(
        genome, locus, igv_payload,
        prediction_id=meta.get("ref_id", ""),
        kind="ref",
        win_start=meta.get("window_start") or 0,
        win_len=meta.get("window_len") or 0,
    )


# ---------------------------------------------------------------------------
#  SNV (single-nucleotide variant) prediction
# ---------------------------------------------------------------------------
def _predict_snv(
    genome: str,
    chromosome: str,
    start: int,
    snv_index,
    snv_base,
    lang: str = DEFAULT_LANG,
) -> str:
    """Single-nucleotide variant prediction -> IGV HTML.  Inputs are 1-based.

    Window length is fixed to ``DEFAULT_WINDOW`` (32768 bp); end is auto-derived.
    """
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    if not chromosome:
        return _error_html(t["err_no_chromosome"])
    if snv_index is None or snv_index == "":
        return _error_html(t["err_no_snv_pos"])
    if not snv_base or str(snv_base).strip().upper() not in ("A", "C", "G", "T", "N"):
        return _error_html(t["err_bad_snv_base"])

    # Convert 1-based user input to a 0-based half-open window & SNV position.
    start_1 = int(start) if start else 1
    start_0 = max(0, start_1 - 1)
    end_0 = start_0 + DEFAULT_WINDOW
    snv_1 = int(snv_index)
    req = {
        "genome": genome,
        "chromosome": chromosome,
        "start": start_0,
        "end": end_0,
        "snv_index": max(0, snv_1 - 1),
        "snv_base": str(snv_base).strip().upper(),
    }

    try:
        result = _call_backend_api("/predict/snv", req)
    except Exception as e:
        return _error_html(f"{t['err_snv_predict']}: {e}")

    if not result.get("success"):
        return _error_html(f"{t['err_snv_predict']}: {result.get('message', '')}")

    meta = result.get("metadata") or {}
    igv_payload = result.get("igv_payload", {})
    locus = igv_payload.get("locus", f"{chromosome}:{start_1:,}-{end_0:,}")
    win_start = meta.get("window_start") or 0
    win_len = meta.get("window_len") or 0
    # Backend metadata is 0-based; show 1-based to the user.
    snv_1based = (meta.get("snv_index") or 0) + 1
    snv_info = t["snv_info"].format(
        ref=meta.get("ref_base", "?"),
        pos=snv_1based,
        alt=meta.get("snv_base", "?"),
        ws=win_start + 1,
        we=win_start + win_len,
    )
    info = f"<p style='color:gray;font-size:12px;'>{snv_info}</p>"
    return info + _igv_html(
        genome, locus, igv_payload,
        prediction_id=meta.get("snv_id", ""),
        kind="snv",
        win_start=win_start,
        win_len=win_len,
    )


# ---------------------------------------------------------------------------
#  Uploaded (custom) genomes — dynamic registry for the genome dropdown
# ---------------------------------------------------------------------------
# genome_id -> {"chromosomes": [...]}
_UPLOADED_GENOMES_FE: dict = {}
# Path of a GFF selected while the FASTA was still uploading; auto-attached
# once the custom genome is registered (race: large FASTA vs small GFF).
_PENDING_GFF_PATH: Optional[str] = None
# True while a FASTA upload request is in flight.
_FASTA_UPLOAD_IN_PROGRESS: bool = False


def _genome_options_all(lang: str = DEFAULT_LANG) -> list:
    """Built-in genomes + uploaded custom genomes + the Custom Genome sentinel."""
    ids = list(GENOME_CONFIGS.keys())
    for gid in _UPLOADED_GENOMES_FE:
        if gid not in ids:
            ids.append(gid)
    choices = [(g, g) for g in ids]
    choices.append((_custom_genome_label(lang), CUSTOM_GENOME_VALUE))
    return choices


def _sync_uploaded_genomes_from_backend() -> None:
    """Populate the frontend uploaded-genome registry from the backend on
    startup, so custom genomes survive a frontend restart (the backend keeps
    its in-memory registry).  Silently degrades if the backend is unreachable."""
    global _UPLOADED_GENOMES_FE
    try:
        with request.urlopen(f"{BACKEND_API_URL}/genomes", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return
    synced: dict = {}
    for gid in (data.get("genomes") or []):
        if gid in GENOME_CONFIGS or gid == CUSTOM_GENOME_VALUE:
            continue
        chroms: list = []
        try:
            with request.urlopen(
                f"{BACKEND_API_URL}/genomes/{gid}/chromosomes", timeout=10
            ) as resp:
                cdata = json.loads(resp.read().decode("utf-8"))
            chroms = cdata.get("chromosomes") or []
        except Exception:
            pass
        if not chroms:
            # FASTA file is missing/unreadable — skip stale custom genomes so
            # deleted/expired uploads don't clutter the genome dropdown.
            continue
        synced[gid] = {"chromosomes": chroms}
    _UPLOADED_GENOMES_FE = synced


_sync_uploaded_genomes_from_backend()


def _on_genome_change(genome, lang: str = DEFAULT_LANG) -> tuple:
    """Handle Genome dropdown changes: the Genome row is one column for built-in
    genomes and splits into three equal columns (Genome | FASTA | GFF) when a
    custom genome / the sentinel is active.  Also refreshes the chromosome list."""
    t = I18N.get(lang, I18N[DEFAULT_LANG])

    if genome == CUSTOM_GENOME_VALUE:
        # Sentinel: Genome -> 1/3, reveal both upload columns (GFF needs FASTA first).
        return (
            gr.update(scale=1),
            gr.update(visible=True, scale=1),
            gr.update(visible=True, scale=1),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(visible=True, value=_fmt_upload_status(t, "idle")),
            gr.update(choices=[], value=None, interactive=False),
        )

    if genome in _UPLOADED_GENOMES_FE:
        # Uploaded custom genome active: keep the split Genome row.
        chroms = _UPLOADED_GENOMES_FE[genome].get("chromosomes") or CHROMOSOME_OPTIONS
        return (
            gr.update(scale=1),
            gr.update(visible=True, scale=1),
            gr.update(visible=True, scale=1),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(visible=True, value=_fmt_upload_status(t, "fasta_ok", gid=genome, n=len(chroms))),
            gr.update(choices=chroms, value=None, interactive=True),
        )

    # Built-in genome: Genome full width, hide both upload columns and the status.
    return (
        gr.update(scale=3),
        gr.update(visible=False, scale=1),
        gr.update(visible=False, scale=1),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(visible=False, value=""),
        gr.update(choices=CHROMOSOME_OPTIONS, value=None, interactive=True),
    )


def _on_upload_genome(file, lang: str = DEFAULT_LANG) -> tuple:
    """Upload a custom genome FASTA; register it and auto-select the new id.

    If a GFF was selected before the FASTA finished uploading (race: small GFF
    finishes first), it is queued in ``_PENDING_GFF_PATH`` and auto-attached
    here once the custom genome is registered.
    """
    global _PENDING_GFF_PATH, _FASTA_UPLOAD_IN_PROGRESS
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    _FASTA_UPLOAD_IN_PROGRESS = True
    local_path = _uploaded_file_path(file)
    if not local_path or not os.path.exists(local_path):
        _FASTA_UPLOAD_IN_PROGRESS = False
        return (gr.update(), gr.update(), gr.update(), gr.update())

    try:
        result = _upload_genome_to_backend(local_path)
    except Exception as e:
        _FASTA_UPLOAD_IN_PROGRESS = False
        return (gr.update(), gr.update(), gr.update(),
                gr.update(visible=True, value=_fmt_upload_status(t, "err_fasta", msg=str(e))))

    if not result.get("success"):
        _FASTA_UPLOAD_IN_PROGRESS = False
        return (gr.update(), gr.update(), gr.update(),
                gr.update(visible=True, value=_fmt_upload_status(t, "err_fasta", msg=result.get("message", "?"))))

    _FASTA_UPLOAD_IN_PROGRESS = False
    genome_id = result.get("genome", "")
    chromosomes = result.get("chromosomes") or []
    _UPLOADED_GENOMES_FE[genome_id] = {"chromosomes": chromosomes}

    # Auto-attach a GFF that was queued while the FASTA was still uploading.
    status_key = "fasta_ok"
    status_kwargs: dict = {"gid": genome_id, "n": len(chromosomes)}
    if _PENDING_GFF_PATH:
        pending = _PENDING_GFF_PATH
        _PENDING_GFF_PATH = None
        try:
            gff_result = _upload_gff_to_backend(pending, genome_id)
        except Exception as e:
            status_key = "fasta_ok_gff_err"
            status_kwargs["msg"] = str(e)
        else:
            if gff_result.get("success"):
                status_key = "both_ok"
            else:
                status_key = "fasta_ok_gff_err"
                status_kwargs["msg"] = gff_result.get("detail", "?")

    # Auto-select the uploaded genome, refresh chromosomes.
    chrom_choices = chromosomes if chromosomes else CHROMOSOME_OPTIONS
    return (
        gr.update(choices=_genome_options_all(lang), value=genome_id),
        gr.update(choices=chrom_choices, value=chrom_choices[0] if chrom_choices else None, interactive=True),
        gr.update(value=None),
        gr.update(visible=True, value=_fmt_upload_status(t, status_key, **status_kwargs)),
    )


def _on_upload_gff(file, genome, lang: str = DEFAULT_LANG) -> dict:
    """Attach an annotation GFF to the currently selected custom genome.

    If no custom genome is registered yet (e.g. the small GFF arrives before
    the large FASTA finishes uploading), the GFF is queued in
    ``_PENDING_GFF_PATH`` and auto-attached once a custom genome is registered
    (or selected), instead of showing a misleading error.

    NOTE: this event has a single output (``upload_status_md``), so it must
    return the plain ``gr.update`` value — NOT a 1-element tuple.  For single-
    output events Gradio wraps the return in a list, so ``(gr.update(),)``
    becomes ``[(gr.update(),)]`` and the inner tuple bypasses the update-dict
    branch, crashing Markdown postprocess with "'tuple' object has no attribute
    'expandtabs'."
    """
    global _PENDING_GFF_PATH
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    local_path = _uploaded_file_path(file)
    if not local_path or not os.path.exists(local_path):
        return gr.update()

    # Custom genome already registered → attach directly.
    if genome in _UPLOADED_GENOMES_FE:
        try:
            result = _upload_gff_to_backend(local_path, genome)
        except Exception as e:
            return gr.update(visible=True, value=_fmt_upload_status(t, "err_gff", msg=str(e)))
        if not result.get("success"):
            return gr.update(visible=True, value=_fmt_upload_status(t, "err_gff", msg=result.get("detail", "?")))
        return gr.update(visible=True, value=_fmt_upload_status(t, "gff_ok", gid=genome))

    # No custom genome ready yet (FASTA may still be uploading) → queue the GFF
    # and auto-attach once a custom genome is registered/selected.
    _PENDING_GFF_PATH = local_path
    return gr.update(visible=True, value=_fmt_upload_status(t, "gff_wait"))


# ---------------------------------------------------------------------------
#  Build Gradio interface
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="OGR-Mutation: DNA → Expression Prediction",
    theme=gr.themes.Default(),
    css="""footer {display:none !important}
    html, body {overflow-x:hidden !important;}
    .gradio-container {max-width:100% !important; overflow-x:hidden !important;}
    body {font-family: Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif !important;}
    /* Let the app content use the full container width (Gradio caps main at 768px). */
    main.app {max-width:none !important; width:100% !important;}

    /* 3:1 layout row — never wrap the two side-by-side columns */
    .layout-row-31 {flex-wrap:nowrap !important;}

    /* ── Header bar (Genos-Mutation style): title left, EN/中文 pills right ── */
    .app-header-bar {display:flex !important; align-items:center !important; justify-content:space-between !important; gap:16px; margin-bottom:4px; flex-wrap:wrap;}
    .app-header-title {margin:0 !important; min-width:0 !important;}
    .app-header-title h1 {margin:0 !important; font-size:1.5rem !important; line-height:1.3 !important;}
    .app-header-actions {display:flex !important; justify-content:flex-end !important; align-items:center !important; gap:8px; flex-wrap:nowrap;}
    .lang-toggle-btn {min-width:88px; min-height:36px; padding:0 18px; border-radius:999px !important; font-weight:600; box-shadow:none !important;}

    .intro-copy {margin-top:2px;}
    .intro-copy p {margin-top:0; color:#6B7280 !important;}

    /* ── Card-style panels (white rounded cards) ── */
    .card-panel {background:#fff !important; border:1px solid #e5e7eb !important; border-radius:12px !important; padding:14px 16px !important; box-shadow:0 1px 2px rgba(16,24,40,.04) !important;}
    .card-panel h2, .card-panel h3 {margin-top:0 !important;}

    /* ── IGV white rounded panel ── */
    .igv-card {background:#fff !important; border:1px solid #e5e7eb !important; border-radius:12px !important; padding:14px 16px !important;}
    .igv-card .html-container {padding:0 !important;}

    /* ── Buttons / upload / bar plot ── */
    #predict-btn {min-height:36px !important; height:36px !important; width:100% !important; margin-top:4px !important;}
    #genome-upload, #gff-upload {height:110px !important;}
    #genome-upload .wrap, #genome-upload .center, #gff-upload .wrap, #gff-upload .center {height:100% !important;}
    #right-col {display:flex; flex-direction:column;}
    #right-col > * {flex-grow:0 !important; flex-shrink:1 !important; flex-basis:auto !important;}
    #right-col > #bar-plot-container {flex:1 1 0% !important; min-height:340px !important; display:flex; flex-direction:column;}
    /* Left column's Genome row splits into three equal columns: Genome | FASTA | GFF */
    .layout-row-3sub {flex-wrap:nowrap !important;}
    .layout-row-3sub > * {min-width:0 !important;}
    #bar-plot-container > div, #bar-plot-container .prose {flex:1 1 auto !important; min-height:0; display:flex; flex-direction:column;}
    #bar-plot-container iframe {flex:1 1 auto !important; width:100% !important; height:100% !important; border:none; border-radius:8px; background:#fff;}
    #bar-plot-container .html-container {padding:0 !important;}

    /* ── SNV optional hint ── */
    .snv-hint p {font-size:0.85rem !important; color:#6b7280 !important; margin:0 0 6px 0 !important;}

    @media (max-width: 900px) {
        .app-header-title h1 {font-size:1.25rem !important;}
        .lang-toggle-btn {min-width:72px; min-height:32px; padding:0 12px;}
    }
    """,
) as demo:

    # Current UI language ("en" / "zh") — kept in sync by the toggle buttons.
    cur_lang = gr.State(DEFAULT_LANG)

    # ── Header bar: title + EN / 中文 pill toggle buttons (Genos-Mutation style) ──
    with gr.Row(equal_height=True, elem_classes=["app-header-bar"]):
        header_md = gr.Markdown(
            f"# {I18N[DEFAULT_LANG]['title']}",
            elem_classes=["app-header-title"],
        )
        with gr.Column(scale=0, min_width=220):
            with gr.Row(elem_classes=["app-header-actions"]):
                btn_lang_en = gr.Button(
                    "English", variant="primary", elem_classes=["lang-toggle-btn"],
                )
                btn_lang_zh = gr.Button(
                    "中文", variant="secondary", elem_classes=["lang-toggle-btn"],
                )

    intro_md = gr.Markdown(
        _get_intro_markdown(DEFAULT_LANG),
        elem_classes=["intro-copy"],
    )

    # --- 3:1 layout: left = params (Genome row splits for custom uploads), right = bar chart ---
    with gr.Row(equal_height=True, elem_classes=["layout-row-31"]):
        # ---- Left column (scale 3) ----
        with gr.Column(scale=3, min_width=380, elem_classes=["card-panel"]):
            required_md = gr.Markdown(I18N[DEFAULT_LANG]["required"])
            # Genome row — splits into 参数 | Upload FASTA | Upload GFF when Custom
            with gr.Row(equal_height=True, elem_classes=["layout-row-3sub"]):
                with gr.Column(scale=3, min_width=0, elem_id="genome-col") as genome_col:
                    genome_dd = gr.Dropdown(
                        choices=_genome_options_all(),
                        value=DEFAULT_GENOME if DEFAULT_GENOME else None,
                        label=I18N[DEFAULT_LANG]["genome"], interactive=True,
                    )
                with gr.Column(visible=False, scale=1, min_width=0, elem_id="fasta-col") as fasta_col:
                    genome_fasta_upload = gr.File(
                        file_count="single",
                        label=I18N[DEFAULT_LANG]["upload_genome_fasta"],
                        file_types=[".fa", ".fasta", ".fna"],
                        interactive=True,
                        elem_id="genome-upload",
                    )
                with gr.Column(visible=False, scale=1, min_width=0, elem_id="gff-col") as gff_col:
                    gff_upload = gr.File(
                        file_count="single",
                        label=I18N[DEFAULT_LANG]["upload_gff"],
                        file_types=[".gff", ".gff3", ".gtf", ".gz"],
                        elem_id="gff-upload",
                    )
            # Upload status — full width, shown only when a custom genome is active
            upload_status_md = gr.Markdown(
                value=_fmt_upload_status(I18N[DEFAULT_LANG], "idle"),
                visible=False,
            )
            # Full-width params below the Genome row
            with gr.Row(equal_height=True):
                chromosome_dd = gr.Dropdown(
                    choices=CHROMOSOME_OPTIONS, value=None,
                    label=I18N[DEFAULT_LANG]["chromosome"], interactive=True, scale=1,
                )
                start_input = gr.Number(
                    value=None, label=I18N[DEFAULT_LANG]["start"], precision=0,
                    interactive=True, scale=1,
                    placeholder="e.g. 20716773",
                )
            snv_section_md = gr.Markdown(I18N[DEFAULT_LANG]["snv_section"])
            snv_hint_md = gr.Markdown(
                I18N[DEFAULT_LANG]["snv_hint"], elem_classes=["snv-hint"],
            )
            with gr.Row(equal_height=True):
                snv_index_input = gr.Number(
                    value=None, label=I18N[DEFAULT_LANG]["snv_pos"], precision=0,
                    placeholder="e.g. 20731844", interactive=True, scale=1,
                )
                snv_base_dd = gr.Dropdown(
                    choices=["A", "C", "G", "T", "N"], value=None,
                    label=I18N[DEFAULT_LANG]["snv_base"], interactive=True, scale=1,
                )
            predict_btn = gr.Button(
                I18N[DEFAULT_LANG]["predict"], variant="primary", elem_id="predict-btn",
            )

        # ---- Right column (scale 1): bar chart only ----
        with gr.Column(scale=1, min_width=200, elem_id="right-col", elem_classes=["card-panel"]):
            bar_plot = gr.HTML(value=_bar_plot_html(), elem_id="bar-plot-container")

    # ── IGV white rounded panel (full width) + track legend ──
    igv_html_component = gr.HTML(
        value=_igv_placeholder_html(DEFAULT_LANG),
        elem_classes=["igv-card"],
    )
    track_legend_md = gr.Markdown(
        value=_get_track_legend_md(DEFAULT_LANG),
    )

    # ── Event wiring ──
    btn_lang_en.click(
        fn=_switch_to_english,
        inputs=[genome_dd],
        outputs=[
            header_md,
            intro_md,
            required_md,
            genome_dd,
            chromosome_dd,
            start_input,
            snv_section_md,
            snv_hint_md,
            snv_index_input,
            snv_base_dd,
            predict_btn,
            genome_fasta_upload,
            gff_upload,
            bar_plot,
            igv_html_component,
            track_legend_md,
            cur_lang,
            btn_lang_en,
            btn_lang_zh,
        ],
        queue=False,
    )
    btn_lang_zh.click(
        fn=_switch_to_chinese,
        inputs=[genome_dd],
        outputs=[
            header_md,
            intro_md,
            required_md,
            genome_dd,
            chromosome_dd,
            start_input,
            snv_section_md,
            snv_hint_md,
            snv_index_input,
            snv_base_dd,
            predict_btn,
            genome_fasta_upload,
            gff_upload,
            bar_plot,
            igv_html_component,
            track_legend_md,
            cur_lang,
            btn_lang_en,
            btn_lang_zh,
        ],
        queue=False,
    )

    # Genome selection -> Genome row is one column (built-in) or three equal
    # columns (Genome | FASTA | GFF) for custom genomes; refresh chromosomes.
    genome_dd.change(
        fn=_on_genome_change,
        inputs=[genome_dd, cur_lang],
        outputs=[genome_col, fasta_col, gff_col, genome_fasta_upload, gff_upload, upload_status_md, chromosome_dd],
        queue=False,
    )

    # Single Predict button (reference or SNV, decided by the SNV inputs)
    predict_btn.click(
        fn=_on_predict,
        inputs=[genome_dd, chromosome_dd, start_input, snv_index_input, snv_base_dd, cur_lang],
        outputs=[igv_html_component],
    )

    # Custom genome uploads (FASTA first, then GFF)
    genome_fasta_upload.change(
        fn=_on_upload_genome,
        inputs=[genome_fasta_upload, cur_lang],
        outputs=[genome_dd, chromosome_dd, gff_upload, upload_status_md],
    )
    gff_upload.change(
        fn=_on_upload_gff,
        inputs=[gff_upload, genome_dd, cur_lang],
        outputs=[upload_status_md],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=FRONTEND_HOST,
        server_port=FRONTEND_PORT,
        show_error=True,
    )
