"""Rice-Reg Frontend — Gradio UI for ATAC → RNA-seq expression prediction.

Style and Custom-Genome workflow aligned with rice-mutation:
- Orange theme (gr.themes.Default + #f97316), header bar with EN/中文 pills.
- Card-style parameter panel (single column) + full-width IGV below (no 3:1,
  no bar chart).
- Genome dropdown includes a "📤 Custom Genome" sentinel; when selected the
  Genome row splits into three equal columns (Genome | Upload FASTA | Upload
  GFF), mirroring rice-mutation.
- ATAC: built-in dropdown + custom bigWig upload (kept from the original).
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib import request, error
from urllib.parse import quote

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
    ATAC_SIGNAL_PATHS,
    GENOME_ATAC_MAP,
    CHROMOSOME_OPTIONS,
    DEFAULT_GENOME,
    BACKEND_API_URL,
    IGV_CDN_URL,
)
from frontend.igv_payload import (
    build_default_prediction_reference,
    resolve_case_configs,
    set_static_base_url,
)

STATIC_DIR_ABS = os.path.join(BASE_DIR, "static")
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "7000"))
ALLOWED_ATAC_UPLOAD_SUFFIXES = (".bw", ".bigwig", ".bigWig", ".BW", ".BIGWIG")

# Fixed prediction window length (bp).  end = start + WINDOW_LEN.
WINDOW_LEN = 32768

# Upload size limits (MB) — read from .env so they stay in sync with backend.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "640"))
MAX_ATAC_UPLOAD_MB = int(os.getenv("MAX_ATAC_UPLOAD_MB", "10240"))

# ---------------------------------------------------------------------------
#  i18n — UI copy.  Default language is English; EN/中文 toggle at top.
# ---------------------------------------------------------------------------
DEFAULT_LANG = "en"
CUSTOM_GENOME_VALUE = "__custom__"
UPLOAD_OPTION_VALUE = "__upload__"

I18N = {
    "en": {
        "title": "OGR-Regulation",
        "subtitle": "Predict RNA-seq coverage from DNA + ATAC. Optionally upload a custom genome FASTA (+ annotation GFF) for other assemblies.",
        "genome": "Genome",
        "chromosome": "Chromosome",
        "start": "Start (prediction window: 32 kb",
        "atac_file": "ATAC-seq_Source",
        "atac_upload_label": "ATAC bigWig path",
        "atac_path_ph": "/abs/path/to/xxx.bw",
        "predict": "🚀 Predict",
        "api_key": "API Key (required)",
        "api_key_ph": "sk-... (required — enter your DCS API key)",
        "custom_genome_option": "📤 Custom Genome",
        "upload_genome_fasta": "Genome FASTA path",
        "upload_gff": "Annotation GFF path",
        "upload_fasta_ph": "/abs/path/to/xxx.fa",
        "upload_gff_ph": "/abs/path/to/xxx.gff3",
        "upload_status_idle": "Select “Custom Genome” and upload a FASTA to register a new genome.",
        "upload_status_fasta_ok": "Genome “{gid}” registered ({n} chromosomes). You can now attach a GFF.",
        "upload_status_gff_ok": "GFF attached to “{gid}”. A Genes track will appear in IGV.",
        "upload_status_gff_wait": "GFF ready — will attach automatically once the FASTA upload finishes.",
        "upload_status_both_ok": "Genome “{gid}” registered ({n} chromosomes). GFF auto-attached — a Genes track will appear in IGV.",
        "upload_err_gff_auto": "GFF auto-attach failed: {msg}. Please re-upload the GFF.",
        "upload_err_fasta": "FASTA upload failed: {msg}",
        "upload_err_gff": "GFF upload failed: {msg}",
        "upload_need_fasta": "Upload a FASTA first, then a GFF can be attached.",
        "placeholder": "Select inputs and click Predict to see results.",
        "err_no_chromosome": "Please select a chromosome.",
        "err_no_atac": "Please select a built-in ATAC source or upload a bigWig file.",
        "err_upload_failed": "Upload failed",
        "err_prediction_failed": "Prediction failed",
    },
    "zh": {
        "title": "OGR-Regulation",
        "subtitle": "由 DNA + ATAC 预测 RNA-seq 覆盖度。可选上传自定义基因组 FASTA（+ 注释 GFF）用于其它组装。",
        "genome": "基因组",
        "chromosome": "染色体",
        "start": "起始位置",
        "atac_file": "ATAC-seq 数据源",
        "atac_upload_label": "ATAC bigWig 文件路径",
        "atac_path_ph": "/abs/path/to/xxx.bw",
        "predict": "🚀 预测",
        "api_key": "API Key（必填）",
        "api_key_ph": "sk-...（必填——请输入你的 DCS API Key）",
        "custom_genome_option": "📤 自定义基因组",
        "upload_genome_fasta": "基因组 FASTA 文件路径",
        "upload_gff": "注释 GFF 文件路径",
        "upload_fasta_ph": "/abs/path/to/xxx.fa",
        "upload_gff_ph": "/abs/path/to/xxx.gff3",
        "upload_status_idle": "选择「自定义基因组」并上传 FASTA 以注册新基因组。",
        "upload_status_fasta_ok": "基因组「{gid}」已注册（{n} 条染色体）。现在可附加 GFF。",
        "upload_status_gff_ok": "GFF 已附加到「{gid}」。IGV 将显示 Genes 轨道。",
        "upload_status_gff_wait": "GFF 已就绪，等待 FASTA 上传完成后自动附加。",
        "upload_status_both_ok": "基因组「{gid}」已注册（{n} 条染色体）。GFF 已自动附加，IGV 将显示 Genes 轨道。",
        "upload_err_gff_auto": "GFF 自动附加失败：{msg}。请重新上传 GFF。",
        "upload_err_fasta": "FASTA 上传失败：{msg}",
        "upload_err_gff": "GFF 上传失败：{msg}",
        "upload_need_fasta": "请先上传 FASTA，之后才能附加 GFF。",
        "placeholder": "选择输入后点击预测查看结果。",
        "err_no_chromosome": "请选择染色体。",
        "err_no_atac": "请选择内置 ATAC 数据或上传 bigWig 文件。",
        "err_upload_failed": "上传失败",
        "err_prediction_failed": "预测失败",
    },
}

# Configure IGV static file serving — served same-origin through the frontend's
# /backend/* reverse proxy (see __main__ below).  Using a same-origin relative
# path means the browser only ever talks to the frontend port, so IGV.js and the
# bigWig / FASTA / GFF tracks no longer depend on the browser being able to
# reach the backend port directly.
set_static_base_url("/backend/static-files")

# Backend responses embed static-file URLs that are absolute
# (http://<host>:<port>/static-files/...) or relative (/static-files/...) — both
# depend on how the backend is configured and are NOT reachable from the
# browser.  Rewrite every one of them to the frontend same-origin proxy path so
# the browser always loads IGV tracks through the frontend port.  This keeps the
# whole setup portable: it works on any machine regardless of the backend
# host/port.
_STATIC_URL_RE = re.compile(r"(?:https?://[^/]+)?/static-files/")

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _genome_options() -> list:
    return list(GENOME_CONFIGS.keys())


def _custom_genome_label(lang: str = DEFAULT_LANG) -> str:
    return I18N.get(lang, I18N[DEFAULT_LANG])["custom_genome_option"]


def _atac_options_for_genome(genome: str) -> list:
    return GENOME_ATAC_MAP.get(genome, [])


def _atac_choices_for_genome(genome: str, lang: str = DEFAULT_LANG) -> list:
    """Built-in ATAC options for a genome, plus the 'upload custom' sentinel."""
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    choices = [(i, i) for i in _atac_options_for_genome(genome)]
    choices.append((t["atac_upload_label"] + "…", UPLOAD_OPTION_VALUE))
    return choices


def _call_backend_api(endpoint: str, data: dict, api_key: str = None) -> dict:
    """Call the backend FastAPI endpoint and return the JSON response.

    Auth: only the per-request ``api_key`` from the UI input box is used (never
    .env).  Empty → no ``Authorization`` header (backend decides; if it has a
    ``DCS_API_KEY`` it replies 401).
    """
    url = f"{BACKEND_API_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    key = (api_key or "").strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = request.Request(
        url,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend error ({e.code}): {body}")
    except error.URLError as e:
        raise RuntimeError(f"Cannot reach backend at {url}: {e}")


def _call_dcs(subpath: str, data: dict, api_key: str = None) -> dict:
    """Call a DCS-style sub-path on the unified backend and return ``result``.

    The web frontend and pure-API (curl) callers share the exact same request
    dialect: camelCase keys (``Genome``/``Chromosome``/``Start``/``ATAC-seq_Source``
    /``Format``), 1-based coordinates, and the same DCS sub-path URLs.  All
    1-based → 0-based conversion happens inside the backend.  Returns the
    unified envelope's ``result`` (raises on non-200).
    """
    endpoint = f"/api/aigress/openai/OGR-Regulation/{subpath}"
    resp = _call_backend_api(endpoint, data, api_key=api_key)
    status = resp.get("status", 200)
    if status != 200:
        msg = resp.get("message", "DCS call failed")
        detail = resp.get("detail")
        if detail:
            msg = f"{msg} ({detail})"
        raise RuntimeError(msg)
    return resp.get("result") or {}


def _upload_genome_to_backend(fasta_path: str, gff_path: str = "") -> dict:
    """Register a custom genome by local FASTA path (zero-copy) via the DCS
    ``RegisterGenome`` sub-path.  Front/back end share the machine, so we send
    the path, not the file bytes.  Returns the UI-shaped dict.
    """
    body = {"fasta_path": fasta_path}
    if gff_path:
        body["gff_path"] = gff_path
    try:
        result = _call_dcs("RegisterGenome", body)
    except Exception as e:
        return {"success": False, "message": str(e)}
    return {
        "success": True,
        "genome": result.get("Genome") or result.get("genome") or "",
        "chromosomes": result.get("chromosomes") or [],
        "fasta_path": result.get("fasta_path", ""),
        "gff_path": result.get("gff_path", ""),
        "message": result.get("message", "genome registered"),
    }


def _upload_gff_to_backend(gff_path: str, genome_id: str) -> dict:
    """Attach an annotation GFF (by local path) to an already-registered custom
    genome via the DCS ``RegisterGenome`` sub-path (attach mode)."""
    try:
        result = _call_dcs("RegisterGenome", {
            "genome_id": genome_id,
            "gff_path": gff_path,
        })
    except Exception as e:
        return {"success": False, "detail": str(e)}
    return {
        "success": True,
        "genome": result.get("Genome") or genome_id,
        "fasta_path": result.get("fasta_path", ""),
        "gff_path": result.get("gff_path", ""),
        "detail": result.get("message", "gff attached"),
    }


def _uploaded_file_path(file) -> Optional[str]:
    """Extract a stable local path from a Gradio-uploaded file object.

    Defensively handles the values Gradio can pass while a file is still
    uploading (None, tuples, dicts, plain strings, objects with .path).
    """
    if file is None:
        return None
    if isinstance(file, tuple):
        # Gradio may pass (path, None) / (None, None) while uploading.
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
) -> str:
    """Generate an HTML snippet that renders IGV.js inside an iframe srcdoc.

    Uses the backend-provided reference when available (supports uploaded
    custom genomes, including their Genes track); falls back to the built-in
    reference builder for local configs.
    """
    # Prefer the backend-provided reference (custom genomes); else built-in.
    ref = None
    if igv_payload and igv_payload.get("reference"):
        ref = igv_payload["reference"]
    else:
        ref = build_default_prediction_reference(genome, GENOME_CONFIGS)
    if ref is None:
        return "<p style='color:red;'>Unknown genome. Check .env configuration.</p>"

    ref_json = _STATIC_URL_RE.sub("/backend/static-files/", json.dumps(ref))
    tracks_json = _STATIC_URL_RE.sub(
        "/backend/static-files/",
        json.dumps(igv_payload.get("tracks", [])) if igv_payload else "[]",
    )
    locus_str = igv_payload.get("locus", locus) if igv_payload else locus

    # Same-origin URL: resolved against the parent page (srcdoc iframe inherits
    # the parent document's base URI), then proxied to the backend by the
    # frontend /backend/* reverse proxy.
    igv_js_url = f"/backend/static-files{STATIC_DIR_ABS}/igv.min.js"

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="{igv_js_url}"></script>
  <style>
    body {{ margin: 0; padding: 0; }}
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

        function installWigContextMenus(browser) {{
            var views = browser.trackViews || [];
            for (var i = 0; i < views.length; i++) {{
                var t = views[i].track;
                if (!t || !t.config || t.config.type !== "wig") continue;
                if (t.__rmWigMenuInstalled) continue;
                t.__rmWigMenuInstalled = true;
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
        }}

        igv.createBrowser(document.getElementById("igv-container"), {{
            genome: ref,
            locus: locus,
            tracks: tracks,
            showNavigation: true,
            showRuler: true,
            genomeList: [],
            customButtons: [
                {{ label: "Save PNG", callback: function(b) {{ saveViewPNG(b); }} }}
            ],
        }}).then(function(browser) {{
            window.__igvBrowser = browser;
            installWigContextMenus(browser);
            browser.on("trackorderchange", function() {{ installWigContextMenus(browser); }});
        }});
    }});
  </script>
</body>
</html>"""

    # Escape the inner HTML for use in an iframe srcdoc
    escaped = (
        inner_html.replace("&", "&amp;").replace('"', "&quot;")
        .replace("'", "&apos;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<iframe srcdoc="{escaped}" style="width:100%;height:540px;border:none;border-radius:8px;"></iframe>"""


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
    startup (survives a frontend restart).  Uses the DCS sub-paths so the web
    frontend and curl callers share one dialect.  Skips stale genomes whose
    FASTA file is no longer readable.
    """
    global _UPLOADED_GENOMES_FE
    try:
        data = _call_dcs("Genome", {})
    except Exception:
        return
    synced: dict = {}
    for gid in (data.get("Genome") or []):
        if gid in GENOME_CONFIGS or gid == CUSTOM_GENOME_VALUE:
            continue
        chroms: list = []
        try:
            cdata = _call_dcs("Chromosome", {"Genome": gid})
            chroms = cdata.get("Chromosome") or []
        except Exception:
            pass
        if not chroms:
            # FASTA file is missing/unreadable — skip stale custom genomes so
            # deleted/expired uploads don't clutter the genome dropdown.
            continue
        synced[gid] = {"chromosomes": chroms}
    _UPLOADED_GENOMES_FE = synced


_sync_uploaded_genomes_from_backend()


# ---------------------------------------------------------------------------
#  Event handlers
# ---------------------------------------------------------------------------
def _on_genome_change(genome, lang: str = DEFAULT_LANG) -> tuple:
    """Handle Genome dropdown changes: the Genome row is one column for built-in
    genomes and splits into three equal columns (Genome | FASTA | GFF) when a
    custom genome / the sentinel is active.  Also refreshes chromosome + ATAC."""
    t = I18N.get(lang, I18N[DEFAULT_LANG])

    if genome == CUSTOM_GENOME_VALUE:
        # Sentinel: Genome -> 1/3, reveal both upload columns.
        return (
            gr.update(scale=1),
            gr.update(visible=True, scale=1),
            gr.update(visible=True, scale=1),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=True, value=_fmt_upload_status(t, "idle")),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=_atac_choices_for_genome(genome, lang), value=None),
        )

    if genome in _UPLOADED_GENOMES_FE:
        # Uploaded custom genome active: keep the split Genome row.
        chroms = _UPLOADED_GENOMES_FE[genome].get("chromosomes") or CHROMOSOME_OPTIONS
        return (
            gr.update(scale=1),
            gr.update(visible=True, scale=1),
            gr.update(visible=True, scale=1),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=True, value=_fmt_upload_status(t, "fasta_ok", gid=genome, n=len(chroms))),
            gr.update(choices=chroms, value=None, interactive=True),
            gr.update(choices=_atac_choices_for_genome(genome, lang), value=None),
        )

    # Built-in genome: Genome full width, hide both upload columns and status.
    return (
        gr.update(scale=3),
        gr.update(visible=False, scale=1),
        gr.update(visible=False, scale=1),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False, value=""),
        gr.update(choices=CHROMOSOME_OPTIONS, value=None, interactive=True),
        gr.update(choices=_atac_choices_for_genome(genome, lang), value=None),
    )


def _on_upload_genome(fasta_path, lang: str = DEFAULT_LANG) -> tuple:
    """Register a custom genome by its local FASTA path (zero-copy, lazy).

    If a GFF path was entered before the FASTA finished registering (race:
    small GFF finishes first), it is queued in ``_PENDING_GFF_PATH`` and
    auto-attached here once the custom genome is registered.
    """
    global _PENDING_GFF_PATH, _FASTA_UPLOAD_IN_PROGRESS
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    _FASTA_UPLOAD_IN_PROGRESS = True
    local_path = (fasta_path or "").strip()
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
    _UPLOADED_GENOMES_FE[genome_id] = {
        "chromosomes": chromosomes,
        "fasta_path": result.get("fasta_path", "") or local_path,
        "gff_path": result.get("gff_path", ""),
    }

    # Auto-attach a GFF that was queued while the FASTA was still registering.
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
                _UPLOADED_GENOMES_FE[genome_id]["gff_path"] = gff_result.get("gff_path", "") or pending
            else:
                status_key = "fasta_ok_gff_err"
                status_kwargs["msg"] = gff_result.get("detail", "?")

    # Auto-select the uploaded genome, refresh chromosomes.
    chrom_choices = chromosomes if chromosomes else CHROMOSOME_OPTIONS
    return (
        gr.update(choices=_genome_options_all(lang), value=genome_id),
        gr.update(choices=chrom_choices, value=chrom_choices[0] if chrom_choices else None, interactive=True),
        gr.update(),
        gr.update(visible=True, value=_fmt_upload_status(t, status_key, **status_kwargs)),
    )


def _on_upload_gff(gff_path, genome, lang: str = DEFAULT_LANG) -> dict:
    """Attach an annotation GFF (by local path) to the selected custom genome.

    If no custom genome is registered yet (e.g. the GFF path is entered before
    the FASTA finishes registering), the GFF is queued in ``_PENDING_GFF_PATH``
    and auto-attached once a custom genome is registered.

    NOTE: single output (``upload_status_md``) → must return a plain gr.update,
    NOT a 1-element tuple (Gradio wraps single-output returns in a list).
    """
    global _PENDING_GFF_PATH
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    local_path = (gff_path or "").strip()
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
        _UPLOADED_GENOMES_FE[genome]["gff_path"] = result.get("gff_path", "") or local_path
        return gr.update(visible=True, value=_fmt_upload_status(t, "gff_ok", gid=genome))

    # No custom genome ready yet (FASTA may still be registering) → queue the GFF.
    _PENDING_GFF_PATH = local_path
    return gr.update(visible=True, value=_fmt_upload_status(t, "gff_wait"))


def _on_atac_select(atac_source: str) -> dict:
    """Reveal the custom bigWig path box when 'upload custom' is chosen; else hide it."""
    if atac_source == UPLOAD_OPTION_VALUE:
        return gr.update(visible=True)
    return gr.update(visible=False, value="")


def _on_atac_upload(upload_state, atac_dropdown: str) -> dict:
    """Deprecated (kept for backward import safety); clears dropdown selection."""
    return gr.update(value=None)


def _on_lang_toggle(lang: str, current_genome=None) -> tuple:
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
        gr.update(value=f"*{t['subtitle']}*"),
        gr.update(choices=choices, value=genome_val, label=t["genome"]),
        gr.update(label=t["chromosome"]),
        gr.update(label=t["start"]),
        gr.update(label=t["atac_file"], choices=_atac_choices_for_genome(genome_val or DEFAULT_GENOME, lang), value=None),
        gr.update(label=t["atac_upload_label"], placeholder=t["atac_path_ph"]),
        gr.update(label=t["api_key"], placeholder=t["api_key_ph"]),
        gr.update(value=t["predict"]),
        gr.update(label=t["upload_genome_fasta"], placeholder=t["upload_fasta_ph"]),
        gr.update(label=t["upload_gff"], placeholder=t["upload_gff_ph"]),
        gr.update(variant="primary" if lang == "en" else "secondary"),
        gr.update(variant="primary" if lang == "zh" else "secondary"),
        gr.update(value=f"<p style='color:#6B7280;'>{t['placeholder']}</p>"),
        gr.update(value=lang),
    )


def _on_lang_en(genome: str):
    return _on_lang_toggle("en", genome)


def _on_lang_zh(genome: str):
    return _on_lang_toggle("zh", genome)


def _on_predict(
    genome: str,
    chromosome: str,
    start: int,
    atac_source: str,
    atac_path: str,
    lang: str = DEFAULT_LANG,
    api_key: str = None,
) -> str:
    """Run ATAC→RNA-seq prediction via the DCS ``Prediction`` sub-path.

    Uses the same camelCase keys + 1-based ``Start`` as a pure-API (curl)
    caller; the backend converts to 0-based internally.  Custom ATAC bigWig is
    passed by local path (``uploaded_atac``); a custom genome carries its
    registered ``fasta_path``/``gff_path`` for lazy registration.
    """
    t = I18N.get(lang, I18N[DEFAULT_LANG])
    if not chromosome:
        return f"<p style='color:red;'>{t['err_no_chromosome']}</p>"

    uploaded_atac = (atac_path or "").strip()
    builtin_atac = atac_source if atac_source and atac_source != UPLOAD_OPTION_VALUE else None
    if not builtin_atac and not uploaded_atac:
        return f"<p style='color:red;'>{t['err_no_atac']}</p>"

    req = {
        "Genome": genome,
        "Chromosome": chromosome,
        "Start": int(start) if start else 1,
        "Format": "Full",
    }
    if builtin_atac:
        req["ATAC-seq_Source"] = builtin_atac
    if uploaded_atac:
        req["uploaded_atac"] = uploaded_atac

    # Custom genome → carry paths so the backend lazy-registers in place.
    entry = _UPLOADED_GENOMES_FE.get(genome)
    if entry:
        if entry.get("fasta_path"):
            req["fasta_path"] = entry["fasta_path"]
        if entry.get("gff_path"):
            req["gff_path"] = entry["gff_path"]

    try:
        data = _call_dcs("Prediction", req, api_key=api_key)
    except Exception as e:
        return f"<p style='color:red;'>{t['err_prediction_failed']}: {e}</p>"

    igv_payload = data.get("igv_payload") or {}
    locus = igv_payload.get("locus", f"{chromosome}:{req['Start']:,}-{req['Start'] + WINDOW_LEN - 1:,}")
    return _igv_html(genome, locus, igv_payload)


# ---------------------------------------------------------------------------
#  Build Gradio interface
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Rice-Reg: ATAC → RNA-seq Prediction",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.gray,
    ),
    css="""footer {display:none !important}
    html, body {overflow-x:hidden !important;}
    .gradio-container {max-width:100% !important; overflow-x:hidden !important;}
    body {font-family: Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif !important;}
    /* Let the app content use the full container width (Gradio caps main at 768px). */
    main.app {max-width:none !important; width:100% !important;}

    /* ── Header bar (rice-mutation style): title left, EN/中文 pills right ── */
    .app-header-bar {display:flex !important; align-items:center !important; justify-content:space-between !important; gap:16px; margin-bottom:4px; flex-wrap:wrap;}
    .app-header-title {margin:0 !important; min-width:0 !important;}
    .app-header-title h1 {margin:0 !important; font-size:1.5rem !important; line-height:1.3 !important;}
    .app-header-actions {display:flex !important; justify-content:flex-end !important; align-items:center !important; gap:8px; flex-wrap:nowrap;}
    .lang-toggle-btn {min-width:88px; min-height:36px; padding:0 18px; border-radius:999px !important; font-weight:600; box-shadow:none !important;}

    .intro-copy {margin-top:2px;}
    .intro-copy p {margin-top:0; color:#6B7280 !important;}

    /* ── Card-style panel (white rounded card) ── */
    .card-panel {background:#fff !important; border:1px solid #e5e7eb !important; border-radius:12px !important; padding:14px 16px !important; box-shadow:0 1px 2px rgba(16,24,40,.04) !important;}
    .card-panel h2, .card-panel h3 {margin-top:0 !important;}

    /* Genome row splits into three equal columns: Genome | FASTA | GFF */
    .layout-row-3sub {flex-wrap:nowrap !important;}
    .layout-row-3sub > * {min-width:0 !important;}
    #genome-upload, #gff-upload {height:110px !important;}
    #genome-upload .wrap, #genome-upload .center, #gff-upload .wrap, #gff-upload .center {height:100% !important;}

    #predict-btn {min-height:36px !important; height:36px !important; width:100% !important; margin-top:4px !important;}
    .atac-upload-btn {min-height:36px !important;}

    @media (max-width: 900px) {
        .app-header-title h1 {font-size:1.25rem !important;}
        .lang-toggle-btn {min-width:72px; min-height:32px; padding:0 12px;}
    }
    """,
) as demo:

    # Current UI language ("en" / "zh") — kept in sync by the toggle buttons.
    cur_lang = gr.State(DEFAULT_LANG)

    # ── Header bar: title + EN / 中文 pill toggle buttons ──
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
        f"*{I18N[DEFAULT_LANG]['subtitle']}*",
        elem_classes=["intro-copy"],
    )

    # ── Parameter card (single column) ──
    with gr.Column(elem_classes=["card-panel"]):
        # Genome row — splits into Genome | Upload FASTA | Upload GFF when Custom
        with gr.Row(equal_height=True, elem_classes=["layout-row-3sub"]):
            with gr.Column(scale=3, min_width=0, elem_id="genome-col") as genome_col:
                genome_dd = gr.Dropdown(
                    choices=_genome_options_all(),
                    value=DEFAULT_GENOME if DEFAULT_GENOME else None,
                    label=I18N[DEFAULT_LANG]["genome"], interactive=True,
                )
            with gr.Column(visible=False, scale=1, min_width=0, elem_id="fasta-col") as fasta_col:
                genome_fasta_upload = gr.Textbox(
                    value="",
                    label=I18N[DEFAULT_LANG]["upload_genome_fasta"],
                    placeholder=I18N[DEFAULT_LANG]["upload_fasta_ph"],
                    interactive=True,
                    elem_id="genome-upload",
                )
            with gr.Column(visible=False, scale=1, min_width=0, elem_id="gff-col") as gff_col:
                gff_upload = gr.Textbox(
                    value="",
                    label=I18N[DEFAULT_LANG]["upload_gff"],
                    placeholder=I18N[DEFAULT_LANG]["upload_gff_ph"],
                    interactive=True,
                    elem_id="gff-upload",
                )
        # Upload status — full width, shown only when a custom genome is active
        upload_status_md = gr.Markdown(
            value=_fmt_upload_status(I18N[DEFAULT_LANG], "idle"),
            visible=False,
        )
        # Chromosome | Start
        with gr.Row(equal_height=True):
            chromosome_dd = gr.Dropdown(
                choices=CHROMOSOME_OPTIONS, value=None,
                label=I18N[DEFAULT_LANG]["chromosome"], interactive=True, scale=2,
            )
            start_input = gr.Number(
                value=None, label=I18N[DEFAULT_LANG]["start"], precision=0,
                interactive=True, scale=1, placeholder="e.g. 10000",
            )
        # ATAC: built-in dropdown + custom bigWig path (same row)
        with gr.Row(equal_height=True):
            atac_dd = gr.Dropdown(
                choices=_atac_choices_for_genome(DEFAULT_GENOME) if DEFAULT_GENOME else [],
                value=None,
                label=I18N[DEFAULT_LANG]["atac_file"],
                interactive=True,
                scale=2,
            )
            atac_upload = gr.Textbox(
                value="",
                label=I18N[DEFAULT_LANG]["atac_upload_label"],
                placeholder=I18N[DEFAULT_LANG]["atac_path_ph"],
                interactive=True,
                visible=False,
                scale=1,
                elem_classes=["atac-upload-btn"],
            )
        api_key_input = gr.Textbox(
            value="",
            label=I18N[DEFAULT_LANG]["api_key"],
            placeholder=I18N[DEFAULT_LANG]["api_key_ph"],
            interactive=True,
            show_label=True,
        )
        predict_btn = gr.Button(
            I18N[DEFAULT_LANG]["predict"], variant="primary", elem_id="predict-btn",
        )

    # ── IGV full width ──
    igv_html_component = gr.HTML(
        value=f"<p style='color:#6B7280;'>{I18N[DEFAULT_LANG]['placeholder']}</p>"
    )

    # ── Event wiring ──
    genome_dd.change(
        fn=_on_genome_change,
        inputs=[genome_dd, cur_lang],
        outputs=[genome_col, fasta_col, gff_col, genome_fasta_upload, gff_upload, upload_status_md, chromosome_dd, atac_dd],
        queue=False,
    )
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
    atac_dd.select(
        fn=_on_atac_select,
        inputs=[atac_dd],
        outputs=[atac_upload],
    )
    btn_lang_en.click(
        fn=_on_lang_en,
        inputs=[genome_dd],
        outputs=[
            header_md,
            intro_md,
            genome_dd,
            chromosome_dd,
            start_input,
            atac_dd,
            atac_upload,
            api_key_input,
            predict_btn,
            genome_fasta_upload,
            gff_upload,
            btn_lang_en,
            btn_lang_zh,
            igv_html_component,
            cur_lang,
        ],
        queue=False,
    )
    btn_lang_zh.click(
        fn=_on_lang_zh,
        inputs=[genome_dd],
        outputs=[
            header_md,
            intro_md,
            genome_dd,
            chromosome_dd,
            start_input,
            atac_dd,
            atac_upload,
            api_key_input,
            predict_btn,
            genome_fasta_upload,
            gff_upload,
            btn_lang_en,
            btn_lang_zh,
            igv_html_component,
            cur_lang,
        ],
        queue=False,
    )
    predict_btn.click(
        fn=_on_predict,
        inputs=[genome_dd, chromosome_dd, start_input, atac_dd, atac_upload, cur_lang, api_key_input],
        outputs=[igv_html_component],
    )


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name=FRONTEND_HOST,
        server_port=FRONTEND_PORT,
        share=False,
        # Don't block: we need to mount the /backend reverse proxy on the
        # running FastAPI app before blocking on the server thread below.
        prevent_thread_lock=True,
    )

    # ------------------------------------------------------------------
    # Same-origin reverse proxy:  /backend/*  ->  backend (port 7001)
    # ------------------------------------------------------------------
    # The IGV iframe and its tracks (bigWig / GFF / FASTA) load resources
    # through the browser.  Pointing those at the raw backend port
    # (BACKEND_API_URL) breaks whenever the browser cannot reach that address
    # (e.g. only the frontend port is exposed / forwarded, or BACKEND_API_URL
    # uses 127.0.0.1 which resolves to the user's machine, not the server).
    # Proxying through the frontend means the browser only talks to the
    # frontend port and the frontend reaches the backend locally.
    import httpx
    from fastapi import Request
    from fastapi.responses import StreamingResponse

    _BACKEND_UPSTREAM = BACKEND_API_URL.rstrip("/")
    _SKIP_REQ_HEADERS = {"host", "content-length", "connection"}
    _SKIP_RESP_HEADERS = {"content-length", "transfer-encoding", "connection"}

    @demo.app.api_route(
        "/backend/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )
    async def _backend_proxy(path: str, request: Request):
        url = f"{_BACKEND_UPSTREAM}/{path}"
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _SKIP_REQ_HEADERS
        }
        body = await request.body()
        async with httpx.AsyncClient(timeout=600) as client:
            upstream = await client.request(
                request.method,
                url,
                params=request.query_params,
                headers=headers,
                content=body,
            )
        resp_headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in _SKIP_RESP_HEADERS
        }
        return StreamingResponse(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=resp_headers,
        )

    demo.block_thread()
