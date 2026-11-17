"""Figure rendering service — matplotlib figures from prediction results.

Reuses the existing ``ResultsViewer`` (backend/src/viewer.py) to render
publication-quality figures with gene annotations and signal tracks.

Supported output formats (``fmt`` argument):

- ``png``  — raster image (default)
- ``svg``  — vector image
- ``html`` — self-contained HTML page with the SVG figure embedded
             (openable / printable / saveable in a browser)

- ``/predict``          → single prediction (reference) tracks
- ``/predict/mutation`` → reference vs mutant overlaid (plot2: blue solid ref,
                          red dashed mut)

Uses the headless ``Agg`` matplotlib backend (no display required).
"""

from __future__ import annotations

import logging
import os
import re
import time
from io import StringIO
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.viewer import ResultsViewer  # noqa: E402

logger = logging.getLogger(__name__)

_PLOTTER: Dict[str, Any] = {"viewer": None, "annotation_path": None}

# Chromosome / track names may contain chars that are unsafe in file names
_SAFE_RE = re.compile(r"[^0-9A-Za-z_.-]")

SUPPORTED_FORMATS = ("png", "svg", "html")


def _get_viewer(annotation_path: str) -> ResultsViewer:
    """Return a cached ResultsViewer (GFF loaded once per path)."""
    if _PLOTTER["viewer"] is None or _PLOTTER["annotation_path"] != annotation_path:
        logger.info("Loading gene annotation for figure rendering: %s", annotation_path)
        viewer = ResultsViewer(
            annotation_path=annotation_path,
            xtick_step=4000,
            dpi=110,
            max_region_length=200_000,
        )
        _PLOTTER["viewer"] = viewer
        _PLOTTER["annotation_path"] = annotation_path
    return _PLOTTER["viewer"]


def _safe_name(name: str) -> str:
    return _SAFE_RE.sub("_", str(name))


def render_prediction_figure(
    ref_values: Dict[str, Dict[str, Any]],
    chromosome: str,
    start: int,
    end: int,
    annotation_path: str,
    out_dir: str,
    mut_values: Optional[Dict[str, Dict[str, Any]]] = None,
    fmt: str = "png",
) -> str:
    """Render prediction tracks to a figure file (png / svg / html).

    Args:
        ref_values: ``{assay: {biosample: np.ndarray[L]}}`` (reference).
        chromosome / start / end: genomic window (from prediction position).
        annotation_path: GFF path for gene track (may be empty to skip genes).
        out_dir: directory for the output file (static-file served).
        mut_values: optional mutant values overlaid (dual-track mode).
        fmt: one of ``png`` / ``svg`` / ``html``.

    Returns:
        Absolute path to the generated file, or ``""`` on failure.
    """
    fmt = (fmt or "png").lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        fmt = "png"

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not ref_values:
        logger.warning("render_prediction_figure: empty ref_values")
        return ""

    ref_results = {"values": ref_values, "position": (chromosome, start, end)}
    mut_results = None
    if mut_values:
        mut_results = {"values": mut_values, "position": (chromosome, start, end)}

    tag = "mut" if mut_results else "ref"
    ext = "html" if fmt == "html" else fmt
    fname = f"{_safe_name(chromosome)}_{start}_{end}_{tag}_{int(time.time())}.{ext}"
    out_path = os.path.join(out_dir, fname)

    try:
        if mut_results is not None:
            fig, axes = viewer_plot2(
                _get_viewer(annotation_path) if annotation_path else None,
                ref_results,
                mut_results,
            )
        else:
            fig, axes = viewer_plot(
                _get_viewer(annotation_path) if annotation_path else None,
                ref_results,
            )
        if fig is None:
            return ""

        if fmt == "html":
            html = _fig_to_html(
                fig, chromosome, start, end,
                mut=mut_results is not None,
            )
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(html)
        else:
            fig.savefig(out_path, dpi=110, format=fmt, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: %s (fmt=%s)", out_path, fmt)
        return out_path
    except Exception as e:
        logger.error("Figure rendering failed: %s", e, exc_info=True)
        return ""


def viewer_plot(viewer: Optional[ResultsViewer], results: Dict[str, Any]):
    """Single-result plot; returns (fig, axes) or (None, None) on failure."""
    if viewer is None:
        return _fallback_plot(results, None)
    try:
        return viewer.plot(results, smoothing_sigma=0, show_legend=True)
    except Exception as e:
        logger.error("viewer.plot failed: %s", e, exc_info=True)
        return _fallback_plot(results, viewer)


def viewer_plot2(
    viewer: Optional[ResultsViewer],
    results: Dict[str, Any],
    results2: Dict[str, Any],
):
    """Dual-result overlay plot (ref vs mut); returns (fig, axes) or (None, None)."""
    if viewer is None:
        return _fallback_plot(results, results2)
    try:
        return viewer.plot2(
            results,
            results2=results2,
            smoothing_sigma=0,
            show_legend=True,
        )
    except Exception as e:
        logger.error("viewer.plot2 failed: %s", e, exc_info=True)
        return _fallback_plot(results, results2)


def _fig_to_svg(fig) -> str:
    """Render a matplotlib figure to a standalone SVG string."""
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    # Drop the XML / DOCTYPE preamble so the SVG can be embedded directly
    start = svg.find("<svg")
    if start == -1:
        return svg
    return svg[start:]


def _fig_to_html(
    fig,
    chromosome: str,
    start: int,
    end: int,
    mut: bool = False,
) -> str:
    """Wrap the figure's SVG into a self-contained HTML page."""
    svg = _fig_to_svg(fig)
    mode = "Reference vs Mutant" if mut else "Reference"
    title = f"Rice-Mutation Prediction — {chromosome}:{start:,}-{end:,} ({mode})"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }}
  h3 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .figure {{ max-width: 100%; overflow-x: auto; }}
  .figure svg {{ max-width: 100%; height: auto; }}
  .note {{ color: #888; font-size: 12px; margin-top: 16px; }}
</style>
</head>
<body>
<h3>{title}</h3>
<div class="meta">{chromosome}:{start:,}-{end:,} &nbsp;·&nbsp; {end - start:,} bp &nbsp;·&nbsp; mode: {mode}</div>
<div class="figure">{svg}</div>
<div class="note">Generated by Rice-Mutation — DNA → multi-omics expression prediction. Use your browser's “Save as…” (Ctrl/Cmd+S) to keep this page.</div>
</body>
</html>
"""


def _fallback_plot(results: Dict[str, Any], results2: Optional[Dict[str, Any]]):
    """Minimal fallback plot (no gene track) — used when GFF is missing/erroneous.

    Draws one subplot per (assay, biosample), overlaying results2 in red dashed.
    """
    values = results.get("values", {})
    position = results.get("position", (None, None, None))
    chrom, start, end = position if len(position) == 3 else (None, None, None)
    if not values or chrom is None:
        return None, None

    values2 = (results2 or {}).get("values", {}) if results2 else {}

    track_names = list(values.keys())
    biosamples = []
    for tn in track_names:
        for b in values[tn]:
            if b not in biosamples:
                biosamples.append(b)

    n_sig = len(track_names) * max(1, len(biosamples))
    fig, axes = plt.subplots(n_sig + 1, 1, figsize=(16, 1.5 * (n_sig + 1)), sharex=True,
                             dpi=110, gridspec_kw={"height_ratios": [0.5] + [1.0] * n_sig})
    if n_sig + 1 == 1:
        axes = [axes]

    ax_gene = axes[0]
    ax_gene.text(0.5, 0.5, "Genes (GFF unavailable)", ha="center", va="center",
                 transform=ax_gene.transAxes, fontsize=9, style="italic")
    ax_gene.set_yticks([])
    ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha="right", va="center")

    # Signal length is taken from the first available track (all tracks share
    # the same prediction window).  Positions are aligned to the array length
    # (single-base), NOT to the bp window, so shorter arrays (e.g. windows
    # clipped at the chromosome end) do not cause a length-mismatch ValueError.
    n_sig_pts = None
    for tn in track_names:
        for b in biosamples:
            arr = values.get(tn, {}).get(b)
            if arr is not None:
                n_sig_pts = len(_to_numpy_1d(arr))
                break
        if n_sig_pts is not None:
            break

    # Downsample very long windows so fallback rendering stays fast.
    # (Normal 32 kb windows stay at full resolution.)
    max_points = 60000
    if n_sig_pts is not None and n_sig_pts > max_points:
        sample_idx = np.linspace(0, n_sig_pts - 1, max_points).astype(int)
        positions = start + sample_idx
    else:
        sample_idx = None
        positions = start + np.arange(n_sig_pts) if n_sig_pts else np.arange(start, end)

    def _sample(yy):
        if yy is None or sample_idx is None:
            return yy
        return yy[sample_idx]

    idx = 1
    for tn in track_names:
        for b in biosamples:
            ax = axes[idx]
            arr = values.get(tn, {}).get(b)
            arr2 = values2.get(tn, {}).get(b) if values2 else None
            y = y2 = None
            if arr is not None:
                y = _sample(_to_numpy_1d(arr))
                ax.plot(positions, y, color="tab:blue", linewidth=1.2)
                ax.fill_between(positions, 0, y, color="tab:blue", alpha=0.15)
            if arr2 is not None:
                y2 = _sample(_to_numpy_1d(arr2))
                ax.plot(positions, y2, color="tab:red", linewidth=1.2, linestyle="--")
            y_max = 0.1
            for yy in (y, y2):
                if yy is not None and yy.size:
                    y_max = max(y_max, float(np.nanmax(yy)) * 1.15)
            ax.set_ylim(0, y_max)
            ax.set_ylabel(f"{tn}\n{b}", fontsize=9, rotation=0, ha="right", va="center")
            ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.4)
            if y is not None and y2 is not None:
                ax.legend(
                    handles=[
                        Line2D([0], [0], color="tab:blue", lw=1.5, label="ref"),
                        Line2D([0], [0], color="tab:red", lw=1.5, ls="--", label="mut"),
                    ],
                    loc="upper right", fontsize=8,
                )
            idx += 1

    for ax in axes:
        ax.set_xlim(start, end)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    axes[-1].set_xlabel(f"{chrom}:{start:,}-{end:,} ({end - start:,} bp)", fontsize=10)
    plt.tight_layout()
    return fig, axes


def _to_numpy_1d(x: Any):
    """Convert array-like to a 1-D numpy array (CPU, float32)."""
    if hasattr(x, "detach"):
        t = x.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        if hasattr(t, "float"):
            t = t.float()
        if hasattr(t, "numpy"):
            x = t.numpy()
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr
