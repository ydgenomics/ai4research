"""IGV payload builder — constructs IGV.js-compatible track configurations.

References:
- rice-reg-server2/backend/rice_reg/igv_payload.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Base URL for static file serving — set by api.py at startup
_STATIC_BASE_URL: str = ""


def set_static_base_url(url: str):
    """Set the base URL for static file serving (called at startup)."""
    global _STATIC_BASE_URL
    _STATIC_BASE_URL = url.rstrip("/")


def _track(
    name: str,
    url: str,
    color: str,
    height: int = 90,
) -> Dict[str, Any]:
    return {
        "name": name,
        "url": _path_to_url(url),
        "type": "wig",
        "format": "bigwig",
        "color": color,
        "height": height,
        "autoscale": True,
    }


# Genos-Mutation style palette: gray = reference, other colors = predictions.
_TRACK_PALETTE = [
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]
_REF_TRACK_COLOR = "#6b7280"


def build_prediction_payload(
    genome: str,
    chromosome: str,
    start: int,
    end: int,
    genome_config: dict,
    track_paths: Optional[Dict[str, str]] = None,
    *,
    ref_track_paths: Optional[Dict[str, str]] = None,
    mut_track_paths: Optional[Dict[str, str]] = None,
    ref_label_fmt: str = "{bios} {assay} result1 (ref)",
    mut_label_fmt: str = "{bios} {assay} result2 (mut)",
) -> Dict[str, Any]:
    """Build an IGV.js payload for prediction tracks.

    Args:
        track_paths: single-prediction case — mapping ``"assay|biosample" -> path``.
        ref_track_paths / mut_track_paths: dual-prediction (mutation) case —
            mapping ``"assay|biosample" -> path`` for each side.
        ref_label_fmt / mut_label_fmt: track label templates with ``{assay}``
            and ``{bios}`` placeholders (dual-track case).

    Returns dict with keys ``reference``, ``locus``, ``tracks``.
    """
    # --- Reference genome ---
    reference = {
        "id": genome,
        "name": genome,
        "fastaURL": _path_to_url(genome_config["fasta"]),
        "indexURL": _path_to_url(genome_config["fai"]) if genome_config.get("fai") else "",
        "tracks": [],
    }

    gff = genome_config.get("gff")
    if gff and os.path.isfile(gff):
        # Auto-detect annotation format so uploaded GTF files also render.
        _gff_lower = gff.lower()
        _gff_fmt = "gtf" if (_gff_lower.endswith(".gtf") or _gff_lower.endswith(".gtf.gz")) else "gff3"
        reference["tracks"].append(
            {
                "name": "Genes",
                "url": _path_to_url(gff),
                "type": "annotation",
                "format": _gff_fmt,
                "displayMode": "EXPANDED",
                "visibilityWindow": 1000000,
            }
        )

    # IGV locus uses 1-based inclusive coordinates (matching the web UI input).
    # `start`/`end` here are internal 0-based (half-open), so convert start+1.
    locus = f"{chromosome}:{start + 1:,}-{end:,}"
    tracks: List[Dict[str, Any]] = []

    if ref_track_paths is not None and mut_track_paths is not None:
        # --- Dual-track (SNV) mode: result1 (ref, gray) vs result2 (mut, colored) ---
        # Matches Genos-Mutation style: gray = reference, other colors = predictions.
        for key, path in ref_track_paths.items():
            assay, bios, _tag = key.split("|")
            tracks.append(_track(
                ref_label_fmt.format(assay=assay, bios=bios), path, _REF_TRACK_COLOR
            ))
        for i, (key, path) in enumerate(mut_track_paths.items()):
            assay, bios, _tag = key.split("|")
            color = _TRACK_PALETTE[i % len(_TRACK_PALETTE)]
            tracks.append(_track(
                mut_label_fmt.format(assay=assay, bios=bios), path, color
            ))
    else:
        # --- Single prediction mode ---
        for i, (key, path) in enumerate(track_paths.items()):
            assay, bios, _tag = key.split("|")
            color = _TRACK_PALETTE[i % len(_TRACK_PALETTE)]
            tracks.append(_track(f"{bios} {assay}", path, color))

    return {
        "reference": reference,
        "locus": locus,
        "tracks": tracks,
    }


def _path_to_url(path: str) -> str:
    """Convert a local file path to an HTTP URL via the static file server."""
    if _STATIC_BASE_URL:
        # Normalize to absolute path (relative cache paths are common in .env)
        rel = os.path.abspath(path).lstrip("/")
        return f"{_STATIC_BASE_URL}/{rel}"
    if path.startswith("file://"):
        return path
    return f"file://{path}"
