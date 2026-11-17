"""IGV payload builder — constructs IGV.js-compatible track configurations.

References:
- genos-reg-server/backend/genos_reg/igv_payload.py
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


def build_prediction_payload(
    genome: str,
    chromosome: str,
    start: int,
    end: int,
    atac_path: str,
    plus_bw_path: str,
    minus_bw_path: str,
    genome_config: dict,
) -> Dict[str, Any]:
    """Build an IGV.js payload dict for displaying prediction results.

    Returns a dict with keys:
    - ``reference``: IGV reference genome config.
    - ``locus``: genomic locus string.
    - ``tracks``: list of IGV track configs (ATAC, predicted RNA-seq + and -).
    """
    # --- Reference genome ---
    reference = {
        "id": genome,
        "name": genome,
        "fastaURL": _path_to_url(genome_config["fasta"]),
        "indexURL": _path_to_url(genome_config["fai"]) if genome_config.get("fai") else "",
        "tracks": [],
    }

    # Add annotation track if GFF available
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

    # --- Locus ---
    # IGV locus uses 1-based inclusive coordinates (matching the web UI input).
    # `start`/`end` here are internal 0-based (half-open), so use start + 1.
    locus = f"{chromosome}:{start + 1:,}-{end:,}"

    # --- Tracks ---
    tracks: List[Dict[str, Any]] = []

    # ATAC signal track
    if atac_path and os.path.isfile(atac_path):
        tracks.append(
            {
                "name": "ATAC-seq",
                "url": _path_to_url(atac_path),
                "type": "wig",
                "format": "bigwig",
                "color": "#1f77b4",
                "height": 60,
                "autoscale": True,
            }
        )

    # Predicted RNA-seq (+)
    if os.path.isfile(plus_bw_path):
        tracks.append(
            {
                "name": "Predicted RNA-seq (+)",
                "url": _path_to_url(plus_bw_path),
                "type": "wig",
                "format": "bigwig",
                "color": "#d62728",
                "height": 60,
                "autoscale": True,
            }
        )

    # Predicted RNA-seq (-)
    if os.path.isfile(minus_bw_path):
        tracks.append(
            {
                "name": "Predicted RNA-seq (-)",
                "url": _path_to_url(minus_bw_path),
                "type": "wig",
                "format": "bigwig",
                "color": "#2ca02c",
                "height": 60,
                "autoscale": True,
            }
        )

    return {
        "reference": reference,
        "locus": locus,
        "tracks": tracks,
    }


def _path_to_url(path: str) -> str:
    """Convert a local file path to an HTTP URL via the static file server."""
    if _STATIC_BASE_URL:
        # Strip leading slash and construct HTTP URL
        rel = path.lstrip("/")
        return f"{_STATIC_BASE_URL}/{rel}"
    # Fallback: file:// (won't work in browser IGV, but keeps tests functional)
    if path.startswith("file://"):
        return path
    return f"file://{path}"
