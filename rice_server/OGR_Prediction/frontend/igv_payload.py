"""Frontend IGV payload helpers — builds default reference and track configs.

References:
- rice-reg-server2/frontend/igv_payload.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Backend static file base URL — set from config at import time
_STATIC_BASE_URL: str = ""


def set_static_base_url(url: str):
    """Set the base URL for backend static file serving."""
    global _STATIC_BASE_URL
    _STATIC_BASE_URL = url.rstrip("/")


def build_default_prediction_reference(
    genome: str,
    genome_configs: dict,
) -> Optional[Dict[str, Any]]:
    """Build the IGV reference genome config for a given genome ID.

    Returns ``None`` if the genome is not found.
    """
    cfg = genome_configs.get(genome)
    if not cfg:
        return None

    ref = {
        "id": genome,
        "name": genome,
        "fastaURL": _path_to_url(cfg["fasta"]),
        "indexURL": _path_to_url(cfg["fai"]) if cfg.get("fai") else "",
        "tracks": [],
    }

    gff = cfg.get("gff")
    if gff and os.path.isfile(gff):
        ref["tracks"].append(
            {
                "name": "Genes",
                "url": _path_to_url(gff),
                "type": "annotation",
                "format": "gff3",
                "displayMode": "EXPANDED",
                "visibilityWindow": 1000000,
            }
        )

    return ref


def _path_to_url(path: str) -> str:
    """Convert a local file path to an HTTP URL via the backend static file server."""
    if _STATIC_BASE_URL:
        rel = path.lstrip("/")
        return f"{_STATIC_BASE_URL}/{rel}"
    # Fallback: file:// (won't work in browser IGV)
    if path.startswith("file://"):
        return path
    return f"file://{path}"
