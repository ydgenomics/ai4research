"""Frontend configuration — reads genome and ATAC config from environment variables.

References:
- genos-reg-server/frontend/config.py
"""

import os


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
#  Genome configurations
# ---------------------------------------------------------------------------
def _build_genome_configs() -> dict:
    """Scan environment for GENOME_<ID>_FASTA / _FAI / _GFF variables."""
    configs = {}
    for key, val in sorted(os.environ.items()):
        if key.startswith("GENOME_") and key.endswith("_FASTA"):
            genome_id = key[len("GENOME_"):-len("_FASTA")]
            fasta = val
            fai = _env_str(f"GENOME_{genome_id}_FAI", "")
            gff = _env_str(f"GENOME_{genome_id}_GFF", "")
            configs[genome_id] = {
                "fasta": fasta,
                "fai": fai,
                "gff": gff,
            }
    return configs


def _build_atac_signal_paths() -> dict:
    """Scan environment for ATAC_PATH_<ID> variables."""
    paths = {}
    for key, val in sorted(os.environ.items()):
        if key.startswith("ATAC_PATH_"):
            atac_id = key[len("ATAC_PATH_"):]
            if val:
                paths[atac_id] = val
    return paths


def _build_genome_atac_map(genome_configs: dict, atac_signal_paths: dict) -> dict:
    """Map each genome to its compatible ATAC options.

    Priority:
      1. Explicit ``ATAC_GENOME_MAP_<GENOME>=<ATAC_ID1>,<ATAC_ID2>`` env mapping.
      2. Fallback: all built-in ATAC sources (if no mapping is defined), so a
         genome without a mapping still works.

    This keeps the ATAC source consistent with the selected genome version.
    """
    mapping = {}
    for genome_id in genome_configs:
        env_key = f"ATAC_GENOME_MAP_{genome_id}"
        raw = _env_str(env_key, "")
        ids = []
        if raw:
            ids = [
                x.strip()
                for x in raw.split(",")
                if x.strip() and x.strip() in atac_signal_paths
            ]
        # Fallback: all built-in ATAC sources for backwards compatibility
        mapping[genome_id] = ids if ids else list(atac_signal_paths.keys())
    return mapping


GENOME_CONFIGS = _build_genome_configs()
ATAC_SIGNAL_PATHS = _build_atac_signal_paths()
GENOME_ATAC_MAP = _build_genome_atac_map(GENOME_CONFIGS, ATAC_SIGNAL_PATHS)

# Chromosome options — unified front-end naming is chr01..chr12.
# The backend normalizes any alias (chr01 / Chr1 / 1 ...) to the real FASTA name.
CHROMOSOME_OPTIONS = [f"chr{i:02d}" for i in range(1, 13)]

# Default genome (first available)
DEFAULT_GENOME = next(iter(GENOME_CONFIGS.keys())) if GENOME_CONFIGS else ""

# Backend API URL
BACKEND_API_URL = _env_str("BACKEND_API_URL", "http://127.0.0.1:7001")

# IGV.js settings
IGV_VERSION = "2.16.2"
IGV_CDN_URL = f"https://cdn.jsdelivr.net/npm/igv@{IGV_VERSION}/dist/igv.min.js"
