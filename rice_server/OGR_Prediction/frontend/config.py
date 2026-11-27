"""Frontend configuration — reads genome and model config from environment variables.

References:
- rice-reg-server2/frontend/config.py
"""

import os


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


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
            configs[genome_id] = {"fasta": fasta, "fai": fai, "gff": gff}
    return configs


GENOME_CONFIGS = _build_genome_configs()

# Chromosome options — unified front-end naming is chr01..chr12.
# The backend normalizes any alias (chr01 / Chr1 / 1 ...) to the real FASTA name.
CHROMOSOME_OPTIONS = [f"chr{i:02d}" for i in range(1, 13)]

# Default genome (first available)
DEFAULT_GENOME = next(iter(GENOME_CONFIGS.keys())) if GENOME_CONFIGS else ""

# Backend API URL
BACKEND_API_URL = _env_str("BACKEND_API_URL", "http://127.0.0.1:8001")

# NOTE: the backend's DCS_API_KEY (.env) is used for server-side validation
# only.  The frontend does NOT read it — API keys come from the UI input box
# at runtime (see frontend/app.py `_call_backend_api`), so an empty value
# here simply means "no default key" for the web frontend.
DCS_API_KEY = _env_str("DCS_API_KEY", "")

# IGV.js settings
IGV_VERSION = "2.16.2"
IGV_CDN_URL = f"https://cdn.jsdelivr.net/npm/igv@{IGV_VERSION}/dist/igv.min.js"
