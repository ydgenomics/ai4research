"""Rice-Introgression backend entry point.

Loads .env and starts uvicorn.  The backend package (rice_introgression) is
imported after adding backend/ to sys.path.
"""

import os
import sys
from pathlib import Path

import uvicorn


def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


ROOT_DIR = Path(__file__).resolve().parents[2]      # rice-intro/
BACKEND_DIR = Path(__file__).resolve().parents[1]    # rice-intro/backend/
_load_env_file(ROOT_DIR / ".env")

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rice_introgression.api import app  # noqa: E402

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "5001")))

if __name__ == "__main__":
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, reload=False)