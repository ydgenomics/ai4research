"""Rice-Reg core: model inference logic ported from inference2.ipynb."""

import os
import sys
from pathlib import Path

# 将 core/ 加入 sys.path，使 from model.config import ... 可工作
_core_dir = Path(__file__).resolve().parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

from .rice_reg import RiceRegPredictor

__all__ = ["RiceRegPredictor"]
