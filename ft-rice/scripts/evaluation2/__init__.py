"""
Gene Expression Prediction Evaluation v2 — 重构的统一评估框架

核心模块：
  - metrics_core: 三层级指标计算（Track/Segment/Feature 级）
  - utils: 数据加载与特征聚合
  - run_evaluation: 统一评估流程
"""

__version__ = "2.0.0"
__date__ = "2026-07-16"

from . import metrics_core
from . import utils

__all__ = [
    "metrics_core",
    "utils",
]
