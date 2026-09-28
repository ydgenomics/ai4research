# -*- coding: utf-8 -*-
"""水稻变异感知 CPT 微调 · 主调用脚本（GenVarLoader + OGR-HF + 变异聚焦 loss_mask）。

用法:
  # CPU 阶段（不需 GPU）：QC/norm → BED → gvl.write
  python main.py --config configs/train_32k.yaml --stage data_preprocess
  # 也可分步：preprocess（QC+norm+bed）/ write（gvl.write）
  python main.py --config configs/train_32k.yaml --stage preprocess
  python main.py --config configs/train_32k.yaml --stage write

  # GPU 阶段（模型训练）：要求 data_preprocess 已跑完（gvl 产物存在）
  python main.py --config configs/train_32k.yaml --stage train
  python main.py --config configs/train_32k.yaml --stage smoke --smoke-steps 20

  # 覆盖任意参数
  python main.py --config configs/train_32k.yaml --override window.length=65536
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _disable_deepspeed():
    """纯代码禁用 deepspeed 探测（不卸载/不改环境）。

    环境里装了 deepspeed 但无 CUDA toolkit（nvcc/CUDA_HOME），其 op_builder 在
    import 时抛 MissingCUDAException（非 ImportError，accelerate 捕获不到）。
    本训练不用 deepspeed，故在 import transformers 前让 importlib 视其不存在。
    """
    import importlib.util

    if getattr(_disable_deepspeed, "_done", False):
        return
    _orig = importlib.util.find_spec

    def _find_spec(name, package=None):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return _orig(name, package)

    importlib.util.find_spec = _find_spec
    _disable_deepspeed._done = True


_disable_deepspeed()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("variant_cpt")


def _norm_scalar(v):
    """把 YAML 误解析为 str 的科学计数法（如 '1e-5'）转回 float。

    只处理科学计数法形式（PyYAML 的坑：无小数点的 1e-5 会被解析成 str）；
    普通数字字符串（如 gpus "1"、max_mem "2g"、region "Chr1:.."）保持原样。
    """
    if isinstance(v, str):
        s = v.strip()
        # 科学计数法：^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$
        import re

        if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+", s):
            return float(s)
    return v


def _norm_cfg(node):
    if isinstance(node, dict):
        return {k: _norm_cfg(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_norm_cfg(v) for v in node]
    return _norm_scalar(node)


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def _set(d, dotted_key, value):
        keys = dotted_key.split(".")
        node = d
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    for ov in overrides or []:
        k, _, v = ov.partition("=")
        if v in ("true", "false"):
            v = v == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        _set(cfg, k, v)
    return _norm_cfg(cfg)


def _ensure_data_ready(cfg: dict):
    """训练前检查 gvl 数据产物是否已就绪（避免 GPU 进程空跑/误触 CPU 大任务）。"""
    from gvl_pipeline import _gvl_path
    gvl_file = _gvl_path(cfg)
    if not gvl_file.exists():
        logger.error(
            f"gvl 数据未就绪: {gvl_file}\n"
            f"  → 请先跑 CPU 阶段数据预处理（不需 GPU）:\n"
            f"      python main.py --config {cfg.get('_config_path', 'configs/*.yaml')} --stage data_preprocess"
        )
        raise SystemExit(1)
    logger.info(f"gvl 数据已就绪: {gvl_file}")


def main():
    ap = argparse.ArgumentParser(description="水稻变异感知 CPT 微调")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage",
                    choices=["preprocess", "write", "data_preprocess", "train", "smoke"],
                    default="train")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-write", action="store_true")
    ap.add_argument("--override", action="append", default=None)
    ap.add_argument("--smoke-steps", type=int, default=20)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    cfg = load_config(args.config, args.override)
    cfg["_config_path"] = args.config

    from gvl_pipeline import env_setup, run_preprocess, run_write
    from trainer_utils import run_smoke, run_training

    env_setup(cfg)

    if args.stage == "data_preprocess":
        # CPU 阶段：QC/norm → BED → gvl.write（全程不需要 GPU）
        run_preprocess(cfg)
        run_write(cfg)
        logger.info("data_preprocess 完成")
        return

    if args.stage in ("preprocess",) and not args.skip_preprocess:
        run_preprocess(cfg)
        logger.info("preprocess 完成")
        return
    if args.stage == "write" and not args.skip_write:
        run_write(cfg)
        logger.info("write 完成")
        return

    # GPU 阶段：训练 / 冒烟（数据必须已由 data_preprocess 生成）
    _ensure_data_ready(cfg)
    if args.stage == "train":
        run_training(cfg)
    elif args.stage == "smoke":
        run_smoke(cfg, max_steps=args.smoke_steps)


if __name__ == "__main__":
    main()