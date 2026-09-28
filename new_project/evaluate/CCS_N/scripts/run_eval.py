"""CCS-N 评测统一入口: 采样 -> GVL 写入 -> 模型评测, 全部由 yaml 配置驱动。

用法:
    python scripts/run_eval.py config/eval.yaml
    bash run_eval.sh config/eval.yaml

配置字段见 config/eval.yaml (数据/采样/GVL/评测 四个块)。
所有路径相对本文件所在项目的根目录 (evaluate/CCS_N)。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_data  # noqa: E402
import sample_carriers  # noqa: E402
from run_ccs_n import main as run_ccs_n_main  # noqa: E402


def resolve(p: str, root: Path = ROOT) -> Path:
    """相对路径相对项目根目录解析, 绝对路径原样返回。"""
    p = Path(p).expanduser()
    return p if p.is_absolute() else root / p


def step_sampling(cfg: dict):
    """步骤1: 采样携带变异的位点生成窗口 BED。"""
    s = cfg.get("sampling", {})
    if not s.get("enabled", True):
        print("[run_eval] 采样已禁用 (sampling.enabled=false), 跳过")
        return
    bed = resolve(s["bed"])
    if bed.exists():
        print(f"[run_eval] BED 已存在, 跳过采样: {bed}")
        return
    vcf = resolve(cfg["data"]["vcf"])
    sample_carriers.build_carrier_bed(
        vcf_path=vcf,
        bed_path=bed,
        sample=s["sample"],
        genotype=s.get("genotype", "hom"),
        window_size=s.get("window_size", 256),
        max_variants=s.get("max_variants"),
        seed=s.get("seed", 42),
    )


def step_gvl(cfg: dict):
    """步骤2: 用采样 BED + 过滤 VCF 写入 GVL 数据集。"""
    g = cfg.get("gvl", {})
    if g.get("skip", False):
        print("[run_eval] GVL 写入已跳过 (gvl.skip=true)")
        return
    gvl_dir = resolve(g["dir"])
    if (gvl_dir / "metadata.json").exists():
        print(f"[run_eval] GVL 数据集已存在, 跳过: {gvl_dir}")
        return
    vcf = resolve(cfg["data"]["vcf"])
    ref = resolve(cfg["data"]["ref"])
    bed = resolve(cfg["sampling"]["bed"])
    samples = [cfg["sampling"]["sample"]] if cfg["sampling"].get("sample") else None
    prepare_data.write_gvl_dataset(bed, vcf, ref, gvl_dir, samples)


def step_eval(cfg: dict):
    """步骤3: 运行 CCS-N 模型评测。"""
    e = cfg.get("eval", {})
    gvl_dir = resolve(cfg["gvl"]["dir"])
    out_dir = resolve(e["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = resolve(cfg["data"]["ref"])

    # 复用 run_ccs_n.main, 通过注入 args 的方式传参
    args = argparse.Namespace(
        model_dir=e["models"],
        data_dir=str(ROOT / cfg["data"]["out_dir"]),
        gvl_dir=str(gvl_dir),
        out_dir=str(out_dir),
        ref=str(ref),
        sample=e.get("sample", "NH001"),
        batch_size=e.get("batch_size", 16),
        device=e.get("device", "cuda:0"),
        max_regions=e.get("max_regions"),
        seed=e.get("seed", 42),
        n_list=tuple(e.get("n_list", (0, 1, 2, 4, 8, 16))),
        save_per_site=e.get("save_per_site", False),
        save_distance_profiles=e.get("save_distance_profiles", False),
    )
    run_ccs_n_main(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="yaml 配置文件路径 (相对项目根目录或绝对)")
    ap.add_argument("--step", choices=["sampling", "gvl", "eval"], default=None,
                    help="只运行指定步骤 (默认全部)")
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[run_eval] 加载配置: {cfg_path}")

    steps = [args.step] if args.step else ["sampling", "gvl", "eval"]
    if "sampling" in steps:
        step_sampling(cfg)
    if "gvl" in steps:
        step_gvl(cfg)
    if "eval" in steps:
        step_eval(cfg)

    print("[run_eval] 全部完成 ✅")


if __name__ == "__main__":
    main()
