#!/bin/bash
# CCS-N 评测一键运行: 采样 -> GVL 写入 -> 双模型评测 (yaml 配置驱动)
# 用法: bash run_eval.sh [config/eval.yaml]
set -e
cd "$(dirname "$0")"

PY=/root/miniconda3/envs/vllm/bin/python
CONFIG=${1:-config/eval.yaml}

echo "=== 使用配置: $CONFIG ==="
# -u: 关闭 Python 输出缓冲, 保证日志实时可见
$PY -u scripts/run_eval.py "$CONFIG"
