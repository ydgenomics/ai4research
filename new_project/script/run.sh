#!/usr/bin/env bash
set -euo pipefail
# ======================================================================
# 统一启动入口（唯一脚本）：配置驱动，一个入口覆盖 data|model|all + 可选 smoke
#
# 用法:
#   bash run.sh <config.yaml> [all|data|model] [smoke]
#
# 示例:
#   bash run.sh configs/test_8k.yaml all smoke    # 8k 全流程: 数据预处理 + 冒烟(不存权重)
#   bash run.sh configs/test_8k.yaml data         # 只数据预处理(CPU, 不需 GPU)
#   bash run.sh configs/test_8k.yaml model smoke  # 只模型冒烟(GPU)
#   bash run.sh configs/train_32k.yaml data       # 正式数据预处理(CPU, 可放后台)
#   bash run.sh configs/train_32k.yaml model smoke  # 正式冒烟定 lr(C5)
#   bash run.sh configs/train_32k.yaml model      # 正式训练(GPU)
#
# 所有配置(窗口/样本/region/gpu/超参…)都在 yaml 里，无参数/=依赖
# ======================================================================
cd "$(dirname "$0")"

PY="${PYTHON:-/root/miniconda3/envs/vllm/bin/python}"
CONFIG="${1:?用法: bash run.sh <config.yaml> [all|data|model] [smoke]}"
MODE="${2:-all}"
EXTRA="${3:-}"

# ---- 参数校验 ----
case "$MODE" in
  all|data|model) ;;
  *) echo "!! MODE 必须是 all|data|model，收到: $MODE" >&2; exit 1 ;;
esac
if [ -n "$EXTRA" ] && [ "$EXTRA" != "smoke" ]; then
  echo "!! 第三参数只支持 smoke，收到: $EXTRA" >&2; exit 1
fi

# ---- 环境 ----
export GVL_NUM_THREADS="${GVL_NUM_THREADS:-8}"
# bcftools/samtools/tabix(数据预处理需要)所在 tools 环境
TOOLS=/mnt/rice/default/Workspace/xuxiaolong/mamba/envs/tools/bin
[ -d "$TOOLS" ] && export PATH="$TOOLS:$PATH"

# ---- 阶段函数 ----
data_stage() {
  echo "===== [data·CPU] 数据预处理: $CONFIG ($(date +%H:%M:%S)) ====="
  "$PY" ./src/main.py --config "$CONFIG" --stage data_preprocess
}

model_stage() {
  GPUS=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['compute']['gpus'])")
  NGPU=$(printf '%s' "$GPUS" | tr ',' '\n' | awk 'NF{n++} END{print n+0}')
  export CUDA_VISIBLE_DEVICES="$GPUS"
  STAGE=train; [ "$EXTRA" = "smoke" ] && STAGE=smoke
  echo "===== [model·GPU×$NGPU] $STAGE: $CONFIG (CUDA=$GPUS, $(date +%H:%M:%S)) ====="
  # 用 vllm 环境 python 启动 torch.distributed.run（避免解析到系统 base python3.13）
  "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" \
    ./src/main.py --config "$CONFIG" --stage "$STAGE"
}

# ---- 分发 ----
case "$MODE" in
  data)  data_stage ;;
  model) model_stage ;;
  all)   data_stage; model_stage ;;
esac
echo "===== 完成 ($(date +%H:%M:%S)) ====="