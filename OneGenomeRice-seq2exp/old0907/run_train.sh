DEVICE=$1  # 可选: "cuda" 或 "mx"


if [[ "$DEVICE" == "mx" ]]; then
    # ==================== MX (MACA) 环境变量 ====================
    echo "🔧 设备模式: MX (MACA)"

    # MACA 路径与 CUDA 桥接
    export MACA_LAUNCH_BLOCKING=1
    export MACA_PATH=/opt/maca
    export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
    export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
    export DEVINFO_ROOT=${MACA_PATH}
    export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
    export CUDA_PATH=${CUCC_PATH}

    export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:$LD_LIBRARY_PATH

    # 通信与性能核心配置
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export MACA_SMALL_PAGESIZE_ENABLE=1
    export SET_DEVICE_NUMA_PREFERRED=1
    export MAX_JOBS=20
    export NVTE_FLASH_ATTN=1
    export NVTE_FUSED_ATTN=0

    # MCCL 通信配置 (替代 NCCL)
    export NCCL_ASYNC_ERROR_HANDLING=3
    export MCCL_SOCKET_IFNAME=bond0
    export GLOO_SOCKET_IFNAME=bond0
    export MCCL_IB_HCA=mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5
    export NCCL_IB_TIMEOUT=22
    export NCCL_COMM_ID_REUSE_TIMEOUT=600
    export NCCL_DEBUG=WARN
    export TORCH_DISTRIBUTED_DEBUG=INFO

    # 运行性能与显存管理
    export MACA_DIRECT_DISPATCH=1
    export PYTORCH_ENABLE_SAME_RAND_CONF=multiprocessor_count:132,maxthreads_per_multiprocessor:2048
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    export MCCL_MAX_NCHANNELS=16
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export MALLOC_THRESHOLD=99

    # # SwanLab 配置 (MX 模式下使用 SwanLab 替代 W&B)
    # # 注意: project/workspace 通过 swanlab.init() 参数传入, 不导出环境变量
    # # 否则 swanlab 的 pydantic_settings 会尝试 json.loads() 解析纯字符串字段导致崩溃
    # export SWANLAB_API_KEY="${SWANLAB_API_KEY:-GAlP3YYw5OCIMwfHysX5P}"

    # ==================== 依赖镜像源配置 ====================
    # 注意: 阿里云正确地址是 .../pypi/simple/ 而不是 .../simple/simple/ (重复 /simple 会 404 → from versions: none)
    PIP_INDEX_ALIYUN="https://mirrors.aliyun.com/pypi/simple/"
    # 阿里云 ECS 内网专属镜像(仅阿里云 VPC 内可达, 不走公网)
    PIP_INDEX_ALIYUN_INTERNAL="http://mirrors.cloud.aliyuncs.com/pypi/simple/"
    PIP_INDEX_TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
    ANTLR_VERSION="4.9.3"
    TRANS_VERSION="4.52.4"
    CORE_DEPS="pybigwig pyfaidx torchmetrics wandb swanlab transformers==$TRANS_VERSION"

    echo "🚀 [$(date +'%H:%M:%S')] 开始配置生物信息与深度学习环境..."

    # ==================== 排查①: pip/python 版本与源配置 ====================
    echo "----- [排查①] pip/python 版本与源配置 -----"
    python3 --version
    pip --version
    echo "[pip config list]"
    pip config list || true
    echo "[PIP/PROXY 相关环境变量]"
    env | grep -iE 'pip|proxy|index' | grep -viE '^_=|LS_COLORS' || echo "(无)"
    echo "---------------------------------------"

    # ==================== 排查②: 探测各镜像源的包页面 ====================
    # 原理: 报 (from versions: none) =
    #   index 页面"拿到了但页面里没有这个包"(404/空页面/被网关劫持), 而不是网络不通;
    #   网络不通通常会卡住或报 Could not reach the index URL
    probe_index() {
        local name="$1" url="$2" pkg="$3"
        echo "----- [排查②] 源[$name] 探测包: $pkg -----"
        python3 - "$url" "$pkg" <<'PY'
import sys, urllib.request, ssl, socket
url, pkg = sys.argv[1], sys.argv[2]
page = url.rstrip("/") + "/" + pkg + "/"
try:
    req = urllib.request.Request(page, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=10, context=ssl._create_unverified_context()) as r:
        body = r.read(65536).decode("utf-8", "ignore")
        print(f"  HTTP {r.status} | Content-Length: {r.headers.get('Content-Length')} | 页面含'{pkg}': {pkg in body}")
        print(f"  页面片段: {body[:120]!r}")
except Exception as e:
    print(f"  ❌ 无法访问: {type(e).__name__}: {e}")
PY
    }

    for src in "阿里云公网:$PIP_INDEX_ALIYUN" "阿里云内网:$PIP_INDEX_ALIYUN_INTERNAL" "清华:$PIP_INDEX_TUNA"; do
        name="${src%%:*}"; url="${src#*:}"
        probe_index "$name" "$url" "pybigwig"
        probe_index "$name" "$url" "antlr4-python3-runtime"
    done

    # ==================== 排查③: 对照实验 ====================
    # 装一个几乎所有镜像都有的包, 区分"源不可达"与"恰好缺这两个包"
    echo "----- [排查③] 对照实验: 安装 requests(人人皆有的包) -----"
    if pip install --no-cache-dir -q -i "$PIP_INDEX_ALIYUN" --root-user-action=ignore requests 2>/tmp/probe_requests.log; then
        echo "  ✅ 对照实验成功: 源可用 → 问题在缺包/包名/依赖解析"
    else
        echo "  ❌ 对照实验失败(见 /tmp/probe_requests.log): 源本身不可用"
        tail -5 /tmp/probe_requests.log
    fi

    # ==================== 安装函数: 默认源优先, 多源 fallback ====================
    # 逐个包安装, 每个包记录完整日志, 便于事后排查
    pip_safe_install() {
        local log=/tmp/pip_install_default.log
        if pip install --no-cache-dir --root-user-action=ignore "$@" >"$log" 2>&1; then
            return 0
        fi
        echo "⚠️  默认源失败: $(tail -2 "$log")" >&2

        log=/tmp/pip_install_aliyun.log
        if pip install --no-cache-dir -i "$PIP_INDEX_ALIYUN" --root-user-action=ignore "$@" >"$log" 2>&1; then
            return 0
        fi
        echo "⚠️  阿里云公网失败: $(tail -2 "$log")" >&2

        log=/tmp/pip_install_aliyun_int.log
        if pip install --no-cache-dir -i "$PIP_INDEX_ALIYUN_INTERNAL" --root-user-action=ignore "$@" >"$log" 2>&1; then
            return 0
        fi
        echo "⚠️  阿里云内网失败: $(tail -2 "$log")" >&2

        log=/tmp/pip_install_tuna.log
        pip install --no-cache-dir -i "$PIP_INDEX_TUNA" --root-user-action=ignore "$@" >"$log" 2>&1
        # 返回 pip 的退出码
    }

    # 1. 逐个安装核心依赖(避免一个包失败拖垮全部; 保留完整报错)
    echo "📦 开始逐个安装核心依赖: $CORE_DEPS"
    for dep in $CORE_DEPS; do
        echo "📦 安装依赖: $dep"
        if pip_safe_install "$dep"; then
            echo "  ✅ $dep 安装成功"
        else
            echo "  ❌ $dep 安装失败(最后一次尝试日志: /tmp/pip_install_tuna.log)"
        fi
    done

    # 2. 修正特定版本 runtime (不安装依赖，防止覆盖已安装的库)
    echo "📦 修正 antlr4-python3-runtime 为 $ANTLR_VERSION ..."
    pip_safe_install "antlr4-python3-runtime==$ANTLR_VERSION" --no-deps --force-reinstall \
        && echo "  ✅ antlr4-python3-runtime 修正成功" \
        || echo "  ❌ antlr4-python3-runtime 修正失败(日志: /tmp/pip_install_tuna.log)"

    # 3. 环境验证
    echo "🔍 正在验证安装状态..."
    python3 -c "
import pyBigWig, pyfaidx, torchmetrics, wandb, swanlab, transformers, torch
print(f'✅ 依赖验证通过 | Transformers: {transformers.__version__} | CUDA Available: {torch.cuda.is_available()}')
" || {
        echo "❌ 验证失败！请检查 pip list。"
        pip list 2>/dev/null | head -30
        echo "----- 最后一次安装日志(末尾30行) -----"
        [ -f /tmp/pip_install_tuna.log ] && tail -30 /tmp/pip_install_tuna.log
        echo "-------------------------------------"
        exit 1
    }

    echo "🎉 环境准备就绪！"
else
    # ==================== CUDA (NVIDIA) 环境变量 ====================
    echo "🔧 设备模式: CUDA (NVIDIA)"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_DEBUG=1
    export NVTE_DEBUG_LEVEL=2
    export NVTE_COMM_OVERLAP=0
    export NCCL_P2P_DISABLE=1
    export NCCL_P2P_DIRECT_DISABLE=1
    # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # export nproc_per_node=8
fi

detect_gpu_count() {
    local detected=0
    # 优先: 从 CUDA_VISIBLE_DEVICES 推断 (云平台常用)
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
        detected="${#gpu_ids[@]}"
        echo "🔍 从 CUDA_VISIBLE_DEVICES 检测到 ${detected} 个 GPU: ${CUDA_VISIBLE_DEVICES}" >&2
    elif [[ "$DEVICE" == "mx" ]]; then
        # MACA: 通过 mx-smi 或 Python torch 检测
        if command -v mx-smi &>/dev/null; then
            detected=$(mx-smi -L 2>/dev/null | grep -c '^GPU' || true)
        fi
        if [[ "$detected" -eq 0 ]]; then
            detected=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
        fi
        echo "🔍 MACA 自动检测到 ${detected} 个 GPU" >&2
    else
        # CUDA: 通过 nvidia-smi 检测
        if command -v nvidia-smi &>/dev/null; then
            detected=$(nvidia-smi -L 2>/dev/null | wc -l)
        fi
        if [[ "$detected" -eq 0 ]]; then
            detected=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
        fi
        echo "🔍 CUDA 自动检测到 ${detected} 个 GPU" >&2
    fi
    echo "$detected"
}
NPROC_PER_NODE=$(detect_gpu_count)
export nproc_per_node="${NPROC_PER_NODE}"


DISTRIBUTED_ARGS=(
    --nnodes 1 
    --nproc_per_node $nproc_per_node
    --node_rank 0  
    --master_addr localhost 
    --master_port 29520
)

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-ydgenomics2-bgi-group}"
export WANDB_PROJECT="${WANDB_PROJECT:-OneGenomeRice}"

torchrun ${DISTRIBUTED_ARGS[@]} train.py \
    --model_path /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
    --tokenizer_dir /mnt/rice/default/Workspace/xuxiaolong/rice_1B_stage2_8k_hf \
    --sequence_split_train_multi data/indices/train_*_multitrack/sequence_split_train.csv \
    --sequence_split_val data/indices/valid_*_multitrack/sequence_split_train.csv \
    --index_stat_multi_json data/indices/train_*_multitrack/index_stat.json \
    --nonzero_means 1.05 1.28 \
    --train_chromosomes Chr5 \
    --output_base_dir output/$(date +%Y%m%d%H%M) \
    --lr 0.00005 \
    --batch_size_per_device 1 \
    --gradient_accumulation_steps 10 \
    --num_train_epochs 20 \
    --loss_func mse \
    --max_sequence_length 32768 \
    --use_flash_attn \
    --gpus_per_node $nproc_per_node \
    --val_chromosomes Chr5 \
    --use_wandb 

