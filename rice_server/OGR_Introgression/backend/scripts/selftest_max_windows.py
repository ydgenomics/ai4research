"""MAX_NUMBER_256W 逻辑自测（不加载模型）：验证 select_windows 各组合。

运行: /root/miniconda3/envs/vllm/bin/python scripts/selftest_max_windows.py
"""
import os
import sys

# 让 import 可用（从 rice_intro 项目根运行）
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "backend")
)

from rice_introgression import analysis as ana  # noqa: E402
import rice_introgression.prediction_service as ps  # noqa: E402

# 与 .env 一致的默认参数
params = ana.AnalysisParams(
    segment_size=8000,
    window_size=256000,
    window_step=64000,
    top_k=10,
    threshold_jap=0.55519,
    threshold_ind=0.53473,
)

CHROM_LEN = 43_900_000  # YF47 chr1 近似
trim_start, trim_end = 0, CHROM_LEN
n_seg = ps._n_segments_after_trim(trim_start, trim_end, 8000)


def show(title, wins):
    print(f"{title}: {tuple(wins)}")


# 1) 整条模式 + 无限制 => 全部网格起点（0, 64k, 128k, ... 步进 64k）
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, None, None, None)
show("整条/无限制(前5)", wins[:5])
assert wins[0] == 0 and wins[1] == 64000 and wins[2] == 128000
window_segments = params.window_size // params.segment_size  # 32
step_segments = params.window_step // params.segment_size    # 8
n_expected = (n_seg - window_segments) // step_segments + 1
assert len(wins) == n_expected, f"n_win={len(wins)} expected={n_expected}"

# 2) 整条模式 + MAX=3 => 前 3 个窗口 [0, 64k, 128k]
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, None, None, 3)
show("整条/MAX=3", wins)
assert wins == [0, 64000, 128000]

# 3) 窗口模式 start=100k,end=800k + MAX=3
#    （覆盖 [100k,800k)，各候选窗口与区间的重叠）=> 取覆盖最大的 3 个
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 100_000, 800_000, 3)
show("start=100k end=800k MAX=3", wins)
# 完全落在 [100k,800k) 内的窗口（重叠=满窗口256k）起点为 128k..512k（7个）；
# 平局取起点更早者 → 前 3 个 = [128k, 192k, 256k]
expected = [128000, 192000, 256000]
assert wins == expected, f"got {wins}"

# 4) 窗口模式只有 start（无 end）+ MAX=2 => 默认只推 1 个覆盖度最大的窗口
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 50_000, None, 2)
show("start=50k(无end) MAX=2", wins)
# user_end=50k+256k=306k。候选：
#   A[0,256k] 重叠 min(256k,306k)-50k = 206k
#   B[64k,320k] 重叠 min(320k,306k)-64k = 306k-64k=242k
#   C[128k,384k] 重叠 min(384k,306k)-128k=178k
# 只填 start（end 空）→ 始终单窗（忽略 MAX）→ 最大覆盖单窗 = B(242k) → [64000]
assert wins == [64000], f"got {wins}"

# 5) 窗口模式完全在染色体外（用户区间 > chrom_len）+ MAX=3
#    => 无正重叠，兜底前 3 个网格起点
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 50_000_000, 51_000_000, 3)
show("超出染色体 end MAX=3", wins)
assert wins == [0, 64000, 128000], f"got {wins}"

# 6) max_windows=None 窗口模式 => 单窗，最大覆盖网格窗口（旧版 behavior）
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 100_000, 800_000, None)
show("窗口/no-MAX (单窗)", wins)
assert len(wins) == 1

print("\nALL LOGIC TESTS PASSED")