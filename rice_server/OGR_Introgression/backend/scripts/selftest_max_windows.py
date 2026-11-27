"""select_windows 逻辑自测（不加载模型）：验证「start 锚定窗口 + 超限/MAX」各组合。

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


# 1) 整条模式 + 无限制 => 全部网格起点（0, 64k, 128k, ... 步进 64k）[保留，与离线一致]
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

# 3) 窗口模式只有 start（无 end）=> 恒 1 个窗口 [start, start+256k)，MAX 不参与
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 50_000, None, None)
show("start=50k(无end) no-MAX", wins)
assert wins == [50000], f"got {wins}"
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 50_000, None, 2)
show("start=50k(无end) MAX=2", wins)
assert wins == [50000], f"got {wins}"

# 4) 窗口模式 start+end：重叠滑窗（起点=start, step=64k），尾部>end 的最后一个保留
#    start=100k, end=800k：起点 100k,164k,...,772k（100k+64k*i ≤ 800k → i≤10 → 11 个）；
#    最后一个 [772k, 1028k) 尾部>end 保留；MAX=3 → 前 3 个
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 100_000, 800_000, 3)
show("start=100k end=800k MAX=3", wins)
assert wins == [100000, 164000, 228000], f"got {wins}"

# 5) 无 MAX：起点≤end 全部输出（含尾部>end 的最后一个）
#    start=100k, end=800k：起点 100k+64k*i ≤ 800k → i=0..10（11 个，最后 740k）；
#    最后一个 [740k, 996k) 尾部>end 保留
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 100_000, 800_000, None)
show("start=100k end=800k no-MAX", wins)
assert wins[0] == 100000
assert wins[-1] == 740000, f"last={wins[-1]}"
assert wins == [100000 + i * 64000 for i in range(11)], f"got {wins}"

# 6) 第 1 个窗口尾部>chrom_len => 整体空
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, CHROM_LEN - 10_000, None, None)
show("start 靠近末端(第1窗超限)", wins)
assert wins == [], f"got {wins}"
# 有 end 也一样：第一个窗口就超 chrom_len → 空
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, CHROM_LEN - 10_000, CHROM_LEN, 3)
show("start 靠近末端(有end)", wins)
assert wins == [], f"got {wins}"

# 7) 后续窗口尾部>chrom_len => 丢弃+break，保留前面的（第 1 个不超）
#    start=43.5M：窗口 [43500000, 43756000) OK；... 直到某起点 s 使 s+256k > 43.9M → break
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 43_500_000, CHROM_LEN, None)
show("start=43.5M end=末端 no-MAX", wins)
assert wins and wins[-1] + 256000 <= CHROM_LEN, f"last overflow: {wins[-1] + 256000}"
if wins:
    nxt = wins[-1] + 64000
    assert nxt + 256000 > CHROM_LEN, f"break 应发生在 {nxt}"

# 8) 有 end 但窗口尾部只超 end（不超 chrom_len）：该窗口保留
#    start=890k, end=900k：唯一窗口 [890000, 1146000) 尾部>end 但 ≤ chrom_len → 输出 1 个
wins = ps.select_windows(params, CHROM_LEN, trim_start, n_seg, 890_000, 900_000, None)
show("start=890k end=900k (尾部超end)", wins)
assert wins == [890000], f"got {wins}"

print("\nALL LOGIC TESTS PASSED")