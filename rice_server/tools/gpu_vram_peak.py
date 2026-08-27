#!/usr/bin/env python3
"""GPU 显存峰值监控器 — 采样 nvidia-smi,统计每个进程在每张卡上的峰值显存。

用途:在跑 API 压测/推理调用时后台运行,结束后打印峰值汇总,评估服务峰值显存。

用法:
    python gpu_vram_peak.py --pids 2149833,2149835,2149838 --interval 0.2
    # Ctrl-C 停止并打印峰值汇总(或 --duration 秒后自动退出)

参数:
    --pids       只跟踪指定 PID(逗号分隔);默认跟踪所有 python 进程
    --interval   采样间隔秒,默认 0.2
    --duration   自动退出秒数,默认 0 = 一直运行直到 Ctrl-C
    --out        追加写入日志文件(每采样一行),可选
    --quiet      采样过程不打印,只在结束时打印汇总
"""

import argparse
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
#  nvidia-smi 采样
# ---------------------------------------------------------------------------
def sample_compute_apps():
    """返回 {pid: {gpu_index: used_mib}},失败返回 {}。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True, timeout=3.0,
        )
    except Exception:
        return {}
    # gpu_uuid → gpu index 映射
    try:
        uu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True, timeout=3.0,
        )
        uuid2idx = {}
        for line in uu.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid2idx[parts[1]] = parts[0]
    except Exception:
        uuid2idx = {}

    result = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            used = int(parts[2])
        except ValueError:
            continue
        gpu = uuid2idx.get(parts[1], "?")
        result.setdefault(pid, {})[gpu] = used
    return result


def proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        return raw[:160]
    except Exception:
        return "(gone)"


# ---------------------------------------------------------------------------
#  主逻辑
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GPU 显存峰值监控")
    ap.add_argument("--pids", default="")
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    track = {int(p) for p in args.pids.split(",") if p.strip()} if args.pids else None
    fout = open(args.out, "a", buffering=1) if args.out else None

    peak = {}          # pid -> {gpu -> max_mib}
    first_cmd = {}     # pid -> cmdline
    n_samples = 0
    t0 = time.time()

    try:
        while True:
            apps = sample_compute_apps()
            n_samples += 1
            ts = time.time() - t0
            line_parts = []
            for pid, mems in apps.items():
                if track is not None and pid not in track:
                    continue
                if pid not in first_cmd:
                    first_cmd[pid] = proc_cmdline(pid)
                peak.setdefault(pid, {})
                for gpu, used in mems.items():
                    peak[pid][gpu] = max(peak[pid].get(gpu, 0), used)
                line_parts.append(f"{pid}:{dict(mems)}")
            msg = f"[{ts:7.1f}s] " + "  ".join(line_parts) if line_parts else f"[{ts:7.1f}s] (no tracked apps)"
            if not args.quiet:
                print(msg, flush=True)
            if fout:
                fout.write(msg + "\n")

            if args.duration and (time.time() - t0) >= args.duration:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if fout:
            fout.close()

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("GPU 显存峰值汇总")
    print("=" * 70)
    print(f"采样次数: {n_samples}  运行时长: {time.time() - t0:.1f}s")
    if not peak:
        print("未采集到任何受跟踪进程的显存。")
        return
    print(f"{'PID':>10}  {'GPU':>4}  {'峰值 MiB':>10}   进程")
    print("-" * 70)
    rows = []
    for pid, gpus in peak.items():
        for gpu, mib in gpus.items():
            rows.append((pid, gpu, mib))
    rows.sort(key=lambda r: -r[2])
    for pid, gpu, mib in rows:
        print(f"{pid:>10}  {gpu:>4}  {mib:>10}   {first_cmd.get(pid, proc_cmdline(pid))}")
    total = sum(mib for _, _, mib in rows)
    print("-" * 70)
    print(f"受跟踪进程峰值显存合计: {total} MiB (≈ {total / 1024:.1f} GiB)")
    print("=" * 70)


if __name__ == "__main__":
    main()