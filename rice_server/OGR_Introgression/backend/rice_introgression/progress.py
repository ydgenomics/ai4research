"""推理进度跟踪器 —— 进程内线程安全单例。

设计：单 uvicorn 进程（当前部署）内维护一个「当前活动任务」进度。
- start(): 真正开始 GPU 推理前调用，重置计数器并记录开始时间
- update(): 由 predictor 的 batch 循环每批回调（低频率，避免锁开销）
- finish()/fail(): 推理结束或异常时置终态
- get(): 供 /progress 端点返回给前端轮询

注意：推理本身由 _INFER_LOCK 串行，同一时刻只有一个任务在 GPU 上跑，
因此单「current」槽位足够；多个请求排队时，后到者会先抢占进度槽，
但阻塞等待 GPU 期间其回调不会被调用（进度停在 0%），语义上仍自洽。
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class ProgressTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[dict] = None

    def start(
        self,
        *,
        key: str,
        genome: str,
        chromosome: str,
        region_start: int,
        region_end: int,
        total_segments: int,
        total_batches: int,
        message: str = "推理中",
    ) -> None:
        with self._lock:
            self._current = {
                "status": "running",
                "_key": key,
                "genome": genome,
                "chromosome": chromosome,
                "region_start": int(region_start),
                "region_end": int(region_end),
                "done_segments": 0,
                "total_segments": int(total_segments),
                "done_batches": 0,
                "total_batches": int(total_batches),
                "percent": 0.0,
                "started_at": self._now(),
                "elapsed_seconds": 0.0,
                "message": message,
                "error": None,
            }

    def update(self, *, done_segments: int, total_segments: int,
               done_batches: int, total_batches: int) -> None:
        """batch 循环回调：只更新 running 状态的任务。"""
        with self._lock:
            cur = self._current
            if cur is None or cur["status"] != "running":
                return
            cur["done_segments"] = int(done_segments)
            cur["total_segments"] = int(total_segments)
            cur["done_batches"] = int(done_batches)
            cur["total_batches"] = int(total_batches)
            total = cur["total_segments"]
            cur["percent"] = (
                100.0 * min(1.0, cur["done_segments"] / total) if total else 100.0
            )
            cur["elapsed_seconds"] = round(self._now() - cur["started_at"], 3)

    def finish(self, message: str = "推理完成") -> None:
        with self._lock:
            if self._current is None:
                return
            self._current["status"] = "done"
            self._current["percent"] = 100.0
            self._current["done_segments"] = self._current["total_segments"]
            self._current["done_batches"] = self._current["total_batches"]
            self._current["elapsed_seconds"] = round(
                self._now() - self._current["started_at"], 3
            )
            self._current["message"] = message

    def fail(self, message: str = "推理失败") -> None:
        with self._lock:
            if self._current is None:
                return
            self._current["status"] = "error"
            self._current["message"] = message
            self._current["error"] = message
            self._current["elapsed_seconds"] = round(
                self._now() - self._current["started_at"], 3
            )

    def get(self) -> dict:
        """返回当前任务快照；无任务时返回 idle。内部 _key 字段不对外暴露。"""
        with self._lock:
            cur = self._current
            if cur is None:
                return {"status": "idle", "message": "无活动推理任务"}
            d = {k: v for k, v in cur.items() if not k.startswith("_")}
            d["elapsed_seconds"] = round(self._now() - cur["started_at"], 3)
            return d

    def is_current(self, key: str) -> bool:
        """当前 running 任务是否仍是 key 对应任务（防止旧回调覆盖新任务进度）。"""
        with self._lock:
            cur = self._current
            return bool(
                cur is not None
                and cur["status"] == "running"
                and cur.get("_key") == key
            )

    @staticmethod
    def _now() -> float:
        return time.time()


# 全局单例（单 worker 进程内共享）
progress_tracker = ProgressTracker()