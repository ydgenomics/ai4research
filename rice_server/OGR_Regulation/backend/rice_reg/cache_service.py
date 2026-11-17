"""Prediction result cache — content-addressed LRU with TTL + in-flight merge.

The model output for a given (genome, chromosome, window, atac) is
deterministic, so identical requests can reuse the previously written bigWig
files and skip GPU inference entirely.  The cache is **shared across all
users** (content-addressed), which is exactly what makes it effective under
multi-user concurrency: popular loci are computed once and reused.

On a hit we return the cached ``igv_payload`` directly; the referenced bigWig
files stay on disk for the cache TTL and are cleaned up in the background.

Also implements *in-flight merge*: when several concurrent requests miss the
cache for the same key, only the first one runs GPU inference and the others
wait for its result instead of each re-computing the same window.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class PredictionResultCache:
    """Content-addressed LRU cache with TTL expiration.

    Entry: ``content_key -> {igv_payload, metadata, ts}``.
    Thread-safe (single lock around every operation).
    """

    def __init__(self, max_entries: int = 128, ttl_seconds: float = 1800.0):
        self._max = max_entries
        self._ttl = ttl_seconds
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @staticmethod
    def build_key(kind: str, **params) -> str:
        """Deterministic content key from a canonical set of input params."""
        canonical = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{kind}|{canonical}"

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            if time.time() - item["ts"] > self._ttl:
                self._data.pop(key, None)  # expired
                return None
            self._data.move_to_end(key)  # refresh LRU position
            return item

    def put(self, key: str, igv_payload: dict, metadata: Optional[dict] = None) -> None:
        with self._lock:
            self._data[key] = {
                "igv_payload": igv_payload,
                "metadata": metadata or {},
                "ts": time.time(),
            }
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)  # evict LRU

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# Singleton shared by the API routes.
prediction_cache = PredictionResultCache()


# ---------------------------------------------------------------------------
#  In-flight request merge (cache-miss coalescing)
# ---------------------------------------------------------------------------
# key -> {"event": threading.Event, "result": Optional[dict]}
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: Dict[str, Dict[str, Any]] = {}


def _finish_inflight(key: str, holder: dict, result: Optional[dict]) -> None:
    """Notify waiters that a key's computation finished (success or failure)."""
    with _INFLIGHT_LOCK:
        if _INFLIGHT.get(key) is holder:
            _INFLIGHT.pop(key, None)
        holder["result"] = result
        holder["event"].set()


def cached_or_compute(key: str, compute: Callable[[], dict]) -> dict:
    """Return a cached payload for ``key``, or compute it once under concurrency.

    - Cache hit: return immediately (no GPU work).
    - Cache miss: the first caller becomes the *leader* and runs ``compute()``;
      concurrent callers for the same key wait for the leader's result
      (in-flight merge) instead of each re-running GPU inference.
    - Leader failure / timeout: followers fall back to computing themselves
      (no recursion — each waiter computes at most once).
    """
    hit = prediction_cache.get(key)
    if hit is not None:
        return hit["igv_payload"]

    with _INFLIGHT_LOCK:
        holder = _INFLIGHT.get(key)
        if holder is None:
            holder = {"event": threading.Event(), "result": None}
            _INFLIGHT[key] = holder
            is_leader = True
        else:
            is_leader = False

    if is_leader:
        payload: Optional[dict] = None
        try:
            payload = compute()
            prediction_cache.put(key, payload)
            return payload
        finally:
            _finish_inflight(key, holder, payload)
    else:
        # Follower: wait for the leader, then reuse its result.
        if holder["event"].wait(timeout=600):
            result = holder["result"]
            if result is not None:
                return result
        # Leader failed or timed out — stop waiting and compute ourselves.
        with _INFLIGHT_LOCK:
            if _INFLIGHT.get(key) is holder:
                _INFLIGHT.pop(key, None)
        return compute()


# ---------------------------------------------------------------------------
#  Background bigWig cleanup
# ---------------------------------------------------------------------------
def clean_old_prediction_bigwigs(cache_dir: str, ttl_seconds: float = 1800.0,
                                 margin: float = 600.0) -> int:
    """Delete stale ``.bw`` files from the prediction cache directory.

    A file is deleted when it is older than ``ttl_seconds + margin``, i.e.
    well past the point where any live content-cache entry could reference it
    (cache entries expire at ``ttl_seconds``).  Returns the number removed.
    """
    cutoff = time.time() - (ttl_seconds + margin)
    removed = 0
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return 0
    for fname in names:
        if not fname.endswith(".bw"):
            continue
        path = os.path.join(cache_dir, fname)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def start_bigwig_cleanup(cache_dir: str, interval_seconds: int = 600,
                         ttl_seconds: float = 1800.0) -> None:
    """Start a daemon thread that periodically deletes stale bigWig files."""
    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            try:
                n = clean_old_prediction_bigwigs(cache_dir, ttl_seconds=ttl_seconds)
                if n:
                    logger.info("[cache] cleaned %d stale bigWig file(s)", n)
            except Exception as e:  # never let cleanup take down the server
                logger.warning("[cache] bigWig cleanup error: %s", e)

    t = threading.Thread(target=_loop, name="bigwig-cleanup", daemon=True)
    t.start()
    logger.info(
        "Started bigWig cleanup thread (dir=%s, interval=%ds, ttl=%ds)",
        cache_dir, interval_seconds, ttl_seconds,
    )
