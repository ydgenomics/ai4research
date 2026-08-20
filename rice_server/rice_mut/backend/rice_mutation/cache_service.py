"""Prediction result cache — content-addressed LRU with TTL.

The model output for a given (genome, window, biosamples [, SNV]) is
deterministic, so identical requests can reuse the previously written bigWig
files and skip GPU inference entirely.  The cache is **shared across all
users** (content-addressed), which is exactly what makes it effective under
multi-user concurrency: popular loci are computed once and reused.

On a hit we return the cached ``igv_payload`` + metadata directly; the value
arrays stay in the API's ``_REF_CACHE`` / ``_SNV_CACHE`` keyed by the same
``prediction_id`` so ``/predict/bar`` and ``/predict/snv/stat`` keep working.

Also contains helpers for the background bigWig cleanup thread.
"""

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Set


class PredictionResultCache:
    """Content-addressed LRU cache with TTL expiration.

    Entry: ``content_key -> {prediction_id, igv_payload, metadata, kind, ts}``.
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

    def put(self, key: str, prediction_id: str, igv_payload: dict,
            metadata: dict, kind: str) -> None:
        with self._lock:
            self._data[key] = {
                "prediction_id": prediction_id,
                "igv_payload": igv_payload,
                "metadata": metadata,
                "kind": kind,
                "ts": time.time(),
            }
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)  # evict LRU

    def live_prediction_ids(self) -> Set[str]:
        """prediction_ids still cached & unexpired (for bigWig cleanup)."""
        now = time.time()
        with self._lock:
            return {
                item["prediction_id"]
                for item in self._data.values()
                if now - item["ts"] <= self._ttl
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# Singleton shared by the API routes.
prediction_cache = PredictionResultCache()


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
                    print(f"[cache] cleaned {n} stale bigWig file(s)")
            except Exception as e:  # never let cleanup take down the server
                print(f"[cache] bigWig cleanup error: {e}")

    threading.Thread(target=_loop, daemon=True, name="bigwig-cleanup").start()
