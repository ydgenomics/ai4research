"""预测结果缓存 —— 内容寻址 LRU + TTL + 磁盘持久化。

模型对同一 (genome, chrom, 标准窗口) 的输出是确定性的，相同请求可跳过
GPU 推理直接复用。缓存跨用户共享（内容寻址），多用户并发时热门位点只算一次。

磁盘持久化（新增）：
- 配置 BACKEND_PREDICTION_CACHE（.env）指向一个目录；每个 entry 落盘为
  ``<sanitized_key>.json``。重新启动后自动预热（load()），使已推理的窗口
  长期留在磁盘、不再触发 GPU 推理。
- 与内存 LRU 同步淘汰：TTL 过期 / 超过 max_entries 时同步删除磁盘文件。
- key 含有 ``|`` 与 ``=`` 等字符，sanitize 后用作文件名避免跨平台问题。

Entry: ``content_key -> {window_df, segments_df, metadata, ts}``
"""

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _sanitize_key(key: str) -> str:
    """把缓存 key（含 | = 等）转换为安全文件名。"""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return safe[:200] or "key"


class PredictionResultCache:
    """内容寻址 LRU 缓存（线程安全），可选磁盘后备。"""

    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: float = 3600.0,
        disk_dir: Optional[str] = None,
        persist: bool = True,
    ):
        self._max = max_entries
        self._ttl = ttl_seconds
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._disk_dir = disk_dir
        self._persist = persist
        if self._disk_dir:
            try:
                os.makedirs(self._disk_dir, exist_ok=True)
            except Exception as e:
                logger.warning("Failed to create cache dir %s: %s", self._disk_dir, e)
                self._disk_dir = None

    # ---- 磁盘方 ─────────────────────────────────────────────
    def _disk_path(self, key: str) -> str:
        return os.path.join(self._disk_dir or "", f"{_sanitize_key(key)}.json")

    def _disk_write(self, key: str, payload: dict) -> None:
        if not self._disk_dir:
            return
        try:
            tmp = self._disk_path(key) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, self._disk_path(key))
        except Exception as e:
            logger.warning("Cache disk write failed for %s: %s", key, e)

    def _disk_remove(self, key: str) -> None:
        if not self._disk_dir:
            return
        try:
            p = self._disk_path(key)
            if os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            logger.warning("Cache disk remove failed for %s: %s", key, e)

    def load(self) -> int:
        """从磁盘预热：读取全部 *.json 还原为内存 entry（TTL 视为加载时刻）。"""
        if not self._disk_dir:
            return 0
        loaded = 0
        now = time.time()
        try:
            for name in sorted(os.listdir(self._disk_dir)):
                if not name.endswith(".json") or name.endswith(".tmp"):
                    continue
                path = os.path.join(self._disk_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        item = json.load(fh)
                    if not isinstance(item, dict) or "payload" not in item:
                        os.remove(path)
                        continue
                    key = item.get("key")
                    if not key:
                        continue
                    item["ts"] = now  # 重启后 TTL 重新计时
                    self._data[key] = item
                    self._data.move_to_end(key)
                    loaded += 1
                except Exception as e:
                    logger.warning("Cache load failed for %s: %s", name, e)
        except Exception as e:
            logger.warning("Cache disk scan failed: %s", e)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
        if loaded:
            logger.info("Warm cache loaded %d entries from %s", loaded, self._disk_dir)
        return loaded

    # ---- 内存主路径 ─────────────────────────────────────────
    @staticmethod
    def build_key(kind: str, **params) -> str:
        canonical = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{kind}|{canonical}"

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                # 磁盘有而内存没有（内存 LRU 淘汰后的冷数据）：直接读盘
                return self._get_disk(key)
            if time.time() - item["ts"] > self._ttl:
                self._data.pop(key, None)
                self._disk_remove(key)
                return None
            self._data.move_to_end(key)
            # 同步刷新磁盘时间戳（顺带保活文件）
            self._disk_write(key, item)
            return item

    def _get_disk(self, key: str) -> Optional[dict]:
        if not self._disk_dir:
            return None
        path = self._disk_path(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                item = json.load(fh)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("Cache disk read failed for %s: %s", key, e)
            return None
        if not isinstance(item, dict) or "payload" not in item:
            self._disk_remove(key)
            return None
        now = time.time()
        if now - item["ts"] > self._ttl:
            self._disk_remove(key)
            return None
        item["ts"] = now
        self._data[key] = item
        self._data.move_to_end(key)
        return item

    def put(self, key: str, entry: dict) -> None:
        """写入缓存。entry 需含 payload 字段（与调用方约定一致，如
        ``{"payload": {...}}``）；内部补充 key/ts 后落盘。"""
        with self._lock:
            entry = dict(entry)
            entry["key"] = key
            entry["ts"] = time.time()
            self._data[key] = entry
            self._data.move_to_end(key)
            if self._persist:
                self._disk_write(key, entry)
            while len(self._data) > self._max:
                _, evicted = self._data.popitem(last=False)
                self._disk_remove(evicted["key"])

    def remove(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._disk_remove(key)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            if self._disk_dir:
                try:
                    for name in os.listdir(self._disk_dir):
                        if name.endswith(".json") or name.endswith(".tmp"):
                            os.remove(os.path.join(self._disk_dir, name))
                except Exception as e:
                    logger.warning("Cache disk clear failed: %s", e)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# 单例缓存：磁盘目录取自 .env 的 BACKEND_PREDICTION_CACHE（未配置则不落盘）
_DEFAULT_DISK_DIR = os.getenv(
    "BACKEND_PREDICTION_CACHE",
    "",
).strip()

_CACHE_MAX_ENTRIES = int(os.getenv("PREDICTION_CACHE_MAX_ENTRIES", "512"))
_CACHE_TTL_SECONDS = float(os.getenv("PREDICTION_CACHE_TTL_SECONDS", "3600.0"))

prediction_cache = PredictionResultCache(
    max_entries=_CACHE_MAX_ENTRIES,
    ttl_seconds=_CACHE_TTL_SECONDS,
    disk_dir=_DEFAULT_DISK_DIR or None,
)


def warm_prediction_cache() -> int:
    """启动预热：从磁盘加载已推理窗口，返回加载条数。"""
    return prediction_cache.load()