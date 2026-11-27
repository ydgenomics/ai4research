"""基因组服务 —— 内置 + 上传基因组的注册、染色体列表、FASTA 读取。

移植自 rice_mut backend/rice_mutation/prediction_service.py 的相关部分，
并针对「按染色体名称原样透传」做了简化：
- 染色体名不做 chrNN 归一化（YF47 的染色体为 GWHBKAR00000001...，直接原样返回）
- FASTA 支持 .gz，读取用 predictor.read_fasta_sequence（流式，不加载整个基因组）
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

import pyfaidx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  上传（custom）基因组注册表 + FASTA 缓存
# ---------------------------------------------------------------------------
_UPLOADED_GENOMES: Dict[str, Dict[str, Any]] = {}
_FASTA_CACHE: Dict[str, pyfaidx.Fasta] = {}
_REGISTRY_LOCK = threading.Lock()


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _builtin_genome_ids() -> list:
    return [
        key[len("GENOME_"):-len("_FASTA")]
        for key, val in sorted(os.environ.items())
        if key.startswith("GENOME_") and key.endswith("_FASTA") and val
    ]


def _load_fai_chroms(fai_path: str) -> list[str]:
    if not fai_path or not os.path.isfile(fai_path):
        return []
    names: list[str] = []
    try:
        with open(fai_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                col = line.split("\t", 1)[0].strip()
                if col:
                    names.append(col)
    except Exception as e:
        logger.warning("Failed to read %s: %s", fai_path, e)
    return names


def _load_fai_lengths(fai_path: str) -> dict[str, int]:
    """读取 .fai -> {染色名: 长度}。"""
    if not fai_path or not os.path.isfile(fai_path):
        return {}
    out: dict[str, int] = {}
    try:
        with open(fai_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] and parts[1].isdigit():
                    out[parts[0]] = int(parts[1])
    except Exception as e:
        logger.warning("Failed to read %s: %s", fai_path, e)
    return out


def _build_fai(fasta_path: str, fai_path: str) -> None:
    """为 FASTA（支持普通 gzip）构建 .fai 索引。

    逐行流式扫描（gzip 不解压落盘），输出标准 5 列 .fai：
        name, length, seq_offset, line_bases, line_width
    注意：对普通 gzip 序列偏移是解压流偏移，仅列 1/2（名字/长度）被使用，
    序列读取始终走 predictor.read_fasta_sequence（流式顺序扫描）。
    """
    import gzip

    is_gz = str(fasta_path).endswith(".gz")
    open_fn = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    records: list[tuple[str, int, int, int, int]] = []
    with open_fn(fasta_path, mode, encoding="utf-8") as fh:
        chrom: str | None = None
        length = 0
        seq_offset = 0
        line_bases = 0
        line_width = 0
        first_seq_line = True
        stream_pos = 0
        for line in fh:
            line_len = len(line)  # 含换行，解压流字节数
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped.startswith(">"):
                if chrom is not None:
                    records.append((chrom, length, seq_offset, line_bases, line_width))
                header = stripped[1:].strip()
                chrom = header.split()[0]
                length = 0
                seq_offset = stream_pos + line_len  # 记录后序列起始偏移
                line_bases = 0
                line_width = 0
                first_seq_line = True
            else:
                if chrom is None:
                    stream_pos += line_len
                    continue
                if first_seq_line:
                    line_bases = len(stripped)
                    line_width = line_len
                    first_seq_line = False
                length += len(stripped)
            stream_pos += line_len
        if chrom is not None:
            records.append((chrom, length, seq_offset, line_bases, line_width))

    if not records:
        raise ValueError(f"No sequence records found in FASTA: {fasta_path}")

    with open(fai_path, "w", encoding="utf-8") as fh:
        for name, length, offset, bases, width in records:
            fh.write(f"{name}\t{length}\t{offset}\t{bases}\t{width}\n")
    logger.info("Built FASTA index (%d records): %s", len(records), fai_path)


def _upload_link_dir() -> str:
    """用于存放 cache-dir FASTA 软链接的目录（在此建 .fai 以规避原目录无写权限）。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.abspath(os.getenv(
        "BACKEND_UPLOADED_FASTA",
        os.path.join(root, "cache", "uploaded_fasta"),
    ))


def _fai_is_valid(fasta_path: str) -> bool:
    """仅当 .fai 存在且不早于 FASTA 时才可用。"""
    fai = str(fasta_path) + ".fai"
    if not os.path.isfile(fai):
        return False
    try:
        return os.path.getmtime(fai) >= os.path.getmtime(fasta_path)
    except OSError:
        return False


def _ensure_fai_link(fasta_path: str, link_dir: str = "") -> tuple[str, str, bool]:
    """确保 ``fasta_path`` 有可用的 ``<fasta>.fai`` 索引（支持普通 gzip）。

    当原生 ``<fasta>.fai`` 缺失或早于 FASTA 时，**不**写入用户目录（可能只读），
    而是把 FASTA 软链接到服务自有的 cache 目录并在那里建索引，返回软链接路径
    供调用方读取缓存索引。

    Returns ``(effective_fasta, fai_path, is_link)``。
    """
    fasta_abs = os.path.abspath(fasta_path)
    if not os.path.isfile(fasta_abs):
        raise FileNotFoundError(f"FASTA not found: {fasta_abs}")

    link_dir = os.path.abspath(link_dir or _upload_link_dir())

    # 1) FASTA 已在 cache 目录（字节上传副本）→ 原地建 .fai
    if os.path.dirname(fasta_abs) == link_dir:
        fai = fasta_abs + ".fai"
        if not _fai_is_valid(fasta_abs):
            _build_fai(fasta_abs, fai)
        return fasta_abs, fai, False

    # 2) 原生 .fai 有效 → 直接用
    if _fai_is_valid(fasta_abs):
        return fasta_abs, fasta_abs + ".fai", False

    # 3) 缺失/过期 → 软链接到 cache 目录再建
    try:
        os.makedirs(link_dir, exist_ok=True)
    except OSError as e:
        logger.warning("Cannot create link dir %s: %s", link_dir, e)
    fname = os.path.basename(fasta_abs) or "custom.fa"
    link_fasta = os.path.join(link_dir, f"{time.time_ns()}_{fname}")
    link_fai = link_fasta + ".fai"
    try:
        if os.path.lexists(link_fasta):
            os.remove(link_fasta)
        os.symlink(fasta_abs, link_fasta)
        _build_fai(link_fasta, link_fai)  # 在 cache 目录内建 .fai
    except Exception as e:
        logger.warning(
            "Failed to build .fai via cache symlink for %s (%s); indexing in place",
            fasta_abs, e,
        )
        try:
            if os.path.lexists(link_fasta):
                os.remove(link_fasta)
        except OSError:
            pass
        _build_fai(fasta_abs, fasta_abs + ".fai")
        return fasta_abs, fasta_abs + ".fai", False
    logger.info("Built .fai via cache symlink: %s", link_fai)
    return link_fasta, link_fai, True


def _ensure_fai(fasta_path: str, link_dir: str = "") -> str | None:
    """确保有可用 .fai（缺失/过期则软链接到 cache 目录再建，返回 fai 路径）。"""
    try:
        _eff, fai_path, _is_link = _ensure_fai_link(fasta_path, link_dir)
        return fai_path
    except Exception as e:
        logger.warning("Failed to build .fai for %s: %s", fasta_path, e)
        return None


def list_genomes() -> list:
    ids = _builtin_genome_ids()
    for gid in _UPLOADED_GENOMES:
        if gid not in ids:
            ids.append(gid)
    return ids


def resolve_genome_config(genome: str) -> dict:
    """解析基因组配置（上传自定义 > 内置环境变量）。"""
    uploaded = _UPLOADED_GENOMES.get(genome)
    if uploaded is not None:
        return dict(uploaded)

    fasta = _env_str(f"GENOME_{genome}_FASTA", "")
    if not fasta:
        raise ValueError(
            f"Unknown genome '{genome}'.  Set GENOME_{genome}_FASTA in .env "
            "or upload a custom FASTA file first."
        )
    if not os.path.isfile(fasta):
        raise FileNotFoundError(f"Genome FASTA not found: {fasta}")
    return {
        "fasta": fasta,
        "fai": _env_str(f"GENOME_{genome}_FAI", ""),
    }


def _touch_uploaded_genome(genome: str) -> None:
    with _REGISTRY_LOCK:
        cfg = _UPLOADED_GENOMES.get(genome)
        if cfg is not None:
            cfg["last_used"] = time.time()
            _UPLOADED_GENOMES[genome] = cfg


def get_genome_chromosomes(genome: str, genome_config: Optional[dict] = None) -> list[str]:
    """返回基因组的染色体名列表（FASTA 实际命名，原样透传）。"""
    cfg = genome_config or resolve_genome_config(genome)
    fasta = cfg["fasta"]

    # 优先使用 .env 显式指定的 .fai
    names = _load_fai_chroms(cfg.get("fai", ""))
    if names:
        return names

    # 自动建 .fai（支持普通 gzip——不依赖 pyfaidx）
    fai_path = _ensure_fai(fasta)
    names = _load_fai_chroms(fai_path) if fai_path else []
    if names:
        return names

    # 极端回退：pyfaidx（仅未压缩 FASTA 可用）
    try:
        fa = pyfaidx.Fasta(fasta)
        return list(fa.keys())
    except Exception as e:
        raise ValueError(f"Could not index FASTA {fasta}: {e}")


def get_chromosome_length(genome: str, chrom: str, genome_config: Optional[dict] = None) -> int:
    """返回某染色体长度（bp）。优先 .fai，其次 pyfaidx（未压缩时）。"""
    _touch_uploaded_genome(genome)
    cfg = genome_config or resolve_genome_config(genome)
    fasta = cfg["fasta"]

    # .env 显式 .fai
    fai = cfg.get("fai", "")
    if not (fai and os.path.isfile(fai)):
        fai = _ensure_fai(fasta) or ""
    lengths = _load_fai_lengths(fai)
    if chrom in lengths:
        return lengths[chrom]

    # 回退：pyfaidx 索引（未压缩 FASTA）
    inst = _FASTA_CACHE.get(genome)
    if inst is None:
        try:
            inst = pyfaidx.Fasta(fasta)
            _FASTA_CACHE[genome] = inst
        except Exception as e:
            raise ValueError(
                f"Chromosome '{chrom}' length unknown for '{genome}': {e}"
            )
    try:
        return int(inst[chrom].end)
    except KeyError:
        raise ValueError(f"Chromosome '{chrom}' not found in genome '{genome}'")


# ---------------------------------------------------------------------------
#  上传基因组注册
# ---------------------------------------------------------------------------
def register_uploaded_genome(fasta_path: str, genome_id: str = "",
                             managed: bool = True, link_dir: str = "") -> dict:
    """注册一个上传的 FASTA 为可用基因组（优先于内置）。

    确保有可用的 .fai：原生 .fai 缺失/过期时，软链接 FASTA 到 cache 目录再建索引
    （不写入原目录，规避无写权限）。

    ``managed=True``（默认）标记文件为本服务所有（字节上传副本），TTL 清理可删除；
    ``managed=False`` 表示路径指向外部/用户文件（懒加载路径注册）——清理只移除
    注册表条目 + 它创建的 cache 软链接/索引，绝不删除用户原始文件。

    未传入 ``genome_id`` 时默认 ``<时间戳ns>_<文件名>``。返回基因组配置（含最终 id）。
    """
    fasta_abs = os.path.abspath(fasta_path)
    if not os.path.isfile(fasta_abs):
        raise FileNotFoundError(f"FASTA not found: {fasta_abs}")
    effective_fasta, fai_path, is_link = _ensure_fai_link(fasta_abs, link_dir)
    names = _load_fai_chroms(fai_path)
    if not names:
        raise ValueError(f"No sequence records found in FASTA: {fasta_abs}")

    if not genome_id:
        genome_id = f"{time.time_ns()}_{os.path.basename(fasta_abs) or 'custom.fa'}"
    while genome_id in _UPLOADED_GENOMES or genome_id in _builtin_genome_ids():
        genome_id = (
            f"{time.time_ns()}_{os.path.basename(fasta_abs) or 'custom.fa'}"
            f"_{len(_UPLOADED_GENOMES)}"
        )

    with _REGISTRY_LOCK:
        cfg = {
            "id": genome_id,
            "fasta": effective_fasta,   # 读取用路径（软链接或原生路径）
            "real_fasta": fasta_abs,    # 用户原始绝对路径
            "fai": fai_path or "",
            "last_used": time.time(), "managed": managed, "is_link": is_link,
        }
        _UPLOADED_GENOMES[genome_id] = cfg
    logger.info(
        "Registered uploaded genome '%s' (%d chromosomes, managed=%s, link=%s): %s",
        genome_id, len(names), managed, is_link, fasta_abs,
    )
    return dict(cfg)


def cleanup_expired_uploaded_genomes(ttl_hours: float) -> int:
    """删除超过 TTL 未使用的上传基因组（内存 + 磁盘 FASTA/.fai）。

    managed=True（本服务复制进 cache 的字节上传副本）删除全部文件；
    is_link=True（外部 FASTA 经 cache 软链接建索引）只删软链接 + 生成的 .fai；
    用户原始 FASTA（real_fasta）永不删除。
    """
    cutoff = time.time() - float(ttl_hours) * 3600
    expired: list = []
    with _REGISTRY_LOCK:
        for gid, cfg in list(_UPLOADED_GENOMES.items()):
            if float(cfg.get("last_used", 0)) < cutoff:
                expired.append((gid, cfg))
        for gid, _ in expired:
            _UPLOADED_GENOMES.pop(gid, None)
            inst = _FASTA_CACHE.pop(gid, None)
            if inst is not None:
                try:
                    inst.close()
                except Exception:
                    pass

    for gid, cfg in expired:
        managed = cfg.get("managed", True)
        is_link = cfg.get("is_link", False)
        if managed or is_link:
            for key in ("fasta", "fai"):
                path = cfg.get(key) or ""
                if not path or not os.path.lexists(path):
                    continue
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", path, e)
        logger.info(
            "Cleaned up expired uploaded genome '%s' (TTL %.1fh, managed=%s, link=%s)",
            gid, ttl_hours, managed, is_link,
        )
    return len(expired)


def start_uploaded_genome_cleanup(ttl_hours: float = 1.0, interval_seconds: int = 300) -> None:
    """后台线程定期清理闲置上传基因组。"""

    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            try:
                n = cleanup_expired_uploaded_genomes(ttl_hours)
                if n:
                    logger.info("[upload] cleaned %d expired uploaded genome(s)", n)
            except Exception as e:
                logger.warning("[upload] cleanup error: %s", e)

    threading.Thread(target=_loop, daemon=True, name="uploaded-genome-cleanup").start()
    logger.info(
        "Started uploaded-genome cleanup thread (ttl=%.1fh, interval=%ds)",
        ttl_hours, interval_seconds,
    )