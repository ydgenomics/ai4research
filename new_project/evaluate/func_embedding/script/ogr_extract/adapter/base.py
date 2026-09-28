# -*- coding: utf-8 -*-
"""SequenceAdapter 抽象基类 + 工厂分发。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ogr_extract.registry import ModelEntry


class SequenceAdapter(ABC):
    def __init__(self, entry: ModelEntry):
        self.entry = entry

    @abstractmethod
    def prepare_seq(self, seq: str, max_len: Optional[int]) -> str:
        """按模型规范预处理单条序列（补齐/大小写/截断回退）。"""

    @abstractmethod
    def tokenize_kwargs(self) -> dict:
        """返回传给 tokenizer 的关键字参数（不加序列）。"""

    def truncate_to_context(self, seq: str) -> tuple[str, bool]:
        """按模型上下文截断。max_context 以 token 计; 对 k-mer 模型换成 bp 上限。"""
        ctx = self.entry.max_context
        if ctx is None:
            return seq, False
        bp_per_tok = getattr(self, "bp_per_token", 1)
        limit = ctx * bp_per_tok
        if len(seq) <= limit:
            return seq, False
        return seq[:limit], True


def get_adapter(entry: ModelEntry) -> "SequenceAdapter":
    from ogr_extract.adapter.char import CharAdapter
    from ogr_extract.adapter.kmer import Kmer3Adapter, Kmer6Adapter, Kmer6ClsAdapter
    from ogr_extract.adapter.singlebase import SingleBaseAdapter

    factory = {
        "singlebase": SingleBaseAdapter,
        "kmer3": Kmer3Adapter,
        "kmer6": Kmer6Adapter,
        "kmer6_cls": Kmer6ClsAdapter,
        "char": CharAdapter,
    }
    try:
        cls = factory[entry.tokenizer]
    except KeyError:
        raise ValueError(f"未注册的 tokenizer 类型: {entry.tokenizer}")
    return cls(entry)