# -*- coding: utf-8 -*-
"""adapter 子包：get_adapter 工厂即各 tokenizer 适配器实现。"""
from ogr_extract.adapter.base import SequenceAdapter, get_adapter
from ogr_extract.adapter.singlebase import SingleBaseAdapter
from ogr_extract.adapter.kmer import Kmer3Adapter, Kmer6Adapter, Kmer6ClsAdapter
from ogr_extract.adapter.char import CharAdapter

__all__ = [
    "SequenceAdapter",
    "get_adapter",
    "SingleBaseAdapter",
    "Kmer3Adapter",
    "Kmer6Adapter",
    "Kmer6ClsAdapter",
    "CharAdapter",
]