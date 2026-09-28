# -*- coding: utf-8 -*-
"""字符级 tokenizer（NTv3 / PlantCAD2：1 碱基 1 token，含 N）。"""
from __future__ import annotations

from typing import Optional

from ogr_extract.adapter.base import SequenceAdapter


class CharAdapter(SequenceAdapter):
    def prepare_seq(self, seq: str, max_len: Optional[int]) -> str:
        s = seq.upper()
        if max_len:
            s = s[:max_len]
        return s

    def tokenize_kwargs(self) -> dict:
        return {"add_special_tokens": False}