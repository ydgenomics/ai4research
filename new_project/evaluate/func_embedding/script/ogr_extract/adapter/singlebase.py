# -*- coding: utf-8 -*-
"""单碱基 tokenizer（rice/muxi 系列：1 bp = 1 token，无 special）。"""
from __future__ import annotations

from typing import Optional

from ogr_extract.adapter.base import SequenceAdapter


class SingleBaseAdapter(SequenceAdapter):
    def prepare_seq(self, seq: str, max_len: Optional[int]) -> str:
        s = seq.upper()
        if max_len:
            s = s[:max_len]
        return s

    def tokenize_kwargs(self) -> dict:
        return {"add_special_tokens": False}