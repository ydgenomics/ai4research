# -*- coding: utf-8 -*-
"""k-mer tokenizer（OryzaG3 3-mer；GENERator 6-mer 补A；Botanic/AgroNT 6-mer+<cls>）。"""
from __future__ import annotations

from typing import Optional

from ogr_extract.adapter.base import SequenceAdapter


class Kmer3Adapter(SequenceAdapter):
    """3-mer（OryzaG3）。"""
    bp_per_token = 3

    def prepare_seq(self, seq: str, max_len: Optional[int]) -> str:
        s = seq.upper()
        if max_len:
            s = s[:max_len]
        # OryzaG3 词表是 WordLevel 纯 ACGT 3-mer; N 会变 <unk>, 统一替换为 A(与 kmer6 一致)。
        if "N" in s:
            s = s.replace("N", "A")
        # 手动按 3bp 切空格让 WordLevel 命中词表(BertPreTokenizer 需要空格预切)。
        # 尾部不足 3 碱基丢弃。
        if len(s) >= 3:
            usable = len(s) - len(s) % 3
            s = " ".join(s[i:i + 3] for i in range(0, usable, 3))
        return s

    def tokenize_kwargs(self) -> dict:
        return {"add_special_tokens": False}


class Kmer6Adapter(SequenceAdapter):
    """6-mer（GENERator）：输入长度必须为 6 的倍数，否则尾部追加 <oov>。"""
    bp_per_token = 6

    def prepare_seq(self, seq: str, max_len: Optional[int]) -> str:
        s = seq.upper()
        if max_len:
            s = s[:max_len]
        # 参考序列中的 N 会落到 <oov>; 而 <oov> 的 embedding 在 GENERator 里
        # fp16 前向会触发 NaN(实测 <oov> 本身权重有限, 但反向后溢出)。先统一替换为 A。
        if "N" in s:
            s = s.replace("N", "A")
        if len(s) % 6 != 0:
            pad = 6 - (len(s) % 6)
            s = "A" * pad + s  # 左侧补 A，保持基因坐标锚定
        return s

    def tokenize_kwargs(self) -> dict:
        return {"add_special_tokens": False}


class Kmer6ClsAdapter(SequenceAdapter):
    """6-mer + <cls>（Botanic0 / AgroNT，ESM 风格，自动前置 <cls>）。"""
    bp_per_token = 6

    def prepare_seq(self, seq: str, max_len: Optional[int]) -> str:
        s = seq.upper()
        if max_len:
            s = s[:max_len]
        # 与 Kmer6 一致：含 N 替换为 A（避免 <oov> 的 NaN 风险；ESM 词表通常无 N）
        if "N" in s:
            s = s.replace("N", "A")
        if len(s) % 6 != 0:
            s = "A" * (6 - (len(s) % 6)) + s
        return s

    def tokenize_kwargs(self) -> dict:
        return {"add_special_tokens": True}

    def truncate_to_context(self, seq: str) -> tuple[str, bool]:
        """Botanic/AgroNT 位置编码 create_position_ids_from_input_ids 有 padding_idx(1) 偏移:
        可用位置 = max_position_embeddings - padding_idx; 真实 token(=kmer+<cls>) 必须再减 1,
        否则 1024 kmer + <cls> 会生成位置 1026 越界(Embedding(1026) 索引 0..1025)。"""
        ctx = self.entry.max_context
        if ctx is None:
            return seq, False
        limit = (ctx - 1) * self.bp_per_token  # 预留 1 个 token 给 <cls>
        if len(seq) <= limit:
            return seq, False
        return seq[:limit], True