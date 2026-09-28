# -*- coding: utf-8 -*-
"""模型注册表：读 models.yaml -> ModelEntry dataclass，含字段校验与兜底。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Union

import yaml

from ogr_extract.embedder import VALID_POOLING

VALID_ARCH = {"causal", "encoder", "mlm"}   # mlm = encoder + AutoModelForMaskedLM(NTv3)
VALID_TOK = {"singlebase", "kmer3", "kmer6", "kmer6_cls", "char"}
VALID_DTYPE = {"float16", "bfloat16", "float32", None}


@dataclass
class ModelEntry:
    name: str
    path: str
    arch: str = "causal"
    tokenizer: str = "singlebase"
    max_context: Optional[int] = None       # None = 不截断
    layer: Optional[int] = None             # None -> 跟随 run 配置
    max_tokens: int = 65536                 # token 限额打包（batch_size=0 时生效）
    batch_size: Optional[int] = None        # >0 用固定条数打包(忽略 max_tokens); None/0 -> max_tokens
    pooling: Optional[str] = None           # 池化方式; None -> 跟随 run 配置（见 embedder.VALID_POOLING）
    device: Optional[str] = None            # 如 "cuda:1"; None -> 跟随 run 的 gpus/device
    flash_attention: bool = False
    torch_dtype: Optional[str] = None       # 加载精度: float16/bfloat16/float32; None -> 跟随默认(见 loader)
    attention_mask: bool = True             # False = 模型 forward 不接受 attention_mask(SSM/Caduceus)
    tokenizer_fix: bool = False             # True = 修正 tokenizer_class=TokenizersBackend 兼容
    pad_to_multiple: Optional[int] = None   # tokenize 后批次长度右补齐到该倍数(NTv3 卷积下采样需 128 倍数)
    # 从 config.json 兜底读出的字段（只读，不参与配置校验）
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    inferred_from_config: bool = False

    def effective_layer(self, run_layer: int) -> int:
        """模型级 layer 优先(可为负, 如 -1=最后一层); 否则沿用 run 配置。

        旧实现返回 num_layers(causal 恰好=最后一层), 但自定义 encoder
        (Botanic0 的 hidden_states 无 embedding 层) 会越界; 统一回退到 run 的
        layer(默认 -1)最安全。
        """
        return self.layer if self.layer is not None else run_layer

    def effective_batch_size(self, run_batch_size: int) -> int:
        """>0 表示按固定条数打包；否则 0 = 用 max_tokens 动态打包。"""
        if self.batch_size:
            return self.batch_size
        return run_batch_size or 0

    def effective_pooling(self, run_pooling: str) -> str:
        """模型级 pooling 优先, 否则用 run 级（默认 mean = 旧行为）。"""
        return self.pooling or run_pooling or "mean"


def _load_config_json(path: str) -> dict:
    cfg_path = Path(path) / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _infer_from_config(cfg: dict, entry: ModelEntry) -> None:
    """从 HF config.json 兜底推断 层数/hidden/上下文，并标记。"""
    if entry.num_layers is None:
        entry.num_layers = (
            cfg.get("num_hidden_layers")
            or cfg.get("n_layer")
            or cfg.get("num_layers")
        )
    if entry.hidden_size is None:
        entry.hidden_size = (
            cfg.get("hidden_size")
            or (cfg.get("d_model") * 2 if cfg.get("rcps") and cfg.get("d_model") else cfg.get("d_model"))
        )
    if entry.max_context is None:
        entry.max_context = (
            cfg.get("max_position_embeddings") or cfg.get("max_seqlen")
        )
    entry.inferred_from_config = True


def load_registry(path: Union[str, Path]) -> dict[str, ModelEntry]:
    """解析 models.yaml，逐条校验 + config.json 兜底。"""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    models_raw = raw.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ValueError(f"models.yaml 缺少 'models:' 字典: {path}")

    entries: dict[str, ModelEntry] = {}
    for name, cfg in models_raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"模型 {name} 配置必须是字典")
        missing = [k for k in ("path",) if k not in cfg]
        if missing:
            raise ValueError(f"模型 {name} 缺少必填字段: {missing}")
        if cfg.get("arch") not in VALID_ARCH:
            raise ValueError(f"模型 {name} arch 非法: {cfg.get('arch')} (可选 {VALID_ARCH})")
        if cfg.get("torch_dtype") not in VALID_DTYPE:
            raise ValueError(f"模型 {name} torch_dtype 非法: {cfg.get('torch_dtype')} "
                             f"(可选 {list(VALID_DTYPE - {None})})")
        if cfg.get("tokenizer") not in VALID_TOK:
            raise ValueError(f"模型 {name} tokenizer 非法: {cfg.get('tokenizer')} (可选 {VALID_TOK})")
        if cfg.get("pooling") is not None and cfg["pooling"] not in VALID_POOLING:
            raise ValueError(f"模型 {name} pooling 非法: {cfg['pooling']} (可选 {VALID_POOLING})")
        # 过滤未知字段，避免 dataclass 报 TypeError（同时容忍注释遗留键）
        known = {f.name for f in fields(ModelEntry)}
        clean = {k: v for k, v in cfg.items() if k in known}
        entry = ModelEntry(name=name, **clean)
        _infer_from_config(_load_config_json(entry.path), entry)
        entries[name] = entry
    return entries