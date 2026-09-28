# -*- coding: utf-8 -*-
"""模型加载：按 arch 选择 AutoModel / AutoModelForCausalLM，统一 remote_code 与 dtype。

device 取值:
    cuda / cuda:0 / cuda:1 ...  -> 指定(单)卡; device_map 传 'cuda' 或 int
    auto                       -> HF device_map='auto' 自动跨卡(大模型)
    cpu                        -> CPU(float32)
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


def _is_cuda(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def _device_map_arg(device: str):
    """把 'cuda' / 'cuda:N' / 'auto' / 'cpu' 转成 HF device_map 可接受的值。"""
    if device == "cuda":
        return "cuda"
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return device  # "auto" / "cpu" 原样


def _fix_tokenizer_legacy(path: str) -> str:
    """处理 tokenizer_class='TokenizersBackend'(transformers 4.57 新格式但本地未实现)。

    返回修正后的临时目录(不污染原模型); 无需修正时返回原 path。
    """
    import json, shutil, tempfile
    cfg_path = Path(path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return path
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("tokenizer_class") != "TokenizersBackend":
            return path
        tmp = tempfile.mkdtemp(prefix="tokfix_")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = Path(path) / name
            if src.exists():
                shutil.copy(src, Path(tmp) / name)
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(Path(tmp) / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info(f"[{path}] tokenizer_class=TokenizersBackend -> PreTrainedTokenizerFast")
        return tmp
    except Exception as e:
        logger.warning(f"tokenizer_fix 失败({e}), 用原目录")
        return path


def load_model(entry, device: str):
    """返回 (model, tokenizer)。model 已 eval; tokenizer 已就绪。"""
    # tokenizer_fix: 仅修正 tokenizer(临时目录), 模型仍从原路径加载(config.json 必须留在原处)
    tok_path = entry.path
    if getattr(entry, "tokenizer_fix", False):
        tok_path = _fix_tokenizer_legacy(entry.path)

    # 归一化 "cuda:N" 设备号; Triton/CUDA 内核(PlantCAD2/mamba)要求默认设备与模型一致,
    # 否则指针在不同设备上下文不可访问 -> set_device 到加载设备
    device_idx = None
    if _is_cuda(device):
        device_idx = int(device.split(":")[1]) if ":" in device else 0
    if device_idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_idx)   # triton autotuner 用默认设备 context launcher

    default_dtype = torch.float16 if _is_cuda(device) or device == "auto" else torch.float32
    torch_dtype = {
        None: default_dtype,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[getattr(entry, "torch_dtype", None)]

    common = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    load_kwargs = dict(common)
    load_kwargs["device_map"] = _device_map_arg(device)
    use_flash = _is_cuda(device) and getattr(entry, "flash_attention", False)
    if use_flash:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    # arch: causal -> CausalLM; mlm -> MaskedLM(NTv3 只注册了 AutoModelForMaskedLM);
    # encoder/bidirectional -> AutoModel(Botanic/Caduceus/agront)
    if entry.arch == "causal":
        model_cls = AutoModelForCausalLM
    elif entry.arch == "mlm":
        model_cls = AutoModelForMaskedLM
    else:
        model_cls = AutoModel
    try:
        model = model_cls.from_pretrained(entry.path, **load_kwargs).eval()
    except Exception as e:  # flash_attention 兜底回退
        if use_flash:
            logger.warning(f"[{entry.name}] flash_attention 加载失败({e}), 回退默认实现")
            load_kwargs.pop("attn_implementation", None)
            model = model_cls.from_pretrained(entry.path, **load_kwargs).eval()
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # 从模型对象确认 hidden / 层数（配置推断优先，运行期以真实为准）
    cfg = model.config
    entry.hidden_size = getattr(cfg, "hidden_size", None) or entry.hidden_size
    entry.num_layers = getattr(cfg, "num_hidden_layers", None) or entry.num_layers
    logger.info(f"[{entry.name}] 加载完成 hidden={entry.hidden_size} "
                f"layers={entry.num_layers} device_map={load_kwargs.get('device_map')}")
    return model, tokenizer


def model_device(model) -> torch.device:
    """可靠的模型所在设备（device_map 场景下比 model.device 稳）。"""
    return next(model.parameters()).device