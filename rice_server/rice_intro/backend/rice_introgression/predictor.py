"""在线渗入预测器 —— 单例生命周期 + 8k 片段切分 + GPU 推理。

移植自 20.introgression_analysis：
- 片段切分：sliding_window(step=segment_size)（离线 0.run_fragments.py 的
  step_size=window_size=8000，即无重叠），并做 trim_n 头尾 N 裁剪
- tokenize：padding="max_length", max_length=segment_size（tokenizer 每碱基
  近似 1 token，8k 序列 ≈ 8194 token，与离线 arrow_cache 一致）
- 前向：FullModel(input_ids, attention_mask) -> logits -> sigmoid 双概率
- checkpoint 加载：safetensors load_model（与离线 _load_checkpoint 一致）

线程安全：模型单例 + 全局推理锁（GPU 推理需串行）。
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  FASTA / 序列读取
# ---------------------------------------------------------------------------
def read_fasta_sequence(fasta_path: str, chrom: str, start: int, end: int) -> str:
    """从 FASTA（支持 .gz）读取 [start, end) 的序列，保证大小写为大写。

    start / end 为 0-based half-open 坐标。一个 256k 窗口读一次再切 8k 片段。
    """
    import gzip

    open_fn = gzip.open if str(fasta_path).endswith(".gz") else open
    mode = "rt" if str(fasta_path).endswith(".gz") else "r"
    seq_buf: list[str] = []
    wanted = (start, end)
    current_header = ""
    found = False
    base_offset = 0  # 当前记录内已累计的碱基数

    with open_fn(fasta_path, mode, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if found:
                    break  # 已过目标记录，无需继续扫描
                current_header = line[1:].strip()
                seq_id = current_header.split()[0]
                if seq_id == chrom:
                    found = True
                base_offset = 0
                continue
            if not found:
                continue
            line_len = len(line)
            rel_start = wanted[0] - base_offset
            rel_end = wanted[1] - base_offset
            if rel_end > 0 and rel_start < line_len:
                lo = max(0, rel_start)
                hi = min(line_len, rel_end)
                if hi > lo:
                    seq_buf.append(line[lo:hi])
            base_offset += line_len
            if base_offset >= wanted[1]:
                break

    if not found:
        raise ValueError(f"chromosome '{chrom}' not found in FASTA: {fasta_path}")
    seq = "".join(seq_buf).upper()
    if len(seq) < (end - start):
        # 超出染色体末端时补齐到期望长度用 N（与 trim_n 语义互补）
        seq = seq.ljust(end - start, "N")
    return seq


def trim_n(sequence: str) -> tuple[str, list[int]]:
    """裁剪头尾 N，返回 (裁剪后序列, [start_offset, end_offset])。"""
    sequence = sequence.upper()
    start = 0
    end = len(sequence)
    while start < end and sequence[start] == "N":
        start += 1
    while end > start and sequence[end - 1] == "N":
        end -= 1
    return sequence[start:end], [start, end]


def sliding_window(sequence: str, window_size: int, step_size: int) -> list[tuple[int, int]]:
    """按滑窗截取序列，返回 [(start, end)] 列表（允许最后一个窗口不足长度）。

    与离线 0.run_fragments.py 完全一致。
    """
    windows: list[tuple[int, int]] = []
    seq_len = len(sequence)
    for start in range(0, seq_len, step_size):
        end = min(start + window_size, seq_len)
        windows.append((start, end))
        if end == seq_len:
            break
    return windows


# ---------------------------------------------------------------------------
#  单例预测器
# ---------------------------------------------------------------------------
_PREDICTOR: dict[str, Any] = {"instance": None}
_INFER_LOCK = threading.Lock()


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _lora_config_from_env() -> Optional[dict]:
    if not _env_bool("LORA_R", False) and not os.getenv("LORA_R"):
        return None
    target = _env_str("LORA_TARGET_MODULES", "q_proj,v_proj")
    return {
        "r": _env_int("LORA_R", 16),
        "alpha": _env_int("LORA_ALPHA", 32),
        "dropout": _env_float("LORA_DROPOUT", 0.1),
        "target_modules": [m.strip() for m in target.split(",") if m.strip()],
    }


def _proj_dims_from_env() -> list[int]:
    raw = _env_str("PROJ_DIMS", "1024,512,128")
    dims = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return dims if dims else [1024, 512, 128]


class IntrogressionPredictor:
    """封装 FullModel + tokenizer + 在线推理。"""

    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "float32",
        segment_size: int = 8000,
        batch_size: int = 8,
        token: Optional[str] = None,
        lora_config: Optional[dict] = None,
        proj_dims: Optional[list[int]] = None,
        num_labels: int = 2,
        pooling_strategy: str = "masked_mean",
        head_dropout: float = 0.1,
        use_bf16: bool = False,
        use_fp16: bool = False,
    ):
        from transformers import AutoTokenizer

        if not os.path.isdir(base_model_path):
            raise FileNotFoundError(f"BASE_MODEL_PATH not found: {base_model_path}")

        # CHECKPOINT_PATH 可为目录（自动拼 model.safetensors）或直接指向 .safetensors
        ckpt = checkpoint_path
        if os.path.isdir(ckpt):
            ckpt = os.path.join(ckpt, "model.safetensors")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"CHECKPOINT_PATH not found: {checkpoint_path}")
        self.checkpoint_path = ckpt
        self.base_model_path = base_model_path
        self.device = device
        self.torch_dtype = self._resolve_dtype(torch_dtype, use_bf16, use_fp16)
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.proj_dims = proj_dims or [1024, 512, 128]
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True, token=token
        )

        from rice_introgression.model import FullModel

        self.model = FullModel(
            model_path=base_model_path,
            proj_dims=self.proj_dims,
            num_labels=num_labels,
            token=token,
            lora_config=lora_config,
            pooling_strategy=pooling_strategy,
            dropout=head_dropout,
        )
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info(
            "IntrogressionPredictor ready on %s (dtype=%s, seg=%d, batch=%d)",
            self.device, self.torch_dtype, segment_size, batch_size,
        )

    @staticmethod
    def _resolve_dtype(torch_dtype: str, use_bf16: bool, use_fp16: bool) -> torch.dtype:
        if use_bf16:
            return torch.bfloat16
        if use_fp16:
            return torch.float16
        name = (torch_dtype or "float32").lower().strip()
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(name, torch.float32)

    def _load_checkpoint(self):
        from safetensors.torch import load_model

        # 先以 float32 加载（与训练脚本一致），随后 .to(dtype) 到目标精度
        load_model(self.model, self.checkpoint_path)
        logger.info("Loaded checkpoint: %s", self.checkpoint_path)

    # ------------------------------------------------------------------
    def segment_bed(self, fasta_path: str, chrom: str, start: int, end: int) -> list[dict]:
        """将 [start, end) 区间切成与离线完全一致的 8k 片段。

        离线 0.run_fragments.py 对整条染色体先 trim_n（去头尾 N），再以
        8000 步长切无重叠片段，坐标 = trim 偏移 + 片段内偏移（绝对坐标）。

        这里对区间做同样的处理：读整段 -> trim_n -> 以 8000 步长切分，
        片段绝对坐标 = abs_start = start + span[0] + s。这样窗口内片段的
        绝对坐标与离线全基因组切分完全一致（含 trim 偏移），窗口聚合时
        用「标准网格 idx」按排序数组切片，与离线 aggregate_windows 一致。

        返回 [{chrom, start, end, sequence, index}]。
        """
        sequence = read_fasta_sequence(fasta_path, chrom, start, end)
        trimmed, span = trim_n(sequence)
        segs: list[dict] = []
        for s, e in sliding_window(trimmed, self.segment_size, self.segment_size):
            abs_start = start + span[0] + s
            abs_end = start + span[0] + e
            segs.append({
                "chrom": chrom,
                "start": abs_start,
                "end": abs_end,
                "sequence": trimmed[s:e],
                "index": abs_start // self.segment_size,
            })
        return segs

    # ------------------------------------------------------------------
    def predict_segments(
        self,
        windows: list[dict],
        progress_cb: Optional[object] = None,
    ) -> dict:
        """对一批 8k 片段序列执行 GPU 推理，返回概率字典。

        windows: [{"start": int, "end": int, "sequence": str}, ...]
        progress_cb: 可选回调 progress_cb(done_batches_after, total_batches)，
                     每个 batch 完成后调用（供进度端点轮询；无回调时零开销）。
        返回 {"prob_jap": np.ndarray, "prob_ind": np.ndarray, "logits": np.ndarray}
        """
        if not windows:
            return {"prob_jap": np.empty(0), "prob_ind": np.empty(0), "logits": np.empty((0, 2))}

        sequences = [w["sequence"] for w in windows]
        n_seq = len(sequences)
        n_batches = (n_seq + self.batch_size - 1) // self.batch_size
        probs_all_jap: list[float] = []
        probs_all_ind: list[float] = []
        logits_all: list[np.ndarray] = []

        with _INFER_LOCK:
            batch_idx = 0
            for i in range(0, len(sequences), self.batch_size):
                batch_seqs = sequences[i:i + self.batch_size]
                batch_idx += 1
                enc = self.tokenizer(
                    batch_seqs,
                    truncation=True,
                    padding="max_length",
                    max_length=self.segment_size,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                # 与离线 Trainer(fp16=True) 一致：权重保持 fp32（checkpoint 加载精度），
                # 前向用 torch.autocast 计算 fp16（省显存且数值与离线一致）。
                # 注意不是 model.to(fp16)——那会同时改权重 dtype。
                logits: torch.Tensor
                with torch.no_grad():
                    if self.torch_dtype == torch.float16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            _z, logits = self.model(input_ids, attention_mask)
                    elif self.torch_dtype == torch.bfloat16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _z, logits = self.model(input_ids, attention_mask)
                    else:
                        _z, logits = self.model(input_ids, attention_mask)
                logits = logits.float()
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, 2): [jap, ind]
                probs_all_jap.extend(probs[:, 0].tolist())
                probs_all_ind.extend(probs[:, 1].tolist())
                logits_all.append(logits.cpu().numpy())
                del enc, input_ids, attention_mask, logits
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if progress_cb is not None:
                    try:
                        # 片段进度按批累计（该批实际处理的序列数）
                        done_segs = min(n_seq, i + len(batch_seqs))
                        progress_cb(done_segments=done_segs, total_segments=n_seq,
                                    done_batches=batch_idx, total_batches=n_batches)
                    except Exception:
                        logger.exception("progress callback failed")

        return {
            "prob_jap": np.asarray(probs_all_jap, dtype=float),
            "prob_ind": np.asarray(probs_all_ind, dtype=float),
            "logits": np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, 2)),
        }

    # ------------------------------------------------------------------
    def release(self):
        try:
            if self.device.startswith("cuda"):
                self.model.cpu()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Predictor released.")


# ---------------------------------------------------------------------------
#  单例管理
# ---------------------------------------------------------------------------
def init_predictor():
    """从 .env 初始化单例预测器。"""
    if _PREDICTOR.get("instance") is not None:
        logger.info("Predictor already initialised, skipping.")
        return

    base_model_path = _env_str("BASE_MODEL_PATH", "")
    checkpoint_path = _env_str("CHECKPOINT_PATH", "")
    if not (base_model_path and checkpoint_path):
        logger.warning(
            "BASE_MODEL_PATH / CHECKPOINT_PATH not set — predictor will not "
            "be available until .env is configured."
        )
        return

    predictor = IntrogressionPredictor(
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        device=_env_str("DEVICE", "cuda:0"),
        torch_dtype=_env_str("MODEL_TORCH_DTYPE", "float32"),
        segment_size=_env_int("SEGMENT_SIZE", 8000),
        batch_size=_env_int("INFERENCE_BATCH_SIZE", 8),
        token=_env_str("SHARE_HF_TOKEN", None) or None,
        lora_config=_lora_config_from_env(),
        proj_dims=_proj_dims_from_env(),
        num_labels=_env_int("NUM_LABELS", 2),
        pooling_strategy=_env_str("POOLING_STRATEGY", "masked_mean"),
        head_dropout=_env_float("HEAD_DROPOUT", 0.1),
        use_bf16=_env_bool("MODEL_USE_BF16", False),
        use_fp16=_env_bool("MODEL_USE_FP16", False),
    )
    _PREDICTOR["instance"] = predictor
    logger.info("IntrogressionPredictor initialised and ready.")


def require_predictor() -> IntrogressionPredictor:
    inst = _PREDICTOR.get("instance")
    if inst is None:
        raise RuntimeError(
            "Predictor not initialised.  Ensure .env is configured and "
            "init_predictor() has been called."
        )
    return inst


def release_predictor():
    inst = _PREDICTOR.get("instance")
    if inst is not None:
        inst.release()
    _PREDICTOR["instance"] = None
    logger.info("Predictor released.")