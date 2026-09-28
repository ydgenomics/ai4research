"""CCS-N 工具: 模型加载 / embedding 提取 / CCS-N 公式计算。

参考: func_embedding/script/ogr_extract/embedder.py 的 fp32 池化修复。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# ---------------------------- 模型加载 ----------------------------

def load_model(model_path: str, device: str = "cuda", model_class: str = "causal_lm"):
    """加载 tokenizer + 模型。返回 (tok, model)。

    model_class:
      - 'causal_lm': AutoModelForCausalLM (单碱基 / k-mer 因果模型)
      - 'auto': AutoModel (ESM 双向 / 自定义架构, 如 agront_1b, Botanic0-L)
      - 'ntv3': NTv3 专用 (自定义配置类, 内部管理 fp32 卷积精度, 不能传 dtype)
      - 'caduceus': PlantCAD2 专用 (加载 Caduceus 主体 backbone, RCPS 双向 SSM)
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # 关键: 设置当前 CUDA 设备, 让 triton autotuner / 默认操作也在目标卡上
    # (否则 triton autotuner 在 cuda:0 上 benchmark, 若 cuda:0 显存不足会崩溃)
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if model_class == "ntv3":
        # NTv3: 自定义 Ntv3PreTrainedConfig + NTv3PreTrained, 卷积层需要内部 fp32 管理
        import sys
        from pathlib import Path as _P

        _p = _P(model_path)
        sys.path.insert(0, str(_p))
        import configuration_ntv3_pretrained as _cmod
        import modeling_ntv3_pretrained as _mmod

        cfg = _cmod.Ntv3PreTrainedConfig.from_pretrained(model_path)
        model = _mmod.NTv3PreTrained.from_pretrained(model_path, config=cfg).to(device)
    elif model_class == "caduceus":
        # PlantCAD2: AutoModelForMaskedLM 加载 CaduceusForMaskedLM (RCPS 双向 SSM)
        # 用 HF trust_remote_code 自动处理相对导入
        from transformers import AutoModelForMaskedLM

        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)
        # Caduceus forward 不接受 attention_mask, 包装一层丢弃
        _orig_forward = model.forward

        def _caduceus_forward(input_ids=None, attention_mask=None, **kwargs):
            return _orig_forward(input_ids=input_ids, **kwargs)

        model.forward = _caduceus_forward
    else:
        cls = AutoModelForCausalLM if model_class == "causal_lm" else AutoModel
        model = cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)
    model.eval()
    return tok, model


def embed_sequences(tok, model, sequences: list[str], device: str = "cuda"):
    """批量对序列前向, 返回最后一层 hidden states [B, T, H] (fp32 numpy) + input_ids。

    返回 (hidden, input_ids, attention_mask):
    - hidden: [B, T, H] fp32
    - input_ids: [B, T] 每序列的 token ids (用于差分定位变异 token)
    - attention_mask: [B, T]

    兼容: HF 模型 (out.hidden_states) 与 NTv3 (out["hidden_states"] dict)。
    """
    import torch

    enc = tok(sequences, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is None:
        # NTv3 等 tokenizer 不返回 attention_mask, 用全 1 (等长序列时无 padding)
        attn = torch.ones_like(input_ids)
    attn = attn.to(device)
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        except TypeError:
            # Caduceus 等 SSM 模型不接受 attention_mask
            out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states if hasattr(out, "hidden_states") else out["hidden_states"]
    hidden = hs[-1].float().cpu().numpy()  # [B, T, H] fp32
    return hidden, input_ids.cpu().numpy(), attn.cpu().numpy()


def find_variant_token(ref_ids: np.ndarray, alt_ids: np.ndarray, var_idxs: np.ndarray, tokenizer_type: str = "kmer_diff") -> np.ndarray:
    """用 token id 差分定位变异 token (适用于非单碱基 tokenizer)。
    
    为了避免在 2048 bp 窗口中由于背景变异导致定位到错误的 Token，
    我们寻找与预期目标 Token 位置 (1 + var_idx // 6) 最接近的、且 ref_ids 与 alt_ids 不同的 Token。

    ref_ids/alt_ids: [B, T] token ids (已 padding)。
    var_idxs: [B] 物理碱基变异位置 (0-based)。
    返回: [B] 每个序列 ref/alt 目标变异 token 的下标 (0-based)。
    """
    B, T = ref_ids.shape
    out = np.full(B, -1, dtype=np.int64)
    for i in range(B):
        expected_tok_idx = 1 + var_idxs[i] // 6
        best_j = -1
        min_dist = float("inf")
        for j in range(T):
            if ref_ids[i, j] != alt_ids[i, j]:
                dist = abs(j - expected_tok_idx)
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
        out[i] = best_j if best_j != -1 else expected_tok_idx
    return out


# ---------------------------- 距离度量 ----------------------------

def token_distances(ref_h: np.ndarray, alt_h: np.ndarray) -> dict[str, np.ndarray]:
    """逐 token 距离 [B, T]。ref_h/alt_h: [B, T, H] fp32。

    - euclidean: ||ref - alt||_2
    - cosine:    1 - cos(ref, alt) (越小越相似, 0=完全一致)
    """
    diff = ref_h - alt_h
    euclid = np.linalg.norm(diff, axis=-1)  # [B, T]
    rn = np.linalg.norm(ref_h, axis=-1, keepdims=True)
    an = np.linalg.norm(alt_h, axis=-1, keepdims=True)
    cos_sim = (ref_h * alt_h).sum(-1, keepdims=True) / (rn * an).clip(min=1e-12)
    cos_dist = 1.0 - cos_sim[..., 0]
    return {"euclidean": euclid, "cosine": cos_dist}


# ---------------------------- CCS-N ----------------------------

def ccs_n_scores(
    dist: np.ndarray,  # [B, T] 逐 token 距离
    var_idx: np.ndarray,  # [B] 每个窗口内变异所在 token 下标
    N_list: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8, 16),
) -> dict[int, np.ndarray]:
    """计算 CCS-N = s0 + Σ(si+ + si-) for i in 1..N。

    Args:
        dist: 逐 token 距离矩阵 [B, T]。
        var_idx: 变异 token 下标 [B]。
        N_list: 累加半径集合。

    Returns:
        {N: [B] 每个窗口的 CCS-N 得分}
    """
    B, T = dist.shape
    out = {}
    for N in N_list:
        s = dist[np.arange(B), var_idx].copy()  # s0
        for i in range(1, N + 1):
            r = var_idx + i
            l = var_idx - i
            sr = dist[np.arange(B), np.clip(r, 0, T - 1)] * (r < T)
            sl = dist[np.arange(B), np.clip(l, 0, T - 1)] * (l >= 0)
            s = s + sr + sl
        out[N] = s
    return out


# ---------------------------- 变异感知评分 ----------------------------

def variant_awareness_scores(
    real_ccs: np.ndarray,  # [B] 真实 SNV 的 CCS-N
    rand_ccs: np.ndarray,  # [B] 随机对照的 CCS-N
) -> dict[str, float]:
    """评估模型对变异 vs 随机对照的区分度。

    - mean_real / mean_rand: 平均 CCS-N
    - auc: 用 real/rand 得分做二分类的 AUC (0.5=无区分度, 1.0=完全区分)
    """
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones_like(real_ccs), np.zeros_like(rand_ccs)])
    x = np.concatenate([real_ccs, rand_ccs])
    auc = roc_auc_score(y, x) if np.unique(y).size == 2 else float("nan")
    return {
        "mean_real": float(np.mean(real_ccs)),
        "mean_rand": float(np.mean(rand_ccs)),
        "ratio": float(np.mean(real_ccs) / max(np.mean(rand_ccs), 1e-12)),
        "auc": float(auc),
    }
