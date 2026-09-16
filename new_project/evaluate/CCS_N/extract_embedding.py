# 序列的构建
# GPU资源的配置
# 模型的配置，token化，前向
# 提取embedding的保存，none和pooling后的

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------- 序列的构建 ----------------------------
# 输入就是一段 DNA 序列；也可以从文件读：# SEQUENCE = Path("seq.fa").read_text().strip()
SEQUENCE = "ACGT" * 64

# ---------------------------- GPU 资源的配置 ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---------------------------- 模型的配置 ----------------------------
MODEL_PATH = "/mnt/rice/default/Workspace/yangdong/ai4research/rice_server/source/rice_mut/rice_1B_stage2_8k_hf"
POOLING = "mean"  # pooled 用哪种池化: mean / last / max
SAVE_TOKENS = True  # 是否保存 per-token(none)的完整 embedding
OUT_PATH = Path(__file__).parent / "output" / "embed.npz"


def load_model():
    """tokenizer + 模型(单碱基 tokenizer, 不加特殊 token)。"""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    model.eval()
    return tok, model


def tokenize_forward(tok, model, sequence):
    """token 化 + 前向 -> 最后一层 hidden [1, T, H] 和 attention_mask [1, T]。

    1 bp = 1 token(单碱基 tokenizer, 不加特殊 token), 故 token 下标 == 序列下标。
    """
    enc = tok(sequence, add_special_tokens=False, return_tensors="pt")
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=mask,
            output_hidden_states=True,
        )
    return out.hidden_states[-1], mask


def pool(hidden, mask, method=POOLING):
    """[1, T, H] -> [1, H]; method='none' 时直接返回 [1, T, H]。"""
    if method == "none":
        return hidden.half()
    h = hidden.float()
    if method == "mean":
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1)
    if method == "last":
        return h[torch.arange(h.size(0)), mask.sum(1) - 1]
    if method == "max":
        return h.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf")).max(1).values
    raise ValueError(f"未知池化方法: {method}")


def main():
    tok, model = load_model()
    hidden, mask = tokenize_forward(tok, model, SEQUENCE)

    arrays = {
        "pooled": pool(hidden, mask)[0].cpu().numpy(),  # [H]
        "seq_len": len(SEQUENCE),
        "token_len": hidden.size(1),
        "pooling": POOLING,
    }
    if SAVE_TOKENS:
        arrays["none"] = hidden[0].half().cpu().numpy()  # [T, H]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **arrays)
    print(f"hidden {tuple(hidden.shape)} -> pooled {arrays['pooled'].shape}, 已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()

'''
import numpy as np

data = np.load("/mnt/rice/default/Workspace/yangdong/ai4research/new_project/evaluate/CCS_N/output/embed.npz")

# 查看里面有哪些数组（键名）
print(data.files)

# 按 key 取出数组
for key in data.files:
    arr = data[key]
    print(key, arr.shape, arr.dtype)
'''