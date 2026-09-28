# -*- coding: utf-8 -*-
"""VariantCPTTrainer(Trainer 子类)：覆写 sampler / dataloader / compute_loss；
驱动 冒烟(定 lr) 与 正式训练。

采样策略（yaml window.sampling_mode）:
  - density        原行为：w = n_variants^power，cap p99（无变异窗权重=0）
  - two_stage     两段采样：①覆盖段——每窗均匀抽 coverage_k 个代表样本（位置全覆盖 100%）
                             ②聚焦段——剩余预算 density 加权抽高变异窗（变异聚焦）
  （两段采样在 Dataset 上用显式 (row, col) 列表，见 get_train_dataloader）
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer, TrainingArguments

from model_head import VariantAwareCausalLM
from variant_dataset import VariantCPTDataset, VariantCollator, build_char_token_map
from window_sampling import build_index_pairs

logger = logging.getLogger("variant_cpt.trainer_utils")


class VariantCPTTrainer(Trainer):
    """自定义 Trainer：GVL 懒加载数据 + 变异聚焦 loss_mask。"""

    def __init__(self, *args, gvl_args=None, **kwargs):
        self.gvl_ds = gvl_args["ds"]
        self.rows = gvl_args["rows"]          # 训练 (row, col) 索引
        self.cols = gvl_args["cols"]
        self.char_map = gvl_args["char_map"]
        self.loss_cfg = gvl_args["loss_cfg"]
        self.window_len = gvl_args["window_len"]
        self.density_power = gvl_args.get("density_power", 0.5)
        self.density_cap_p = gvl_args.get("density_cap_p", 0.99)
        self.sampling_mode = gvl_args.get("sampling_mode", "density")
        self.coverage_k = int(gvl_args.get("coverage_k", 8))
        # 两段采样：覆盖段固定代表样本列表（每窗 coverage_k 个，均匀），聚焦段 density 池
        self.cover_rows = None
        self.cover_cols = None
        self.cover_frac = float(gvl_args.get("cover_frac", 0.6))   # 覆盖段占预算比例
        self._build_cover_pool()
        super().__init__(*args, **kwargs)

    def _step_budget(self):
        """理论步预算（不截断）：max_steps × gradient_accumulation_steps × per_device_batch_size。

        两段采样用它 × world_size 构建全局数据集：每 rank 恰好消费 max_steps 步对应的样本数，
        一个 epoch 跑满 max_steps，无需 dataloader 重建（避免跨 epoch 重复）。
        """
        ms = int(getattr(self.args, "max_steps", -1))
        accum = max(int(getattr(self.args, "gradient_accumulation_steps", 1)), 1)
        bs = max(int(self.args.per_device_train_batch_size), 1)
        if ms <= 0:          # 未设 max_steps（epoch 驱动）时退回：全部 (row,col) 对
            return len(self.rows)
        return ms * accum * bs

    def _train_budget(self):
        """单卡样本预算（截断到 n_total）：max_steps × accum × bs。

        供 density 模式（WeightedRandomSampler 的 num_samples 不能超过 len(weights)）使用。
        """
        n_total = len(self.rows)
        return min(n_total, self._step_budget())

    def _world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return max(int(os.environ.get("WORLD_SIZE", "1")), 1)

    def _rank(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", "0"))

    def _build_cover_pool(self):
        """预构建两段采样·覆盖段池：每窗 coverage_k 个代表样本（确定性 seed 42）。

        排序关键：窗口外循环、样本内循环 → Sequential 消费前 n 个时最大化"不同窗覆盖"。
        e.g. 预算 5120 微批 ≥ 窗数时，前 22773 个恰好每窗都算到 1 次 → 位置全覆盖。
        """
        if self.sampling_mode != "two_stage":
            return
        rng = np.random.default_rng(42)
        windows = np.unique(self.rows)
        n_cols = self.gvl_ds.shape[1]
        k = min(self.coverage_k, n_cols)
        # 每窗抽 k 个代表样本（不重复）
        cols_by_win = [rng.choice(n_cols, size=k, replace=False) for _ in windows]
        cover_rows, cover_cols = [], []
        for j in range(k):                       # 样本轮 → 窗循环：保证前 n_win 个覆盖全部窗
            for i, w in enumerate(windows):
                cover_rows.append(int(w))
                cover_cols.append(int(cols_by_win[i][j]))
        self.cover_rows = np.asarray(cover_rows, dtype=np.int64)
        self.cover_cols = np.asarray(cover_cols, dtype=np.int64)
        logger.info(f"两段采样·覆盖段: {len(windows)} 窗 × K={k} 样本 → {len(self.cover_rows)} 索引对")

    # --- sampler：两段采样（覆盖段 → 聚焦段）或 原 density ---
    def get_train_sampler(self):
        if self.sampling_mode == "two_stage":
            # 两段采样的索引合并已在 get_train_dataloader 预构建（self._rows_use/_cols_use）
            # 数据集长度 = 全局预算（所有 rank 共享同一 dataset），DistributedSampler 交错切分：
            #   rank r 消费 dataset[r], dataset[r+ws], ... → 不同 rank 取不同样本，零重复、批梯度真实×ws
            n_use = getattr(self, "_n_use", 0)
            return torch.utils.data.distributed.DistributedSampler(
                list(range(n_use)), num_replicas=self._world_size(),
                rank=self._rank(), shuffle=False)
        # 原 density：加权随机采样；不同 rank 用不同 seed → 抽到不同样本（错误机会均等）
        w = self._weights()
        gen = torch.Generator().manual_seed(
            int(np.random.default_rng(0).integers(0, 2**31)) + self._rank())
        return torch.utils.data.WeightedRandomSampler(
            torch.tensor(w, dtype=torch.float64),
            num_samples=self._train_budget(), replacement=False, generator=gen)

    def _density_weights_for(self, rows: np.ndarray):
        """对给定 rows 数组生成 density 权重（窗级计数映射到样本级）。"""
        ds = self.gvl_ds
        ridx = np.unique(rows)
        try:
            n = ds.n_variants(regions=ridx)
            cnt = np.asarray(n).reshape(len(ridx), -1).max(axis=-1)
        except Exception as e:
            logger.warning(f"n_variants 不可用({e})，全窗等权")
            cnt = np.ones(len(ridx))
        cnt_map = {int(r): float(int(c)) for r, c in zip(ridx, cnt)}
        w = np.array([cnt_map.get(int(r), 1.0) for r in rows], dtype=np.float64)
        w = w ** self.density_power
        cap = np.quantile(w, self.density_cap_p)
        return np.minimum(w, cap)

    def _weights(self):
        # density 采样：w = n_variants^density_power，cap 到 density_cap_p（无变异窗权重=0，不参与采样）
        return self._density_weights_for(self.rows)

    # --- dataloader：GVL collate ---
    def get_train_dataloader(self):
        if self.sampling_mode == "two_stage":
            self._rows_use, self._cols_use, self._n_use = self._build_two_stage_pairs()
            ds = VariantCPTDataset(self.gvl_ds, self._rows_use, self._cols_use)
        else:
            self._n_use = len(self.rows)
            ds = VariantCPTDataset(self.gvl_ds, self.rows, self.cols)
        collate = VariantCollator(self.gvl_ds, self.char_map, self.loss_cfg, self.window_len)
        return DataLoader(ds, batch_size=self.args.per_device_train_batch_size,
                          sampler=self.get_train_sampler(), collate_fn=collate,
                          num_workers=self.args.dataloader_num_workers or 0,
                          pin_memory=torch.cuda.is_available(), drop_last=True)

    def _build_two_stage_pairs(self):
        """两段采样最终索引对：覆盖段（每窗 cover_rows/cover_cols 均匀）+ 聚焦段（density 补充）。

        比例：覆盖段 ≤ budget×cover_frac（至少 1 轮全窗位置覆盖）；剩余 → 聚焦段。
        预算不足一轮全窗覆盖时（冒烟小步数）：全部预算给覆盖段，聚焦段 0。
        """
        # 全局预算：数据集按「理论步预算 × world_size」构建（覆盖池 + 聚焦段），
        # 再由 DistributedSampler 切分 → 每 rank 恰好消费 max_steps×accum×bs 个不同样本，
        # 一个 epoch 跑满 max_steps；覆盖段取满覆盖池（位置全覆盖），剩余归聚焦段
        budget = self._step_budget() * self._world_size()
        budget_cover = max(int(budget * self.cover_frac), len(np.unique(self.cover_rows)))
        n_cover = min(len(self.cover_rows), budget_cover)
        if n_cover > budget:
            n_cover = budget                     # 预算不足：全部给覆盖段
        # 覆盖段：前 n_cover 个（窗口外循环，前 n_win 个已覆盖全部窗）
        rows_use = self.cover_rows[:n_cover].copy()
        cols_use = self.cover_cols[:n_cover].copy()
        n_focus = max(budget - n_cover, 0)
        logger.info(f"两段采样: 覆盖段 {n_cover} / 聚焦段 {n_focus}（总预算 {budget}，cover_frac={self.cover_frac}）")
        if n_focus > 0:
            # 聚焦段：从全 (row,col) 对按 density 权重无放回抽样，避免与覆盖段重复
            w = self._density_weights_for(self.rows)
            gen = torch.Generator().manual_seed(int(np.random.default_rng(0).integers(0, 2**31)))
            idx_all = np.arange(len(self.rows))
            # 已覆盖对的哈希去重
            covered = set(zip(rows_use.tolist(), cols_use.tolist()))
            remain = np.array([i for i in idx_all
                               if (int(self.rows[i]), int(self.cols[i])) not in covered])
            w_remain = w[remain] if len(remain) > 0 else np.array([], dtype=np.float64)
            if len(remain) > 0:
                pick = torch.utils.data.WeightedRandomSampler(
                    torch.tensor(w_remain, dtype=torch.float64),
                    num_samples=min(n_focus, len(remain)), replacement=False, generator=gen)
                picked = remain[list(pick)]
                rows_use = np.concatenate([rows_use, self.rows[picked]])
                cols_use = np.concatenate([cols_use, self.cols[picked]])
        return rows_use, cols_use, len(rows_use)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        import torch.nn.functional as F

        input_ids = inputs["input_ids"]
        loss_mask = inputs.get("loss_mask")
        out = model(input_ids=input_ids, loss_mask=loss_mask)
        loss = out.loss
        # 分离信号：从 logits 直接重算（DataParallel 会重构 ModelOutput 丢附加属性，
        # 故不依赖 model_head 附加的 variant_ce/background_ce，保证多卡模式也可靠）
        sig = {}
        if loss_mask is not None and out.logits is not None:
            sl = out.logits.detach()[:, :-1].contiguous()
            st = input_ids[:, 1:].contiguous()
            ce = F.cross_entropy(sl.transpose(1, 2), st, reduction="none")   # (B, L-1)
            m = loss_mask[:, 1:].float()
            is_var = (m > 0.1).float()
            weighted = (m > 0).float()
            if is_var.sum() > 0:
                sig["variant_ce"] = float((ce * is_var).sum() / is_var.sum().clamp(min=1e-8))
            bg = weighted * (1 - is_var)
            if bg.sum() > 0:
                sig["background_ce"] = float((ce * bg).sum() / bg.sum().clamp(min=1e-8))
            sig["variant_frac"] = float(is_var.mean())
        self._last_sig = sig          # 供 log() 打印
        if return_outputs:
            return loss, out
        return loss

    def log(self, logs, start_time=None):
        """在每步日志中附加分离信号：variant_ce（变异区 CE）+ background_ce（背景区 CE）。

        信号解读：variant_ce 下降 = 学到变异；background_ce 保持 = 无灾难性遗忘。
        """
        for k, v in (getattr(self, "_last_sig", None) or {}).items():
            logs[k] = v
        super().log(logs, start_time)


def _build(config, split):
    from gvl_pipeline import open_dataset
    return open_dataset(config, chroms=config["window"][f"{split}_chroms"])


def _disable_deepspeed():
    """同 main.py：从 trainer_utils 独立入口再确保一次（保险，幂等）。"""
    import importlib.util

    if getattr(_disable_deepspeed, "_done", False):
        return
    _orig = importlib.util.find_spec

    def _find_spec(name, package=None):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return _orig(name, package)

    importlib.util.find_spec = _find_spec
    _disable_deepspeed._done = True


def run_training(cfg: dict, save_final: bool = True):
    from gvl_pipeline import open_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _disable_deepspeed()          # 保险：任何入口都先屏蔽 deepspeed 探测

    # 1. 数据
    ds_train = open_dataset(cfg, cfg["window"]["train_chroms"])
    rows, cols = build_index_pairs(cfg, ds_train, cfg["window"]["train_chroms"])
    # 2. 模型
    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["path"], trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg["compute"]["bf16"] else torch.float32,
        attn_implementation="flash_attention_2" if cfg["compute"]["flash_attn"] else None,
        device_map=None,
    ).to("cuda")
    # 2.1 LoRA 可选：lora=false 全参微调；lora=true 或 {r/alpha/dropout/target_modules} 冻结基模只训适配器
    lora_cfg = cfg["train"].get("lora", False)
    lora_enabled = bool(lora_cfg) if not isinstance(lora_cfg, dict) else lora_cfg.get("enabled", True)
    if lora_enabled:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError("lora=true 但未安装 peft，请 pip install peft") from e
        lora_r = lora_cfg.get("r", 8) if isinstance(lora_cfg, dict) else 8
        lora_alpha = lora_cfg.get("alpha", 16) if isinstance(lora_cfg, dict) else 16
        lora_dropout = lora_cfg.get("dropout", 0.05) if isinstance(lora_cfg, dict) else 0.05
        lora_targets = (lora_cfg.get("target_modules", None) if isinstance(lora_cfg, dict)
                        else None) or ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_conf = LoraConfig(
            task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=lora_targets,
        )
        base = get_peft_model(base, peft_conf)
        # grad checkpointing + 冻结基模：输入需要梯度占位，否则 checkpoint 梯度链断裂
        # （"None of the inputs have requires_grad=True" → backward 报 no grad_fn）
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        base.print_trainable_parameters()
        logger.info(f"LoRA 微调: r={lora_r} alpha={lora_alpha} dropout={lora_dropout} targets={lora_targets}")
    model = VariantAwareCausalLM(base, aux_loss_coef=cfg["train"].get("aux_loss_coef", 1e-3))

    # ── DDP find_unused_parameters（自动判断，yaml 可显式覆盖）──
    # 只对 requires_grad=True 的参数起作用：
    #   · 全参微调 / 纯 attention LoRA（q/k/v/o_proj）→ 每步都用得到 → False（省 autograd 遍历，更快）
    #   · LoRA 覆盖 MoE expert 层（experts/w1/w2/w3，top-2 路由会跳过部分）→ 未使用参数存在，必须 True
    destr = "auto"
    if lora_enabled:
        destr = ",".join(lora_targets).lower() if isinstance(cfg["train"].get("lora"), dict) else ""
        if "expert" in destr or any(k in destr for k in ("w1", "w2", "w3", "block_sparse", "moe.")):
            destr = "experts"            # 覆盖 MoE → 必须 True
        else:
            destr = "attn"               # 纯 attention → False 安全
    lora_touched_experts = (destr == "experts")
    find_unused = cfg["train"].get("ddp_find_unused_parameters", "auto")
    if find_unused == "auto":
        find_unused = bool(lora_enabled and lora_touched_experts)
    logger.info(f"DDP find_unused_parameters = {find_unused}"
                f"（LoRA 覆盖{'MoE experts→必须 True' if lora_touched_experts else '纯 attention→可 False'}，yaml 显式值={cfg['train'].get('ddp_find_unused_parameters', 'auto')}）")

    args = TrainingArguments(
        output_dir=cfg["train"]["output_dir"], overwrite_output_dir=True,
        per_device_train_batch_size=cfg["compute"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["compute"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["lr"], warmup_steps=cfg["train"]["warmup_steps"],
        weight_decay=cfg["train"]["weight_decay"], max_steps=cfg["train"]["max_steps"],
        save_steps=cfg["train"]["save_steps"], logging_steps=cfg["train"]["logging_steps"],
        bf16=cfg["compute"]["bf16"], gradient_checkpointing=cfg["compute"]["grad_ckpt"],
        # ★ MoE + DDP 必须非重入 checkpoint：reentrant 模式 backward 重放 forward，
        #   同一 expert 参数被多个 token 复用 → DDP 报 "marked as ready twice"（已踩坑）。
        #   非重入只存激活不重放，天然规避；transformers 4.57 默认 use_reentrant=True，必须显式关。
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg["compute"].get("dataloader_workers", 0),
        remove_unused_columns=False, report_to=[],
        # 自动判定：全参/纯 attention LoRA → False（性能）；LoRA 覆盖 expert → True（必须）
        ddp_find_unused_parameters=find_unused,
    )
    trainer = VariantCPTTrainer(
        model=model, args=args, tokenizer=tok,
        data_collator=lambda x: x,           # collate 已由自定义 DataLoader 完成
        gvl_args=dict(ds=ds_train, rows=rows, cols=cols,
                      char_map=build_char_token_map(tok),
                      loss_cfg=cfg["loss"], window_len=cfg["window"]["length"],
                      density_power=cfg["window"].get("density_power", 0.5),
                      density_cap_p=cfg["window"].get("density_cap_p", 0.99),
                      sampling_mode=cfg["window"].get("sampling_mode", "density"),
                      coverage_k=int(cfg["window"].get("coverage_k", 8)),
                      cover_frac=float(cfg["window"].get("cover_frac", 0.6))),
    )
    trainer.train()

    # ★ 保存基模本身（无 base. 前缀 + config.json + tokenizer）→ 可直接当 OGR 基模加载
    #   包装模型 VariantAwareCausalLM 只是计算层，参数全在 base；存 trainer 会带 base. 前缀不可复用
    if save_final:
        final_dir = Path(cfg["train"]["output_dir"]) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        # LoRA：先 merge 回原始权重再保存，得完整 OGR 基模格式
        if lora_enabled:
            if hasattr(base, "merge_and_unload"):
                base = base.merge_and_unload()
        if hasattr(base, "save_pretrained"):
            base.save_pretrained(final_dir, safe_serialization=True)   # model.safetensors + config.json
        else:
            # 双保险：若 peft 包装异常，退化保存当前状态
            base.model.save_pretrained(final_dir, safe_serialization=True)
        tok.save_pretrained(final_dir)                             # tokenizer 文件
        logger.info(f"微调权重已保存（OGR 基模格式，可直接加载）: {final_dir}")
    logger.info("训练完成。")


def run_smoke(cfg: dict, max_steps: int = 20):
    """冒烟：20 步内打印 loss、variant_loss_frac、等位分辨提示，辅助定 lr（C5）。"""
    cfg = {**cfg, "train": {**cfg["train"], "max_steps": max_steps,
                            "save_steps": 1_000_000, "logging_steps": 1,
                            "output_dir": cfg["train"]["output_dir"] + "_smoke"}}
    run_training(cfg, save_final=False)   # 冒烟只定 lr，不产出最终权重