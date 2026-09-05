"""模型定义 —— 与 20.introgression_analysis/models/model.py 完全一致。

结构：
- BackboneModule: Mixtral-1B DNA 基座（AutoModel + 可选 LoRA）
- PoolingLayer: masked_mean / cls_token / attention_weighted
- ProjectionHead: 对比投影头（含 L2 归一化）
- ClassificationHead: 双标签分类头（Jap / Ind）
- FullModel: 组装器 forward -> (z, logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class PoolingLayer(nn.Module):
    """Pooling 策略模块，支持多种池化方式"""

    def __init__(self, strategy="masked_mean"):
        super().__init__()
        self.strategy = strategy
        assert strategy in ["masked_mean", "cls_token", "attention_weighted"], \
            f"Unknown pooling strategy: {strategy}"

    def forward(self, hidden, attention_mask):
        """
        Args:
            hidden: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        if self.strategy == "masked_mean":
            # Masked mean pooling: ignore paddings
            # （保持与离线 models/model.py 完全一致：autocast 下运行，无需改动）
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden).float()
            sum_hidden = (hidden * attention_mask_expanded).sum(dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.strategy == "cls_token":
            return hidden[:, 0, :]

        elif self.strategy == "attention_weighted":
            attention_mask = attention_mask.unsqueeze(-1)
            attention_weights = attention_mask.float() / attention_mask.float().sum(dim=1, keepdim=True)
            return (hidden * attention_weights).sum(dim=1)


class BackboneModule(nn.Module):
    """Backbone 模块，支持 LoRA 微调"""

    def __init__(self, model_path, token=None, lora_config=None):
        super().__init__()
        self.lora_enabled = lora_config is not None

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=token,
        )

        if lora_config:
            peft_conf = LoraConfig(
                r=lora_config["r"],
                lora_alpha=lora_config["alpha"],
                lora_dropout=lora_config["dropout"],
                target_modules=lora_config["target_modules"],
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, peft_conf)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class ProjectionHead(nn.Module):
    """投影头，用于对比学习"""

    def __init__(self, dims, is_bn=False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if is_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


class ClassificationHead(nn.Module):
    """分类头"""

    def __init__(self, input_dim, num_labels, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.fc(self.dropout(x))


class FullModel(nn.Module):
    """完整模型：组装 backbone、pooling、projection、classifier"""

    def __init__(
        self,
        model_path,
        proj_dims,
        num_labels,
        token=None,
        lora_config=None,
        pooling_strategy="masked_mean",
        dropout=0.1,
        is_bn=False,
    ):
        super().__init__()

        self.backbone = BackboneModule(model_path, token, lora_config)
        self.pooling = PoolingLayer(strategy=pooling_strategy)
        self.proj = ProjectionHead(proj_dims, is_bn=is_bn)
        self.cls = ClassificationHead(proj_dims[-1], num_labels, dropout=dropout)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Returns:
            z: 对比学习的嵌入 [batch_size, proj_dims[-1]]
            logits: 分类器输出 [batch_size, num_labels]
        """
        del labels, kwargs
        hidden = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden, attention_mask)
        z = self.proj(pooled)
        logits = self.cls(z)
        return z, logits