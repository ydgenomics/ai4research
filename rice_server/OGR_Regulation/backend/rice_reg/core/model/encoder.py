"""ATAC signal encoder (1D CNN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATAC_Encoder(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(512, output_dim, kernel_size=1)
        for module in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            module.weight.data = module.weight.data.float()
            if module.bias is not None:
                module.bias.data = module.bias.data.float()

    def forward(self, atac_signal):
        x = atac_signal.unsqueeze(1).float()
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        return self.conv5(x)
