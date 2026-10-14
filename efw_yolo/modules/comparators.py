# Derived from IR_YOLO/ultralytics/ultralytics/nn/Addmodules/AttentionTriplet.py.
# Source file: AGPL-3.0. This module is a comparison candidate, not evidence that
# this exact implementation/configuration generated the manuscript's results.
"""Explicit attention comparator kept separate from the final EFW-YOLO modules."""

import torch
from torch import nn


class _BasicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class _ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class _AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = _ZPool()
        self.conv = _BasicConv()

    def forward(self, x):
        return x * torch.sigmoid_(self.conv(self.compress(x)))


class TripletAttention(nn.Module):
    """Three rotated attention gates, copied mathematically from local reference.

    No local evidence connects this candidate's layer placement to the original
    paper experiment. Its configuration is explicitly marked reconstructed.
    """

    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = _AttentionGate()
        self.hc = _AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = _AttentionGate()

    def forward(self, x):
        first = self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        second = self.hc(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        return (self.hw(x) + first + second) / 3 if not self.no_spatial else (first + second) / 2
