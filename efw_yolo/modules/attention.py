# Derived from the project-modified Ultralytics runtime; AGPL-3.0.
"""Paper LKAA = 7x7 depthwise convolution followed by parameter-free PfAAM.

Historical ``nn/add_modules/LSKA.py:LSKA`` actually implements PfAAM, not the
large separable-kernel attention implemented by that file's unused ``LSKA1``.
The model YAML keeps DWConv and PfAAM as separate nodes to preserve checkpoint
keys and all lateral/skip indices. LKAA below is a reusable mathematical wrapper,
not a replacement node in the compatibility configuration.
"""

import torch
from torch import nn

from ultralytics.nn.modules.conv import DWConv


class PfAAM(nn.Module):
    """Reweight U by sigmoid(mean_HW(U) * mean_C(U)); no trainable parameters.

    c2 and ratio are accepted solely for the historical parser call signature;
    they never changed the old computation or channel count.
    """

    def __init__(self, c1=None, c2=None, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        channel_descriptor = self.avg_pool(x).view(batch, channels, 1, 1).expand_as(x)
        spatial_descriptor = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        return x * self.sigmoid(channel_descriptor * spatial_descriptor)


class LKAA(nn.Sequential):
    """Convenience composition; EFW-YOLO YAML uses these as two separate nodes.

    The source DWConv includes BatchNorm and SiLU. Its 7x7 operation is not a
    bare torch.nn.Conv2d; this wrapper deliberately retains that distinction.
    """

    def __init__(self, channels):
        super().__init__(DWConv(channels, channels, 7, 1, 3), PfAAM(channels, channels))
