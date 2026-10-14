# Derived from ultralytics/nn/add_modules/ShuffleNetV2.py; AGPL-3.0.
"""The actual 0.5x ShuffleNetV2 backbone used by the source configuration.

Stage repeats are [8, 4, 4], not the standard [4, 8, 4]. Four outputs and
legacy state-dict spelling ``banch1``/``banch2`` are retained for compatibility
with the project's multi-output parser and historical weights. The first
output is an early P3 feature; the next three are final P3, P4, P5 features.
"""

import torch
from torch import nn


def channel_shuffle(x, groups=2):
    batch, channels, height, width = x.size()
    x = x.view(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().view(batch, channels, height, width)


def _branch2(c1, c2, stride):
    return nn.Sequential(
        nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
        nn.BatchNorm2d(c2),
        nn.ReLU(inplace=True),
        nn.Conv2d(c2, c2, 3, stride, 1, groups=c2, bias=False),
        nn.BatchNorm2d(c2),
        nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
        nn.BatchNorm2d(c2),
        nn.ReLU(inplace=True),
    )


class InvertedResidual(nn.Module):
    """ShuffleNetV2 split/downsample block, preserving source tensor ordering."""

    def __init__(self, inp, oup, stride, benchmodel):
        super().__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        if stride not in (1, 2):
            raise ValueError("ShuffleNetV2 stride must be 1 or 2")
        half = oup // 2
        if benchmodel == 1:
            self.banch2 = _branch2(half, half, stride)
        else:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, half, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half),
                nn.ReLU(inplace=True),
            )
            # Historical source initializes and then replaces banch2. Retain
            # this RNG consumption so seeded source/refactor weights match.
            self.banch2 = _branch2(inp, half, stride)
            self.banch2 = _branch2(inp, half, stride)

    def forward(self, x):
        if self.benchmodel == 1:
            first = x[:, : x.shape[1] // 2, :, :]
            second = x[:, x.shape[1] // 2 :, :, :]
            out = torch.cat((first, self.banch2(second)), 1)
        else:
            out = torch.cat((self.banch1(x), self.banch2(x)), 1)
        return channel_shuffle(out)


class ShuffleNetV2Backbone(nn.Module):
    """Frozen 0.5x paper backbone; classification layers are not constructed."""

    def __init__(self):
        super().__init__()
        self.stage_repeats = [8, 4, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        input_channels = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        features = []
        for stage, repeats in enumerate(self.stage_repeats):
            output_channels = self.stage_out_channels[stage + 2]
            for repeat in range(repeats):
                features.append(InvertedResidual(input_channels, output_channels, 2 if repeat == 0 else 1,
                                                 2 if repeat == 0 else 1))
                input_channels = output_channels
        self.features = nn.Sequential(*features)
        self.index = self.stage_out_channels[2:5]
        # Preserve the original constructor's RNG consumption and BatchNorm
        # warm-up. Removing this requires a separate initialization protocol.
        self.width_list = [item.size(1) for item in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        outputs = [None, None, None, None]
        for index, block in enumerate(self.features):
            x = block(x)
            if index == 0:
                outputs[0] = x
            if x.size(1) in self.index:
                outputs[self.index.index(x.size(1)) + 1] = x
        return outputs


def shufflenetv2_05():
    """Factory recognized by the frozen parser's multi-output backbone branch."""
    return ShuffleNetV2Backbone()
