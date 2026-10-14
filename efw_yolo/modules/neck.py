# Derived from ultralytics/nn/add_modules/ESPBlock.py; AGPL-3.0.
"""Small residual stacking module (SRSM), the building block of the MFMN neck.

MFMN is the complete lateral/top-down/bottom-up topology in configs/models;
it is not a synonym for one ESPBlock. Source feature widths are nominal 256,
scaled to 64 by the nano width multiplier. The current source SRSM uses n=1,
e=0.5 at its outer projection, and e=1.0 inside its DSBottleneck. The inner
shortcut is inactive because its input and output widths differ (64 -> 32).
These source facts must not silently be changed to match prose descriptions.
"""

from torch import nn

from ultralytics.nn.modules import Bottleneck, C3
from ultralytics.nn.modules.conv import Conv


class DSConv(nn.Module):
    """Depthwise then pointwise convolution, each with BatchNorm and ReLU.

    Both convolutions retain source biases. Removing them would change the
    checkpoint layout and parameter count even though BatchNorm follows them.
    """

    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, p, groups=c1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        return self.act(self.bn2(self.pointwise(x)))


class DSBottleneck(nn.Module):
    """Two DSConv units with an identity connection only at equal widths."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = DSConv(c1, hidden, k[0], 1)
        self.cv2 = DSConv(hidden, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        result = self.cv2(self.cv1(x))
        return x + result if self.add else result


class _C3k(C3):
    """Preserved optional source branch; the paper's nano SRSM does not use it."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class SRSM(nn.Module):
    """Source-compatible ESPBlock with explicit paper naming.

    This compatibility profile intentionally supports the actual nano case
    (n=1). Historical ESPSerial connects 2*c -> c blocks in a loop and sizes
    cv2 for n*c despite forwarding only the last output; n>1 is not a valid
    serial stack. Reject it instead of advertising an untested larger model.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        if n != 1:
            raise ValueError("The frozen EFW-YOLO SRSM supports n=1 only; larger repeats require a new architecture.")
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(n * self.c, c2, 1)
        # The source superclass constructed this sequence before ESPBlock
        # replaced it. Preserve RNG consumption for same-seed initialization.
        self.m = nn.ModuleList(
            Bottleneck(2 * self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.m = nn.ModuleList(
            _C3k(2 * self.c, self.c, 2, shortcut, g)
            if c3k else DSBottleneck(2 * self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        x = self.cv1(x)
        for block in self.m:
            x = block(x)
        return self.cv2(x)
