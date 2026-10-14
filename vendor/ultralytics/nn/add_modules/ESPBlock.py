# ESPDet 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Block modules."""
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, C3, Bottleneck
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import DFL, PSABlock
from ultralytics.nn.modules.head import Detect


__all__ = (
    # "DSBottleneck",
    # "DSC3k2",
    # "ESPSerial",
    # "ESPSerialLite",
    "ESPBlock",
    "ESPBlockLite",
    # "DSConv",
    "ESPDetect",
)


class ESPDetect(Detect):
    def __init__(self, nc=1, ch=()):
        """Initializes the ESP detection layer with specified number of classes and channels."""
        super().__init__(nc, ch)
        self.reg_max = 1
        self.no = nc + self.reg_max * 4  # number of outputs per anchor =9

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(DSConv(x, c2, 3), DSConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        ) #cv2 is CIoU branch
        self.cv3 = nn.ModuleList(
            nn.Sequential(   #cv3 branch is CLS loss branch
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  #reg_max=2

    def export_onnx_forward(self, x):
        # self.nl = 3
        box0 = self.cv2[0](x[0])
        score0 = self.cv3[0](x[0])

        box1 = self.cv2[1](x[1])
        score1 = self.cv3[1](x[1])

        box2 = self.cv2[2](x[2])
        score2 = self.cv3[2](x[2])

        return box0, score0, box1, score1, box2, score2


class DSConv(nn.Module):  #depthwise separable conv
    """ Implementation of Depthwise Separable Convolution. """
    def __init__(self, c1, c2, k=3, s=1, p=1, dilation=1, groups=1, act=nn.ReLU(inplace=False)):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, p, groups=c1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = act

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class C3k(C3):
    """
    C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks.
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DSBottleneck(nn.Module):
    """Replace Conv in standard bottleneck with DSConv."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a ds_bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DSConv(c1, c_, k[0], 1)
        self.cv2 = DSConv(c_, c2, k[1], 1) #g is removed
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DSC3(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DSBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DSC3k2(C2f):
    """Replace the standard bottleneck in C3k2 with DSBottleneck."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else DSBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class ESPSerial(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): # True in pico
        """
        Initialize a ESPBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(n * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(2 * self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])


class ESPBlock(ESPSerial):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(2 * self.c, self.c, 2, shortcut, g)
            if c3k else DSBottleneck(2 * self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class ESPSerialLite(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])


class ESPBlockLite(ESPSerialLite):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=False):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else DSBottleneck(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )