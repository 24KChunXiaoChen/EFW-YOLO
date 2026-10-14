import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, DWConv

__all__ = ("SerialBottleneck")


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, act=nn.ReLU(inplace=False)):
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

class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DSConv(c1, c_, k[0], 1)
        self.cv2 = DSConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SerialBottleneck(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, g=1, shortcut=True): # True in pico
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(n * self.c, c2, 1)
        self.m = nn.ModuleList( DSBottleneck(2 * self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
    def forward(self, x):
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])





class C2PSA(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = DWConv(c1, 2 * self.c, 1, 1)
        self.cv2 = DWConv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c , attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class PSABlock(nn.Module):

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x







class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y

