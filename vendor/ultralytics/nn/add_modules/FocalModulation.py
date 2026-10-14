import torch
import torch.nn as nn

__all__ = ['FocalModulation', 'C2PSA_FM']

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import PSABlock

class C2PSA_FM(nn.Module):
    # 默认初始化
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2 # 断言输入出相等
        self.c = int(c1 * e) # 中间通道数
        self.fm = FocalModulation(self.c)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1) # 卷积2，最开始的卷积模块
        self.cv2 = Conv(3 * self.c, c1, 1) # 卷积2，最后的卷积模块
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    # 实际逻辑方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量进过卷积1，然后分割为两个通道数相同的ab中间张量
        a, b, c = self.cv1(x).split((self.c, self.c, self.c), dim=1)
        # 将a,b使用m处理
        b = self.m(b)
        c = self.fm(c)

        # 然后在将a,b合并输出
        return self.cv2(torch.cat((a, b, c), 1))


class FocalModulation(nn.Module):
    """
    Focal Modulation
    """
    def __init__(
            self,
            dim: int, # 输入通道数
            focal_window: int = 3, # 最小卷积核大小
            focal_level: int = 2, # 多尺度卷积层数
            focal_factor: int = 2, # 每层 kernel 的扩张/增长因子
            bias: bool = True, # 卷积层是否带有偏置
            proj_drop: float = 0.0, # 输出投影层的 dropout 概率
            use_postln_in_modulation: bool = False, # 是否在 modulation 后加归一化
            normalize_modulator: bool = True, # 是否对多尺度上下文ctx_all做幅度归一化
    ):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        # 预投影：生成 q, ctx, gates
        self.f_linear = nn.Conv2d(dim, 2 * dim + (focal_level + 1), kernel_size=1, bias=bias)
        # 生成调制器
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        # 多尺度深度可分离卷积
        self.focal_layers = nn.ModuleList()
        for k in range(focal_level):
            ks = focal_factor * k + focal_window
            self.focal_layers.append(nn.Sequential(nn.Conv2d(dim, dim, kernel_size=ks, stride=1, groups=dim, padding=ks // 2, bias=False),nn.GELU()))
        # 对 NCHW，用 GroupNorm(1, C) 作为“LN 等价”更方便
        if self.use_postln_in_modulation: self.ln = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 预线性：拆分为 q, ctx, gates
        f = self.f_linear(x) # (B, 2C + L+1, H, W)
        q, ctx, gates = torch.split(f, (C, C, self.focal_level + 1), dim=1)
        # gates 归一化（关键改动）
        gates = torch.softmax(gates, dim=1)            # 在通道维度做 softmax，更稳
        # 多尺度上下文
        ctx_all = 0.0
        for l in range(self.focal_level):
            y = self.focal_layers[l](ctx)
            ctx_all = ctx_all + y * gates[:, l:l + 1]
        # 全局上下文
        ctx_global = self.act(ctx.mean(dim=(2, 3), keepdim=True))# 全局平均特征
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]
        # 幅度归一（默认开启）
        if self.normalize_modulator: ctx_all = ctx_all / float(self.focal_level + 1)

        # 调制：tanh 压幅（关键改动）
        mod = torch.tanh(self.h(ctx_all))
        #mod = self.h(ctx_all)
        x_out = q * mod
        if self.use_postln_in_modulation: x_out = self.ln(x_out)
        # 后投影
        # 可学习线性投影（1×1 Conv），用于重新融合、重标定和变换特征
        x_out = self.proj(x_out)
        # 尝试丢弃一些权重，提高鲁棒性
        x_out = self.proj_drop(x_out)
        return x_out

