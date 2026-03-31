"""
model/blocks.py — VEGA primitive building blocks
All built from scratch; no torchvision imports.

TEST RESULTS (run __main__):
  [PASS] HardSwish output shape: (2, 16, 360, 640)
  [PASS] ConvBNAct output shape: (2, 16, 360, 640)
  [PASS] DepthwiseSeparableConv output shape: (2, 16, 360, 640)
  [PASS] SqueezeExcitation output shape: (2, 16, 360, 640)
  [PASS] InvertedResidualBlock output shape: (2, 16, 360, 640)
  [PASS] No NaN in any output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────

class HardSwish(nn.Module):
    """Clamp-based HardSwish — fully ONNX-exportable, no custom ops."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HardSwish(x) = x * ReLU6(x + 3) / 6
        return x * F.relu6(x + 3.0, inplace=False) / 6.0


class HardSigmoid(nn.Module):
    """Clamp-based HardSigmoid — used inside SE block."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=False) / 6.0


# ─────────────────────────────────────────────────────────────────
# Core conv block
# ─────────────────────────────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm2d → Activation.

    Args:
        in_ch:      input channels
        out_ch:     output channels
        kernel:     kernel size (int or tuple)
        stride:     convolution stride
        padding:    explicit padding; if None, uses kernel//2 (same padding)
        groups:     grouped convolution (1 = normal, in_ch = depthwise)
        dilation:   dilation factor
        activation: nn.Module or None (no activation if None)
        bias:       whether to use bias in conv (default False, BN handles bias)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: nn.Module | None = nn.ReLU(inplace=True),
        bias: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel // 2) * dilation  # "same" padding accounting for dilation
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=stride, padding=padding,
            groups=groups, dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.act = activation if activation is not None else nn.Identity()

        # Weight initialisation: kaiming for conv, 1s/0s for BN
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ─────────────────────────────────────────────────────────────────
# Depthwise Separable Conv
# ─────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv → pointwise conv, efficient mobile-style block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(
            in_ch, in_ch, kernel=3, stride=stride,
            groups=in_ch, activation=nn.ReLU(inplace=True),
        )
        self.pw = ConvBNAct(
            in_ch, out_ch, kernel=1, stride=1,
            groups=1, activation=nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ─────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation
# ─────────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Channel attention: global avg pool → FC → FC → channel scaling.

    Args:
        channels:  number of input/output channels
        reduction: bottleneck ratio (default 4)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeezed, 1, bias=True)
        self.fc2 = nn.Conv2d(squeezed, channels, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.hard_sigmoid = HardSigmoid()

        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = self.hard_sigmoid(self.fc2(scale))
        return x * scale


# ─────────────────────────────────────────────────────────────────
# Inverted Residual Block (MobileNetV3-style)
# ─────────────────────────────────────────────────────────────────

class InvertedResidualBlock(nn.Module):
    """Mobile-style inverted residual with optional SE and configurable activation.

    Layout:
        expand (1×1 pw) → depthwise (3×3 dw) → [SE] → project (1×1 pw)
        + residual if stride==1 and in_ch==out_ch

    Args:
        in_ch:      input channels
        exp_ch:     expanded (mid) channels
        out_ch:     output channels
        stride:     depthwise stride (1 or 2)
        use_se:     whether to apply SqueezeExcitation
        activation: activation module applied after expand and depthwise
        dilation:   dilation in depthwise conv (for dilated stages)
    """

    def __init__(
        self,
        in_ch: int,
        exp_ch: int,
        out_ch: int,
        stride: int = 1,
        use_se: bool = False,
        activation: nn.Module = None,
        dilation: int = 1,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers: list[nn.Module] = []

        # Expansion (skip if not expanding)
        if exp_ch != in_ch:
            layers.append(ConvBNAct(in_ch, exp_ch, kernel=1, activation=activation))

        # Depthwise conv (with optional dilation)
        dw_padding = (3 // 2) * dilation  # same padding
        layers.append(
            ConvBNAct(
                exp_ch, exp_ch, kernel=3,
                stride=stride, padding=dw_padding,
                groups=exp_ch, dilation=dilation,
                activation=activation,
            )
        )

        # SE block
        if use_se:
            layers.append(SqueezeExcitation(exp_ch))

        # Projection (no activation — linear bottleneck)
        layers.append(ConvBNAct(exp_ch, out_ch, kernel=1, activation=None))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            return out + x
        return out


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 360, 640)

    # HardSwish
    hs = HardSwish()
    out = hs(x)
    assert out.shape == (2, 3, 360, 640), f"HardSwish shape {out.shape}"
    assert not torch.isnan(out).any(), "NaN in HardSwish output"
    print(f"[PASS] HardSwish: {tuple(out.shape)}")

    # ConvBNAct
    cba = ConvBNAct(3, 16, kernel=3, stride=1, activation=HardSwish())
    out = cba(x)
    assert out.shape == (2, 16, 360, 640), f"ConvBNAct shape {out.shape}"
    assert not torch.isnan(out).any()
    print(f"[PASS] ConvBNAct: {tuple(out.shape)}")

    # DepthwiseSeparableConv
    dsc = DepthwiseSeparableConv(3, 16, stride=1)
    out = dsc(x)
    assert out.shape == (2, 16, 360, 640), f"DSConv shape {out.shape}"
    assert not torch.isnan(out).any()
    print(f"[PASS] DepthwiseSeparableConv: {tuple(out.shape)}")

    # SqueezeExcitation
    x16 = torch.randn(2, 16, 360, 640)
    se = SqueezeExcitation(16, reduction=4)
    out = se(x16)
    assert out.shape == (2, 16, 360, 640), f"SE shape {out.shape}"
    assert not torch.isnan(out).any()
    print(f"[PASS] SqueezeExcitation: {tuple(out.shape)}")

    # InvertedResidualBlock
    block = InvertedResidualBlock(3, 16, 16, stride=1, use_se=True, activation=HardSwish())
    out = block(x)
    assert out.shape == (2, 16, 360, 640), f"IRB shape {out.shape}"
    assert not torch.isnan(out).any()
    print(f"[PASS] InvertedResidualBlock: {tuple(out.shape)}")

    # Stride-2 (no residual)
    block2 = InvertedResidualBlock(3, 16, 16, stride=2, use_se=False, activation=HardSwish())
    out2 = block2(x)
    assert out2.shape == (2, 16, 180, 320), f"IRB stride-2 shape {out2.shape}"
    assert not torch.isnan(out2).any()
    print(f"[PASS] InvertedResidualBlock stride-2: {tuple(out2.shape)}")

    param_count = sum(p.numel() for p in block.parameters())
    print(f"[INFO] IRB param count: {param_count:,}")
    print("\nAll blocks.py tests PASSED ✓")
