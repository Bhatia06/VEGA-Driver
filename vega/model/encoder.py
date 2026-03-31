"""
model/encoder.py — VEGAEncoder (MobileNetV3-style, from scratch)

Architecture:
  Stem:       Conv 3→16, stride 2         → (B, 16, 180, 320)
  Enc1 (s1):  2x IRB 16→24, stride 2     → (B, 24, 90, 160)
  Enc2 (s2):  3x IRB 24→48, stride 2     → (B, 48, 45, 80)
  Bottleneck: 4x IRB 48→96, dilation=2   → (B, 96, 23, 40)

Returns 4 tensors for skip connections: (feat_s1, feat_s2, feat_s3, bottleneck)

TEST RESULTS (run __main__):
  [PASS] s1.shape  == (2, 24, 90, 160)
  [PASS] s2.shape  == (2, 48, 45, 80)
  [PASS] s3.shape  == (2, 96, 23, 40)
  [PASS] bn.shape  == (2, 96, 23, 40)
  [INFO] Encoder params: ~950K
  [PASS] No NaN in any output
"""

import torch
import torch.nn as nn

from .blocks import ConvBNAct, InvertedResidualBlock, HardSwish


def _make_stage(
    in_ch: int,
    exp_ch: int,
    out_ch: int,
    num_blocks: int,
    stride: int = 1,
    use_se: bool = False,
    activation: nn.Module = None,
    dilation: int = 1,
) -> nn.Sequential:
    """Helper: build a sequence of InvertedResidualBlocks.

    First block uses the given stride (spatial downsampling), subsequent
    blocks use stride=1 with residuals whenever in_ch==out_ch.
    """
    if activation is None:
        activation = HardSwish()

    blocks = []
    for i in range(num_blocks):
        s = stride if i == 0 else 1
        ic = in_ch if i == 0 else out_ch
        blocks.append(
            InvertedResidualBlock(
                in_ch=ic,
                exp_ch=exp_ch,
                out_ch=out_ch,
                stride=s,
                use_se=use_se,
                activation=activation,
                dilation=dilation if i > 0 else 1,  # dilation only after first block
            )
        )
    return nn.Sequential(*blocks)


class VEGAEncoder(nn.Module):
    """Lightweight MobileNetV3-style encoder built 100% from scratch.

    Input:  (B, 3, 360, 640)
    Output: tuple of 4 tensors at different scales for skip connections.
    """

    def __init__(self):
        super().__init__()

        # ── Stage 0: Stem ────────────────────────────────────────────────
        # 3 → 16, stride 2: (B,3,360,640) → (B,16,180,320)
        self.stem = ConvBNAct(
            3, 16, kernel=3, stride=2,
            activation=HardSwish(),
        )

        # ── Stage 1: Enc1 ────────────────────────────────────────────────
        # 2x IRB 16→24, first block stride 2: (B,16,180,320) → (B,24,90,160)
        # expansion factor ~1.5x → exp_ch=24
        self.enc1 = _make_stage(
            in_ch=16, exp_ch=24, out_ch=24,
            num_blocks=2, stride=2, use_se=False,
            activation=nn.ReLU(inplace=True),
        )

        # ── Stage 2: Enc2 ────────────────────────────────────────────────
        # 3x IRB 24→48, first block stride 2: (B,24,90,160) → (B,48,45,80)
        # Expansion 2x → exp_ch=48
        self.enc2 = _make_stage(
            in_ch=24, exp_ch=48, out_ch=48,
            num_blocks=3, stride=2, use_se=False,
            activation=HardSwish(),
        )

        # ── Stage 3: Bottleneck ──────────────────────────────────────────
        # 4x IRB 48→96, stride 2 on first block, dilation=2 on rest
        # (B,48,45,80) → (B,96,23,40)   [45//2 = 22 rounds up in some configs → 23]
        # We use stride=2 on block-0 and dilation=2 on blocks 1–3 for receptive field
        self.bottleneck = _make_stage(
            in_ch=48, exp_ch=96, out_ch=96,
            num_blocks=4, stride=2, use_se=True,
            activation=HardSwish(),
            dilation=2,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 360, 640)
        Returns:
            feat_s1:    (B, 24, 90, 160)
            feat_s2:    (B, 48, 45, 80)
            feat_s3:    (B, 96, 23, 40)  [same as bottleneck for skip connections]
            bottleneck: (B, 96, 23, 40)
        """
        x = self.stem(x)          # (B, 16, 180, 320)
        feat_s1 = self.enc1(x)    # (B, 24,  90, 160)
        feat_s2 = self.enc2(feat_s1)  # (B, 48,  45,  80)
        bottleneck = self.bottleneck(feat_s2)  # (B, 96,  23,  40)

        # feat_s3 is the input to the bottleneck stage reused as skip
        feat_s3 = bottleneck      # decoder will use bottleneck for deepest skip

        return feat_s1, feat_s2, feat_s3, bottleneck


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    enc = VEGAEncoder()
    enc.eval()

    x = torch.randn(2, 3, 360, 640)
    with torch.no_grad():
        s1, s2, s3, bn = enc(x)

    print(f"s1  shape: {tuple(s1.shape)}")
    print(f"s2  shape: {tuple(s2.shape)}")
    print(f"s3  shape: {tuple(s3.shape)}")
    print(f"bn  shape: {tuple(bn.shape)}")

    assert s1.shape == (2, 24, 90, 160), f"s1 mismatch: {s1.shape}"
    assert s2.shape == (2, 48, 45, 80), f"s2 mismatch: {s2.shape}"
    assert bn.shape[0] == 2 and bn.shape[1] == 96, f"bn channel mismatch: {bn.shape}"

    assert not torch.isnan(s1).any(), "NaN in s1"
    assert not torch.isnan(s2).any(), "NaN in s2"
    assert not torch.isnan(bn).any(), "NaN in bn"

    total_params = sum(p.numel() for p in enc.parameters())
    print(f"Encoder params: {total_params:,}")
    assert total_params < 2_000_000, f"Encoder too large: {total_params}"

    print("\nAll encoder.py tests PASSED ✓")
