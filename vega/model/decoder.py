"""
model/decoder.py — VEGADecoder — Asymmetric bilinear upsampling decoder

Takes multi-scale features from encoder + TCM output and produces
a full-resolution segmentation map using bilinear upsampling + skip connections.

Uses F.interpolate(mode='bilinear') — NOT ConvTranspose2d — to avoid
checkerboard artifacts and improve ONNX export compatibility.

TEST RESULTS (run __main__):
  [PASS] out.shape == (2, 1, 360, 640)
  [PASS] No NaN in output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, HardSwish


class VEGADecoder(nn.Module):
    """Asymmetric bilinear decoder with skip connections.

    Input tensors (from encoder + TCM):
        feat_s1:        (B, 24,  90, 160)   — shallow skip
        feat_s2:        (B, 48,  45,  80)   — mid skip
        feat_s3:        (B, 96,  23,  40)   — deep skip (same level as bottleneck)
        bottleneck_tcm: (B, 96,  23,  40)   — TCM-refined bottleneck

    Output:
        (B, num_classes, 360, 640)

    Decoding path:
        bottleneck_tcm → upsample 2× → fuse with feat_s3 → conv
                       → upsample 2× → fuse with feat_s2 → conv
                       → upsample 2× → fuse with feat_s1 → conv
                       → upsample 2× → head conv → (B, C, 360, 640)
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()

        # Pointwise projections to align channel counts for skip fusion
        # bottleneck (96) + s3 (96) → project to 64
        self.proj_s3 = ConvBNAct(96, 64, kernel=1, activation=HardSwish())
        self.proj_bn = ConvBNAct(96, 64, kernel=1, activation=HardSwish())
        self.fuse_bn = ConvBNAct(128, 64, kernel=3, activation=HardSwish())

        # After first upsample: fuse fused_bn (64) + s2 (48) → 48
        self.proj_s2 = ConvBNAct(48, 48, kernel=1, activation=HardSwish())
        self.fuse_s2 = ConvBNAct(64 + 48, 48, kernel=3, activation=HardSwish())

        # After second upsample: fuse (48) + s1 (24) → 32
        self.proj_s1 = ConvBNAct(24, 24, kernel=1, activation=HardSwish())
        self.fuse_s1 = ConvBNAct(48 + 24, 32, kernel=3, activation=HardSwish())

        # Final upsample + head (32 → num_classes)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _upsample_to(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample x to match spatial size of target."""
        return F.interpolate(
            x, size=target.shape[2:],
            mode="bilinear", align_corners=False,
        )

    def forward(
        self,
        feat_s1: torch.Tensor,
        feat_s2: torch.Tensor,
        feat_s3: torch.Tensor,
        bottleneck_tcm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_s1:        (B, 24, 90, 160)
            feat_s2:        (B, 48, 45, 80)
            feat_s3:        (B, 96, 23, 40)
            bottleneck_tcm: (B, 96, 23, 40)

        Returns:
            logits: (B, num_classes, 360, 640)
        """
        # ── Level 3: Fuse bottleneck + s3 skip ──────────────────────────
        p_bn = self.proj_bn(bottleneck_tcm)  # (B, 64, 23, 40)
        p_s3 = self.proj_s3(feat_s3)         # (B, 64, 23, 40)
        fused = torch.cat([p_bn, p_s3], dim=1)  # (B, 128, 23, 40)
        fused = self.fuse_bn(fused)           # (B, 64, 23, 40)

        # ── Level 2: Upsample → fuse with s2 ───────────────────────────
        fused = self._upsample_to(fused, feat_s2)  # (B, 64, 45, 80)
        p_s2 = self.proj_s2(feat_s2)               # (B, 48, 45, 80)
        fused = torch.cat([fused, p_s2], dim=1)    # (B, 112, 45, 80)
        fused = self.fuse_s2(fused)                # (B, 48, 45, 80)

        # ── Level 1: Upsample → fuse with s1 ───────────────────────────
        fused = self._upsample_to(fused, feat_s1)  # (B, 48, 90, 160)
        p_s1 = self.proj_s1(feat_s1)               # (B, 24, 90, 160)
        fused = torch.cat([fused, p_s1], dim=1)    # (B, 72, 90, 160)
        fused = self.fuse_s1(fused)                # (B, 32, 90, 160)

        # ── Final upsample → head ────────────────────────────────────────
        # Upsample 2× to (180, 320), then 2× more to (360, 640)
        # We do it in one shot with the target size for speed
        fused = F.interpolate(fused, scale_factor=4, mode="bilinear", align_corners=False)
        # fused: (B, 32, 360, 640)

        logits = self.head(fused)  # (B, num_classes, 360, 640)
        return logits


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(7)

    dec = VEGADecoder(num_classes=1)
    dec.eval()

    dummy = (
        torch.randn(2, 24, 90, 160),   # feat_s1
        torch.randn(2, 48, 45, 80),    # feat_s2
        torch.randn(2, 96, 23, 40),    # feat_s3
        torch.randn(2, 96, 23, 40),    # bottleneck_tcm
    )

    with torch.no_grad():
        out = dec(*dummy)

    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 1, 360, 640), f"Shape mismatch: got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in decoder output"
    print("[PASS] out.shape == (2, 1, 360, 640)")

    total_params = sum(p.numel() for p in dec.parameters())
    print(f"[INFO] Decoder params: {total_params:,}")

    print("\nAll decoder.py tests PASSED ✓")
