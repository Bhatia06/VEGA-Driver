"""
model/vega.py — Full VEGA model assembly

Wires together:
  VEGAEncoder → ConvGRUCell (TCM) → VEGADecoder

The model maintains a temporal hidden state (self.hidden) across forward
calls for the same video sequence. Call reset_temporal_state() between clips.

TEST RESULTS (run __main__):
  [PASS] out.shape == (2, 1, 360, 640)
  [PASS] No NaN in output
  [INFO] Total params: ~1.9M (target: <2.5M)
"""

import torch
import torch.nn as nn

from .encoder import VEGAEncoder
from .tcm import ConvGRUCell
from .decoder import VEGADecoder


class VEGA(nn.Module):
    """End-to-end drivable space segmentation model.

    - Encoder extracts multi-scale features (s1, s2, s3, bottleneck)
    - TCM (ConvGRU) applies temporal coherence to the bottleneck feature
    - Decoder upsamples + fuses skip connections to produce full-res logits

    The temporal state (self.hidden) is maintained across frames of the
    same scene. It must be explicitly reset between different scenes/clips.

    Args:
        num_classes: number of output segmentation classes (default 1 = binary drivable)
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()

        self.encoder = VEGAEncoder()
        self.tcm = ConvGRUCell(input_dim=96, hidden_dim=96, kernel_size=3)
        self.decoder = VEGADecoder(num_classes=num_classes)

        # Temporal hidden state — managed externally via reset_temporal_state()
        # Stored as a plain attribute (NOT a Parameter) to avoid ONNX issues
        self.hidden: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        reset_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, 3, H, W) — normalized input frame(s)
            reset_hidden: if True, clears the temporal hidden state before processing

        Returns:
            logits: (B, num_classes, H, W) — raw logits (no sigmoid applied)
        """
        if reset_hidden:
            self.hidden = None

        # Encoder: extract 4-scale features
        feat_s1, feat_s2, feat_s3, bottleneck = self.encoder(x)

        # TCM: propagate temporal state
        self.hidden = self.tcm(bottleneck, self.hidden)

        # Decoder: upsample + skip fusion
        logits = self.decoder(feat_s1, feat_s2, feat_s3, self.hidden)

        return logits

    def reset_temporal_state(self) -> None:
        """Explicitly reset the temporal hidden state (call between sequences)."""
        self.hidden = None

    def get_hidden_state(self) -> torch.Tensor | None:
        """Return a detached copy of the current hidden state (for inspection)."""
        if self.hidden is None:
            return None
        return self.hidden.detach()

    def set_hidden_state(self, hidden: torch.Tensor | None) -> None:
        """Inject an external hidden state (e.g. from distributed inference)."""
        self.hidden = hidden


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(99)

    model = VEGA(num_classes=1)
    model.eval()

    x = torch.randn(2, 3, 360, 640)

    # Frame 1 (hidden=None, auto-initialised)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1, 360, 640), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output (frame 1)"
    print(f"[PASS] Frame 1 output: {tuple(out.shape)}")

    # Frame 2 (temporal state should propagate)
    with torch.no_grad():
        out2 = model(x)

    assert out2.shape == (2, 1, 360, 640)
    assert not torch.isnan(out2).any(), "NaN in output (frame 2)"
    print(f"[PASS] Frame 2 output: {tuple(out2.shape)}")
    assert not torch.allclose(out, out2), "Temporal state had no effect!"
    print("[PASS] Temporal state propagates between frames")

    # Reset test
    model.reset_temporal_state()
    assert model.hidden is None
    with torch.no_grad():
        out3 = model(x, reset_hidden=True)
    assert out3.shape == (2, 1, 360, 640)
    print("[PASS] reset_hidden=True works")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Total params:     {total_params:,}")
    print(f"[INFO] Trainable params: {trainable:,}")
    assert total_params < 2_500_000, f"Model too large: {total_params}"
    print("[PASS] Model size < 2.5M params")

    # Breakdown by submodule
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    tcm_p = sum(p.numel() for p in model.tcm.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n  Encoder:  {enc_p:,}")
    print(f"  TCM:      {tcm_p:,}")
    print(f"  Decoder:  {dec_p:,}")

    print("\nAll vega.py tests PASSED ✓")
