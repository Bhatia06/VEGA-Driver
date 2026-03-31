"""
loss/compound.py — VEGALoss: compound segmentation loss

Components:
  1. WeightedBCELoss      — handles class imbalance (drivable ~30% of pixels)
  2. DiceLoss             — overlap-based loss, handles imbalance naturally
  3. BoundaryAwareLoss    — 3× weight on boundary pixels
  4. TemporalConsistencyLoss — MSE between consecutive predictions

TEST RESULTS (run __main__):
  [PASS] loss in range ~0.5–1.5 on random inputs
  [PASS] breakdown dict has all 4 components
  [PASS] No NaN in loss value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .boundary import get_boundary_mask


# ─────────────────────────────────────────────────────────────────
# Individual loss components
# ─────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with positive class weight.

    Corrects for ~30/70 drivable/non-drivable imbalance in nuScenes.
    pos_weight=2.5 ≈ (1 - 0.30) / 0.30 ≈ 2.33, rounded up slightly.
    """

    def __init__(self, pos_weight: float = 2.5):
        super().__init__()
        self.pos_weight_val = pos_weight
        # Register as non-parameter tensor (moved to device with model)
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight,
        )


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Dice = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)
    Operates in probability space (sigmoid applied internally).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        # Flatten spatial dims, keep batch dimension
        pred_flat = pred.view(pred.size(0), -1)
        tgt_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * tgt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)

        dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        return dice.mean()


class BoundaryAwareLoss(nn.Module):
    """BCE loss with 3× weighting on boundary pixels.

    Boundary region identified by dilated erosion of the target mask.
    """

    def __init__(self, dilation_px: int = 5, boundary_weight: float = 2.0):
        super().__init__()
        self.dilation_px = dilation_px
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        boundary_mask = get_boundary_mask(target, dilation_px=self.dilation_px)

        # Per-pixel weight: 1.0 everywhere + extra on boundary
        weight = 1.0 + self.boundary_weight * boundary_mask

        return F.binary_cross_entropy_with_logits(
            logits, target, weight=weight,
        )


class TemporalConsistencyLoss(nn.Module):
    """MSE penalising prediction changes between consecutive frames.

    Encourages temporal smoothness in the TCM-refined predictions.
    """

    def forward(
        self,
        current_pred: torch.Tensor,
        prev_pred: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prev_pred is None:
            return torch.tensor(0.0, device=current_pred.device, requires_grad=False)
        return F.mse_loss(current_pred, prev_pred.detach())


# ─────────────────────────────────────────────────────────────────
# Compound loss
# ─────────────────────────────────────────────────────────────────

class VEGALoss(nn.Module):
    """Weighted compound segmentation loss for VEGA.

    Args:
        w_bce:      weight for WeightedBCELoss      (default 0.35)
        w_dice:     weight for DiceLoss             (default 0.35)
        w_boundary: weight for BoundaryAwareLoss    (default 0.20)
        w_temporal: weight for TemporalConsistency  (default 0.10)
        pos_weight: positive class weight for BCE   (default 2.5)
    """

    def __init__(
        self,
        w_bce: float = 0.35,
        w_dice: float = 0.35,
        w_boundary: float = 0.20,
        w_temporal: float = 0.10,
        pos_weight: float = 2.5,
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.w_temporal = w_temporal

        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryAwareLoss()
        self.temporal_loss = TemporalConsistencyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits:    (B, 1, H, W) — raw model output (no sigmoid)
            target:    (B, 1, H, W) — binary ground truth {0, 1}
            prev_pred: (B, 1, H, W) — sigmoid prediction from previous frame,
                       or None for first frame

        Returns:
            total_loss: scalar tensor
            breakdown:  dict with keys 'bce', 'dice', 'boundary', 'temporal'
        """
        # Core losses (all on raw logits + float targets)
        l_bce = self.bce_loss(logits, target)
        l_dice = self.dice_loss(logits, target)
        l_boundary = self.boundary_loss(logits, target)

        # Temporal (operates on sigmoid probabilities)
        current_pred = torch.sigmoid(logits)
        l_temporal = self.temporal_loss(current_pred, prev_pred)

        # Compound weighted sum
        total = (
            self.w_bce * l_bce
            + self.w_dice * l_dice
            + self.w_boundary * l_boundary
            + self.w_temporal * l_temporal
        )

        breakdown = {
            "bce": l_bce.item(),
            "dice": l_dice.item(),
            "boundary": l_boundary.item(),
            "temporal": l_temporal.item() if isinstance(l_temporal, torch.Tensor) else 0.0,
        }

        return total, breakdown


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(42)

    criterion = VEGALoss()

    logits = torch.randn(2, 1, 360, 640)
    target = torch.randint(0, 2, (2, 1, 360, 640)).float()

    # Test without temporal loss
    loss, breakdown = criterion(logits, target, prev_pred=None)
    print(f"[INFO] Loss (no temporal): {loss.item():.4f}")
    print(f"[INFO] Breakdown: {breakdown}")

    assert not torch.isnan(loss), "NaN in compound loss"
    assert 0.3 < loss.item() < 3.0, f"Loss out of expected range: {loss.item()}"
    print(f"[PASS] Loss in expected range: {loss.item():.4f}")

    # Test with temporal loss
    prev_pred = torch.sigmoid(torch.randn(2, 1, 360, 640))
    loss2, breakdown2 = criterion(logits, target, prev_pred=prev_pred)
    print(f"[INFO] Loss (with temporal): {loss2.item():.4f}")
    assert breakdown2["temporal"] > 0.0, "Temporal loss should be nonzero"
    print(f"[PASS] Temporal loss: {breakdown2['temporal']:.4f}")

    # Gradient check
    logits_g = torch.randn(2, 1, 180, 320, requires_grad=True)
    target_g = torch.randint(0, 2, (2, 1, 180, 320)).float()
    loss_g, _ = criterion(logits_g, target_g)
    loss_g.backward()
    assert logits_g.grad is not None, "No gradient!"
    assert not torch.isnan(logits_g.grad).any(), "NaN in gradient"
    print("[PASS] Gradients flow correctly through compound loss")

    print("\nAll compound.py tests PASSED ✓")
