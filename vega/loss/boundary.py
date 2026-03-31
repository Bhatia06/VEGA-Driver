"""
loss/boundary.py — Dilated boundary mask extraction

Uses morphological erosion via max_pool2d to identify the
boundary region of a binary segmentation mask.

TEST RESULTS (run __main__):
  [PASS] boundary.shape == (2, 1, 360, 640)
  [PASS] boundary values in [0, 1]
  [PASS] No NaN in boundary mask
"""

import torch
import torch.nn.functional as F


def get_boundary_mask(
    mask: torch.Tensor,
    dilation_px: int = 5,
) -> torch.Tensor:
    """Extract a dilated boundary mask from a binary segmentation mask.

    Strategy:
        eroded = morphological erosion of mask (via max_pool2d on inverted mask)
        boundary = mask XOR eroded_mask

    This gives a band of `dilation_px` pixels around the mask boundary.

    Args:
        mask:        (B, 1, H, W) float tensor with values in {0.0, 1.0}
        dilation_px: width of the boundary band in pixels (default 5)

    Returns:
        boundary: (B, 1, H, W) float tensor — 1.0 on boundary pixels, 0.0 elsewhere
    """
    kernel_size = 2 * dilation_px + 1
    padding = dilation_px

    # Erosion: erode "foreground" pixels
    # max_pool2d on (1 - mask) = dilation of background = erosion of foreground
    inverted = 1.0 - mask
    dilated_bg = F.max_pool2d(
        inverted, kernel_size=kernel_size, stride=1, padding=padding,
    )
    eroded = 1.0 - dilated_bg  # eroded foreground

    # Boundary = pixels in mask but not in eroded mask (or vice versa)
    # = absolute difference (works for soft masks too)
    boundary = torch.abs(mask - eroded)

    # Clamp to [0, 1] — small floating point drift possible
    boundary = boundary.clamp(0.0, 1.0)

    return boundary


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # Create a synthetic binary mask (circle in centre)
    B, H, W = 2, 360, 640
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 100:260, 200:440] = 1.0   # road-like rectangle

    boundary = get_boundary_mask(mask, dilation_px=5)

    assert boundary.shape == (B, 1, H, W), f"Shape: {boundary.shape}"
    assert not torch.isnan(boundary).any(), "NaN in boundary"
    assert boundary.min() >= 0.0 and boundary.max() <= 1.0, "Boundary out of [0,1]"
    print(f"[PASS] boundary.shape == {tuple(boundary.shape)}")

    nonzero_pct = boundary.mean().item() * 100
    print(f"[INFO] Boundary covers {nonzero_pct:.2f}% of pixels")
    assert nonzero_pct > 0.0, "Boundary mask is all zeros — erosion might be wrong"
    assert nonzero_pct < 20.0, f"Boundary mask covers too much ({nonzero_pct:.1f}%)"

    print("\nAll boundary.py tests PASSED ✓")
