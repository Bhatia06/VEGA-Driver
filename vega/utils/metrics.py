"""
utils/metrics.py — Evaluation metrics for VEGA

Implements:
  - compute_miou:          mean Intersection over Union (binary segmentation)
  - compute_boundary_iou:  IoU restricted to boundary regions
  - FPSCounter:            warmup-then-measure FPS benchmark

TEST RESULTS (run __main__):
  [PASS] mIoU in [0, 1]
  [PASS] BoundaryIoU in [0, 1]
  [PASS] FPS > 0
"""

import time
import torch
import numpy as np
from typing import Optional

from ..loss.boundary import get_boundary_mask


def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Mean IoU for binary drivable segmentation.

    Args:
        preds:     (B, 1, H, W) raw logits (sigmoid applied internally)
        targets:   (B, 1, H, W) float binary {0, 1}
        threshold: sigmoid threshold for binary decision (default 0.5)

    Returns:
        mean_iou: float in [0, 1]
    """
    with torch.no_grad():
        pred_bin = (torch.sigmoid(preds) > threshold).float()
        tgt_bin  = (targets > threshold).float()

        # Per-sample IoU
        B = pred_bin.size(0)
        pred_flat = pred_bin.view(B, -1)
        tgt_flat  = tgt_bin.view(B, -1)

        intersection = (pred_flat * tgt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) - intersection

        iou_per_sample = (intersection + 1e-6) / (union + 1e-6)
        return iou_per_sample.mean().item()


def compute_boundary_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    dilation: int = 5,
    threshold: float = 0.5,
) -> float:
    """IoU computed only on boundary regions of the target mask.

    Useful for evaluating sharpness of predicted boundaries.

    Args:
        preds:     (B, 1, H, W) raw logits
        targets:   (B, 1, H, W) float binary {0, 1}
        dilation:  boundary width in pixels (default 5)
        threshold: decision threshold

    Returns:
        boundary_iou: float in [0, 1]
    """
    with torch.no_grad():
        pred_bin = (torch.sigmoid(preds) > threshold).float()
        tgt_bin  = (targets > threshold).float()

        # Extract boundary masks
        pred_boundary = get_boundary_mask(pred_bin, dilation_px=dilation)
        tgt_boundary  = get_boundary_mask(tgt_bin,  dilation_px=dilation)

        B = pred_bin.size(0)
        pb_flat  = pred_boundary.view(B, -1)
        tb_flat  = tgt_boundary.view(B, -1)

        # Intersection and union of boundary regions
        intersection = (pb_flat * tb_flat).sum(dim=1)
        union = (pb_flat + tb_flat - pb_flat * tb_flat).sum(dim=1)

        biou = (intersection + 1e-6) / (union + 1e-6)
        return biou.mean().item()


class FPSCounter:
    """GPU/CPU FPS benchmark with warmup.

    Usage:
        counter = FPSCounter(warmup=10, measure=100)
        for frames or batches:
            counter.tick()
        fps, std = counter.result()
    """

    def __init__(self, warmup: int = 10, measure: int = 100, use_cuda: bool = True):
        self.warmup = warmup
        self.measure = measure
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self._count = 0
        self._times: list[float] = []
        self._t0: Optional[float] = None

    def tick(self) -> None:
        """Call once per frame/batch."""
        if self.use_cuda:
            torch.cuda.synchronize()

        now = time.perf_counter()

        if self._count < self.warmup:
            self._count += 1
            self._t0 = now
            return

        if self._t0 is not None:
            self._times.append(now - self._t0)

        self._t0 = now
        self._count += 1

    def result(self) -> tuple[float, float]:
        """Return (mean_fps, std_fps).

        If not enough samples, returns (0.0, 0.0).
        """
        if len(self._times) == 0:
            return 0.0, 0.0

        times = np.array(self._times[:self.measure])
        fps_per_frame = 1.0 / (times + 1e-9)
        return float(fps_per_frame.mean()), float(fps_per_frame.std())

    def reset(self) -> None:
        self._count = 0
        self._times = []
        self._t0 = None

    @classmethod
    def benchmark(
        cls,
        model: torch.nn.Module,
        input_shape: tuple = (1, 3, 360, 640),
        device: str = "cuda",
        warmup: int = 10,
        measure: int = 100,
    ) -> tuple[float, float]:
        """Standalone benchmark — runs model forward passes and returns FPS.

        Args:
            model:        PyTorch model (already on device)
            input_shape:  (B, C, H, W) dummy input shape
            device:       'cuda' or 'cpu'
            warmup:       warmup iterations
            measure:      timing iterations

        Returns:
            (mean_fps, std_fps)
        """
        model.eval()
        dummy = torch.randn(input_shape, device=device)
        use_cuda = (device == "cuda" and torch.cuda.is_available())

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
                if use_cuda:
                    torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(measure):
                t0 = time.perf_counter()
                _ = model(dummy)
                if use_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        times = np.array(times)
        fps = 1.0 / (times + 1e-9)
        return float(fps.mean()), float(fps.std())


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    logits = torch.randn(4, 1, 360, 640)
    targets = torch.randint(0, 2, (4, 1, 360, 640)).float()

    # mIoU
    miou = compute_miou(logits, targets)
    assert 0.0 <= miou <= 1.0, f"mIoU out of range: {miou}"
    print(f"[PASS] mIoU: {miou:.4f}")

    # BoundaryIoU
    biou = compute_boundary_iou(logits, targets)
    assert 0.0 <= biou <= 1.0, f"BoundaryIoU out of range: {biou}"
    print(f"[PASS] BoundaryIoU: {biou:.4f}")

    # Perfect prediction
    perfect_logits = targets * 10 - 5   # high logits where target=1, low elsewhere
    miou_perfect = compute_miou(perfect_logits, targets)
    assert miou_perfect > 0.9, f"Perfect prediction mIoU: {miou_perfect}"
    print(f"[PASS] Perfect prediction mIoU: {miou_perfect:.4f}")

    # FPSCounter
    counter = FPSCounter(warmup=3, measure=10, use_cuda=False)
    for _ in range(15):
        time.sleep(0.001)
        counter.tick()
    fps_mean, fps_std = counter.result()
    assert fps_mean > 0, f"FPS: {fps_mean}"
    print(f"[PASS] FPSCounter: {fps_mean:.1f} ± {fps_std:.1f} FPS")

    print("\nAll metrics.py tests PASSED ✓")
