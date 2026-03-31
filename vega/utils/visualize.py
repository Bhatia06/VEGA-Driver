"""
utils/visualize.py — Overlay mask on image, save side-by-side comparisons
"""

import cv2
import numpy as np
import torch
from pathlib import Path


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) float ImageNet-normalized tensor → uint8 BGR numpy."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 80),   # lime green in BGR
    alpha: float = 0.5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Overlay a binary/probability mask on a BGR image.

    Args:
        image:     (H, W, 3) uint8 BGR
        mask:      (H, W) float [0,1] or binary
        color:     BGR overlay color (default lime green)
        alpha:     blend factor for mask overlay
        threshold: binarize mask at this value

    Returns:
        (H, W, 3) uint8 BGR image with colored overlay
    """
    mask_bin = (mask > threshold).astype(np.uint8)

    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[mask_bin == 1] = color

    # Blend only on masked pixels
    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    result = image.copy()
    result[mask_bin == 1] = blended[mask_bin == 1]

    return result


def save_comparison(
    image_tensor: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    save_path: str,
    fps: float = 0.0,
) -> None:
    """Save a 3-panel comparison: original | GT mask | predicted mask.

    Args:
        image_tensor: (3, H, W) float normalized tensor
        pred_logits:  (1, H, W) raw logits
        gt_mask:      (1, H, W) float binary tensor
        save_path:    path to save PNG
        fps:          if >0, show FPS counter on prediction panel
    """
    # Decode tensors
    img_bgr = tensor_to_bgr(image_tensor)

    pred_prob = torch.sigmoid(pred_logits[0]).cpu().numpy()  # (H, W)
    gt_np     = gt_mask[0].cpu().numpy()                     # (H, W)

    # Overlay panels
    panel_gt   = overlay_mask(img_bgr, gt_np,   color=(0, 255, 80),  alpha=0.5)
    panel_pred = overlay_mask(img_bgr, pred_prob, color=(255, 140, 0), alpha=0.5)

    # Confidence heatmap on pred
    heatmap = cv2.applyColorMap(
        (pred_prob * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    panel_heat = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, label in [
        (img_bgr, "Input"),
        (panel_gt, "Ground Truth"),
        (panel_pred, f"Predicted  FPS:{fps:.1f}" if fps > 0 else "Predicted"),
    ]:
        cv2.putText(panel, label, (10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, label, (10, 25), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # Stack horizontally
    combined = np.hstack([img_bgr, panel_gt, panel_pred])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, combined)


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter in top-left corner of frame."""
    frame = frame.copy()
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    import torch
    import numpy as np

    # Test overlay
    img = np.random.randint(50, 200, (360, 640, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=np.float32)
    mask[100:260, 150:490] = 1.0

    result = overlay_mask(img, mask)
    assert result.shape == (360, 640, 3)
    print(f"[PASS] overlay_mask shape: {result.shape}")

    # Test save_comparison
    img_t = torch.randn(3, 360, 640)
    pred  = torch.randn(1, 360, 640)
    gt    = torch.randint(0, 2, (1, 360, 640)).float()
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    save_comparison(img_t, pred, gt, path, fps=32.5)
    assert os.path.exists(path)
    print(f"[PASS] save_comparison wrote: {path}")
    os.unlink(path)

    print("\nAll visualize.py tests PASSED ✓")
