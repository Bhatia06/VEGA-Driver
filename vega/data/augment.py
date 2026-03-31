"""
data/augment.py — VEGAAugment: edge-case augmentation pipeline for drivable segmentation

All augmentations apply the same spatial transform to both image AND mask.
Color/appearance augmentations apply to image only.

TEST RESULTS (run __main__):
  [PASS] augmented image shape: (3, 360, 640)
  [PASS] augmented mask shape:  (1, 360, 640)
  [PASS] mask values in {0, 1}
  [PASS] No NaN
"""

import random
import math
from typing import Tuple

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# ─────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize uint8 HWC image → float32 HWC in ImageNet space."""
    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def denormalize(image: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 HWC."""
    img = image * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────
# Spatial augmentations (applied to image + mask)
# ─────────────────────────────────────────────────────────────────

def random_horizontal_flip(
    image: np.ndarray, mask: np.ndarray, p: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def random_scale_crop(
    image: np.ndarray,
    mask: np.ndarray,
    scale_range: Tuple[float, float] = (0.75, 1.25),
    output_size: Tuple[int, int] = (640, 360),   # (W, H)
) -> Tuple[np.ndarray, np.ndarray]:
    """Random scale then centre crop to output_size."""
    h, w = image.shape[:2]
    out_w, out_h = output_size

    scale = random.uniform(*scale_range)
    new_w = max(out_w, int(w * scale))
    new_h = max(out_h, int(h * scale))

    # Resize image and mask to new_w × new_h
    image_r = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_r  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Random crop
    x0 = random.randint(0, max(0, new_w - out_w))
    y0 = random.randint(0, max(0, new_h - out_h))
    image_c = image_r[y0:y0 + out_h, x0:x0 + out_w]
    mask_c  = mask_r[y0:y0 + out_h, x0:x0 + out_w]

    # Ensure exact output size (in case of rounding)
    if image_c.shape[:2] != (out_h, out_w):
        image_c = cv2.resize(image_c, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mask_c  = cv2.resize(mask_c,  (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    return image_c, mask_c


# ─────────────────────────────────────────────────────────────────
# Appearance augmentations (image only)
# ─────────────────────────────────────────────────────────────────

def color_jitter(
    image: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.2,
    saturation: float = 0.25,
    hue: float = 0.1,
) -> np.ndarray:
    """Apply random brightness, contrast, saturation, hue jitter to uint8 BGR image."""
    # Brightness
    if brightness > 0:
        delta = random.uniform(-brightness, brightness) * 255
        image = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # Contrast
    if contrast > 0:
        factor = random.uniform(1 - contrast, 1 + contrast)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation
    if saturation > 0:
        factor = random.uniform(1 - saturation, 1 + saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

    # Hue
    if hue > 0:
        delta = random.uniform(-hue, hue) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180

    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image


def shadow_augment(image: np.ndarray, p: float = 0.3) -> np.ndarray:
    """Apply random Bezier-curve shadow (darkening below the curve)."""
    if random.random() >= p:
        return image

    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Random quadratic Bezier: 3 control points
    p0x, p0y = random.randint(0, w), random.randint(0, h // 2)
    p1x, p1y = random.randint(0, w), random.randint(0, h)
    p2x, p2y = random.randint(0, w), random.randint(h // 2, h)

    # Build shadow mask
    shadow_mask = np.zeros((h, w), dtype=np.float32)
    for i in range(1000):
        t = i / 999.0
        # Bezier curve at t
        bx = int((1 - t) ** 2 * p0x + 2 * (1 - t) * t * p1x + t ** 2 * p2x)
        by = int((1 - t) ** 2 * p0y + 2 * (1 - t) * t * p1y + t ** 2 * p2y)
        bx = np.clip(bx, 0, w - 1)
        by = np.clip(by, 0, h - 1)
        shadow_mask[by:, bx] = 1.0   # below the curve

    # Apply shadow (darken by 0.3–0.7 factor)
    strength = random.uniform(0.3, 0.7)
    shadow_mask = shadow_mask[:, :, np.newaxis]  # HxWx1
    img = img * (1.0 - shadow_mask * (1.0 - strength))
    return np.clip(img, 0, 255).astype(np.uint8)


def puddle_augment(
    image: np.ndarray, mask: np.ndarray, p: float = 0.2
) -> np.ndarray:
    """Simulate reflective puddle on drivable region (image only)."""
    if random.random() >= p:
        return image

    h, w = image.shape[:2]
    road_pixels = np.argwhere(mask > 0)
    if len(road_pixels) < 100:
        return image

    # Random circle centre on drivable region
    idx = random.randint(0, len(road_pixels) - 1)
    cy, cx = road_pixels[idx]
    radius = random.randint(15, min(h, w) // 6)

    # Sky region = top 20% of image, flipped horizontally
    sky_h = max(1, h // 5)
    sky = image[:sky_h, :, :]
    sky_flip = cv2.flip(sky, 1)
    sky_resized = cv2.resize(sky_flip, (2 * radius, 2 * radius), interpolation=cv2.INTER_LINEAR)

    # Blend sky into circle region
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    sy1, sy2 = y1 - (cy - radius), y1 - (cy - radius) + (y2 - y1)
    sx1, sx2 = x1 - (cx - radius), x1 - (cx - radius) + (x2 - x1)

    region = image[y1:y2, x1:x2].astype(np.float32)
    blend = sky_resized[sy1:sy2, sx1:sx2].astype(np.float32)

    if region.shape == blend.shape and region.size > 0:
        alpha = 0.5
        image = image.copy()
        image[y1:y2, x1:x2] = np.clip(alpha * blend + (1 - alpha) * region, 0, 255).astype(np.uint8)

    return image


def fog_augment(image: np.ndarray, p: float = 0.15) -> np.ndarray:
    """Atmospheric scattering fog simulation."""
    if random.random() >= p:
        return image

    t = random.uniform(0.4, 0.8)
    A = np.array([0.85, 0.88, 0.90], dtype=np.float32) * 255  # fog "color" in BGR

    img = image.astype(np.float32)
    img = img * t + A * (1 - t)
    return np.clip(img, 0, 255).astype(np.uint8)


def night_augment(image: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Gamma correction + Gaussian noise to simulate night driving."""
    if random.random() >= p:
        return image

    gamma = random.uniform(0.15, 0.35)
    img = image.astype(np.float32) / 255.0
    img = np.power(np.clip(img, 1e-6, 1.0), gamma)

    noise = np.random.normal(0.0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def construction_zone_erase(
    image: np.ndarray, mask: np.ndarray, p: float = 0.25
) -> np.ndarray:
    """Draw horizontal noise rectangles on road region (missing lane paint sim)."""
    if random.random() >= p:
        return image

    h, w = image.shape[:2]
    road_rows = np.where(mask.sum(axis=1) > 0)[0]
    if len(road_rows) == 0:
        return image

    num_stripes = random.randint(2, 5)
    img = image.copy()

    for _ in range(num_stripes):
        row_idx = np.random.choice(road_rows)
        stripe_h = random.randint(2, 8)
        y1 = np.clip(row_idx, 0, h - 1)
        y2 = np.clip(row_idx + stripe_h, 0, h)

        # Road-colored noise (sample mean from that row)
        road_color = image[y1, :, :].mean(axis=0)  # (3,)
        noise = np.random.normal(road_color, 20, (y2 - y1, w, 3))
        img[y1:y2, :] = np.clip(noise, 0, 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────────────────────────────
# Main augmentation class
# ─────────────────────────────────────────────────────────────────

class VEGAAugment:
    """Augmentation pipeline for VEGA training / validation.

    Input:  uint8 BGR image (H, W, 3) + binary mask (H, W) numpy arrays
    Output: float32 torch tensors — image (3, H, W), mask (1, H, W)

    Args:
        mode: 'train' (full augmentation) or 'val' (resize + normalize only)
        img_size: (W, H) output size tuple
    """

    def __init__(self, mode: str = "train", img_size: Tuple[int, int] = (640, 360)):
        assert mode in ("train", "val"), f"Unknown mode: {mode}"
        self.mode = mode
        self.img_size = img_size  # (W, H)

    def __call__(
        self,
        image: np.ndarray,  # (H, W, 3) uint8 BGR
        mask: np.ndarray,   # (H, W) uint8 binary
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_t: (3, H, W) float32 tensor, ImageNet-normalized
            mask_t:  (1, H, W) float32 tensor, binary {0, 1}
        """
        out_w, out_h = self.img_size

        if self.mode == "train":
            # ── 1. Random horizontal flip ─────────────────────────────
            image, mask = random_horizontal_flip(image, mask, p=0.5)

            # ── 2. Random scale + crop ────────────────────────────────
            image, mask = random_scale_crop(
                image, mask,
                scale_range=(0.75, 1.25),
                output_size=(out_w, out_h),
            )

            # ── 3. Color jitter ───────────────────────────────────────
            image = color_jitter(image, brightness=0.3, contrast=0.2,
                                 saturation=0.25, hue=0.1)

            # ── 4. Shadow ─────────────────────────────────────────────
            image = shadow_augment(image, p=0.3)

            # ── 5. Puddle ─────────────────────────────────────────────
            image = puddle_augment(image, mask, p=0.2)

            # ── 6. Fog ────────────────────────────────────────────────
            image = fog_augment(image, p=0.15)

            # ── 7. Night ──────────────────────────────────────────────
            image = night_augment(image, p=0.1)

            # ── 8. Construction zone erase ────────────────────────────
            image = construction_zone_erase(image, mask, p=0.25)

        else:
            # Val: only resize to target size
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask,  (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # ── Ensure correct output size ────────────────────────────────
        if image.shape[:2] != (out_h, out_w):
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask,  (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # ── Convert BGR → RGB for ImageNet normalisation ──────────────
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = normalize(image_rgb)  # (H, W, 3) float32

        # ── To tensors ────────────────────────────────────────────────
        image_t = torch.from_numpy(image_norm.transpose(2, 0, 1))  # (3, H, W)
        mask_t  = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)  # (1, H, W)

        return image_t, mask_t


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)
    random.seed(0)

    # Dummy 720×1280 BGR image + binary mask (mimics nuScenes CAM_FRONT)
    H, W = 360, 640
    dummy_img  = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_mask = np.zeros((H, W), dtype=np.uint8)
    dummy_mask[100:260, 150:490] = 1   # road region

    for mode in ("train", "val"):
        aug = VEGAAugment(mode=mode, img_size=(640, 360))
        img_t, msk_t = aug(dummy_img.copy(), dummy_mask.copy())

        assert img_t.shape == (3, 360, 640), f"[{mode}] img shape: {img_t.shape}"
        assert msk_t.shape == (1, 360, 640), f"[{mode}] mask shape: {msk_t.shape}"
        assert msk_t.min() >= 0.0 and msk_t.max() <= 1.0, "Mask out of [0,1]"
        assert not torch.isnan(img_t).any(), f"NaN in {mode} image"
        assert not torch.isnan(msk_t).any(), f"NaN in {mode} mask"
        print(f"[PASS] mode={mode}: img {tuple(img_t.shape)}, mask {tuple(msk_t.shape)}")

    print("\nAll augment.py tests PASSED ✓")
