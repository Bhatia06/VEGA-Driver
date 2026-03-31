"""
test_all.py — Comprehensive shape tests for all VEGA modules.
Run from project root: python vega/test_all.py
"""

import sys
import os
import time
import torch
import numpy as np

# Add parent to path so 'vega' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vega.model.blocks import (
    HardSwish, ConvBNAct, DepthwiseSeparableConv,
    SqueezeExcitation, InvertedResidualBlock,
)
from vega.model.encoder import VEGAEncoder
from vega.model.tcm import ConvGRUCell
from vega.model.decoder import VEGADecoder
from vega.model.vega import VEGA
from vega.loss.boundary import get_boundary_mask
from vega.loss.compound import VEGALoss
from vega.data.augment import VEGAAugment
from vega.utils.metrics import compute_miou, compute_boundary_iou, FPSCounter
from vega.config import Config

PASS = "✓"
FAIL = "✗"

def section(name: str):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


def check(cond: bool, msg: str):
    status = PASS if cond else FAIL
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


# ─────────────────────────────────────────────────────────────────
# Phase 1 — Model architecture
# ─────────────────────────────────────────────────────────────────

section("Phase 1a — model/blocks.py")
torch.manual_seed(0)
x = torch.randn(2, 3, 360, 640)

hs = HardSwish()
out = hs(x)
check(out.shape == (2, 3, 360, 640), f"HardSwish shape: {tuple(out.shape)}")
check(not torch.isnan(out).any(), "HardSwish no NaN")

cba = ConvBNAct(3, 16, kernel=3, stride=1, activation=HardSwish())
out = cba(x)
check(out.shape == (2, 16, 360, 640), f"ConvBNAct shape: {tuple(out.shape)}")
check(not torch.isnan(out).any(), "ConvBNAct no NaN")

dsc = DepthwiseSeparableConv(3, 16, stride=1)
out = dsc(x)
check(out.shape == (2, 16, 360, 640), f"DSConv shape: {tuple(out.shape)}")

se = SqueezeExcitation(16, reduction=4)
x16 = torch.randn(2, 16, 360, 640)
out = se(x16)
check(out.shape == (2, 16, 360, 640), f"SE shape: {tuple(out.shape)}")

block = InvertedResidualBlock(3, 16, 16, stride=1, use_se=True, activation=HardSwish())
out = block(x)
check(out.shape == (2, 16, 360, 640), f"IRB shape: {tuple(out.shape)}")
check(not torch.isnan(out).any(), "IRB no NaN")
print(f"  [i] IRB params: {sum(p.numel() for p in block.parameters()):,}")


section("Phase 1b — model/encoder.py")
torch.manual_seed(1)
enc = VEGAEncoder()
enc.eval()
x = torch.randn(2, 3, 360, 640)
with torch.no_grad():
    s1, s2, s3, bn = enc(x)

check(s1.shape == (2, 24, 90, 160), f"s1 shape: {tuple(s1.shape)}")
check(s2.shape == (2, 48, 45, 80),  f"s2 shape: {tuple(s2.shape)}")
check(bn.shape[1] == 96,            f"bn channels: {bn.shape[1]}")
check(bn.shape[0] == 2,             f"bn batch: {bn.shape[0]}")
check(not torch.isnan(s1).any(), "s1 no NaN")
check(not torch.isnan(bn).any(), "bn no NaN")

enc_params = sum(p.numel() for p in enc.parameters())
print(f"  [i] Encoder params: {enc_params:,}")
check(enc_params < 2_000_000, f"Encoder < 2M params: {enc_params:,}")


section("Phase 1c — model/tcm.py")
torch.manual_seed(42)
gru = ConvGRUCell(input_dim=96, hidden_dim=96, kernel_size=3)
gru.eval()
x_gru = torch.randn(2, 96, 23, 40)

h1 = gru(x_gru, None)
check(h1.shape == (2, 96, 23, 40), f"h1 shape (frame 1): {tuple(h1.shape)}")
check(not torch.isnan(h1).any(), "h1 no NaN")

h2 = gru(x_gru, h1)
check(h2.shape == (2, 96, 23, 40), f"h2 shape (frame 2): {tuple(h2.shape)}")
check(not torch.allclose(h1, h2), "Temporal state updates between frames")
print(f"  [i] ConvGRU params: {sum(p.numel() for p in gru.parameters()):,}")


section("Phase 1d — model/decoder.py")
torch.manual_seed(7)
dec = VEGADecoder(num_classes=1)
dec.eval()
dummy_feats = (
    torch.randn(2, 24, 90, 160),
    torch.randn(2, 48, 45, 80),
    torch.randn(2, 96, 23, 40),
    torch.randn(2, 96, 23, 40),
)
with torch.no_grad():
    out = dec(*dummy_feats)
check(out.shape == (2, 1, 360, 640), f"Decoder output shape: {tuple(out.shape)}")
check(not torch.isnan(out).any(), "Decoder no NaN")
print(f"  [i] Decoder params: {sum(p.numel() for p in dec.parameters()):,}")


section("Phase 1e — model/vega.py (full model)")
torch.manual_seed(99)
model = VEGA(num_classes=1)
model.eval()
x = torch.randn(2, 3, 360, 640)
with torch.no_grad():
    out1 = model(x)
check(out1.shape == (2, 1, 360, 640), f"VEGA output shape: {tuple(out1.shape)}")
check(not torch.isnan(out1).any(), "VEGA no NaN (frame 1)")

with torch.no_grad():
    out2 = model(x)
check(out2.shape == (2, 1, 360, 640), "Frame 2 shape ok")
check(not torch.allclose(out1, out2), "Temporal state propagates")

model.reset_temporal_state()
check(model.hidden is None, "reset_temporal_state() works")

total_params = sum(p.numel() for p in model.parameters())
print(f"  [i] TOTAL model params: {total_params:,}")
check(total_params < 2_500_000, f"Model < 2.5M params: {total_params:,}")

# Parameter breakdown
enc_p = sum(p.numel() for p in model.encoder.parameters())
tcm_p = sum(p.numel() for p in model.tcm.parameters())
dec_p = sum(p.numel() for p in model.decoder.parameters())
print(f"       Encoder: {enc_p:,}  |  TCM: {tcm_p:,}  |  Decoder: {dec_p:,}")


# ─────────────────────────────────────────────────────────────────
# Phase 2 — Loss functions
# ─────────────────────────────────────────────────────────────────

section("Phase 2a — loss/boundary.py")
mask = torch.zeros(2, 1, 360, 640)
mask[:, :, 100:260, 200:440] = 1.0
boundary = get_boundary_mask(mask, dilation_px=5)
check(boundary.shape == (2, 1, 360, 640), f"Boundary shape: {tuple(boundary.shape)}")
check(not torch.isnan(boundary).any(), "Boundary no NaN")
check(boundary.min() >= 0.0 and boundary.max() <= 1.0, "Boundary in [0,1]")
pct = boundary.mean().item() * 100
check(0 < pct < 20, f"Boundary coverage {pct:.2f}% in (0, 20)")


section("Phase 2b — loss/compound.py")
torch.manual_seed(42)
criterion = VEGALoss()
logits = torch.randn(2, 1, 360, 640)
target = torch.randint(0, 2, (2, 1, 360, 640)).float()
loss, breakdown = criterion(logits, target)
check(not torch.isnan(loss), "Compound loss no NaN")
check(0.3 < loss.item() < 3.0, f"Loss in expected range: {loss.item():.4f}")
check(all(k in breakdown for k in ["bce", "dice", "boundary", "temporal"]), 
      "breakdown has all 4 keys")

# With temporal
prev_p = torch.sigmoid(torch.randn(2, 1, 360, 640))
loss2, bd2 = criterion(logits, target, prev_pred=prev_p)
check(bd2["temporal"] > 0, f"Temporal loss > 0: {bd2['temporal']:.4f}")
print(f"  [i] Loss={loss.item():.4f}  Components: bce={breakdown['bce']:.4f} "
      f"dice={breakdown['dice']:.4f} boundary={breakdown['boundary']:.4f}")

# Gradient check
lg = torch.randn(2, 1, 180, 320, requires_grad=True)
tg = torch.randint(0, 2, (2, 1, 180, 320)).float()
l, _ = criterion(lg, tg)
l.backward()
check(lg.grad is not None, "Gradient flows")
check(not torch.isnan(lg.grad).any(), "No NaN in gradient")


# ─────────────────────────────────────────────────────────────────
# Phase 3 — Data pipeline
# ─────────────────────────────────────────────────────────────────

section("Phase 3a — data/augment.py")
import cv2
import random
random.seed(0)
np.random.seed(0)

dummy_img  = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
dummy_mask = np.zeros((900, 1600), dtype=np.uint8)
dummy_mask[200:700, 300:1300] = 1

for mode in ("train", "val"):
    aug = VEGAAugment(mode=mode, img_size=(640, 360))
    img_t, msk_t = aug(dummy_img.copy(), dummy_mask.copy())
    check(img_t.shape == (3, 360, 640), f"[{mode}] img shape: {tuple(img_t.shape)}")
    check(msk_t.shape == (1, 360, 640), f"[{mode}] mask shape: {tuple(msk_t.shape)}")
    check(msk_t.min() >= 0.0 and msk_t.max() <= 1.0, f"[{mode}] mask in [0,1]")
    check(not torch.isnan(img_t).any(), f"[{mode}] img no NaN")


section("Phase 3b — data/nuscenes_seg.py (synthetic fallback)")
from vega.data.nuscenes_seg import NuScenesDrivableDataset, scene_aware_collate
from torch.utils.data import DataLoader

ds = NuScenesDrivableDataset(nusc_root="./data/nuscenes", split="train", img_size=(640, 360))
check(len(ds) > 0, f"Dataset length: {len(ds)}")
item = ds[0]
check(item["image"].shape == (3, 360, 640), f"item image shape: {tuple(item['image'].shape)}")
check(item["mask"].shape  == (1, 360, 640), f"item mask shape: {tuple(item['mask'].shape)}")
check(not torch.isnan(item["image"]).any(), "Dataset image no NaN")

loader = DataLoader(ds, batch_size=4, collate_fn=scene_aware_collate, num_workers=0)
batch = next(iter(loader))
check(batch["image"].shape == (4, 3, 360, 640), f"Batch image: {tuple(batch['image'].shape)}")
check(batch["mask"].shape  == (4, 1, 360, 640), f"Batch mask: {tuple(batch['mask'].shape)}")


# ─────────────────────────────────────────────────────────────────
# Phase 4 — Metrics & config
# ─────────────────────────────────────────────────────────────────

section("Phase 4a — utils/metrics.py")
logits_m = torch.randn(4, 1, 360, 640)
targets_m = torch.randint(0, 2, (4, 1, 360, 640)).float()
miou = compute_miou(logits_m, targets_m)
biou = compute_boundary_iou(logits_m, targets_m)
check(0.0 <= miou <= 1.0, f"mIoU in [0,1]: {miou:.4f}")
check(0.0 <= biou <= 1.0, f"BoundaryIoU in [0,1]: {biou:.4f}")

# Perfect prediction test
perf_logits = targets_m * 10 - 5
miou_perf = compute_miou(perf_logits, targets_m)
check(miou_perf > 0.9, f"Perfect prediction mIoU: {miou_perf:.4f}")
print(f"  [i] mIoU={miou:.4f} | BoundaryIoU={biou:.4f} | Perfect={miou_perf:.4f}")


section("Phase 4b — config.py")
cfg = Config()
check(cfg.epochs == 150, f"Default epochs: {cfg.epochs}")
check(cfg.img_size == (640, 360), f"Default img_size: {cfg.img_size}")
check(cfg.batch_size == 8, f"Default batch_size: {cfg.batch_size}")
check(cfg.device in ("cuda", "cpu"), f"Device: {cfg.device}")
print(f"  [i] Device: {cfg.device}, AMP: {cfg.amp}")


# ─────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"  ALL TESTS PASSED ✓  [{total_params:,} params]")
print(f"{'═'*60}")

# Verification checklist
print("\n  Verification Checklist:")
print(f"  [ ] Total model params < 2.5M:  {total_params:,}  {'✓' if total_params < 2_500_000 else '✗'}")
print(f"  [ ] No pretrained weights: check with grep (see below)")
print(f"  [ ] ONNX export: run vega/export.py")
print(f"  [ ] Full training: python -m vega.train --epochs 2 --subset_n 50")
