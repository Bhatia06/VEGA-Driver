"""
eval.py — Evaluation on the nuScenes val set

Computes:
  - mIoU (binary drivable segmentation)
  - BoundaryIoU (accuracy at mask boundaries)
  - Per-scene breakdown
  - Saves sample visual comparisons to logs/

Usage:
  python -m vega.eval --checkpoint checkpoints/vega_best.pth
  python -m vega.eval --checkpoint checkpoints/vega_best.pth --subset_n 50
"""

import os
import sys
import argparse

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vega.config import Config
from vega.model.vega import VEGA
from vega.loss.compound import VEGALoss
from vega.data.nuscenes_seg import NuScenesDrivableDataset, scene_aware_collate
from vega.utils.metrics import compute_miou, compute_boundary_iou, FPSCounter
from vega.utils.visualize import save_comparison
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: VEGA,
    loader: DataLoader,
    criterion: VEGALoss,
    cfg: Config,
    save_dir: str = "./logs/eval",
    max_save: int = 10,
) -> dict:
    """Run full evaluation loop.

    Returns:
        dict with keys: miou, boundary_iou, loss
    """
    model.eval()
    model.reset_temporal_state()

    device = cfg.device
    total_loss  = 0.0
    total_miou  = 0.0
    total_biou  = 0.0
    n_batches   = 0
    saved_count = 0

    os.makedirs(save_dir, exist_ok=True)

    fps_counter = FPSCounter(warmup=5, measure=500, use_cuda=(device == "cuda"))

    pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        if any(batch["is_scene_start"]):
            model.reset_temporal_state()

        if cfg.amp and device == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss, _ = criterion(logits, masks)
        else:
            logits = model(images)
            loss, _ = criterion(logits, masks)

        fps_counter.tick()

        miou = compute_miou(logits, masks)
        biou = compute_boundary_iou(logits, masks)

        total_loss  += loss.item()
        total_miou  += miou
        total_biou  += biou
        n_batches   += 1

        pbar.set_postfix(miou=f"{miou:.4f}", biou=f"{biou:.4f}")

        # Save sample visualizations
        if saved_count < max_save:
            for b in range(min(2, images.size(0))):
                out_path = os.path.join(save_dir, f"sample_{saved_count:04d}.png")
                save_comparison(images[b].cpu(), logits[b].cpu(), masks[b].cpu(), out_path)
                saved_count += 1

    fps_mean, fps_std = fps_counter.result()

    results = {
        "miou":         total_miou / max(n_batches, 1),
        "boundary_iou": total_biou / max(n_batches, 1),
        "loss":         total_loss / max(n_batches, 1),
        "fps":          fps_mean,
        "fps_std":      fps_std,
    }

    print("\n─── Evaluation Results ───────────────────────────────")
    print(f"  mIoU:        {results['miou']:.4f}")
    print(f"  BoundaryIoU: {results['boundary_iou']:.4f}")
    print(f"  Val Loss:    {results['loss']:.4f}")
    print(f"  Throughput:  {fps_mean:.1f} ± {fps_std:.1f} FPS")
    print(f"  Samples saved to: {save_dir}")
    print("──────────────────────────────────────────────────────")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VEGA")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    parser.add_argument("--nusc_root",  default="./data/nuscenes")
    parser.add_argument("--subset_n",   type=int, default=None)
    parser.add_argument("--save_dir",   default="./logs/eval")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_amp",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        nusc_root=args.nusc_root,
        subset_n=args.subset_n,
        amp=not args.no_amp,
        batch_size=args.batch_size,
    )

    model = VEGA(num_classes=cfg.num_classes).to(cfg.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        print(f"[VEGA] Loaded checkpoint: {args.checkpoint}")
        print(f"[VEGA] Checkpoint epoch: {ckpt.get('epoch', 'unknown')}, "
              f"mIoU: {ckpt.get('miou', 'unknown')}")
    else:
        print("[VEGA] No checkpoint provided — evaluating with random weights")

    val_ds = NuScenesDrivableDataset(
        nusc_root=cfg.nusc_root,
        split="val",
        img_size=cfg.img_size,
        subset_n=cfg.subset_n,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        collate_fn=scene_aware_collate,
    )

    criterion = VEGALoss().to(cfg.device)

    results = evaluate(model, val_loader, criterion, cfg, save_dir=args.save_dir)
