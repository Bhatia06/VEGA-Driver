"""
train.py — Full VEGA training loop

Features:
  - AdamW + CosineAnnealingWarmRestarts scheduler
  - Linear LR warmup for first N epochs
  - Mixed precision (torch.cuda.amp)
  - Gradient accumulation (effective batch × grad_accum_steps)
  - Scene-boundary TCM reset
  - Validation every val_every epochs
  - Best-checkpoint saving by val mIoU
  - Training curve PNG plots
  - Resume from checkpoint

Usage:
  python -m vega.train --epochs 150 --batch_size 8
  python -m vega.train --epochs 2 --subset_n 50 --dry_run   # debug
  python -m vega.train --resume checkpoints/best.pth
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from typing import Optional

# Make sure vega package is importable when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vega.config import Config
from vega.model.vega import VEGA
from vega.loss.compound import VEGALoss
from vega.data.nuscenes_seg import NuScenesDrivableDataset, scene_aware_collate
from vega.utils.metrics import compute_miou
from vega.utils.logger import VEGALogger
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────
# Warmup scheduler helper
# ─────────────────────────────────────────────────────────────────

def warmup_lr(optimizer: AdamW, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    """Linear LR warmup from 1e-7 → base_lr over warmup_epochs epochs."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


# ─────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: VEGA,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: VEGALoss,
    scaler,
    cfg: Config,
    epoch: int,
    logger: VEGALogger,
    global_step: list,   # mutable list to allow mutation in closure
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        (mean_loss, mean_miou)
    """
    model.train()
    device = cfg.device

    total_loss = 0.0
    total_miou = 0.0
    n_batches  = 0

    prev_pred: Optional[torch.Tensor] = None
    current_scene: Optional[str] = None

    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch:03d}", leave=False, dynamic_ncols=True)

    for step, batch in pbar:
        images = batch["image"].to(device, non_blocking=True)   # (B,3,H,W)
        masks  = batch["mask"].to(device, non_blocking=True)    # (B,1,H,W)
        scene_tokens   = batch["scene_token"]
        is_scene_start = batch["is_scene_start"]

        # ── Scene boundary: reset TCM hidden state ────────────────
        if any(is_scene_start):
            model.reset_temporal_state()
            prev_pred = None
            current_scene = scene_tokens[0]

        # ── Forward pass ──────────────────────────────────────────
        if cfg.amp and device == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss, breakdown = criterion(logits, masks, prev_pred=prev_pred)
                loss = loss / cfg.grad_accum_steps
        else:
            logits = model(images)
            loss, breakdown = criterion(logits, masks, prev_pred=prev_pred)
            loss = loss / cfg.grad_accum_steps

        # CRITICAL: detach TCM hidden state from computation graph after each step
        # This prevents "backward through graph twice" errors across accumulation steps
        if model.hidden is not None:
            model.hidden = model.hidden.detach()

        # Update prev_pred for temporal loss
        with torch.no_grad():
            prev_pred = torch.sigmoid(logits).detach()

        # ── Backward ──────────────────────────────────────────────
        if cfg.amp and device == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── Gradient accumulation step ────────────────────────────
        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
            if cfg.amp and device == "cuda":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # ── Metrics ───────────────────────────────────────────────
        real_loss = loss.item() * cfg.grad_accum_steps
        miou = compute_miou(logits.detach(), masks)

        total_loss += real_loss
        total_miou += miou
        n_batches += 1
        global_step[0] += 1

        # ── Logging ───────────────────────────────────────────────
        if global_step[0] % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            gpu_mem = 0.0
            if device == "cuda" and torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
            logger.log_step(global_step[0], real_loss, breakdown, lr)
            pbar.set_postfix(loss=f"{real_loss:.4f}", miou=f"{miou:.4f}",
                             lr=f"{lr:.2e}", mem=f"{gpu_mem:.1f}G")

        if cfg.dry_run:
            break

    return total_loss / max(n_batches, 1), total_miou / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: VEGA,
    loader: DataLoader,
    criterion: VEGALoss,
    cfg: Config,
) -> tuple[float, float]:
    """Run validation loop.

    Returns:
        (mean_val_loss, mean_val_miou)
    """
    model.eval()
    model.reset_temporal_state()

    device = cfg.device
    total_loss = 0.0
    total_miou = 0.0
    n_batches  = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)
        is_scene_start = batch["is_scene_start"]

        if any(is_scene_start):
            model.reset_temporal_state()

        if cfg.amp and device == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss, _ = criterion(logits, masks)
        else:
            logits = model(images)
            loss, _ = criterion(logits, masks)

        total_loss += loss.item()
        total_miou += compute_miou(logits, masks)
        n_batches += 1

        if cfg.dry_run:
            break

    return total_loss / max(n_batches, 1), total_miou / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────

def train(cfg: Config, resume: Optional[str] = None) -> None:
    print(f"[VEGA] Training on device: {cfg.device}")
    print(f"[VEGA] AMP: {cfg.amp}, batch_size: {cfg.batch_size}, "
          f"grad_accum: {cfg.grad_accum_steps} → eff. batch: "
          f"{cfg.batch_size * cfg.grad_accum_steps}")

    # ── Datasets & Loaders ────────────────────────────────────────
    train_ds = NuScenesDrivableDataset(
        nusc_root=cfg.nusc_root,
        split="train",
        img_size=cfg.img_size,
        subset_n=cfg.subset_n,
    )
    val_ds = NuScenesDrivableDataset(
        nusc_root=cfg.nusc_root,
        split="val",
        img_size=cfg.img_size,
        subset_n=cfg.subset_n,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and cfg.device == "cuda",
        drop_last=True,
        collate_fn=scene_aware_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=max(0, cfg.num_workers // 2),
        drop_last=False,
        collate_fn=scene_aware_collate,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = VEGA(num_classes=cfg.num_classes).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[VEGA] Model params: {total_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── Scheduler (post-warmup) ───────────────────────────────────
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.min_lr,
    )

    # ── Scaler for AMP ────────────────────────────────────────────
    scaler = None
    if cfg.amp and cfg.device == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # ── Loss ──────────────────────────────────────────────────────
    criterion = VEGALoss(
        w_bce=cfg.w_bce, w_dice=cfg.w_dice,
        w_boundary=cfg.w_boundary, w_temporal=cfg.w_temporal,
        pos_weight=cfg.pos_weight,
    ).to(cfg.device)

    # ── Logger ────────────────────────────────────────────────────
    logger = VEGALogger(log_dir=cfg.log_dir, run_name=cfg.run_name)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 1
    best_miou   = 0.0
    global_step = [0]

    if resume is not None and os.path.exists(resume):
        print(f"[VEGA] Resuming from {resume}")
        ckpt = torch.load(resume, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("miou", 0.0)
        global_step = [ckpt.get("global_step", 0)]
        print(f"[VEGA] Resumed at epoch {start_epoch}, best_miou={best_miou:.4f}")

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        t_start = time.time()

        # Warmup
        if epoch <= cfg.warmup_epochs:
            warmup_lr(optimizer, epoch - 1, cfg.warmup_epochs, cfg.lr)
        else:
            scheduler.step(epoch - cfg.warmup_epochs)

        # Train
        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, cfg, epoch, logger, global_step,
        )

        # Validate
        val_loss, val_miou = None, None
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_loss, val_miou = validate(model, val_loader, criterion, cfg)

            # Save best checkpoint
            if val_miou > best_miou:
                best_miou = val_miou
                ckpt_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "miou": best_miou,
                    "global_step": global_step[0],
                    "config": vars(cfg),
                }, ckpt_path)
                print(f"[VEGA] ✓ New best mIoU: {best_miou:.4f} → saved {ckpt_path}")

        # Periodic checkpoint
        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_ep{epoch:04d}.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "miou": best_miou,
                "global_step": global_step[0],
            }, ckpt_path)

        epoch_time = time.time() - t_start
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_miou=train_miou,
            val_loss=val_loss,
            val_miou=val_miou,
            epoch_time=epoch_time,
        )
        logger.plot_curves(save_every=10, current_epoch=epoch)

    print(f"\n[VEGA] Training complete. Best Val mIoU: {best_miou:.4f}")
    logger.print_summary()


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train VEGA drivable segmentation")
    parser.add_argument("--nusc_root",  default="./data/nuscenes")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--subset_n",   type=int,   default=None,
                        help="Use only first N samples (for debugging)")
    parser.add_argument("--resume",     default=None, help="Path to checkpoint")
    parser.add_argument("--dry_run",    action="store_true",
                        help="1 batch per epoch for sanity check")
    parser.add_argument("--no_amp",     action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir",   default="./checkpoints")
    parser.add_argument("--log_dir",    default="./logs")
    parser.add_argument("--run_name",   default="vega")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        nusc_root=args.nusc_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_n=args.subset_n,
        dry_run=args.dry_run,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )
    train(cfg, resume=args.resume)
