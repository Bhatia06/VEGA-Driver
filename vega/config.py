"""
config.py — All VEGA hyperparameters as a single Config dataclass.

Usage:
    from vega.config import Config
    cfg = Config()
    cfg.epochs = 10  # override

Or from CLI (see train.py):
    python train.py --epochs 150 --batch_size 8
"""

from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ─── Data ─────────────────────────────────────────────────────
    nusc_root: str = "./data/nuscenes"
    img_size: tuple = (640, 360)       # (W, H)
    num_classes: int = 1

    # ─── Training ─────────────────────────────────────────────────
    epochs: int = 150
    batch_size: int = 8
    grad_accum_steps: int = 2          # effective batch size = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6              # cosine annealing floor

    # ─── Loss weights ─────────────────────────────────────────────
    w_bce: float = 0.35
    w_dice: float = 0.35
    w_boundary: float = 0.20
    w_temporal: float = 0.10
    pos_weight: float = 2.5            # BCE positive class weight

    # ─── Scheduler ────────────────────────────────────────────────
    T_0: int = 30                      # CosineAnnealingWarmRestarts period
    T_mult: int = 2

    # ─── Evaluation ───────────────────────────────────────────────
    val_every: int = 5                 # validate every N epochs
    save_every: int = 10              # save checkpoint every N epochs

    # ─── System ───────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp: bool = True                   # mixed precision (torch.cuda.amp)
    num_workers: int = 4
    pin_memory: bool = True

    # ─── Paths ────────────────────────────────────────────────────
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    run_name: str = "vega"

    # ─── Logging ──────────────────────────────────────────────────
    log_interval: int = 50             # log every N steps

    # ─── Debug / subset training ──────────────────────────────────
    subset_n: int = None              # if set, use only first N samples
    dry_run: bool = False             # 1 batch per epoch (sanity check)

    def __post_init__(self):
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


if __name__ == "__main__":
    cfg = Config()
    print("Config:")
    for k, v in vars(cfg).items():
        print(f"  {k:25s}: {v}")
    print(f"\n[PASS] Config instantiated. Device: {cfg.device}")
