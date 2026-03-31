"""
utils/logger.py — Training logger with loss/mIoU curve plotting
"""

import os
import json
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
from pathlib import Path
from typing import Optional


class VEGALogger:
    """Training progress logger with loss/mIoU curve plotting.

    Args:
        log_dir:   directory to save logs and plots
        run_name:  experiment name prefix
    """

    def __init__(self, log_dir: str = "./logs", run_name: str = "vega"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_mious: list[float] = []
        self.val_mious: list[float] = []
        self.epochs: list[int] = []
        self.val_epochs: list[int] = []

        self.step_losses: list[float] = []
        self.step_ids: list[int] = []

        self._t_epoch_start: Optional[float] = None

        # Log file
        self.log_path = self.log_dir / f"{run_name}_log.jsonl"

    def log_step(self, step: int, loss: float, breakdown: dict, lr: float) -> None:
        """Log a training step."""
        self.step_losses.append(loss)
        self.step_ids.append(step)

        entry = {
            "type": "step",
            "step": step,
            "loss": loss,
            "lr": lr,
            **{f"loss_{k}": v for k, v in breakdown.items()},
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_miou: float,
        val_loss: Optional[float] = None,
        val_miou: Optional[float] = None,
        epoch_time: Optional[float] = None,
    ) -> None:
        """Log one epoch summary."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_mious.append(train_miou)

        msg = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train mIoU: {train_miou:.4f}"
        )

        if val_loss is not None:
            self.val_epochs.append(epoch)
            self.val_losses.append(val_loss)
            self.val_mious.append(val_miou or 0.0)
            msg += f" | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}"

        if epoch_time is not None:
            msg += f" | {epoch_time:.1f}s"

        print(msg)

        entry = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "epoch_time": epoch_time,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def plot_curves(self, save_every: int = 10, current_epoch: int = 0) -> None:
        """Save loss and mIoU curves as PNG if current_epoch % save_every == 0."""
        if current_epoch % save_every != 0 or len(self.epochs) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"VEGA Training — {self.run_name}", fontsize=14, fontweight="bold")

        # Loss curve
        axes[0].plot(self.epochs, self.train_losses, label="Train", color="#4C9BE8", lw=2)
        if self.val_losses:
            axes[0].plot(self.val_epochs, self.val_losses, label="Val", color="#E84C4C",
                         lw=2, marker="o", markersize=4)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Compound Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # mIoU curve
        axes[1].plot(self.epochs, self.train_mious, label="Train mIoU", color="#4CE877", lw=2)
        if self.val_mious:
            axes[1].plot(self.val_epochs, self.val_mious, label="Val mIoU", color="#E8964C",
                         lw=2, marker="o", markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mIoU")
        axes[1].set_title("Segmentation mIoU")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.log_dir / f"{self.run_name}_curves_ep{current_epoch:04d}.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[Logger] Curves saved: {save_path}")

    def print_summary(self, gpu_mem_gb: float = 0.0) -> None:
        """Print current best metrics summary."""
        if not self.val_mious:
            return
        best_miou = max(self.val_mious)
        best_ep   = self.val_epochs[self.val_mious.index(best_miou)]
        print(f"  Best Val mIoU: {best_miou:.4f} @ Epoch {best_ep}")
        if gpu_mem_gb > 0:
            print(f"  GPU Memory: {gpu_mem_gb:.2f} GB")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = VEGALogger(log_dir=tmpdir, run_name="test")
        for epoch in range(1, 11):
            logger.log_epoch(
                epoch=epoch,
                train_loss=1.0 - epoch * 0.05,
                train_miou=epoch * 0.04,
                val_loss=(1.0 - epoch * 0.05) * 1.05 if epoch % 2 == 0 else None,
                val_miou=epoch * 0.035 if epoch % 2 == 0 else None,
            )
            logger.plot_curves(save_every=5, current_epoch=epoch)

        assert (Path(tmpdir) / "test_curves_ep0010.png").exists()
        print("[PASS] Logger curves saved correctly")

    print("\nAll logger.py tests PASSED ✓")
