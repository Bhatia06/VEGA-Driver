"""
infer.py — Real-time VEGA inference with temporal state

Accepts:
  --source 0            (webcam index)
  --source video.mp4    (video file)
  --source /path/imgs/  (folder of images, sorted)

Outputs:
  - Real-time display with mask overlay + FPS counter
  - Output video saved to --output path

Usage:
  python -m vega.infer --checkpoint checkpoints/vega_best.pth --source sample.mp4
  python -m vega.infer --checkpoint checkpoints/vega_best.pth --source 0
"""

import os
import sys
import time
import argparse
import glob
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vega.config import Config
from vega.model.vega import VEGA
from vega.utils.visualize import overlay_mask, draw_fps


# ImageNet normalisation
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame: np.ndarray, img_size: tuple) -> torch.Tensor:
    """BGR frame → (1, 3, H, W) normalized tensor."""
    out_w, out_h = img_size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rsz = cv2.resize(frame_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    img = frame_rsz.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    return tensor


@torch.no_grad()
def infer(cfg: Config, checkpoint: str, source: str, output: str) -> None:
    """Run real-time inference on source and save to output."""

    # ── Load model ────────────────────────────────────────────────
    model = VEGA(num_classes=cfg.num_classes).to(cfg.device)
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        print(f"[VEGA] Loaded: {checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[VEGA] No checkpoint — using random weights (demo mode)")
    model.eval()
    model.reset_temporal_state()

    # ── Open source ───────────────────────────────────────────────
    is_webcam   = False
    is_folder   = False
    image_paths = []

    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        # Webcam
        cap = cv2.VideoCapture(int(source))
        is_webcam = True
    elif os.path.isdir(source):
        # Image folder
        image_paths = sorted(
            glob.glob(os.path.join(source, "*.jpg")) +
            glob.glob(os.path.join(source, "*.png")) +
            glob.glob(os.path.join(source, "*.jpeg"))
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {source}")
        is_folder = True
        cap = None
    else:
        # Video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

    # ── Output writer ─────────────────────────────────────────────
    writer = None
    if output:
        if is_folder:
            ref_frame = cv2.imread(image_paths[0])
            fh, fw = ref_frame.shape[:2]
        elif cap is not None:
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            fw, fh = cfg.img_size

        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = 30.0 if is_webcam or is_folder else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        writer = cv2.VideoWriter(output, fourcc, fps_out, (fw, fh))
        print(f"[VEGA] Writing output to: {output}")

    # ── Warmup ────────────────────────────────────────────────────
    dummy = torch.zeros(1, 3, cfg.img_size[1], cfg.img_size[0], device=cfg.device)
    for _ in range(5):
        _ = model(dummy)
    model.reset_temporal_state()

    # ── Inference loop ────────────────────────────────────────────
    times = []
    frame_idx = 0
    avg_fps   = 0.0

    frame_iter = iter(image_paths) if is_folder else None

    print("[VEGA] Starting inference. Press 'q' to quit.")

    while True:
        # Get next frame
        if is_folder:
            try:
                path = next(frame_iter)
                frame = cv2.imread(path)
                if frame is None:
                    break
            except StopIteration:
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break

        original_h, original_w = frame.shape[:2]

        # Preprocess
        tensor = preprocess(frame, cfg.img_size).to(cfg.device)

        # Inference
        t0 = time.perf_counter()
        if cfg.amp and cfg.device == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(tensor)
        else:
            logits = model(tensor)

        if cfg.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        times.append(dt)
        if len(times) > 10:
            avg_fps = 1.0 / (sum(times[-30:]) / min(len(times), 30))

        # Postprocess mask
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()  # (H, W)
        mask_rsz = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Overlay
        result = overlay_mask(frame, mask_rsz, color=(0, 255, 80), alpha=0.5)
        result = draw_fps(result, avg_fps)

        # Write frame
        if writer is not None:
            writer.write(result)

        # Display
        cv2.imshow("VEGA — Drivable Surface Segmentation", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx:5d} | FPS: {avg_fps:.1f} | "
                  f"Latency: {dt*1000:.1f}ms")

    # ── Cleanup ───────────────────────────────────────────────────
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if times:
        import numpy as np
        arr = np.array(times[10:] or times)
        mean_fps = 1.0 / arr.mean()
        print(f"\n[VEGA] Done. {frame_idx} frames | Mean FPS: {mean_fps:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="VEGA Real-time Inference")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--source",     default="0",
                        help="Webcam int, video path, or image folder")
    parser.add_argument("--output",     default="./output_vega.mp4")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--no_amp",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(amp=not args.no_amp)

    source = args.source
    if source.isdigit():
        source = int(source)

    infer(cfg, checkpoint=args.checkpoint, source=source, output=args.output)
