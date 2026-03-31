"""
export.py — ONNX export and FPS benchmark for VEGA

Exports:
  - vega.onnx (opset 17, single-frame, batch-dynamic)

Benchmarks:
  - GPU FPS via CUDAExecutionProvider (if available)
  - CPU FPS via CPUExecutionProvider
  - PyTorch GPU FPS (for reference)

Usage:
  python -m vega.export --checkpoint checkpoints/vega_best.pth
  python -m vega.export --output vega.onnx
"""

import os
import sys
import time
import argparse
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vega.config import Config
from vega.model.vega import VEGA


def export_onnx(
    model: VEGA,
    onnx_path: str,
    img_size: tuple,
    device: str,
) -> None:
    """Export VEGA to ONNX format.

    The model is exported in single-frame mode (batch=1) with a static
    TCM hidden state (reset to zeros) — suitable for sequential inference
    where the GRU state is managed externally.

    Args:
        model:     VEGA model (eval mode)
        onnx_path: output .onnx file path
        img_size:  (W, H) tuple
        device:    torch device string
    """
    model.eval()
    model.reset_temporal_state()

    W, H = img_size
    dummy = torch.randn(1, 3, H, W, device=device)

    print(f"[Export] Tracing VEGA: input shape {tuple(dummy.shape)}")

    torch.onnx.export(
        model,
        (dummy,),                    # model inputs (positional)
        onnx_path,
        opset_version=17,
        input_names=["image"],
        output_names=["drivable_logits"],
        dynamic_axes={
            "image":            {0: "batch"},
            "drivable_logits":  {0: "batch"},
        },
        do_constant_folding=True,
        verbose=False,
    )

    file_size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[Export] Saved: {onnx_path}  ({file_size_mb:.2f} MB)")


def verify_onnx(onnx_path: str, img_size: tuple) -> None:
    """Verify ONNX model loads and produces correct output shape."""
    import onnx
    import onnxruntime as ort

    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    print("[Verify] ONNX model check passed ✓")

    W, H = img_size
    dummy_np = np.random.randn(1, 3, H, W).astype(np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(["drivable_logits"], {"image": dummy_np})[0]

    assert out.shape == (1, 1, H, W), f"Output shape mismatch: {out.shape}"
    print(f"[Verify] ONNX output shape: {out.shape} ✓")


def benchmark_onnx(
    onnx_path: str,
    img_size: tuple,
    warmup: int = 20,
    measure: int = 200,
) -> dict:
    """Benchmark ONNX model on CPU and GPU.

    Returns:
        dict with 'cpu_fps', 'gpu_fps' (None if CUDA not available)
    """
    import onnxruntime as ort

    W, H = img_size
    dummy = np.random.randn(1, 3, H, W).astype(np.float32)
    results = {}

    # ── CPU benchmark ─────────────────────────────────────────────
    sess_cpu = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    # Warmup
    for _ in range(warmup):
        sess_cpu.run(["drivable_logits"], {"image": dummy})

    # Measure
    times_cpu = []
    for _ in range(measure):
        t0 = time.perf_counter()
        sess_cpu.run(["drivable_logits"], {"image": dummy})
        times_cpu.append(time.perf_counter() - t0)

    cpu_fps = 1.0 / (np.mean(times_cpu) + 1e-9)
    results["cpu_fps"] = cpu_fps
    print(f"[Benchmark] ONNX CPU: {cpu_fps:.1f} FPS  "
          f"(latency: {np.mean(times_cpu)*1000:.1f}ms ± {np.std(times_cpu)*1000:.1f}ms)")

    # ── GPU benchmark ─────────────────────────────────────────────
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        sess_gpu = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Warmup
        for _ in range(warmup):
            sess_gpu.run(["drivable_logits"], {"image": dummy})

        # Measure
        times_gpu = []
        for _ in range(measure):
            t0 = time.perf_counter()
            sess_gpu.run(["drivable_logits"], {"image": dummy})
            times_gpu.append(time.perf_counter() - t0)

        gpu_fps = 1.0 / (np.mean(times_gpu) + 1e-9)
        results["gpu_fps"] = gpu_fps
        print(f"[Benchmark] ONNX GPU: {gpu_fps:.1f} FPS  "
              f"(latency: {np.mean(times_gpu)*1000:.1f}ms ± {np.std(times_gpu)*1000:.1f}ms)")
    else:
        results["gpu_fps"] = None
        print("[Benchmark] ONNX GPU: CUDA provider not available")

    return results


def benchmark_pytorch(
    model: VEGA,
    img_size: tuple,
    device: str,
    warmup: int = 20,
    measure: int = 200,
) -> float:
    """Benchmark PyTorch model FPS for reference."""
    model.eval()
    model.reset_temporal_state()

    W, H = img_size
    dummy = torch.randn(1, 3, H, W, device=device)
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if use_cuda:
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(measure):
            t0 = time.perf_counter()
            _ = model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    fps = 1.0 / (np.mean(times) + 1e-9)
    print(f"[Benchmark] PyTorch {device.upper()}: {fps:.1f} FPS  "
          f"(latency: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms)")
    return fps


def parse_args():
    parser = argparse.ArgumentParser(description="Export VEGA to ONNX")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output",     default="vega.onnx")
    parser.add_argument("--no_verify",  action="store_true")
    parser.add_argument("--warmup",     type=int, default=20)
    parser.add_argument("--measure",    type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()

    model = VEGA(num_classes=cfg.num_classes).to(cfg.device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        print(f"[Export] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[Export] Using random weights (no checkpoint provided)")

    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1e6  # float32
    print(f"[Export] Model: {total_params:,} params, ~{model_size_mb:.2f} MB (fp32)")

    # ── PyTorch benchmark ─────────────────────────────────────────
    pt_fps = benchmark_pytorch(model, cfg.img_size, cfg.device,
                               warmup=args.warmup, measure=args.measure)

    # ── ONNX export (requires onnx package) ───────────────────────
    try:
        import onnx as _onnx_test
        onnx_available = True
    except ImportError:
        onnx_available = False
        print("[Export] WARNING: 'onnx' package not installed.")
        print("[Export] To enable ONNX export, run:")
        print("  pip install onnx onnxruntime")
        print(f"\n[Summary] PyTorch {cfg.device.upper()}: {pt_fps:.1f} FPS")
        print(f"[Summary] Model: {total_params:,} params (~{model_size_mb:.2f} MB)")

    if onnx_available:
        export_onnx(model, args.output, cfg.img_size, cfg.device)

        # ── ONNX verify + benchmark ───────────────────────────────
        if not args.no_verify:
            try:
                verify_onnx(args.output, cfg.img_size)
                results = benchmark_onnx(
                    args.output, cfg.img_size,
                    warmup=args.warmup, measure=args.measure,
                )

                print("\n--- Export Summary ---")
                print(f"  ONNX file:      {args.output}")
                print(f"  File size:      {os.path.getsize(args.output)/1e6:.2f} MB")
                print(f"  PyTorch {cfg.device.upper()}:   {pt_fps:.1f} FPS")
                cpu_ok = results['cpu_fps'] > 18
                print(f"  ONNX CPU:       {results['cpu_fps']:.1f} FPS "
                      f"({'OK >18' if cpu_ok else 'SLOW target >18'})")
                if results.get("gpu_fps"):
                    gpu_ok = results['gpu_fps'] > 30
                    print(f"  ONNX GPU:       {results['gpu_fps']:.1f} FPS "
                          f"({'OK >30' if gpu_ok else 'SLOW target >30'})")
                print("-----------------------------------------------------")

            except ImportError as e:
                print(f"[Export] Skipping ONNX verify/benchmark: {e}")
                print("[Export] Install: pip install onnx onnxruntime")
