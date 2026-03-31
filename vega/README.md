<div align="center">

# 🚗 VEGA
### Vehicle Edge Guidance Architecture
**Real-Time Drivable Surface Segmentation for Level 4 Autonomous Vehicles**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Params](https://img.shields.io/badge/Params-735K-purple)]()
[![FPS](https://img.shields.io/badge/GPU%20FPS-231%2B-brightgreen)]()

</div>

---

## Overview

VEGA is a lightweight, real-time drivable surface segmentation system built **100% from scratch** in PyTorch — no pretrained weights, no torchvision models, no external backbone libraries.

It uses a MobileNetV3-style encoder, a Convolutional GRU (ConvGRU) for temporal coherence across video frames, and an asymmetric bilinear decoder — all under **735K parameters**.

| Metric | Value |
|--------|-------|
| Total Parameters | **734,961 (~0.73M)** |
| Model Size (fp32) | **2.94 MB** |
| PyTorch GPU FPS | **>230 FPS** (RTX series) |
| ONNX CPU FPS Target | **>18 FPS** |
| Input Resolution | 640 × 360 |
| Dataset | nuScenes (binary drivable surface) |

---

## Architecture

```
Input (B, 3, 360, 640)
        │
        ▼
  ┌─────────────┐
  │ VEGAEncoder │  MobileNetV3-style, 4 stages, HardSwish + SE blocks
  └─────────────┘
  s1 (B,24,90,160) ──────────────────────────────────┐
  s2 (B,48,45,80)  ─────────────────────────┐        │
  bn (B,96,23,40)  ──► ConvGRUCell (TCM) ──►│        │
                        temporal hidden state │        │
        │                                    │        │
        ▼                                    │        │
  ┌─────────────┐                            │        │
  │ VEGADecoder │◄───────────────────────────┘────────┘
  └─────────────┘  bilinear upsample + skip fusions
        │
        ▼
  Logits (B, 1, 360, 640)
```

**Compound Loss:** Weighted BCE (0.35) + Soft Dice (0.35) + Boundary-Aware BCE (0.20) + Temporal MSE (0.10)

---

## Project Structure

```
vega/
├── model/
│   ├── blocks.py        # HardSwish, ConvBNAct, SE, InvertedResidualBlock
│   ├── encoder.py       # VEGAEncoder — 4-stage MobileNetV3-style
│   ├── decoder.py       # VEGADecoder — asymmetric bilinear upsampling
│   ├── tcm.py           # ConvGRUCell — temporal coherence module
│   └── vega.py          # Full VEGA model assembly
├── loss/
│   ├── boundary.py      # Dilated boundary mask extraction
│   └── compound.py      # VEGALoss — 4-component compound loss
├── data/
│   ├── nuscenes_seg.py  # nuScenes dataloader + synthetic fallback
│   └── augment.py       # 8-stage augmentation pipeline
├── utils/
│   ├── metrics.py       # mIoU, BoundaryIoU, FPSCounter
│   ├── visualize.py     # Overlay, comparison saves, FPS HUD
│   └── logger.py        # JSONL + matplotlib training curves
├── train.py             # Full training loop
├── eval.py              # Validation evaluation
├── infer.py             # Real-time inference (webcam/video/folder)
├── export.py            # ONNX export + FPS benchmark
├── config.py            # All hyperparameters as Config dataclass
├── test_all.py          # Comprehensive shape test suite
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourname/vega.git
cd vega

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) ONNX export support
pip install onnx onnxruntime
```

---

## Quick Start — No Data Required

Run the full shape test suite to verify everything works:

```bash
python vega/test_all.py
```

Expected:
```
ALL TESTS PASSED ✓  [734,961 params]
```

Train on synthetic data (no nuScenes needed):

```bash
python -m vega.train --epochs 5 --subset_n 50 --num_workers 0 --batch_size 4
```

---

## With nuScenes Dataset

### 1. Download nuScenes

Download the **nuScenes Full Dataset (v1.0)** from [nuscenes.org](https://www.nuscenes.org/nuscenes#download) and extract to `data/nuscenes/`.

```
data/nuscenes/
├── v1.0-trainval/
├── samples/
│   └── CAM_FRONT/
├── sweeps/
└── panoptic/   ← for drivable surface masks
```

### 2. Install the devkit

```bash
pip install nuscenes-devkit
```

### 3. Train

```bash
python -m vega.train \
  --nusc_root data/nuscenes \
  --epochs 150 \
  --batch_size 8 \
  --num_workers 4
```

Training produces:
- `checkpoints/vega_best.pth` — best val mIoU checkpoint
- `logs/vega_curves_*.png` — loss/mIoU plots every 10 epochs

---

## Evaluation

```bash
python -m vega.eval \
  --checkpoint checkpoints/vega_best.pth \
  --nusc_root data/nuscenes
```

Output:
```
── Evaluation Results ──────────────────────
  mIoU:        0.XXXX
  BoundaryIoU: 0.XXXX
  Val Loss:    0.XXXX
  Throughput:  XXX FPS
  Samples saved to: ./logs/eval
────────────────────────────────────────────
```

Visual comparisons (3-panel: input | GT | prediction) are saved to `logs/eval/`.

---

## Real-Time Inference

```bash
# Webcam (live)
python -m vega.infer \
  --checkpoint checkpoints/vega_best.pth \
  --source 0 \
  --output output_webcam.mp4

# Video file
python -m vega.infer \
  --checkpoint checkpoints/vega_best.pth \
  --source dashcam.mp4 \
  --output output_vega.mp4

# Image folder
python -m vega.infer \
  --checkpoint checkpoints/vega_best.pth \
  --source frames/ \
  --output output_vega.mp4
```

The live window shows the **lime-green drivable surface overlay** with a real-time FPS counter. Press `q` to quit.

---

## ONNX Export & Benchmarking

```bash
# Export + benchmark (requires pip install onnx onnxruntime)
python -m vega.export \
  --checkpoint checkpoints/vega_best.pth \
  --output vega.onnx

# PyTorch FPS only (no onnx needed)
python -m vega.export --no_verify --warmup 20 --measure 200
```

Output:
```
[Benchmark] PyTorch CUDA: 231.2 FPS  (latency: 4.3ms ± 0.1ms)
[Benchmark] ONNX CPU:      XX.X FPS
[Benchmark] ONNX GPU:      XX.X FPS
```

---

## Training Configuration

All hyperparameters are in `config.py`:

```python
@dataclass
class Config:
    nusc_root: str = "./data/nuscenes"
    img_size: tuple = (640, 360)      # W × H
    epochs: int = 150
    batch_size: int = 8
    grad_accum_steps: int = 2         # effective batch = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    w_bce: float = 0.35
    w_dice: float = 0.35
    w_boundary: float = 0.20
    w_temporal: float = 0.10
    amp: bool = True                  # mixed precision
```

Resume from checkpoint:
```bash
python -m vega.train --resume checkpoints/vega_best.pth --epochs 150
```

---

## Data Augmentation Pipeline

| Stage | Augmentation | Probability |
|-------|-------------|-------------|
| 1 | Random Horizontal Flip | 50% |
| 2 | Random Scale + Crop (0.75–1.25×) | Always |
| 3 | Color Jitter (brightness/contrast/sat/hue) | Always |
| 4 | Shadow (Bezier curve darkening) | 30% |
| 5 | Puddle (sky reflection on road) | 20% |
| 6 | Fog (atmospheric scattering) | 15% |
| 7 | Night (gamma + Gaussian noise) | 10% |
| 8 | Construction Zone Erase (road stripes) | 25% |

---

## Design Principles

- **No pretrained weights** — all parameters initialised from scratch (Kaiming Normal for conv, Orthogonal for ConvGRU)
- **ONNX-safe** — no Python conditionals in forward passes; hidden state init via `torch.zeros()` not in-graph conditionals
- **Temporal-aware** — ConvGRU maintains spatial hidden state across frames; reset at scene boundaries
- **Boundary-weighted loss** — 3× pixel weight on mask boundaries to sharpen edges
- **AMP + Gradient Accumulation** — trains efficiently on a single RTX 3060/3070/T4

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
nuscenes-devkit>=1.1.11
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
onnx>=1.15.0          (optional, for export)
onnxruntime>=1.16.0   (optional, for export)
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@misc{vega2026,
  title   = {VEGA: Vehicle Edge Guidance Architecture for Real-Time Drivable Surface Segmentation},
  author  = {Bhatia, Divya},
  year    = {2026},
  url     = {https://github.com/Bhatia06/VEGA-Driver}
}
```
