#!/usr/bin/env bash
# setup.sh — VEGA environment setup

set -e

echo "[VEGA] Installing dependencies..."
pip install -r requirements.txt

echo "[VEGA] Creating directories..."
mkdir -p checkpoints logs data/nuscenes

echo ""
echo "[VEGA] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download nuScenes dataset to data/nuscenes/"
echo "     https://www.nuscenes.org/nuscenes#download"
echo ""
echo "  2. Quick sanity check (synthetic data, no nuScenes needed):"
echo "     cd vega && python model/blocks.py"
echo "     python model/encoder.py"
echo "     python model/vega.py"
echo ""
echo "  3. Train on a 50-sample subset:"
echo "     python -m vega.train --epochs 2 --subset_n 50"
echo ""
echo "  4. Full training:"
echo "     python -m vega.train --epochs 150"
echo ""
echo "  5. Export to ONNX:"
echo "     python -m vega.export --checkpoint checkpoints/vega_best.pth"
echo ""
echo "  6. Real-time inference:"
echo "     python -m vega.infer --checkpoint checkpoints/vega_best.pth --source sample.mp4"
