#!/usr/bin/env bash
# Bootstrap Strawberry Rover on Jetson Orin 8GB.
# Usage: bash bootstrap/orin_bootstrap.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Strawberry Rover Orin bootstrap"
echo "    repo: $REPO_ROOT"

if [ -f "$REPO_ROOT/scripts/jetson_gpu_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/jetson_gpu_env.sh"
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system packages (may require sudo)..."
  sudo apt-get update -qq || true
  sudo apt-get install -y -qq \
    libopenblas0-pthread \
    python3-pip python3-venv \
    python3-pyqt5 \
    curl git \
    || true
fi

VENV="$REPO_ROOT/.venv_cuda"
if [ -d "$VENV" ] && [ ! -f "$VENV/bin/activate" ]; then
  echo "==> Removing broken venv (missing bin/activate): $VENV"
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
  echo "==> Creating venv: $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -U pip wheel setuptools

JETSON_TORCH_WHL="$REPO_ROOT/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
if [ -f "$JETSON_TORCH_WHL" ]; then
  echo "==> Installing Jetson torch from local wheel"
  pip install "$JETSON_TORCH_WHL"
else
  echo "WARN: Jetson torch wheel not found at $JETSON_TORCH_WHL"
  echo "      Copy wheel to repo root or install JetPack PyTorch manually."
fi

if [ -f "$REPO_ROOT/bootstrap/requirements-orin.txt" ]; then
  pip install -r "$REPO_ROOT/bootstrap/requirements-orin.txt" --no-deps || true
  pip install matplotlib scipy pyyaml requests psutil polars ultralytics-thop \
    opencv-python opencv-python-headless numpy pillow packaging || true
  pip install ultralytics || true
else
  pip install ultralytics opencv-python-headless numpy pillow
fi

if [ -d "$REPO_ROOT/third_party/vision" ]; then
  echo "==> Installing torchvision from third_party/vision (Jetson)"
  pip install -e "$REPO_ROOT/third_party/vision" || true
else
  pip install torchvision || true
fi

if [ -f "$JETSON_TORCH_WHL" ]; then
  pip install --force-reinstall --no-deps "$JETSON_TORCH_WHL" || true
fi

pip install onnx || true
pip install onnxsim || true

if [ -f /usr/lib/python3.8/dist-packages/tensorrt/tensorrt.so ]; then
  echo "==> Linking system TensorRT into venv"
  SITE_PACKAGES="$VENV/lib/python3.8/site-packages"
  for pkg in tensorrt cuda cublas cudnn; do
    if [ -d "/usr/lib/python3.8/dist-packages/$pkg" ]; then
      ln -sfn "/usr/lib/python3.8/dist-packages/$pkg" "$SITE_PACKAGES/$pkg"
    elif [ -f "/usr/lib/python3.8/dist-packages/${pkg}.py" ]; then
      ln -sfn "/usr/lib/python3.8/dist-packages/${pkg}.py" "$SITE_PACKAGES/${pkg}.py"
    fi
  done
fi

python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

python3 - <<'PY'
import json
from pathlib import Path
m = Path("models/weights_manifest.json")
if not m.exists():
    print("WARN: models/weights_manifest.json not found")
else:
    d = json.loads(m.read_text())
    missing = []
    for gname, g in d.get("groups", {}).items():
        for role, info in g.get("weights", {}).items():
            if not info.get("exists"):
                missing.append(f"{gname}/{role}: {info.get('path')}")
    if missing:
        print("MISSING weights (copy from Drive or train):")
        for x in missing:
            print(" ", x)
    else:
        print("All manifest weights present.")
PY

echo ""
echo "==> Bootstrap done."
echo "    Activate: source $VENV/bin/activate"
echo "    Read: AGENTS.md and docs/orin_new_device_setup.md"
echo "    RealSense D405: bash bootstrap/realsense_d405_orin.sh"
echo "    TensorRT export: COOLDOWN_SEC=60 bash tools/export_tensorrt/export_all_group02.sh"
