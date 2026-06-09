#!/usr/bin/env bash
# Export group-02 detector + segmenter TensorRT engines for all benchmark imgsz presets.
# Run on target Jetson. Engines are gitignored (runs/export_tensorrt/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/jetson_gpu_env.sh"
source "$REPO_ROOT/.venv_cuda/bin/activate"
export PATH="/usr/src/tensorrt/bin:${PATH:-}"

DET="$REPO_ROOT/models/model_groups/02_lightened_current/detector_best.pt"
SEG="$REPO_ROOT/models/model_groups/02_lightened_current/segmenter_best.pt"
OUT="$REPO_ROOT/runs/export_tensorrt/group02"
mkdir -p "$OUT"

COOLDOWN_SEC="${COOLDOWN_SEC:-45}"

_temp_c() {
  local t
  t="$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1 || echo 0)"
  echo $((t / 1000))
}

export_one() {
  local weights="$1"
  local task="$2"
  local imgsz="$3"
  local name="$4"
  local engine="$OUT/${name}_imgsz${imgsz}.engine"
  if [ -f "$engine" ]; then
    echo "SKIP exists: $engine"
    return 0
  fi
  echo "==> Export $name imgsz=$imgsz (temp ~$(_temp_c)C)"
  python3 "$REPO_ROOT/tools/export_tensorrt/export_yolo.py" \
    --weights "$weights" \
    --task "$task" \
    --imgsz "$imgsz" \
    --half \
    --outdir "$OUT/${name}_imgsz${imgsz}" \
    --engine-out "$engine" \
    --device 0
  if [ ! -f "$engine" ]; then
    echo "ERROR: engine missing after export: $engine"
    exit 1
  fi
  echo "Wrote $engine (temp ~$(_temp_c)C)"
  if [ "$COOLDOWN_SEC" -gt 0 ]; then
    echo "Cooling ${COOLDOWN_SEC}s..."
    sleep "$COOLDOWN_SEC"
  fi
}

# Smallest first (faster builds, lower thermals during bring-up).
for sz in 416 480 512 640; do
  export_one "$DET" detect "$sz" detector
done

for sz in 256 320 384; do
  export_one "$SEG" segment "$sz" segmenter
done

echo "==> Export complete. Engines in $OUT"
ls -la "$OUT"/*.engine 2>/dev/null || ls -la "$OUT"
