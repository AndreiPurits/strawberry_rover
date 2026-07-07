#!/usr/bin/env bash
# Copy production PyTorch weights (group 02) from old Orin or backup host.
#
# Usage:
#   bash scripts/sync_group02_weights.sh user@OLD_ORIN_IP
#   bash scripts/sync_group02_weights.sh user@OLD_ORIN_IP:/home/andrei/project/strawberry_rover_ws
#
# Expected on source (see models/model_groups/02_lightened_current/group.json):
#   runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt
#   runs/classification_benchmark_v2/efficientnet_b0/best.pt
#   runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC="${1:-}"
if [ -z "$SRC" ]; then
  echo "Usage: $0 user@host[:/path/to/strawberry_rover_ws]" >&2
  echo "Example: $0 andrei@192.168.1.50:/home/andrei/project/strawberry_rover_ws" >&2
  exit 2
fi

REMOTE="${SRC%%:*}"
REMOTE_DIR="${SRC#*:}"
if [ "$REMOTE_DIR" = "$SRC" ]; then
  REMOTE_DIR="~/project"
fi

PATHS=(
  "runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt"
  "runs/classification_benchmark_v2/efficientnet_b0/best.pt"
  "runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt"
)

echo "=== Sync group02 weights from ${REMOTE}:${REMOTE_DIR} ==="
for rel in "${PATHS[@]}"; do
  dest="$REPO_ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  echo "→ $rel"
  rsync -av --progress "${REMOTE}:${REMOTE_DIR}/${rel}" "$dest"
done

echo ""
echo "Optional symlinks for TensorRT export:"
mkdir -p "$REPO_ROOT/models/model_groups/02_lightened_current"
ln -sf "../../runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt" \
  "$REPO_ROOT/models/model_groups/02_lightened_current/detector_best.pt"
ln -sf "../../runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt" \
  "$REPO_ROOT/models/model_groups/02_lightened_current/segmenter_best.pt"

echo ""
bash "$REPO_ROOT/tools/check_orin_setup.sh" | grep -E 'weight:|TRT|PyTorch' || true
echo "Done. Restart fleet-agent: bash scripts/restart_fleet_agent.sh"
