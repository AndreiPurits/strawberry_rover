#!/usr/bin/env bash
# RealSense D405 stereo (+ RGB relay for /stereo_camera/image_raw) + RPLidar.
# Used on Orin boot so hub stereo telemetry has live frames.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_orin_env.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/local_prefix_env.sh" 2>/dev/null || true
# shellcheck disable=SC1091
[ -f "$REPO_ROOT/install/setup.bash" ] && source "$REPO_ROOT/install/setup.bash"

LIDAR_PORT="${LIDAR_PORT:-/dev/ttyUSB0}"
USE_FAKE_LIDAR="${USE_FAKE_LIDAR:-false}"
LOG_DIR="${AXM_LOG_DIR:-$HOME/.local/log/axm}"
mkdir -p "$LOG_DIR"

echo "[perception] stereo_source=realsense lidar=${LIDAR_PORT} fake_lidar=${USE_FAKE_LIDAR}"

cleanup() {
  [[ -n "${RS_PID:-}" ]] && kill "$RS_PID" 2>/dev/null || true
  [[ -n "${RELAY_PID:-}" ]] && kill "$RELAY_PID" 2>/dev/null || true
  [[ -n "${LIDAR_PID:-}" ]] && kill "$LIDAR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 launch rover_perception stereo_realsense_depth.launch.py \
  >>"$LOG_DIR/realsense.log" 2>&1 &
RS_PID=$!

"$REPO_ROOT/.venv_cuda/bin/python3" "$REPO_ROOT/scripts/stereo_rgb_relay_boot.py" \
  >>"$LOG_DIR/stereo-relay.log" 2>&1 &
RELAY_PID=$!

if [[ "$USE_FAKE_LIDAR" != "true" && -e "$LIDAR_PORT" ]]; then
  ros2 launch rplidar_ros rplidar_c1_launch.py \
    serial_port:="$LIDAR_PORT" frame_id:=lidar_link \
    >>"$LOG_DIR/lidar.log" 2>&1 &
  LIDAR_PID=$!
elif [[ "$USE_FAKE_LIDAR" == "true" ]]; then
  ros2 run rover_fake_lidar fake_lidar_node >>"$LOG_DIR/lidar.log" 2>&1 &
  LIDAR_PID=$!
fi

wait "$RS_PID"
