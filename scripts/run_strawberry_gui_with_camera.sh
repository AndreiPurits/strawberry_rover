#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/andrei/project/strawberry_rover_ws"

# If launched from desktop, DISPLAY may be missing in some environments.
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/andrei/.Xauthority}"

cd "$REPO_ROOT"
source "/opt/ros/foxy/setup.bash"
source "$REPO_ROOT/install/setup.bash"

CAMERA_NAME="${CAMERA_NAME:-camera}"

# Camera desired profiles (override via env if needed)
COLOR_WIDTH="${COLOR_WIDTH:-1280}"
COLOR_HEIGHT="${COLOR_HEIGHT:-720}"
COLOR_FPS="${COLOR_FPS:-30}"
COLOR_FORMAT="${COLOR_FORMAT:-MJPG}"

DEPTH_WIDTH="${DEPTH_WIDTH:-640}"
DEPTH_HEIGHT="${DEPTH_HEIGHT:-400}"
DEPTH_FPS="${DEPTH_FPS:-30}"
DEPTH_FORMAT="${DEPTH_FORMAT:-ANY}"

ALIGN_MODE="${ALIGN_MODE:-HW}"
DEPTH_REGISTRATION="${DEPTH_REGISTRATION:-true}"

echo "[strawberry_gui] starting camera..."
ros2 launch orbbec_camera gemini210.launch.py \
  camera_name:="$CAMERA_NAME" \
  depth_registration:="$DEPTH_REGISTRATION" \
  align_mode:="$ALIGN_MODE" \
  color_width:="$COLOR_WIDTH" \
  color_height:="$COLOR_HEIGHT" \
  color_fps:="$COLOR_FPS" \
  color_format:="$COLOR_FORMAT" \
  depth_width:="$DEPTH_WIDTH" \
  depth_height:="$DEPTH_HEIGHT" \
  depth_fps:="$DEPTH_FPS" \
  depth_format:="$DEPTH_FORMAT" &
CAM_PID=$!

cleanup() {
  echo "[strawberry_gui] stopping camera (pid=$CAM_PID)..."
  kill "$CAM_PID" >/dev/null 2>&1 || true
  wait "$CAM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

RGB_TOPIC="/${CAMERA_NAME}/color/image_raw"
DEPTH_TOPIC="/${CAMERA_NAME}/depth/image_raw"

echo "[strawberry_gui] waiting for topics:"
echo "  - $RGB_TOPIC"
echo "  - $DEPTH_TOPIC"
for _ in $(seq 1 60); do
  if ros2 topic list 2>/dev/null | grep -qx "$RGB_TOPIC" && ros2 topic list 2>/dev/null | grep -qx "$DEPTH_TOPIC"; then
    echo "[strawberry_gui] topics are up."
    break
  fi
  sleep 0.5
done

echo "[strawberry_gui] starting GUI..."
python3 -u scripts/run_strawberry_ensemble.py \
  --rgb-topic "$RGB_TOPIC" \
  --depth-topic "$DEPTH_TOPIC"

