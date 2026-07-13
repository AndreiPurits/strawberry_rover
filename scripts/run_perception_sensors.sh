#!/usr/bin/env bash
# Stereo camera (+ optional front RGB) + RPLidar via ROS2.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_orin_env.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/axm_single_instance.sh"

LIDAR_PORT="${LIDAR_PORT:-/dev/ttyUSB1}"
CAMERA_DEVICE="${CAMERA_DEVICE:-0}"
STEREO_CAMERA_DEVICE="${STEREO_CAMERA_DEVICE:-4}"
ENABLE_RGB_CAMERA="${ENABLE_RGB_CAMERA:-false}"
ENABLE_STEREO_CAMERA="${ENABLE_STEREO_CAMERA:-true}"
USE_FAKE_LIDAR="${USE_FAKE_LIDAR:-false}"
MEGA_PORT="${MEGA_PORT:-/dev/ttyUSB0}"

if [ ! -e "$LIDAR_PORT" ]; then
  if detected="$(axm_detect_lidar_port "$MEGA_PORT" 2>/dev/null)"; then
    echo "[perception] lidar port $LIDAR_PORT missing — using $detected"
    LIDAR_PORT="$detected"
  else
    echo "[perception] WARN: no LiDAR serial port found except Mega $MEGA_PORT" >&2
    echo "[perception]       plug RPLidar USB or set LIDAR_PORT — camera only for now" >&2
    USE_FAKE_LIDAR="${USE_FAKE_LIDAR:-true}"
  fi
fi

if ! axm_acquire_lock perception-sensors; then
  if axm_perception_healthy "$CAMERA_DEVICE" "$LIDAR_PORT" "$USE_FAKE_LIDAR"; then
    axm_print_perception_status "$CAMERA_DEVICE" "$LIDAR_PORT"
    echo "[perception] already running — skip start"
    exit 0
  fi
  echo "[perception] lock busy but stack unhealthy — waiting 3s..."
  sleep 3
  if ! axm_acquire_lock perception-sensors; then
    echo "[perception] ERROR: another start in progress" >&2
    exit 1
  fi
fi

set +e
axm_prepare_perception_start "$CAMERA_DEVICE" "$LIDAR_PORT" "$USE_FAKE_LIDAR"
prep_rc=$?
set -e
if [ "$prep_rc" -eq 2 ]; then
  exit 0
fi

echo "[perception] front=${ENABLE_RGB_CAMERA} camera=/dev/video${CAMERA_DEVICE} stereo=${ENABLE_STEREO_CAMERA} stereo=/dev/video${STEREO_CAMERA_DEVICE} lidar=${LIDAR_PORT} fake_lidar=${USE_FAKE_LIDAR}"

exec ros2 launch rover_bringup perception_sensors.launch.py \
  lidar_serial_port:="$LIDAR_PORT" \
  camera_device_index:="$CAMERA_DEVICE" \
  stereo_camera_device_index:="$STEREO_CAMERA_DEVICE" \
  enable_rgb_camera:="$ENABLE_RGB_CAMERA" \
  enable_stereo_camera:="$ENABLE_STEREO_CAMERA" \
  use_fake_lidar:="$USE_FAKE_LIDAR"
