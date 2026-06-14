#!/usr/bin/env bash
# Front camera (/dev/video0) + RPLidar via ROS2.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_orin_env.sh"

LIDAR_PORT="${LIDAR_PORT:-/dev/ttyUSB1}"
CAMERA_DEVICE="${CAMERA_DEVICE:-0}"
USE_FAKE_LIDAR="${USE_FAKE_LIDAR:-false}"

echo "[perception] camera=/dev/video${CAMERA_DEVICE} lidar=${LIDAR_PORT} fake_lidar=${USE_FAKE_LIDAR}"

exec ros2 launch rover_bringup perception_sensors.launch.py \
  lidar_serial_port:="$LIDAR_PORT" \
  camera_device_index:="$CAMERA_DEVICE" \
  use_fake_lidar:="$USE_FAKE_LIDAR"
