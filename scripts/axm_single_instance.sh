#!/usr/bin/env bash
# Shared helpers: one perception stack (camera + lidar) and optional process locks.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/axm_single_instance.sh"

AXM_LOCK_DIR="${AXM_LOCK_DIR:-$HOME/.local/run/axm}"
mkdir -p "$AXM_LOCK_DIR"

axm_proc_count() {
  local pattern=$1
  pgrep -f "$pattern" 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}

axm_stop_fleet_chassis() {
  echo "[axm] stopping fleet agent + chassis web..."
  pkill -f 'fleet_agent.py' 2>/dev/null || true
  pkill -f 'rover_web_interface.backend.api_server' 2>/dev/null || true
  pkill -f 'rover_arduino.rover_arduino_driver' 2>/dev/null || true
  pkill -f 'run_chassis_web.sh' 2>/dev/null || true
  sleep 2
  pkill -9 -f 'fleet_agent.py' 2>/dev/null || true
  pkill -9 -f 'rover_web_interface.backend.api_server' 2>/dev/null || true
  pkill -9 -f 'rover_arduino.rover_arduino_driver' 2>/dev/null || true
  sleep 1
}

# Pick LiDAR serial port: first ttyUSB that is not the Mega port.
axm_detect_lidar_port() {
  local mega_port="${1:-/dev/ttyUSB0}"
  local d
  for d in /dev/ttyUSB*; do
    [ -e "$d" ] || continue
    if [ "$d" != "$mega_port" ]; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

axm_stop_perception() {
  echo "[perception] stopping camera/lidar stack..."
  pkill -f 'perception_sensors.launch.py' 2>/dev/null || true
  pkill -f 'rgb_camera_node' 2>/dev/null || true
  pkill -f 'stereo_camera_node' 2>/dev/null || true
  pkill -f 'rplidar_node' 2>/dev/null || true
  pkill -f 'fake_lidar_node' 2>/dev/null || true
  sleep 2
  pkill -9 -f 'rgb_camera_node' 2>/dev/null || true
  pkill -9 -f 'stereo_camera_node' 2>/dev/null || true
  pkill -9 -f 'rplidar_node' 2>/dev/null || true
  pkill -9 -f 'fake_lidar_node' 2>/dev/null || true
  sleep 1
}

axm_perception_pids() {
  PERCEPTION_LAUNCH_PIDS="$(pgrep -f 'perception_sensors.launch.py' 2>/dev/null || true)"
  PERCEPTION_CAMERA_PIDS="$(pgrep -f 'rgb_camera_node' 2>/dev/null || true)"
  PERCEPTION_LIDAR_PIDS="$(pgrep -f 'rplidar_node|fake_lidar_node' 2>/dev/null || true)"
}

axm_device_used_by() {
  local dev=$1
  local pid=$2
  fuser "$dev" 2>/dev/null | grep -qw "$pid"
}

# 0 = healthy single camera + lidar + launch supervisor
axm_perception_healthy() {
  local camera_device=${1:-0}
  local lidar_port=${2:-/dev/ttyUSB1}
  local use_fake_lidar=${3:-false}

  axm_perception_pids

  local launch_n cam_n lidar_n
  launch_n=$(printf '%s\n' "$PERCEPTION_LAUNCH_PIDS" | sed '/^$/d' | wc -l)
  cam_n=$(printf '%s\n' "$PERCEPTION_CAMERA_PIDS" | sed '/^$/d' | wc -l)
  lidar_n=$(printf '%s\n' "$PERCEPTION_LIDAR_PIDS" | sed '/^$/d' | wc -l)

  if [ "$launch_n" -ne 1 ] || [ "$cam_n" -ne 1 ] || [ "$lidar_n" -ne 1 ]; then
    return 1
  fi

  local cam_pid lidar_pid video_dev
  cam_pid=$(printf '%s\n' "$PERCEPTION_CAMERA_PIDS" | head -1)
  lidar_pid=$(printf '%s\n' "$PERCEPTION_LIDAR_PIDS" | head -1)
  video_dev="/dev/video${camera_device}"

  if [ ! -e "$video_dev" ]; then
    return 1
  fi
  if [ "$use_fake_lidar" != true ] && [ ! -e "$lidar_port" ]; then
    return 1
  fi

  axm_device_used_by "$video_dev" "$cam_pid" || return 1
  if [ "$use_fake_lidar" != true ]; then
    axm_device_used_by "$lidar_port" "$lidar_pid" || return 1
  fi

  return 0
}

axm_print_perception_status() {
  local camera_device=${1:-0}
  local lidar_port=${2:-/dev/ttyUSB1}

  axm_perception_pids
  echo "[perception] launch pids: ${PERCEPTION_LAUNCH_PIDS:-none}"
  echo "[perception] camera pids: ${PERCEPTION_CAMERA_PIDS:-none} (/dev/video${camera_device})"
  echo "[perception] lidar pids:  ${PERCEPTION_LIDAR_PIDS:-none} (${lidar_port})"
}

# Ensure exactly one stack. Exits 0 if already healthy; otherwise stops stale and returns 0.
axm_prepare_perception_start() {
  local camera_device=${1:-0}
  local lidar_port=${2:-/dev/ttyUSB1}
  local use_fake_lidar=${3:-false}

  if axm_perception_healthy "$camera_device" "$lidar_port" "$use_fake_lidar"; then
    axm_print_perception_status "$camera_device" "$lidar_port"
    echo "[perception] already running (single camera + lidar) — skip start"
    return 2
  fi

  local launch_n cam_n lidar_n
  launch_n=$(axm_proc_count 'perception_sensors.launch.py')
  cam_n=$(axm_proc_count 'rgb_camera_node')
  lidar_n=$(axm_proc_count 'rplidar_node|fake_lidar_node')

  if [ "$launch_n" -gt 0 ] || [ "$cam_n" -gt 0 ] || [ "$lidar_n" -gt 0 ]; then
    echo "[perception] stale or duplicate stack (launch=$launch_n camera=$cam_n lidar=$lidar_n) — restarting"
    axm_stop_perception
  fi

  return 0
}

# Exclusive lock for long-running supervisor scripts.
# Usage: axm_acquire_lock rover-online || exit 0
axm_acquire_lock() {
  local name=$1
  local lock_file="$AXM_LOCK_DIR/${name}.lock"
  exec {AXM_LOCK_FD}>"$lock_file"
  if ! flock -n "$AXM_LOCK_FD"; then
    echo "[${name}] already running (lock: $lock_file)"
    echo "[${name}] stop it first, e.g. pkill -f '${name}' or close the terminal"
    return 1
  fi
  return 0
}
