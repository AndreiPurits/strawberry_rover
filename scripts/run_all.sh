#!/bin/bash
set -Ee

REPO="/home/andrei/project/strawberry_rover_ws"
CAM_LOG="/tmp/camera.log"

cd "$REPO"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/andrei/.Xauthority}"

source /opt/ros/foxy/setup.bash
source install/setup.bash

cleanup() {
  if [[ -n "${CAMERA_PGID:-}" ]]; then
    echo "[run_all] stopping camera process group: $CAMERA_PGID"
    kill -- "-$CAMERA_PGID" >/dev/null 2>&1 || true
    sleep 0.5
    kill -9 -- "-$CAMERA_PGID" >/dev/null 2>&1 || true
  fi
}

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  echo
  echo "[run_all] ERROR: script failed (exit_code=$exit_code, line=$line_no)"
  echo "[run_all] Camera log: $CAM_LOG"
  if [[ -f "$CAM_LOG" ]]; then
    echo "[run_all] --- last 200 lines of camera log ---"
    tail -n 200 "$CAM_LOG" || true
    echo "[run_all] --- end of camera log ---"
  else
    echo "[run_all] (camera log file not found)"
  fi
  cleanup
  echo
  read -r -p "[run_all] Press Enter to close this window..." _ || true
  exit "$exit_code"
}

trap 'on_error $LINENO' ERR
trap cleanup EXIT INT TERM

echo "[run_all] starting camera, logging to $CAM_LOG"
mkdir -p "$(dirname "$CAM_LOG")"
rm -f "$CAM_LOG"
touch "$CAM_LOG"

preclean_orbbec_camera() {
  echo "[run_all] pre-clean: looking for stale Orbbec camera processes (namespace=/camera)"
  ps aux | egrep 'ros2 launch orbbec_camera gemini210\.launch\.py|orbbec_camera_node' | egrep -v egrep || true

  # Be precise: only kill the known launch invocation and the node in /camera namespace.
  pkill -f 'ros2 launch orbbec_camera gemini210\.launch\.py' >/dev/null 2>&1 || true
  pkill -f 'orbbec_camera_node.*__ns:=/camera' >/dev/null 2>&1 || true

  # Give the UVC device time to be released.
  sleep 2.0

  echo "[run_all] pre-clean done; remaining Orbbec processes:"
  ps aux | egrep 'ros2 launch orbbec_camera gemini210\.launch\.py|orbbec_camera_node' | egrep -v egrep || true
}

start_camera() {
  local depth_registration="$1"
  local align_mode="$2"
  local color_w="$3"
  local color_h="$4"
  local color_fps="$5"
  local color_fmt="$6"
  local depth_w="$7"
  local depth_h="$8"
  local depth_fps="$9"
  local depth_fmt="${10}"

  echo "[run_all] launching camera color=${color_w}x${color_h}@${color_fps} fmt=${color_fmt} depth=${depth_w}x${depth_h}@${depth_fps} fmt=${depth_fmt} depth_registration=${depth_registration} align_mode=${align_mode}"

  setsid bash -lc "
    cd \"$REPO\"
    source /opt/ros/foxy/setup.bash
    source install/setup.bash
    ros2 launch orbbec_camera gemini210.launch.py \
      camera_name:=camera \
      depth_registration:=${depth_registration} \
      align_mode:=${align_mode} \
      color_width:=${color_w} color_height:=${color_h} color_fps:=${color_fps} color_format:=${color_fmt} \
      depth_width:=${depth_w} depth_height:=${depth_h} depth_fps:=${depth_fps} depth_format:=${depth_fmt}
  " >>"$CAM_LOG" 2>&1 &
  CAM_PID=$!
  CAMERA_PGID=$CAM_PID

  echo "[run_all] camera started: launch_pid=$CAM_PID (process group id=$CAMERA_PGID)"
  echo "[run_all] active Orbbec processes after start:"
  ps aux | egrep 'ros2 launch orbbec_camera gemini210\.launch\.py|orbbec_camera_node' | egrep -v egrep || true
}

# Рабочий профиль:
# RGB   1280x720 @30 MJPG
# Depth 1280x800 @30 Y16
# Align SW для стабильного старта depth
preclean_orbbec_camera
start_camera true SW 1280 720 30 MJPG 1280 800 30 Y16

RGB_TOPIC="/camera/color/image_raw"
DEPTH_TOPIC="/camera/depth/image_raw"
INFO_TOPIC="/camera/color/camera_info"

wait_for_topic_message() {
  local topic="$1"
  local timeout_s="${2:-5}"

  python3 - "$topic" "$timeout_s" <<'PY'
import sys
import time

topic = sys.argv[1]
timeout_s = float(sys.argv[2])

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo

msg_t = CameraInfo if topic.endswith("/camera_info") else Image

def wait_once(reliability):
    rclpy.init(args=None)
    node = rclpy.create_node("strawberry_wait_for_msg")
    got = {"ok": False}

    def cb(_msg):
        got["ok"] = True

    qos = QoSProfile(depth=1)
    qos.history = HistoryPolicy.KEEP_LAST
    qos.reliability = reliability
    qos.durability = DurabilityPolicy.VOLATILE

    sub = node.create_subscription(msg_t, topic, cb, qos)

    deadline = time.time() + timeout_s
    while rclpy.ok() and time.time() < deadline and not got["ok"]:
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        node.destroy_subscription(sub)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    return bool(got["ok"])

ok = wait_once(ReliabilityPolicy.RELIABLE) or wait_once(ReliabilityPolicy.BEST_EFFORT)
raise SystemExit(0 if ok else 1)
PY
}

echo "[run_all] waiting for topics to publish at least 1 message (up to 30s):"
echo "  - $RGB_TOPIC"
echo "  - $DEPTH_TOPIC"
echo "  - $INFO_TOPIC"

deadline=$(( $(date +%s) + 30 ))
got_rgb=0
got_depth=0
got_info=0

while (( $(date +%s) < deadline )); do
  if (( got_rgb == 0 )); then
    echo "[run_all] waiting for first RGB..."
    if wait_for_topic_message "$RGB_TOPIC" 2; then
      got_rgb=1
      echo "[run_all] RGB OK"
    fi
  fi

  if (( got_rgb == 1 && got_depth == 0 )); then
    echo "[run_all] waiting for first Depth..."
    if wait_for_topic_message "$DEPTH_TOPIC" 2; then
      got_depth=1
      echo "[run_all] Depth OK"
    fi
  fi

  if (( got_rgb == 1 && got_depth == 1 && got_info == 0 )); then
    echo "[run_all] waiting for first CameraInfo..."
    if wait_for_topic_message "$INFO_TOPIC" 2; then
      got_info=1
      echo "[run_all] CameraInfo OK"
    fi
  fi

  if (( got_rgb == 1 && got_depth == 1 && got_info == 1 )); then
    echo "[run_all] topics are publishing (messages received)."
    break
  fi

  sleep 0.2
done

if (( got_rgb == 0 || got_depth == 0 || got_info == 0 )); then
  echo "[run_all] ERROR: camera topics did not publish messages within 30 seconds."
  echo "[run_all] Last lines from $CAM_LOG:"
  tail -n 200 "$CAM_LOG" || true
  exit 1
fi

echo "[run_all] starting GUI..."
python3 -u scripts/run_strawberry_ensemble.py \
  --rgb-topic "$RGB_TOPIC" \
  --depth-topic "$DEPTH_TOPIC" \
  --no-sync \
  --fps-cap 30