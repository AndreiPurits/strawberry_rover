#!/usr/bin/env bash
# Keep rover ONLINE on rover.axm.tech: chassis web + fleet agent.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/axm_single_instance.sh"

ENV_FILE="${AXM_FLEET_ENV:-$HOME/.config/axm/fleet-agent.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${AXM_HUB_URL:?Set AXM_HUB_URL in $ENV_FILE}"
: "${AXM_ROVER_TOKEN:?Set AXM_ROVER_TOKEN in $ENV_FILE}"

source "$REPO_ROOT/scripts/activate_orin_env.sh" 2>/dev/null || true
pip install -q pyserial 2>/dev/null || true

MEGA_PORT="${MEGA_PORT:-/dev/ttyUSB0}"
LOG_DIR="${AXM_LOG_DIR:-$HOME/.local/log/axm}"
mkdir -p "$LOG_DIR"

run_chassis() {
  exec MEGA_PORT="$MEGA_PORT" "$REPO_ROOT/scripts/run_chassis_web.sh"
}

run_agent() {
  exec "$REPO_ROOT/scripts/run_fleet_agent.sh"
}

case "${1:-all}" in
  perception)
    exec LIDAR_PORT="${LIDAR_PORT:-/dev/ttyUSB1}" CAMERA_DEVICE="${CAMERA_DEVICE:-0}" \
      bash "$REPO_ROOT/scripts/run_perception_sensors.sh"
    ;;
  chassis)
    run_chassis
    ;;
  agent)
    run_agent
    ;;
  all|*)
    if ! axm_acquire_lock rover-online; then
      exit 0
    fi

    echo "[rover_online] logs: $LOG_DIR"
    echo "[rover_online] hub: $AXM_HUB_URL rover: ${AXM_ROVER_ID:-rover-01}"
    LIDAR_PORT="${LIDAR_PORT:-/dev/ttyUSB1}"
    CAMERA_DEVICE="${CAMERA_DEVICE:-0}"
    STEREO_CAMERA_DEVICE="${STEREO_CAMERA_DEVICE:-4}"
    MEGA_PORT="${MEGA_PORT:-/dev/ttyUSB0}"
    if [ ! -e "$LIDAR_PORT" ] && detected="$(axm_detect_lidar_port "$MEGA_PORT" 2>/dev/null || true)"; then
      LIDAR_PORT="$detected"
    fi
    echo "[rover_online] perception camera=${CAMERA_DEVICE} stereo=${STEREO_CAMERA_DEVICE} lidar=${LIDAR_PORT}"

    axm_stop_fleet_chassis

    LIDAR_PORT="$LIDAR_PORT" CAMERA_DEVICE="$CAMERA_DEVICE" STEREO_CAMERA_DEVICE="$STEREO_CAMERA_DEVICE" \
      USE_FAKE_LIDAR="${USE_FAKE_LIDAR:-}" \
      bash "$REPO_ROOT/scripts/run_perception_sensors.sh" >>"$LOG_DIR/perception.log" 2>&1 &
    PERC_PID=$!
    sleep 6
    if ! kill -0 "$PERC_PID" 2>/dev/null; then
      PERC_PID="$(pgrep -f 'perception_sensors.launch.py' | head -1 || true)"
    fi

    if groups | grep -q dialout; then
      MEGA_PORT="$MEGA_PORT" "$REPO_ROOT/scripts/run_chassis_web.sh" >>"$LOG_DIR/chassis.log" 2>&1 &
    else
      echo "[rover_online] WARN: not in dialout — run: newgrp dialout"
      MEGA_PORT="$MEGA_PORT" "$REPO_ROOT/scripts/run_chassis_web.sh" >>"$LOG_DIR/chassis.log" 2>&1 &
    fi
    CH_PID=$!
    sleep 8
    "$REPO_ROOT/scripts/run_fleet_agent.sh" >>"$LOG_DIR/fleet-agent.log" 2>&1 &
    AG_PID=$!
    echo "[rover_online] chassis pid=$CH_PID agent pid=$AG_PID perception pid=${PERC_PID:-skipped}"

    sleep 3
    if ! wget -qO- --timeout=3 http://127.0.0.1:8080/api/health >/dev/null 2>&1; then
      echo "[rover_online] ERROR: api_server not responding on :8080 — see $LOG_DIR/chassis.log" >&2
      tail -15 "$LOG_DIR/chassis.log" 2>/dev/null || true
      exit 1
    fi
    if ! kill -0 "$AG_PID" 2>/dev/null; then
      echo "[rover_online] ERROR: fleet agent exited immediately — see $LOG_DIR/fleet-agent.log" >&2
      tail -10 "$LOG_DIR/fleet-agent.log" 2>/dev/null || true
      exit 1
    fi
    if ! kill -0 "$CH_PID" 2>/dev/null; then
      echo "[rover_online] ERROR: chassis web exited immediately — see $LOG_DIR/chassis.log" >&2
      tail -10 "$LOG_DIR/chassis.log" 2>/dev/null || true
      exit 1
    fi

    cleanup() {
      kill "$AG_PID" 2>/dev/null || true
      kill "$CH_PID" 2>/dev/null || true
      if [ -n "${PERC_PID:-}" ]; then
        kill "$PERC_PID" 2>/dev/null || true
      fi
      axm_stop_perception
    }
    trap cleanup EXIT INT TERM
    wait $AG_PID
    ;;
esac
