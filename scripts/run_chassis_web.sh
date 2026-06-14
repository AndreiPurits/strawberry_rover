#!/usr/bin/env bash
# Web UI + Arduino driver for chassis (Gecoma / Xbox gamepad via browser).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_orin_env.sh"
pip install -q pyserial uvicorn fastapi websockets 2>/dev/null || true
# ROS entrypoint uses local_ros python; expose venv wheels to pkg_resources.
if [ -d "$REPO_ROOT/.venv_cuda/lib/python3.8/site-packages" ]; then
  export PYTHONPATH="$REPO_ROOT/.venv_cuda/lib/python3.8/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi

MEGA_PORT="${MEGA_PORT:-/dev/ttyUSB0}"

echo "[chassis_web] Mega port: $MEGA_PORT"
echo "[chassis_web] Open dashboard, click Start -> Manual, use keyboard or Xbox gamepad."

PY="$REPO_ROOT/.venv_cuda/bin/python3"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi
"$PY" -m rover_arduino.rover_arduino_driver --ros-args -p serial_port:="$MEGA_PORT" &
ARDUINO_PID=$!

cleanup() {
  kill "$ARDUINO_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$PY" -m rover_web_interface.backend.api_server --host 0.0.0.0 --port 8080
