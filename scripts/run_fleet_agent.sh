#!/usr/bin/env bash
# Run on Orin: push telemetry to AXM hub while chassis web is up.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$REPO_ROOT/ops/axm-monitor/agent"
cd "$AGENT_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/axm_single_instance.sh"
if ! axm_acquire_lock fleet-agent; then
  echo "[fleet-agent] already running (lock: $AXM_LOCK_DIR/fleet-agent.lock)" >&2
  exit 0
fi

ENV_FILE="${AXM_FLEET_ENV:-$HOME/.config/axm/fleet-agent.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

source "$REPO_ROOT/scripts/activate_orin_env.sh" 2>/dev/null || true
AXM_DEVICES_ENV="${AXM_DEVICES_ENV:-$HOME/.config/axm/devices.env}"
if [ -f "$AXM_DEVICES_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$AXM_DEVICES_ENV"
  set +a
fi
PYTHON_BIN="${AXM_PYTHON:-$REPO_ROOT/.venv_cuda/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi
"$PYTHON_BIN" -m pip install -q pyserial 2>/dev/null || true

: "${AXM_HUB_URL:?Set AXM_HUB_URL e.g. https://rover.axm.tech}"
: "${AXM_ROVER_ID:?Set AXM_ROVER_ID e.g. rover-01}"
: "${AXM_ROVER_TOKEN:?Set AXM_ROVER_TOKEN (same as in hub .env on VPS)}"

export AXM_LOCAL_WEB="${AXM_LOCAL_WEB:-http://127.0.0.1:8080}"
export MEGA_PORT="${MEGA_PORT:-/dev/ttyUSB1}"

exec "$PYTHON_BIN" "$AGENT_DIR/fleet_agent.py" "$@"
