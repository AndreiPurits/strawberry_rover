#!/usr/bin/env bash
# Run on Orin: push telemetry to AXM hub while chassis web is up.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$REPO_ROOT/ops/axm-monitor/agent"
cd "$AGENT_DIR"

ENV_FILE="${AXM_FLEET_ENV:-$HOME/.config/axm/fleet-agent.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${AXM_HUB_URL:?Set AXM_HUB_URL e.g. https://rover.axm.tech}"
: "${AXM_ROVER_ID:?Set AXM_ROVER_ID e.g. rover-01}"
: "${AXM_ROVER_TOKEN:?Set AXM_ROVER_TOKEN (same as in hub .env on VPS)}"

export AXM_LOCAL_WEB="${AXM_LOCAL_WEB:-http://127.0.0.1:8080}"
export MEGA_PORT="${MEGA_PORT:-/dev/ttyUSB0}"

python3 "$AGENT_DIR/fleet_agent.py" "$@"
