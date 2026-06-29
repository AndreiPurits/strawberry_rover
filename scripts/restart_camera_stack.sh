#!/usr/bin/env bash
# Restart perception + api + fleet-agent (safe for remote agent shells).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/axm_single_instance.sh"

axm_stop_fleet_chassis
axm_stop_perception
rm -f "$AXM_LOCK_DIR"/chassis-web.lock "$AXM_LOCK_DIR"/perception-sensors.lock "$AXM_LOCK_DIR"/fleet-agent.lock
sleep 2

bash "$REPO_ROOT/scripts/run_perception_sensors.sh" >> /tmp/perception_restart.log 2>&1 &
sleep 14

source "$REPO_ROOT/scripts/local_ros_env.sh"
source "$REPO_ROOT/install/setup.bash"
setsid bash "$REPO_ROOT/scripts/start_chassis_api.sh" < /dev/null &
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if "$REPO_ROOT/.venv_cuda/bin/python3" - <<'PY' 2>/dev/null; then
import json, urllib.request
print(json.loads(urllib.request.urlopen("http://127.0.0.1:8080/api/health", timeout=3).read()).get("bridge_active"))
PY
    break
  fi
  sleep 1
done
"$REPO_ROOT/.venv_cuda/bin/python3" - <<'PY' || true
import json, urllib.request
try:
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8080/api/health", timeout=6).read())
    print("[restart_stack] api health", h.get("bridge_active"))
except Exception as exc:
    print("[restart_stack] api health FAIL", exc)
PY

setsid bash "$REPO_ROOT/scripts/run_fleet_agent.sh" >> /tmp/fleet_agent.log 2>&1 < /dev/null &
sleep 4

echo "[restart_stack] done"
pgrep -af 'node:=stereo_camera_node' | head -1 || true
pgrep -af 'backend.api_server --host' | head -1 || true
pgrep -af 'ops/axm-monitor/agent/fleet_agent.py' | head -1 || true
