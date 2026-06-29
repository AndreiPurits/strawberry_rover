#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for pid in $(pgrep -f 'backend.api_server --host' || true); do
  cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$cmd" == *rover_web_interface.backend.api_server* ]] && kill "$pid" 2>/dev/null || true
done
sleep 2
for pid in $(pgrep -f 'backend.api_server --host' || true); do
  cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$cmd" == *rover_web_interface.backend.api_server* ]] && kill -9 "$pid" 2>/dev/null || true
done
sleep 1
source "$REPO_ROOT/scripts/local_ros_env.sh"
source "$REPO_ROOT/install/setup.bash"
exec setsid "$REPO_ROOT/.venv_cuda/bin/python3" -m rover_web_interface.backend.api_server \
  --host 0.0.0.0 --port 8080 >> /tmp/chassis_web.log 2>&1
