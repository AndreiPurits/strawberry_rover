#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/jetson_gpu_env.sh"
source "$REPO_ROOT/scripts/local_ros_env.sh"
source "$REPO_ROOT/scripts/local_prefix_env.sh"
# shellcheck disable=SC1091
[ -f "$REPO_ROOT/install/setup.bash" ] && source "$REPO_ROOT/install/setup.bash"
# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.venv_cuda/bin/activate" ] && source "$REPO_ROOT/.venv_cuda/bin/activate"
export PYTHONPATH="$REPO_ROOT/.venv_cuda/lib/python3.8/site-packages:$REPO_ROOT:$REPO_ROOT/ops/axm-monitor/agent${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export AXM_GRIPPER_MASK_PROFILE="${AXM_GRIPPER_MASK_PROFILE:-left_only}"
exec python3 "$REPO_ROOT/scripts/roarm_one_shot_approach.py" "$@"
