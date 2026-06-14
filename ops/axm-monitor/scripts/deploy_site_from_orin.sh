#!/usr/bin/env bash
# Деплой hub на VPS одной командой с Orin (SSH-ключ в ~/.ssh/id_ed25519).
#
# Usage:
#   bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
#
# После push в GitHub:
#   ~/project/scripts/git_push.sh && bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AXM_SSH_HOST="${AXM_SSH_HOST:-geekom_vlad}"
export AXM_VPS_HUB_DIR="${AXM_VPS_HUB_DIR:-/c/Users/redro/project/ops/axm-monitor}"
export AXM_VPS_BASH="${AXM_VPS_BASH:-C:\Program Files\Git\bin\bash.exe}"

if [ ! -f "${HOME}/.ssh/id_ed25519" ]; then
  echo "ERROR: ~/.ssh/id_ed25519 not found (copy PC key to Orin)." >&2
  exit 2
fi

exec bash "${ROOT}/scripts/deploy_site_from_pc.sh"
