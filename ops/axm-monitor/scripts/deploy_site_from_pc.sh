#!/usr/bin/env bash
# Деплой hub на VPS (Windows geekom) — с PC (Git Bash) или с Orin.
#
# Usage:
#   bash ops/axm-monitor/scripts/deploy_site_from_pc.sh
#
# Опционально: ~/pc_deploy.env или PC_DEPLOY_ENV (см. pc_deploy.env.example)
#
set -euo pipefail

PC_ENV="${PC_DEPLOY_ENV:-$HOME/pc_deploy.env}"
if [ -f "$PC_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$PC_ENV"
  set +a
fi

SSH_HOST="${AXM_SSH_HOST:-geekom_vlad}"
VPS_HUB_DIR="${AXM_VPS_HUB_DIR:-/c/Users/redro/project/ops/axm-monitor}"
VPS_BASH="${AXM_VPS_BASH:-C:\Program Files\Git\bin\bash.exe}"

echo "[deploy] SSH ${SSH_HOST} → ${VPS_HUB_DIR}"

# shellcheck disable=SC2029
ssh -o BatchMode=yes "$SSH_HOST" "\"${VPS_BASH}\" -lc \"cd ${VPS_HUB_DIR} && bash scripts/deploy_hub_vps.sh\""

echo "[deploy] OK — https://rover.axm.tech"
