#!/usr/bin/env bash
# Опционально: cron на VPS — git pull + rebuild hub каждые N минут (без PC).
# Запускать ОДИН РАЗ на VPS: bash scripts/install_vps_autodeploy_cron.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${AXM_AUTODEPLOY_MIN:-15}"
MARKER="# axm-hub-autodeploy"
CRON_LINE="*/${INTERVAL} * * * * cd ${ROOT} && git fetch origin main && git reset --hard origin/main && docker compose up -d --build hub >> ${HOME}/axm-hub-autodeploy.log 2>&1 ${MARKER}"

if crontab -l 2>/dev/null | grep -q "$MARKER"; then
  echo "Cron already installed."
  crontab -l | grep "$MARKER"
  exit 0
fi

( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -
echo "Installed cron every ${INTERVAL} min. Log: ~/axm-hub-autodeploy.log"
