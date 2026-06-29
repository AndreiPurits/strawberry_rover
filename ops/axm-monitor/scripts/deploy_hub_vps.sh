#!/usr/bin/env bash
# Запускать НА VPS (после git pull) — пересборка hub rover.axm.tech
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "${ROOT}/hub/data"
if [ -f "${ROOT}/hub/data/roarm_points.json" ]; then
  cp -a "${ROOT}/hub/data/roarm_points.json" "${ROOT}/hub/data/roarm_points.json.bak"
fi
if [ -f "${ROOT}/hub/data/roarm_home.json" ]; then
  cp -a "${ROOT}/hub/data/roarm_home.json" "${ROOT}/hub/data/roarm_home.json.bak"
fi
echo "[deploy] dir=$ROOT"
git pull origin main
docker compose build hub
docker compose up -d hub
docker compose ps hub

if curl -fsS --max-time 10 https://rover.axm.tech/healthz >/dev/null 2>&1; then
  curl -fsS https://rover.axm.tech/healthz
  echo ""
else
  echo "[deploy] WARN: healthz check failed (Caddy/hub starting?)"
fi

echo "[deploy] Done. If needed: docker restart n8n-caddy"
