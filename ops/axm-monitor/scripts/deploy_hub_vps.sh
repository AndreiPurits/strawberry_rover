#!/usr/bin/env bash
# Запускать НА VPS (после git pull) — пересборка hub rover.axm.tech
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
