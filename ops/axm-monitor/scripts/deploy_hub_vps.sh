#!/usr/bin/env bash
# Rebuild hub on VPS (run where Docker + git repo live, SSH session).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git pull origin main
docker compose build hub
docker compose up -d hub
docker compose ps hub
curl -fsS https://rover.axm.tech/healthz
echo ""
echo "Hub updated. Reload Caddy if needed: docker restart n8n-caddy"
