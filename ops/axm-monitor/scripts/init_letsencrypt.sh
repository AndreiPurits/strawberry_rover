#!/usr/bin/env bash
# First-time TLS for rover.axm.tech on VPS (after DNS A-record points to server IP).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOMAIN="${AXM_DOMAIN:-rover.axm.tech}"
EMAIL="${AXM_LETSENCRYPT_EMAIL:-admin@axm.tech}"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — set AXM_ADMIN_PASSWORD and AXM_AGENT_TOKENS, then re-run."
  exit 1
fi

cp nginx/rover.axm.tech.bootstrap.conf nginx/active.conf
docker compose up -d hub nginx

docker compose run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  --email "$EMAIL" \
  --agree-tos --no-eff-email \
  -d "$DOMAIN"

cp nginx/rover.axm.tech.conf nginx/active.conf
docker compose restart nginx
echo "HTTPS ready: https://$DOMAIN"
