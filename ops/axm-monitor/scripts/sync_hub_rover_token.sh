#!/usr/bin/env bash
# Sync Orin AXM_ROVER_TOKEN into hub .env on Geekom VPS and restart hub.
set -euo pipefail
ENV_FILE="${AXM_FLEET_ENV:-$HOME/.config/axm/fleet-agent.env}"
SSH_HOST="${AXM_SSH_HOST:-geekom_vlad}"
VPS_BASH="${AXM_VPS_BASH:-C:\Program Files\Git\bin\bash.exe}"
VPS_HUB_DIR="${AXM_VPS_HUB_DIR:-/c/Users/redro/project/ops/axm-monitor}"

# shellcheck disable=SC1090
source "$ENV_FILE"
: "${AXM_ROVER_TOKEN:?Set AXM_ROVER_TOKEN in $ENV_FILE}"
: "${AXM_ROVER_ID:=rover-01}"

TOK_JSON=$(python3 -c "import json,sys; print(json.dumps({sys.argv[1]: sys.argv[2]}))" "$AXM_ROVER_ID" "$AXM_ROVER_TOKEN")

ssh -o BatchMode=yes "$SSH_HOST" "\"${VPS_BASH}\" -lc \"
cd ${VPS_HUB_DIR}
python3 -c \\\"import json; from pathlib import Path; tok=${TOK_JSON}; p=Path('.env'); lines=p.read_text(encoding='utf-8').splitlines(); out=[]; found=False
for line in lines:
    if line.startswith('AXM_AGENT_TOKENS='):
        out.append('AXM_AGENT_TOKENS='+json.dumps(tok)); found=True
    else:
        out.append(line)
if not found:
    out.append('AXM_AGENT_TOKENS='+json.dumps(tok))
p.write_text('\\\\n'.join(out)+'\\\\n', encoding='utf-8')
print('synced', list(tok.keys()))\\\"
docker compose up -d hub
sleep 2
curl -fsS https://rover.axm.tech/healthz
echo
\""

echo "[sync] OK token for ${AXM_ROVER_ID}"
