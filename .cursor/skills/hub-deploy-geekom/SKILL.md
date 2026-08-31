---
name: hub-deploy-geekom
description: >-
  Deploy rover.axm.tech hub/UI via GitHub main then SSH geekom_vlad
  (git pull + docker compose). Use when updating the site, dashboard,
  hub, axm-monitor static/API, deploying to VPS/geekom, or when the
  user mentions push site, deploy_site_from_orin, UPDATE_SITE, or
  rover.axm.tech release sync.
---

# Hub deploy: GitHub → geekom

Canonical human doc: `ops/axm-monitor/UPDATE_SITE.md`.  
Cursor rule (auto-deploy expectation): `.cursor/rules/hub-deploy-via-github-vps.mdc`.

## When this applies

Any change under `ops/axm-monitor/hub/`, hub-facing agent APIs that the
site depends on, or docs/scripts for site deploy — after the code is
ready, **do not ask** whether to deploy. Run the full cycle unless the
operator said «только локально / без push / без деплоя».

## Mandatory cycle

1. **Commit** on Orin/PC — only needed files (no `runs/`, `data/`, weights, `.local_*`, secrets).
2. **Push** to GitHub `main`:
   ```bash
   cd ~/project
   ~/project/scripts/git_push.sh
   ```
   Token: `~/.config/axm/github.env` (`GITHUB_TOKEN`).
3. **Deploy on geekom** (SSH pull + rebuild hub):
   ```bash
   bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
   ```
   Or from PC: `bash ops/axm-monitor/scripts/deploy_site_from_pc.sh`.

On VPS the script runs `scripts/deploy_hub_vps.sh`:
`git pull origin main` → `docker compose build hub` → `up -d hub` → healthz.

## SSH target

```
Host geekom_vlad
    HostName 195.211.38.61
    Port 2222
    User geekom
    IdentityFile ~/.ssh/id_ed25519
```

VPS hub dir default: `/c/Users/redro/project/ops/axm-monitor`.

## After deploy

- Tell operator: **Ctrl+F5** on https://rover.axm.tech
- Verify VPS `git log -1` matches the pushed commit
- Bump `?v=` in `dashboard.html` when static assets change
- Restart Orin fleet-agent **only** if `ops/axm-monitor/agent/` changed

## Network failure (Orin)

If ICMP works but TCP to GitHub/`195.211.38.61:2222` returns
`No route to host`, Orin cannot push or SSH. Commit locally, then
push+deploy from a PC with working outbound TCP, or retry when the
route/firewall is fixed. Do not claim the site is updated until
deploy healthz succeeds.

## Not this skill

- Orin-only fleet-agent / RoArm / ROS changes with **no** hub/UI impact → no VPS deploy required.
- Mega/chassis drive, grasp/cut — unrelated; follow safety rules.
