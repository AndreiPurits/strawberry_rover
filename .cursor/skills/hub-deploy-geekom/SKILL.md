---
name: hub-deploy-geekom
description: >-
  Deploy rover.axm.tech from current Orin: push site code to GitHub,
  then SSH with key to geekom_vlad and run git pull (hub rebuild) there.
  Use when updating the site/dashboard/hub, deploy to geekom/VPS,
  deploy_site_from_orin, UPDATE_SITE, or rover.axm.tech sync.
---

# Hub: Orin push → geekom pull

Каноника: `ops/axm-monitor/UPDATE_SITE.md`.  
Правило автодеплоя: `.cursor/rules/hub-deploy-via-github-vps.mdc`.

## Модель (как у нас)

1. **Текущая машина (Orin)** — commit и **push в GitHub** кода сайта/hub.
2. **С этой же Orin** — SSH по ключу на **geekom** (`geekom_vlad`).
3. **Уже на geekom** — команда **git pull** (+ `docker compose` hub).

Orin сам сайт не раздаёт. Geekom тянет `main` из GitHub.

## Цикл (обязательный после правок hub/UI)

```bash
cd ~/project
# 1) push с Orin в GitHub
~/project/scripts/git_push.sh

# 2) SSH на машину с ключом → там pull
bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
```

`deploy_site_from_orin.sh` только открывает SSH на `geekom_vlad` и на VPS запускает:

```bash
cd /c/Users/redro/project/ops/axm-monitor
bash scripts/deploy_hub_vps.sh
# = git pull origin main && docker compose build hub && up -d hub && healthz
```

Токен push: `~/.config/axm/github.env` (`GITHUB_TOKEN`).  
Ключ SSH: `~/.ssh/id_ed25519` → Host `geekom_vlad` (195.211.38.61:2222, user `geekom`).

## После успеха

- Оператору: Ctrl+F5 на https://rover.axm.tech
- На geekom: `git log -1` == commit с Orin
- При смене статики — bump `?v=` в `dashboard.html`
- `systemctl --user restart axm-rover-online` только если менялся `ops/axm-monitor/agent/`

## Если не вышло

- `git_push.sh` не достучался до GitHub → push с Orin не прошёл (сайт на VPS не обновить).
- `ssh geekom_vlad` не проходит → ключ/маршрут до geekom; pull на VPS не запустить.

Не объявлять сайт актуальным, пока оба шага не OK.

## Не этот skill

Orin-only (RoArm/ROS) без hub/UI — push/deploy сайта не нужны.
