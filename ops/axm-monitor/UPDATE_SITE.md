# Обновление сайта rover.axm.tech

Технология после утверждения изменений в коде.

## Участники

| Машина | Роль |
|--------|------|
| **Orin** | разработка, commit, **git push** |
| **GitHub** | хранение `main` |
| **PC или Orin** | SSH на VPS (`geekom_vlad`), деплой hub |
| **VPS geekom** | `git pull` + Docker hub + Caddy |

SSH-конфиг на **PC и Orin** (`~/.ssh/config`, ключ `id_ed25519`):

```
Host geekom_vlad
    HostName 195.211.38.61
    Port 2222
    User geekom
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

---

## Шаг 1 — Orin: push в GitHub (1 команда)

После commit на Orin:

```bash
cd ~/project
git add …
git commit -m "…"
~/project/scripts/git_push.sh
```

Требуется `~/.config/axm/github.env` с `GITHUB_TOKEN` (fine-grained, Contents: write).

---

## Шаг 2 — деплой hub на VPS (одна команда)

### Вариант A — с Orin (рекомендуется)

```bash
bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
```

### Вариант B — с PC (Git Bash)

```bash
bash ops/axm-monitor/scripts/deploy_site_from_pc.sh
```

Пути VPS заданы по умолчанию (`/c/Users/redro/project/ops/axm-monitor`).  
Переопределение: `~/pc_deploy.env` (см. `scripts/pc_deploy.env.example`).

### Вариант B — вручную по SSH

```bash
ssh geekom_vlad
```

На VPS:

```bash
cd ~/project/ops/axm-monitor
bash scripts/deploy_hub_vps.sh
exit
```

### Что делает deploy_hub_vps.sh

```bash
git pull origin main
docker compose build hub
docker compose up -d hub
curl https://rover.axm.tech/healthz
```

При проблемах с HTTPS: `docker restart n8n-caddy`

---

## Шаг 3 — проверка

- https://rover.axm.tech — новый UI / healthz
- Orin: fleet-agent без перезапуска (если менялся только hub)

---

## Orin после обновления hub

Fleet-agent перезапускать **только** если менялся `ops/axm-monitor/agent/`:

```bash
pkill -f fleet_agent.py
nohup ~/project/scripts/run_fleet_agent.sh >> ~/.local/log/axm/fleet-agent.log 2>&1 &
```

---

## Краткая шпаргалка

```text
[Orin]  commit → ~/project/scripts/git_push.sh
[Orin]  bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
[Браузер] https://rover.axm.tech
```

---

## Автодеплой (опционально)

Чтобы **не заходить с PC каждый раз** — cron на VPS (см. `scripts/install_vps_autodeploy_cron.sh` после push в git).
