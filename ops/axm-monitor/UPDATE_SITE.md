# Обновление сайта rover.axm.tech

Каноническая инструкция. Агенту: skill `.cursor/skills/hub-deploy-geekom/SKILL.md`
и правило `.cursor/rules/hub-deploy-via-github-vps.mdc`.

## Участники

| Машина | Роль |
|--------|------|
| **Orin** | разработка, commit, **git push** |
| **GitHub** | `main` → https://github.com/AndreiPurits/strawberry_rover |
| **Orin или PC** | SSH на VPS (`geekom_vlad`), деплой hub |
| **VPS geekom** | `git pull` + `docker compose` hub + Caddy |

SSH (`~/.ssh/config`, ключ `~/.ssh/id_ed25519`):

```
Host geekom_vlad
    HostName 195.211.38.61
    Port 2222
    User geekom
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

---

## Шаг 1 — Orin: push в GitHub

```bash
cd ~/project
git add …   # только нужные файлы (без runs/, data/, весов, .local_*)
git commit -m "…"
~/project/scripts/git_push.sh
```

Токен: `~/.config/axm/github.env` → `GITHUB_TOKEN` (Contents: write).

Проверка: `git status` показывает `main` синхрон с `origin/main` (не ahead).

---

## Шаг 2 — деплой hub на geekom

### С Orin (рекомендуется)

```bash
bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
```

### С PC (Git Bash)

```bash
bash ops/axm-monitor/scripts/deploy_site_from_pc.sh
```

Путь на VPS по умолчанию: `/c/Users/redro/project/ops/axm-monitor`.  
Переопределение: `~/pc_deploy.env` (см. `scripts/pc_deploy.env.example`).

### Вручную

```bash
ssh geekom_vlad
cd /c/Users/redro/project/ops/axm-monitor   # или ~/project/ops/axm-monitor
bash scripts/deploy_hub_vps.sh
```

`deploy_hub_vps.sh` делает:

```bash
git pull origin main
docker compose build hub
docker compose up -d hub
curl https://rover.axm.tech/healthz
```

---

## Шаг 3 — проверка

- https://rover.axm.tech — Ctrl+F5 (кэш статики)
- На VPS: `git log -1` в `ops/axm-monitor` == ожидаемый commit
- При смене ассетов — bump `?v=` в `dashboard.html`

Fleet-agent на Orin перезапускать **только** если менялся `ops/axm-monitor/agent/`:

```bash
systemctl --user restart axm-rover-online
# или: scripts/restart_fleet_agent.sh
```

---

## Если Orin не достучится до GitHub / geekom

Симптом: ICMP до хоста есть, TCP (`443`/`2222`) → `No route to host`.

Тогда push/deploy **с PC** в той же сети/VPN, где открыт исходящий TCP:

```bash
git push origin main
bash ops/axm-monitor/scripts/deploy_site_from_pc.sh
```

На Orin после появления сети: `~/project/scripts/git_push.sh && bash ops/axm-monitor/scripts/deploy_site_from_orin.sh`.

---

## Шпаргалка

```text
[Orin] commit → ~/project/scripts/git_push.sh
[Orin] bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
[Браузер] https://rover.axm.tech  (Ctrl+F5)
```
