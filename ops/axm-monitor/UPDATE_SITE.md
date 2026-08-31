# Обновление сайта rover.axm.tech

Как у нас устроено:

1. **Orin (эта машина)** → commit + **push в GitHub** (код сайта/hub).
2. **Orin** → SSH по ключу на **geekom**.
3. **На geekom** → **git pull** (+ пересборка hub).

Агенту: `.cursor/skills/hub-deploy-geekom/SKILL.md`,  
`.cursor/rules/hub-deploy-via-github-vps.mdc`.

## SSH к geekom

`~/.ssh/config`:

```
Host geekom_vlad
    HostName 195.211.38.61
    Port 2222
    User geekom
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Каталог hub на geekom: `/c/Users/redro/project/ops/axm-monitor`.

---

## Шаг 1 — с Orin: push в GitHub

```bash
cd ~/project
git add …
git commit -m "…"
~/project/scripts/git_push.sh
```

Токен: `~/.config/axm/github.env` → `GITHUB_TOKEN`.

---

## Шаг 2 — с Orin: SSH на geekom → pull

```bash
bash ops/axm-monitor/scripts/deploy_site_from_orin.sh
```

Это не «деплой с Orin напрямую», а **подключение к geekom** и запуск там:

```bash
git pull origin main
docker compose build hub
docker compose up -d hub
```

Вручную:

```bash
ssh geekom_vlad
cd /c/Users/redro/project/ops/axm-monitor
bash scripts/deploy_hub_vps.sh
```

---

## Проверка

- https://rover.axm.tech — Ctrl+F5
- На geekom: `git log -1` совпадает с push с Orin

Если менялся `ops/axm-monitor/agent/` на Orin:

```bash
systemctl --user restart axm-rover-online
```

---

## Шпаргалка

```text
[Orin]   commit → ~/project/scripts/git_push.sh
[Orin]   ssh geekom_vlad → на VPS git pull (deploy_site_from_orin.sh)
[Браузер] https://rover.axm.tech  (Ctrl+F5)
```
