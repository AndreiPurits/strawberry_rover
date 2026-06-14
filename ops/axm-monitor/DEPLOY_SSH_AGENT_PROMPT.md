# Промпт для агента с SSH на VPS (rover.axm.tech)

Скопируй текст ниже в новый чат Cursor, подключённый по SSH к серверу с белым IP.

---

## PROMPT (copy from here)

Ты деплоишь **защищённый fleet monitoring hub** для роверов на этом VPS.

### Цель
Поднять сайт **https://rover.axm.tech** — дашборд мониторинга роверов (логин + HTTPS). Роверы на Jetson Orin будут слать телеметрию **исходящими** HTTPS-запросами (порты Orin в интернет не открываем).

### Исходники
Репозиторий: **strawberry_rover** (GitHub). Деплой из каталога:
```
ops/axm-monitor/
```

Стек:
- **hub** — FastAPI (Python), порт 8090 внутри Docker
- **nginx** — reverse proxy + TLS
- **certbot** — Let's Encrypt

Файлы уже в репо:
- `ops/axm-monitor/docker-compose.yml`
- `ops/axm-monitor/hub/` — приложение + Dockerfile
- `ops/axm-monitor/nginx/rover.axm.tech.conf` — HTTPS
- `ops/axm-monitor/nginx/rover.axm.tech.bootstrap.conf` — HTTP до сертификата
- `ops/axm-monitor/nginx/active.conf` — монтируется в nginx
- `ops/axm-monitor/scripts/init_letsencrypt.sh` — первичный SSL
- `ops/axm-monitor/.env.example` — шаблон секретов

### Предусловия (проверь сам)
1. **DNS reg.ru**: A-запись `rover.axm.tech` → публичный IP **этого** VPS  
   ```bash
   dig +short rover.axm.tech
   ```
   IP должен совпадать с внешним IP сервера.

2. Порты **80** и **443** открыты (ufw/security group).

3. Установлен **Docker** + **docker compose**:
   ```bash
   docker --version && docker compose version
   ```
   Если нет: `curl -fsSL https://get.docker.com | sh`

### Шаги деплоя

```bash
# 1. Клонировать или обновить репо
cd ~
git clone <REPO_URL> project   # или git pull если уже есть
cd project/ops/axm-monitor

# 2. Секреты
cp .env.example .env
# Сгенерировать сильный пароль админа и токен для первого ровера:
python3 - <<'PY'
import json, secrets
print("AXM_ADMIN_PASSWORD=" + secrets.token_urlsafe(16))
tok = secrets.token_urlsafe(32)
print('AXM_AGENT_TOKENS=' + json.dumps({"rover-01": tok}))
print("# Сохрани токен rover-01 для Orin: AXM_ROVER_TOKEN=" + tok)
PY
# Вставь вывод в .env (nano/vim)

# 3. Bootstrap + Let's Encrypt
chmod +x scripts/init_letsencrypt.sh
AXM_LETSENCRYPT_EMAIL=admin@axm.tech bash scripts/init_letsencrypt.sh

# 4. Поднять всё
docker compose up -d
docker compose ps
```

### Проверка после деплоя

```bash
curl -sS https://rover.axm.tech/healthz
# ожидается: {"ok":true,"rovers":0}

curl -sS -o /dev/null -w "%{http_code}\n" https://rover.axm.tech/
# ожидается: 302 (редirect на /login без cookie)
```

В браузере: **https://rover.axm.tech** → страница входа → логин из `.env` → пустой дашборд «Роверы пока не подключены».

Тест heartbeat (подставь токен из `.env`):
```bash
curl -sS -X POST https://rover.axm.tech/api/agents/heartbeat \
  -H 'Content-Type: application/json' \
  -d '{"rover_id":"rover-01","token":"TOKEN_FROM_ENV","name":"Test Rover","telemetry":{"hostname":"vps-test"}}'
```

После логина на дашборде должна появиться карточка **Test Rover** ONLINE.

### Что НЕ делать
- Не коммить `.env` в git
- Не открывать наружу порты роверов (8080 и т.д.)
- Не использовать self-signed cert — только Let's Encrypt через скрипт

### После деплоя — сообщи оператору
1. URL: `https://rover.axm.tech`
2. `AXM_ADMIN_USER` и сгенерированный пароль
3. `AXM_ROVER_TOKEN` для `rover-01` (для Orin):
   ```bash
   export AXM_HUB_URL=https://rover.axm.tech
   export AXM_ROVER_ID=rover-01
   export AXM_ROVER_TOKEN=<token>
   ~/project/scripts/run_fleet_agent.sh
   ```
4. Логи при проблемах:
   ```bash
   docker compose logs -f hub nginx
   ```

### Troubleshooting
| Проблема | Решение |
|----------|---------|
| certbot fail | DNS не резолвится на этот IP — подождать TTL reg.ru |
| 502 nginx | `docker compose logs hub` — hub не стартовал |
| 403 heartbeat | неверный token в POST vs `AXM_AGENT_TOKENS` в `.env` |
| WebSocket не обновляет | проверить nginx location `/ws/` и `wss://` в браузере |

### Вариант B — Caddy уже занимает :80/:443 (n8n-caddy)

```bash
docker compose up -d hub   # hub на 127.0.0.1:8090
```

Caddyfile (`deploy/caddy-snippet.caddyfile`):

```
rover.axm.tech {
    reverse_proxy host.docker.internal:8090
}
```

TLS выдаёт Caddy. nginx/certbot не нужны (`--profile nginx-edge`).

Выполни деплой полностью, проверь healthz и тестовый heartbeat, отчитайся URL + credentials + rover token.

## PROMPT (copy until here)
