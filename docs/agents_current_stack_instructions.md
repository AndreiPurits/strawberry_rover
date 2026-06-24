# Инструкция агентам: текущий рабочий стек

Этот документ — практический runbook для агентов, работающих с текущим стеком rover + hub.
Цель: быстро вносить изменения без потери стабильности и без рассинхрона между UI, агентом и железом.

## 1) Что считается текущим стеком

- `ops/axm-monitor/hub` — веб-хаб (FastAPI + dashboard).
- `ops/axm-monitor/agent/fleet_agent.py` — агент на Orin (телеметрия, LiDAR guard, команды движения).
- `src/rover_web_interface` — локальный web bridge на Orin (`:8080`), через который агент шлет `cmd_vel`.
- `src/rover_arduino/rover_arduino_driver.py` + Mega sketch — выход на моторы.
- `scripts/run_rover_online.sh` — основной запуск стека на Orin (perception + chassis + fleet-agent).

## 2) Главные правила безопасности движения

- LiDAR guard блокирует **только движение вперёд**.
- Зона guard — **конус ~35° строго перед ровером** (геометрия: ширина 25 см при дистанции 40 см → `atan(0.125/0.40)` ≈ **17.4°** от оси, **34.7°** полный угол). Точки **вне конуса полностью игнорируются** для STOP.
- STOP срабатывает только если **внутри конуса** есть точка ближе `0.40 m` (release — `0.48 m`, latch/hysteresis).
- Назад — разрешено. **Override** (Shift / E / Space / R2) — снимает UI-блок и разрешает проезд вперёд (удерживать вместе с W).
- Любой STOP → нейтраль моторов (`1500/1500/1500/1500`) на Mega.
- Изменения движения: сначала UI + логи, потом реальное движение.
- Никогда не ослаблять safety без явного запроса владельца.

## 3) Где что менять

- Визуал dashboard, поведение кнопок, блоки LiDAR/STOP:
  - `ops/axm-monitor/hub/static/dashboard.js`
  - `ops/axm-monitor/hub/static/style.css`
  - `ops/axm-monitor/hub/static/dashboard.html`
- Логика guard/команд (технический STOP, latch/hysteresis, разрешение/запрет forward/back):
  - `ops/axm-monitor/agent/fleet_agent.py`
- Преобразование `cmd_vel` в PWM и связь с Mega:
  - `src/rover_arduino/rover_arduino/rover_arduino_driver.py`
  - `arduino/MEGA_Rover_Gecoma/MEGA_Rover_Gecoma.ino`

## 4) Базовый запуск и логи (Orin)

- Запуск полного онлайн-стека:
  - `bash scripts/run_rover_online.sh`
- Ключевые логи:
  - `~/.local/log/axm/fleet-agent.log`
  - `~/.local/log/axm/chassis.log`
  - `~/.local/log/axm/perception.log`
- Проверка локального bridge:
  - `http://127.0.0.1:8080/api/health`

## 5) Обязательный регламент внесения изменений

Ниже обязательный порядок для любого изменения (особенно в движении/безопасности):

1. **Определи слой изменения**
   - UI only, agent logic, driver/firmware, или несколько слоев.
2. **Измени минимально необходимое**
   - Не смешивать в одном шаге “визуал” и “полный рефактор безопасности”.
3. **Локальная проверка перед коммитом**
   - Python: `py_compile` измененных `.py`.
   - Проверить логи на `Traceback`, `error`, `LIDAR_GUARD_STOP`.
   - Проверить UI: нет флапа STOP, нет layout-сдвигов, кнопки не отправляют unsafe-команды.
4. **Обнови версию статики при изменении frontend**
   - В `dashboard.html` поднять query-параметр `style.css?v=...` и `dashboard.js?v=...`.
   - Иначе можно получить “старый UI из кеша”.
5. **Коммит и push**
   - Сделать атомарный коммит с понятным сообщением.
   - Push через стандартный процесс репозитория.
6. **Деплой**
   - Для hub/UI: деплой `ops/axm-monitor` на VPS.
   - Для agent-изменений: обновить код на Orin и перезапустить fleet agent.
7. **Пост-деплой проверка**
   - `https://rover.axm.tech/healthz`
   - Проверка dashboard в браузере (forced refresh/новая версия статики).
   - Проверка реального поведения STOP/движения по логам и телеметрии.

## 6) Деплой на прод (агент делает сам)

Источник истины: `main` на GitHub. Прод-сайт: VPS `geekom_vlad` → https://rover.axm.tech

### SSH (Orin и PC)

Ключ: `~/.ssh/id_ed25519`. Конфиг `~/.ssh/config`:

```
Host geekom_vlad
    HostName 195.211.38.61
    Port 2222
    User geekom
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Проверка: `ssh -o BatchMode=yes geekom_vlad echo ok`

### Цепочка после изменений (Orin)

```bash
cd ~/project

# 1) синтаксис
python3 -m py_compile ops/axm-monitor/agent/fleet_agent.py ops/axm-monitor/hub/app.py

# 2) commit (только нужные файлы)
git add ops/axm-monitor/ docs/agents_current_stack_instructions.md
git commit -m "fix: lidar guard …"

# 3) push в GitHub (нужен ~/.config/axm/github.env с GITHUB_TOKEN)
~/project/scripts/git_push.sh

# 4) деплой hub на VPS одной командой (SSH → git pull → docker compose)
bash ops/axm-monitor/scripts/deploy_site_from_orin.sh

# 5) если менялся agent/ — перезапуск fleet-agent на Orin
pkill -f fleet_agent.py
nohup ~/project/scripts/run_fleet_agent.sh >> ~/.local/log/axm/fleet-agent.log 2>&1 &
```

Скрипт `deploy_site_from_orin.sh` вызывает `deploy_site_from_pc.sh`, который по SSH запускает на VPS:

```bash
cd /c/Users/redro/project/ops/axm-monitor && bash scripts/deploy_hub_vps.sh
```

(`deploy_hub_vps.sh`: `git pull origin main` → `docker compose build hub` → `docker compose up -d hub` → `curl healthz`)

### Пост-деплой

- `curl -s https://rover.axm.tech/healthz`
- Dashboard: Ctrl+F5 (версия статики в `dashboard.html`: `?v=…`)
- Лог: `tail -f ~/.local/log/axm/fleet-agent.log` — стабильный `guard=True` при препятствии в конусе

Подробнее: `ops/axm-monitor/UPDATE_SITE.md`

## 7) Изменения через GitHub + SSH (кратко)

1. Изменения на Orin → commit → `git_push.sh`
2. `deploy_site_from_orin.sh` (hub/UI)
3. Перезапуск fleet-agent, если менялся `ops/axm-monitor/agent/`

## 8) Что обязательно указывать в отчете после изменений

- Какие файлы изменены.
- Что изменилось по факту в поведении.
- Как проверено (команды/логи/UI).
- Что еще не проверено и какие риски остались.

## 9) Частые ошибки, которых нельзя допускать

- Изменили frontend, но не подняли версию статики в `dashboard.html`.
- Смотрят только UI, не проверяют фактические команды в логах.
- Вносят изменения в движение без явной проверки STOP на реальном пороге.
- Деплойнули hub, но забыли перезапустить fleet-agent после изменений в `ops/axm-monitor/agent`.
- Учитывают боковые точки LiDAR для STOP (должен быть только конус ~40° впереди).

## 10) Мини-чеклист перед завершением задачи

- [ ] Изменения внесены только в нужный слой.
- [ ] Safety-поведение не деградировало.
- [ ] Линтер/синтаксис чистые.
- [ ] Статика версионирована (если менялся frontend).
- [ ] Push и деплой выполнены по цепочке GitHub + SSH.
- [ ] В отчете есть проверка, риски и следующий шаг.

