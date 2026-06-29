# RoArm pick: vision Auto + joint-space RL

План: **камера → ML (det/cls/seg) → подвод руки** без плохого T:104 IK, с обучением policy в **пространстве шарниров**.

## Почему T:104 (XYZ) плохо

По [Waveshare RoArm-M3-S](https://www.waveshare.com/wiki/RoArm-M3-S_Robotic_Arm_Control):

| Команда | Поведение |
|---------|-----------|
| **T:104** | IK в прошивке ESP32, **блокирует** до конца движения |
| **T:1041** | IK без интерполяции, «рывками» |
| **T:103** | Jog по **одной** оси XYZ (мм), блокирует |
| **T:101** | Один шарнир (rad), предсказуемо |
| **T:102** | Все шарниры сразу — задевает препятствия |
| **T:105** | Feedback: `x,y,z`, `b,s,e,t,r,g` (rad), нагрузки `tB…tG` |

Наблюдение на стенде: T:104 по одной оси даёт **cross-axis drift** (команда −Z, уходит Y) — это артефакт IK, не HTTP.

**Вывод:** для подвода к ягоде — **joint space (T:101)** или наш **staged T:101** (base → shoulder → elbow → …), не слепой T:104 из `berry_to_arm_mm()`.

---

## RoArm-M3: шарниры (эталон Waveshare)

| ID | Имя | Ключ T:105 | Диапазон (rad) | Начало |
|----|-----|------------|----------------|--------|
| 1 | BASE | `b` | 3.14 … −3.14 | 0 |
| 2 | SHOULDER | `s` | 1.57 … −1.57 | 0 |
| 3 | ELBOW | `e` | 3.14 … 0 | 1.57 |
| 4 | WRIST | `t` | 1.57 … −1.57 | 0 |
| 5 | ROLL | `r` | 3.14 … −3.14 | 0 |
| 6 | GRIP | `g` | 1.08 … 3.14 (open→close) | 3.14 |

Рабочая зона: диаметр до **~1120 mm**, повторяемость **±5 mm** @ 0.2 kg @ 0.5 m.

Полные DH-параметры Waveshare **не публикует** — IK закрыт в ESP32. Для симуляции/RL:

1. **Калибровка по T:105** — сетка joint poses → `(x,y,z)` → numeric FK или lookup.
2. **ROS2 / MoveIt** (опционально): `roarm-moveit-ikfast-plugins` — аналитический IK для планирования offline.
3. **Operational space без IK:** vision даёт ошибку в **пикселях + depth**, policy выдаёт **Δq** (6 шарниров).

---

## Архитектура Auto (блок управления ровера)

```
[Стерео RGB + depth] → strawberry_ensemble (group 02)
        ↓ bbox + mask + ripeness + distance_m
[Выбор цели] largest ripe bbox / центр маски
        ↓
[Perception → joint target]
   Phase A: heuristic / calibrated Δq from pixel error (MVP)
   Phase B: learned policy π(q | image, q, goal_px)
        ↓
[Motion] staged T:101 или sequence шагов
        ↓
[Verify] T:105 — berry в центре кадра, distance в допуске
        ↓
[Grip] T:106 + retreat → «Корзина левая»
```

### Уже есть в репо

| Компонент | Путь |
|-----------|------|
| ML pipeline | `pipelines/strawberry_ensemble.py` |
| Headless approach | `scripts/run_strawberry_roarm_approach.py` |
| Калибровка (грубая) | `config/roarm_approach.yaml` |
| Staged joints | `ops/axm-monitor/agent/roarm_staged_move.py` |
| Сохранённые точки | hub `roarm_points.json` (Дом, Корзина левая) |

### Следующий код (этапы)

1. **Auto UI** — вместо «Авто — скоро»: статус детекта, кнопка «Искать / Подойти», overlay на WebRTC.
2. **`rover_pick_auto` на Orin** — subprocess из `fleet_agent` при `drive_mode=auto`: ensemble + pick largest + joint heuristic.
3. **Связка с hub** — telemetry `pick: {phase, berry_px, distance_m, target_joints}`.

---

## Математическая модель (без закрытого IK)

### State (вектор наблюдения)

```
s = [
  q_1..q_6,              # rad из T:105 (b,s,e,t,r,g)
  x,y,z,                 # EoAT mm
  u, v,                  # центр bbox ягоды в кадре (норм. −1..1)
  log(distance_m),       # depth по маске
  mask_area_norm,        # площадь / площадь кадра
  ripeness_onehot(3),    # unripe / turning / ripe
  q_home_delta(6),       # q - q_home (безопасность)
]
```

### Action (единицы перемещения — **эталон для RL**)

**Не** сырые XYZ мм из T:104. Рекомендуемые единицы:

| Вариант | Action `a` | Плюсы |
|---------|------------|-------|
| **A (рекомендуется)** | `Δq_i` rad, шаг **±0.03…0.08** на шаг, joint order base→elbow | Совпадает с T:101, нет IK drift |
| B | Нормированный `Δq_i ∈ [-1,1]` × `q_max_i` | Удобно для PPO/SAC |
| C | T:103 jog: одна ось XYZ, `Δ ∈ {−5,0,+5}` mm | Медленнее, но interpretable |

**Порядок на шаге policy (как staged move):**

1. `Δq_base` — центрировать ягоду по `u` (горизонталь кадра)
2. `Δq_shoulder`, `Δq_elbow` — по `v` и `distance_m`
3. `Δq_wrist`, `Δq_roll`, `Δq_grip` — только на финальном pregrasp

### Reward (скаляр на шаг)

```
r = w1 * exp(-|u|)                    # ягода по центру X
  + w2 * exp(-|v|)                    # по центру Y
  + w3 * exp(-|d - d*|)               # d* ≈ 0.15 m pregrasp
  - w4 * Σ_i |Δq_i - Δq_i^cmd|       # выполнена ли команда (из T:105 after)
  - w5 * cross_axis_penalty            # если cmd только base, а |Δy| большой
  - w6 * collision_penalty             # |tE|,|tS| > порог из T:105
  - w7 * floor_penalty                 # z < z_min
  - w8 * timeout
  + w9 * human_ok                      # оператор: «хорошо» в UI
```

Человеческие метки: кнопки **👍 / 👎 / повтор** после каждого шага в Manual — offline RL / reward model (см. `docs/orin_migration_and_optimization_plan.md`).

### Goal (эталон траектории)

Не одна точка XYZ, а **последовательность манипуляторных фаз**:

1. `q_home` — точка «Дом»
2. `q_approach` — промежуточная (основание на корзину, локоть сложен) — **сохранить как именованную точку**
3. `q_pregrasp` — из policy / калибровки у ягоды
4. `q_basket` — «Корзина левая»

Метрика успеха эпизода: после T:106 ягода в grip + retreat без столкновения.

---

## Как обучать (практический план)

### Фаза 0 — Данные (2–3 дня на стенде)

Логировать каждый шаг в `runs/roarm_rl_logs/`:

```json
{
  "ts": "...",
  "command": {"op": "joint_move", "joint": 1, "rad": 0.12},
  "q_before": {"b":0,"s":0,"e":1.57,...},
  "q_after": {...},
  "xyz_before": {...},
  "vision": {"u":0.2,"v":-0.1,"distance_m":0.42,"conf":0.81},
  "human": "ok|bad|repeat|null"
}
```

Источник: доработка `roarm_proxy.execute_rpc` + опционально запись кадра.

### Фаза 1 — Behavioral cloning (BC)

- Демо: оператор ведёт руку слайдерами к 50–100 ягодам (разные позы).
- Обучить `π(a|s)` предсказывать `Δq` по `(u,v,distance,q)`.
- Baseline без нейросети: **правило** `Δq_base = -k_u * u`, `Δq_shoulder = -k_v * v`, `Δq_elbow = f(distance)`.

### Фаза 2 — Offline RL / reward model

- Датасет из логов + human labels.
- Алгоритмы: **IQL**, **CQL** или простой **reward model** (MLP) → фильтрация плохих шагов.
- Штраф cross-axis из реальных логов T:104 vs T:101.

### Фаза 3 — Online fine-tune (осторожно)

- Малые `Δq`, лимиты joint, estop на LiDAR / по току серво (`tE` из T:105).
- **Никогда** T:210 cmd:0 в автоматике.

### Фаза 4 — Sim (опционально)

- MoveIt / PyBullet с приближённой моделью RoArm.
- Sim2real gap большой без калибровки — приоритет **real logs + BC**.

---

## Калибровка camera → arm (до RL)

1. Charuco / chessboard: camera optical frame → base RoArm.
2. При `q_home` и ягоде на известном расстоянии подстроить `y_mm_per_px`, `z_mm_per_px` в `config/roarm_approach.yaml`.
3. Вместо T:104 из пикселей — **итеративный joint jog**:

```
while |u| > 0.05:  T:101 joint=1, Δq_base
while |v| > 0.05:  T:101 joint=2,3 малыми шагами
while |d-d*| > 0.02: T:101 shoulder/elbow
```

Это уже реализуемо поверх текущего staged move.

---

## Связь с двумя точками (Дом ↔ Корзина левая)

| Точка | Роль |
|-------|------|
| **Дом** | `q_home`, старт/финиш эпизода, HOME |
| **Корзина левая** | `q_place`, после grip |

Траектория pick: **Дом → (policy к ягоде) → grip → Корзина левая → Дом**.

Промежуточную «безопасную» позу (base на цель, локоть вверх) стоит **сохранить третьей точкой** — убирает задевание корзины при переходе.

---

## Приоритеты реализации

1. ✅ Staged T:101 (сделано)
2. 🔧 Auto subprocess: ensemble + joint heuristic (без RL)
3. 📝 Лог шагов + human labels
4. 🤖 BC policy `Δq` от `(u,v,d)`
5. 🎯 Grip + retreat через sequence + сохранённые точки
