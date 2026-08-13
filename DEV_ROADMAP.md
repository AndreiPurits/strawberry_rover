# Дорожная карта Strawberry Rover (логичные stages пайплайна)

Этот репозиторий покрывает **два контура**:
- **CV**: датасеты → 3 замороженные модели → ensemble → stereo/depth distance → входы для grasp
- **ROS2 rover**: симуляция/bringup → сенсоры → навигация → мониторинг/web

Ниже — stages в порядке реальной зависимости (с минимумом текста).

---

## Stage 0 — Репозиторий/архитектура (foundation)

- **Цель**: структура, правила, стабильные интерфейсы.
- **Выход**: понятные каталоги, документация, контракт тем.
- **Статус**: выполнено (baseline есть).

---

## Stage 1 — Датасет (сбор + нормализация + отчёты)

- **Цель**: воспроизводимый датасет для детекции и классификации зрелости.
- **Документы**:
  - `docs/datasets_overview.md`
  - `docs/final_detection_dataset_summary.md`
- **Статус**: выполнено (датасет собран, статистики зафиксированы).

---

## Stage 2 — Модели (train/benchmark → freeze)

- **Цель**: зафиксировать рабочие чекпоинты под Jetson Orin.
- **Выход**: 3 базовые production модели (detector + classifier + segmenter) и отчёты.
- **Документ**: `docs/model_selection.md`
- **Статус**: выполнено (модели выбраны и “frozen”).

---

## Stage 3 — Ensemble pipeline (детект → классиф → сегм → distance)

- **Цель**: реалтайм выдача per-strawberry результатов + расстояние по depth внутри маски.
- **Документ**: `reports/ensemble_pipeline/benchmark.md`
- **Статус**: в работе / активный контур (особенно stereo/depth).

---

## Stage 4 — Стерео/Depth камера (интеграция IO + синхронизация + alignment)

- **Цель**: стабильный RGB+Depth поток и корректная метрика расстояния.
- **Требования**:
  - ROS2 sync RGB/Depth (approx-time, slop)
  - контракт единиц depth (mm/m)
  - поведение при несовпадении разрешений (не падать; либо align)
- **Статус**: в работе (текущая приоритетная стадия).

---

## Stage 5 — Манипулятор (RoArm-M3) MVP bridge + безопасные команды

- **Цель**: базовый ROS2 канал к роборуке для тестов pick-sequence.
- **Документы**:
  - `src/roarm_ros2_http/README.md`
  - `tools/roarm_local_gui/README.md`
- **Статус**: выполнено (MVP bridge + selftest + demo pick).

---

## Stage 6 — ROS2 rover stack (sim/sensors/navigation)

- **Цель**: воспроизводимая симуляция + сенсоры + управляющий контур `/cmd_vel`.
- **Контракт**: `TOPIC_CONTRACT.md`
- **Статус**: выполнено (симуляция/сенсоры), навигация — “in progress” (как отдельная подстройка).

---

## Stage 7 — Web monitoring/control (операторский слой)

- **Цель**: web dashboard поверх ROS2 (monitoring + manual control + routes lifecycle).
- **Статус**: выполнено для MVP (manual control, route record/edit), execution — запланировано.

---

## Stage 8 — Full autonomy loop (row follow + perception + grasp)

- **Цель**: связать ROS навигацию + CV perception + arm grasp в единый автономный цикл.
- **Статус**: запланировано (после стабилизации stereo/distance).
