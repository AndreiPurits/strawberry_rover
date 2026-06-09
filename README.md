# strawberry_rover

Проект `strawberry_rover` — это CV‑пайплайн для клубники: детектирование ягод, классификация зрелости, сегментация, а также задел под stereo/distance/grasp‑поток.

В этом репозитории **хранится только код, конфиги и документация**. Датасеты, изображения и тяжёлые артефакты обучения — **вне git** (например, на Google Drive).

## Зафиксированные production‑модели (кратко)

Текущий набор production/рабочих моделей фиксируется в документе:
- `docs/model_selection.md`

Там же перечислены:
- production detector
- auto‑label detector
- production segmentation
- quality segmentation
- production classifier
- fast fallback classifier

## Архитектура пайплайна (упрощённо)

1) **Detector** → находит ягоды (bbox)  
2) **Classifier** → оценивает зрелость (ripeness)  
3) **Segmentation** → уточняет маску/контуры  
4) Далее: **stereo / distance / grasp pipeline** (следующий слой интеграции)

## Новый Jetson Orin

```bash
git clone git@github.com:AndreiPurits/strawberry_rover.git
cd strawberry_rover
git submodule update --init --recursive
bash bootstrap/orin_bootstrap.sh
```

Полный гайд: `docs/orin_new_device_setup.md` · Промпт для агента: `docs/NEW_ORIN_AGENT_PROMPT.md`

## Структура репозитория

- `AGENTS.md` — onboarding для Cursor-агента
- `bootstrap/` — установка Orin (Python, CUDA, TensorRT, RealSense)
- `pipelines/` — ensemble inference
- `models/model_groups/` — группы моделей + manifest
- `tools/` — бенчмарки FPS, TensorRT export, labeling
- `scripts/` — обучение, GUI, jetson fixes
- `docs/` — документация
- `reports/` — отчёты бенчмарков
- `src/` — ROS2 workspace (rover_*, Orbbec submodule, rplidar submodule)
- `arduino/` — скетчи Mega

**Не храним в git:**
- `diagnostics/`, `runs/`
- `data/` (датасеты)
- веса/бинарники (`*.pt`, `*.onnx`, `*.engine`, `*.whl`, `*.deb`)
- картинки/preview‑артефакты

## Быстрый запуск (типовые команды)

Примеры запуска смотри в `scripts/`:
- инференс детектора
- инференс классификатора
- сегментационный бенчмарк / инференс

Ключевой документ для понимания текущей сборки: `docs/model_selection.md`.

## Где лежат данные

Датасеты, изображения и тяжёлые артефакты обучения хранятся **вне git** (например, на Google Drive).