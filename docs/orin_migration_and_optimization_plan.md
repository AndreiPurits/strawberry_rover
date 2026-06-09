# Orin 8GB: миграция на GitHub + пиковый FPS (50W) + onboarding агента

Дата: 2026-04-24  
Контекст: собранный ровер (Orin 8GB @ 50W, стереокамера, LiDAR, навигация, Arduino, RoArm-M3).

Цель в двух словах:
1. **Один GitHub-репозиторий**, в котором агент в Cursor сразу понимает проект и может работать на Orin.
2. **Пиковый FPS** vision pipeline на Orin при power cap 50W — узнать «пик силы» до/после TensorRT.
3. **Три группы моделей** сравниваются честно на `data/ФПС ДАТАСЕТ/`.

---

## Что уже есть (не пересоздавать)

| Компонент | Где |
|-----------|-----|
| 3 группы моделей | `models/model_groups/01_fast_initial`, `02_lightened_current`, `03_finetune_future` |
| Production frozen weights | `docs/model_selection.md` |
| Ensemble pipeline | `pipelines/strawberry_ensemble.py`, `scripts/run_strawberry_ensemble.py` |
| FPS benchmark | `tools/benchmark_holdout_full_pipeline_fps.py` |
| FPS датасет (auto-label) | `data/ФПС ДАТАСЕТ/` (~8090+ позитивных кадров) |
| GUI стерео (OpenCV) | `scripts/run_strawberry_ensemble.py` + `scripts/run_strawberry_gui_with_camera.sh` |
| GUI роборуки | `tools/roarm_local_gui/` |
| Web MVP | `src/rover_web_interface/` |
| ROS2 stack | `src/rover_*`, `src/roarm_ros2_http/`, `src/rplidar_ros` (submodule) |
| Jetson fixes | `scripts/jetson_gpu_env.sh`, `scripts/yolo_jetson_compat.py` |

---

## Фаза A — Подготовка GitHub-репозитория (переезд)

### A1. Структура репозитория `strawberry_rover` (monorepo)

```
strawberry_rover/
├── README.md                 # entry point для агента
├── ORCHESTRATOR.md           # правила работы агента
├── CURRENT_STATE.md
├── DEV_ROADMAP.md
├── AGENTS.md                 # NEW: быстрый onboarding для Cursor
├── docs/
├── config/
├── models/model_groups/      # symlinks + group.json (без .pt в git)
├── pipelines/
├── scripts/
├── tools/
│   ├── benchmark_holdout_full_pipeline_fps.py
│   ├── label_fps_dataset_from_reserve.py
│   └── roarm_local_gui/
├── arduino/
├── install/                  # NEW: bootstrap Orin
│   ├── orin_bootstrap.sh
│   ├── requirements-orin.txt
│   └── ros_deps.sh
└── src/                      # ROS2 colcon workspace
    ├── rover_*/
    ├── roarm_ros2_http/
    └── .gitmodules           # rplidar_ros, Orbbec (submodule)
```

### A2. Что в git / что вне git

**В git:** код, конфиги, `group.json`, документация, launch-файлы, скрипты установки.  
**Вне git (Google Drive / локально на Orin):**
- `data/` (датасеты, `ФПС ДАТАСЕТ`)
- `runs/` (все `best.pt`, train artifacts)
- `*.engine`, `*.onnx`, `*.pt`, `build/`, `install/`, `log/`

**Weights manifest (NEW):** `models/weights_manifest.json` — пути и SHA256, без бинарников в git.

### A3. Onboarding агента (`AGENTS.md`)

Агент при старте читает (в порядке):
1. `README.md`
2. `AGENTS.md`
3. `ORCHESTRATOR.md`
4. `CURRENT_STATE.md`
5. `docs/model_selection.md`
6. `docs/orin_migration_and_optimization_plan.md` (этот файл)

В `AGENTS.md` зафиксировать:
- Jetson: `source scripts/jetson_gpu_env.sh`
- CUDA check: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Запуск GUI камеры: `scripts/run_strawberry_gui_with_camera.sh`
- Запуск FPS bench: `tools/benchmark_holdout_full_pipeline_fps.py --preset all`
- Где лежат веса на Orin (manifest)
- Power mode 50W: `sudo nvpmodel -m 0` + `sudo jetson_clocks` (уточнить на железе)

### A4. Bootstrap на Orin (`install/orin_bootstrap.sh`)

Один скрипт:
- ROS2 Foxy/Humble (что стоит на Orin)
- Python venv + torch (Jetson wheel)
- ultralytics, opencv, PyQt5 (roarm gui)
- colcon build
- проверка камеры / lidar / arm HTTP
- скачивание weights по manifest (rsync / gdrive)

### A5. Third-party → submodules

| Сейчас vendored | Действие |
|-----------------|----------|
| `src/rplidar_ros` | уже submodule |
| `src/OrbbecSDK_ROS2` | submodule или apt |
| `src/image_pipeline`, `image_common` | submodule ros-perception |
| `src/backward_ros` | submodule или rosdep |

---

## Фаза B — Оптимизация 3 групп моделей (пик FPS @ 50W)

Power cap **50W** — все замеры только в этом режиме.  
Benchmark holdout: **`data/ФПС ДАТАСЕТ/images/`** (только кадры с ≥1 детекцией).

### B0. Baseline замеры (уже частично сделано)

| Preset | FPS (new_photos) | Примечание |
|--------|------------------|------------|
| baseline | ~13.9 | det640/seg384, seg_every=2, cls_every=8 |
| fast | ~15.3 | det512/seg320 |
| very_fast | ~18.7 | det480, seg_every=3 |

**Следующий шаг:** повторить все 3 **model groups** × presets на **ФПС ДАТАСЕТ** на Orin @ 50W.

### B1. Уровни оптимизации (без переобучения)

| Уровень | Что делаем | Ожидаемый выигрыш |
|---------|------------|-------------------|
| L1 Runtime | imgsz, conf, max_det, seg_every, cls_every | уже ~+35% (baseline→very_fast) |
| L2 Pipeline | batch crops classifier, async det/seg, zero-copy ROS | +10–20% |
| L3 Export | ONNX → TensorRT FP16/INT8 (det + seg) | +2–4× на YOLO |
| L4 Classifier | TorchScript / TensorRT EfficientNet | +1.5–2× |
| L5 System | jetson_clocks, pinned memory, CUDA streams, power lock | стабильность +5–10% |

### B2. TensorRT pipeline (NEW scripts)

```
tools/export_tensorrt/
├── export_yolo_det.py      # best.pt → ONNX → .engine
├── export_yolo_seg.py
├── export_classifier.py    # optional phase 2
├── benchmark_engines.py    # compare pt vs engine FPS
└── README.md
```

Порядок export:
1. Group 02 detector (YOLOv8s) @ imgsz 512 FP16
2. Group 02 segmenter (YOLOv8n-seg) @ imgsz 320 FP16
3. Group 01 (fast initial) — те же скрипты, другие weights
4. Classifier — после det/seg стабилизированы

### B3. Сравнение 3 групп (контракт)

Для каждой группы из `models/model_groups/*/group.json`:

**FPS метрики** (на ФПС ДАТАСЕТ):
- end-to-end FPS, ms/img
- det / cls / seg ms/img
- avg detections, avg masks

**Quality метрики** (на `data/fps_dataset_yolo/val` + labels):
- detection: mAP50, recall (Ultralytics val)
- segmentation: mask mAP50
- classification: accuracy (crop-from-gt-bbox, если есть GT)

**Выход:**
- `reports/model_groups_fps_dataset_comparison.{md,csv,json}`
- `reports/model_groups_tensorrt_comparison.{md,csv,json}` (после export)

### B4. Целевые KPI @ 50W (Orin 8GB)

| Режим | Target FPS | Комментарий |
|-------|------------|-------------|
| L1 only (runtime presets) | ≥20 | уже близко (18.7 very_fast) |
| L1 + L2 (pipeline) | ≥22 | cls batch + меньше overhead |
| L3 TensorRT FP16 | ≥30 | realistic для det+seg |
| L3 + L4 full TRT | ≥35 | stretch goal |

Classifier на MobileNet (group 01) может дать ≥40 FPS end-to-end, но ниже quality.

### B5. Group 03 (finetune) — после B3 baseline

1. Finetune detector + segmenter на `data/fps_dataset_yolo/` (уже подготовлен split 80/20)
2. Собрать `models/model_groups/03_finetuned_fps_dataset/`
3. Повторить B3 + TensorRT export
4. Сравнить quality vs FPS tradeoff

---

## Фаза C — Интеграция на ровере (после оптимизации)

1. ROS node: `rover_perception` ← ensemble с TRT engines + model group selector
2. Публикация: `/perception/strawberries` (bbox, class, mask, distance)
3. Web dashboard: real telemetry вместо mock
4. RoArm: target pose из perception → `/roarm/target_pose`

---

## Порядок работ (что делаем прямо сейчас)

### Sprint 1 (сейчас): документы + infra оптимизации
- [x] План (этот файл)
- [ ] `AGENTS.md` — onboarding для Cursor на Orin
- [ ] `models/weights_manifest.json` — manifest весов
- [ ] `install/orin_bootstrap.sh` — bootstrap скрипт
- [ ] `tools/export_tensorrt/` — каркас export + benchmark
- [ ] Обновить `tools/benchmark_holdout_full_pipeline_fps.py`: `--model-group` selector

### Sprint 2: baseline на Orin @ 50W
- [ ] Зафиксировать power mode (50W) и логировать в summary
- [ ] Benchmark 3 groups × 3 presets на `data/ФПС ДАТАСЕТ`
- [ ] Отчёт `reports/model_groups_fps_dataset_comparison.*`

### Sprint 3: TensorRT
- [ ] Export det+seg для groups 01 и 02
- [ ] Benchmark .pt vs .engine
- [ ] Интеграция engine inference в `pipelines/strawberry_ensemble.py` (opt-in flag)

### Sprint 4: GitHub migration
- [ ] Создать repo, push code-only
- [ ] Submodules для third-party
- [ ] CI: lint + smoke test (no GPU)
- [ ] Release notes + weights download instructions

### Sprint 5: Group 03 finetune
- [ ] Train detector + segmenter на fps_dataset_yolo
- [ ] Benchmark vs 01/02
- [ ] TensorRT export group 03

---

## Команды (шпаргалка)

```bash
# Power lock (Orin, уточнить модель на железе)
sudo nvpmodel -m 0 && sudo jetson_clocks

# Jetson env
source scripts/jetson_gpu_env.sh

# FPS benchmark all presets
python3 tools/benchmark_holdout_full_pipeline_fps.py \
  --holdout "data/ФПС ДАТАСЕТ/images" \
  --preset all \
  --outdir runs/holdout_pipeline_speed_presets

# Label FPS dataset (positives only)
python3 tools/label_fps_dataset_from_reserve.py \
  --limit 0 --only-positive --skip-existing

# Ensemble live
./scripts/run_strawberry_gui_with_camera.sh

# RoArm GUI
python3 tools/roarm_local_gui/roarm_test_gui.py
```

---

## Риски

| Риск | Митигация |
|------|-----------|
| Weights не на Orin | weights_manifest + bootstrap download |
| TensorRT version mismatch | export на целевом Orin, не cross-compile |
| ABI torch/torchvision | `yolo_jetson_compat.py` уже есть |
| ФПС ДАТАСЕТ auto-label noise | quality metrics на val split с GT labels |
| Repo слишком большой | submodules + git-lfs только для tiny assets |

---

## Definition of Done (миграция + оптимизация)

- [ ] GitHub repo: clone → bootstrap → `colcon build` → camera GUI работает
- [ ] Agent (`AGENTS.md`): новый чат понимает проект без объяснений
- [ ] 3 model groups benchmarked на ФПС ДАТАСЕТ @ 50W
- [ ] TensorRT engines для det+seg, FPS ≥30 end-to-end (group 02)
- [ ] Отчёт с таблицей FPS + quality для всех групп
- [ ] Group 03 finetuned и включён в сравнение
