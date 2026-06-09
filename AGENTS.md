# Cursor Agent — Strawberry Rover (Orin 8GB)

Быстрый onboarding для работы на собранном ровере или dev-машине.

## Читать первым (порядок)

1. `README.md` — что это за проект
2. `AGENTS.md` — этот файл
3. `ORCHESTRATOR.md` — правила работы
4. `CURRENT_STATE.md` — что уже сделано
5. `docs/model_selection.md` — frozen production weights
6. `docs/orin_migration_and_optimization_plan.md` — план миграции + оптимизации FPS

## Железо (целевое)

- **Jetson Orin 8GB**, power cap **50W**
- **Стереокамера** Orbbec Gemini (ROS2: `src/OrbbecSDK_ROS2/`)
- **LiDAR** RPLidar C1 (`src/rplidar_ros/`, submodule)
- **Навигация** `src/rover_navigation/`
- **Arduino** Mega — PWM моторы (`arduino/MEGA_Rover_4x_RC_PWM/`)
- **RoArm-M3** — HTTP + ROS2 (`tools/roarm_local_gui/`, `src/roarm_ros2_http/`)

## Три группы моделей

| Group | Path | Назначение |
|-------|------|------------|
| 01_fast_initial | `models/model_groups/01_fast_initial/` | YOLOv8n + MobileNetV3 + YOLOv8n-seg |
| 02_lightened_current | `models/model_groups/02_lightened_current/` | YOLOv8s v3 + EfficientNet-B0 + YOLOv8n-seg |
| 03_finetuned_fps_dataset | `models/model_groups/03_finetuned_fps_dataset/` | после дообучения на ФПС ДАТАСЕТ |

Manifest весов (пути, размеры): `models/weights_manifest.json`

**Веса не в git.** Лежат в `runs/` локально или скачиваются по manifest.

## Vision pipeline

```
camera → detector (bbox) → crop → classifier (ripeness) → segmenter (mask) → depth distance
```

Код:
- `pipelines/strawberry_ensemble.py` — ядро
- `scripts/run_strawberry_ensemble.py` — live GUI / headless

Benchmark FPS:
```bash
source scripts/jetson_gpu_env.sh
python3 tools/benchmark_holdout_full_pipeline_fps.py \
  --holdout "data/ФПС ДАТАСЕТ/images" \
  --model-group 02_lightened_current \
  --preset all \
  --outdir runs/holdout_pipeline_speed_presets
```

Holdout для quality/FPS: `data/ФПС ДАТАСЕТ/` (auto-label, positives only).

## GUI

| GUI | Запуск |
|-----|--------|
| Стерео + ML overlay | `./scripts/run_strawberry_gui_with_camera.sh` |
| Роборука (без ROS) | `python3 tools/roarm_local_gui/roarm_test_gui.py` |
| Web dashboard | `ros2 launch rover_web_interface web_interface.launch.py` |

## ROS2

```bash
source /opt/ros/foxy/setup.bash   # или humble — что установлено на Orin
source install/setup.bash
ros2 launch rover_bringup bringup.launch.py
```

Кастомные пакеты: `src/rover_*`, `src/roarm_ros2_http/`.

## Jetson setup

```bash
# Power mode (уточнить nvpmodel -q на железе)
sudo nvpmodel -m 0
sudo jetson_clocks

# OpenBLAS fix для PyTorch
source scripts/jetson_gpu_env.sh

# Bootstrap (первый раз)
bash bootstrap/orin_bootstrap.sh
```

Проверка CUDA:
```bash
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Правила для агента

- **Не обучать** и **не менять датасеты** без явного запроса
- **Не перезаписывать** старые `best.pt` — новые runs только в `runs/finetune_*` или `runs/export_*`
- **Не коммитить** `data/`, `runs/`, `*.pt`, `*.engine`
- Benchmark только на holdout (`ФПС ДАТАСЕТ`), не train/val/test сплиты обучения
- Перед оптимизацией: baseline FPS @ 50W, потом TensorRT, потом сравнение quality

## TensorRT (бенчмарк 8099 кадров, group 02)

| preset | PT FPS | TRT FPS |
|--------|--------|---------|
| very_fast (480/320) | 23.1 | **38.0** |

Export: `COOLDOWN_SEC=60 bash tools/export_tensorrt/export_all_group02.sh`  
Отчёт: `reports/fps_dataset_trt_comparison.md`

## Новый Orin

1. `docs/orin_new_device_setup.md` — установка с нуля
2. `docs/NEW_ORIN_AGENT_PROMPT.md` — промпт для агента
3. `bootstrap/orin_bootstrap.sh` — Python/CUDA/TensorRT venv

## Текущий приоритет

1. Поднять стек на новом Orin (bootstrap → weights → TRT export)
2. Group 03 finetune на `data/ФПС ДАТАСЕТ/` (по запросу)
3. TRT classifier / дальнейший FPS

План: `docs/orin_migration_and_optimization_plan.md`
