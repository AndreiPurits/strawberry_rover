# Текущее состояние проекта

## CV pipeline (production)

- Ensemble: detector → classifier → segmenter → depth (`pipelines/strawberry_ensemble.py`)
- Frozen weights: `docs/model_selection.md`, manifest: `models/weights_manifest.json`
- 3 model groups: `models/model_groups/01_fast_initial`, `02_lightened_current`, `03_finetune_future`

## FPS бенчмарк (ФПС ДАТАСЕТ, 8099 кадров, group 02)

| preset | PT FPS | TRT FPS |
|--------|--------|---------|
| baseline (640/384) | 21.8 | 31.9 |
| fast (512/320) | 23.3 | 31.5 |
| very_fast (480/320) | 23.1 | **38.0** |
| ultra_low (416/256) | 23.1 | 34.1 |

Отчёт: `reports/fps_dataset_trt_comparison.md`  
TensorRT export: `tools/export_tensorrt/export_all_group02.sh`

## Камеры

- Orbbec Gemini: `src/OrbbecSDK_ROS2/` (submodule)
- RealSense D405: `bootstrap/realsense_d405_orin.sh`, `docs/realsense_d405_orin.md`

## Миграция на новый Orin

- Bootstrap: `bootstrap/orin_bootstrap.sh`
- Гайд: `docs/orin_new_device_setup.md`
- Onboarding агента: `AGENTS.md`, `docs/NEW_ORIN_AGENT_PROMPT.md`

## Вне git

- `data/`, `runs/`, `*.pt`, `*.engine`, torch wheel, `third_party/vision/`

## Следующее

1. Развернуть на новом Orin (git clone → bootstrap → weights → TRT export)
2. Group 03 finetune (по запросу)
3. TRT / оптимизация classifier
