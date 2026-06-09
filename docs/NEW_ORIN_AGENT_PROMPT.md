# Промпт для нового агента (Cursor на новом Orin)

Скопируй блок ниже в первое сообщение агенту:

---

Ты работаешь в репозитории **strawberry_rover** на Jetson Orin 8GB @ 50W.

**Прочитай первым:** `AGENTS.md` → `docs/orin_new_device_setup.md` → `CURRENT_STATE.md` → `reports/fps_dataset_trt_comparison.md`

**Железо:** Orbbec Gemini (ROS2), RealSense D405 (отдельно), RPLidar C1, Arduino Mega, RoArm-M3.

**Vision pipeline:** detector → crop → classifier → segmenter → depth. Код: `pipelines/strawberry_ensemble.py`, `scripts/run_strawberry_ensemble.py`.

**Production модели (group 02):** `models/model_groups/02_lightened_current/`. Веса в `runs/` — **не в git**. Manifest: `models/weights_manifest.json`.

**TensorRT (уже бенчмаркнуто на старом Orin):**
- PT very_fast: 23.1 FPS
- TRT very_fast: 38.0 FPS
- Engines: `runs/export_tensorrt/group02/` (экспорт на целевом Orin: `bash tools/export_tensorrt/export_all_group02.sh`)

**Bootstrap:** `bash bootstrap/orin_bootstrap.sh` + `source scripts/jetson_gpu_env.sh`

**Правила:**
- Не обучать и не менять датасеты без явного запроса
- Не перезаписывать старые `best.pt`
- Не коммитить `data/`, `runs/`, `*.pt`, `*.engine`
- Не трогать Orbbec при работе с RealSense
- Benchmark только на holdout `data/ФПС ДАТАСЕТ/`

**Текущая задача:** [опиши здесь — например: «поднять ROS2 + камеру, проверить TRT FPS на новом Orin»]

---
