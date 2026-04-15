# Strawberry detection benchmark (YOLO)

- Generated (UTC): `2026-04-12T14:23:19Z`
- CUDA available: **True**
- Device: `Orin`
- Ultralytics YOLO11: файл весов называется `yolo11n.pt` (имени `yolov11n.pt` в релизах нет). Run-папка для отчёта: `yolov11n_benchmark`. 
- В логе обучения возможны предупреждения torchvision (Jetson) и `NMS time limit exceeded` на val — на итоговые метрики после `model.val()` на `best.pt` это не влияет.

## Сводная таблица

| model | batch | epochs | train (s) | mAP50 | mAP50-95 | P | R | ms/img | FPS | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolov8n | 64 | 50 | 1387.146 | 0.922563 | 0.767641 | 0.905571 | 0.875697 | 76.0222 | 13.1540 | ok |
| yolov8s | 64 | 50 | 2124.632 | 0.928053 | 0.772015 | 0.928146 | 0.881960 | 87.9919 | 11.3647 | ok |
| yolov8m | 64 | 50 | 3761.005 | 0.932360 | 0.770892 | 0.923589 | 0.861438 | 117.8285 | 8.4869 | ok |
| yolo11n | 64 | 50 | 1531.192 | 0.933320 | 0.761461 | 0.924286 | 0.870035 | 76.9169 | 13.0010 | ok |

## Выводы

- **Лучшая по качеству** (mAP50-95, при равенстве — recall): `yolov8s`
- **Лучшая по скорости инференса** (минимум ms/изображение на полном val, см. скрипт): `yolov8n`
- **Баланс speed/quality** (recall ≥ max−0.02, затем FPS, затем mAP50): `yolov8n`
- **Авторазметка** (приоритет recall, затем mAP50): `yolov8s`
- **Production на Orin** (recall ≥ max−0.03, затем минимальная задержка): `yolov8n`

### Пути к весам

- Авторазметка: `/home/andrei/project/strawberry_rover_ws/runs/detect_benchmark/yolov8s_benchmark/weights/best.pt`
- Production (рекомендация): `/home/andrei/project/strawberry_rover_ws/runs/detect_benchmark/yolov8n_benchmark/weights/best.pt`

Превью (30 фиксированных val, seed=42): `data/benchmark_preview/<model_name>/`.
