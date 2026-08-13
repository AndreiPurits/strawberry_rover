# Strawberry Rover — материалы к заявке

Код этой ветки: https://github.com/AndreiPurits/strawberry_rover/tree/Патент  
Полный репозиторий разработки: https://github.com/AndreiPurits/strawberry_rover/tree/main  
Размеченные обучающие данные и веса моделей: https://huggingface.co/datasets/AndreiPurits/strawBerry

Сторонние SDK (Orbbec, RPLidar, ROS `image_pipeline`) в объём изобретения не входят.

## Предмет

Бортовая система сбора клубники (Jetson Orin + RGB-D камера + манипулятор):

1. Обнаружение ягод на RGB-кадре.
2. Классификация зрелости по кропу ягоды.
3. Сегментация маски ягоды.
4. Оценка расстояния по карте глубины **только внутри маски**.
5. Построение close-up ROI у чашечки, детекция ориентированных боксов плодоножки, **ассоциация стебля с конкретной ягодой**, расчёт **точки реза**, проверка стабильности по нескольким кадрам.
6. Передача цели манипулятору и шасси.

## Реализация в этой ветке

| Компонент | Путь |
|---|---|
| Конвейер detect → classify → seg → distance-in-mask | `pipelines/strawberry_ensemble.py` |
| Захват по плодоножке (calyx, ROI, OBB, ассоциация, рез, temporal) | `pipelines/peduncle_grasp/` |
| Параметры ассоциации и реза | `config/peduncle_grasp_v1.yaml` |
| Геометрия ягода–стебель | `scripts/peduncle_berry_geometry.py` |
| Полевой демо-проход | `scripts/run_peduncle_grasp_field_demo.py` |
| Юнит-тесты геометрии | `tests/test_peduncle_grasp_v1.py` |
| Зафиксированные production-модели | `docs/model_selection.md` |
| Описание датасетов | `docs/datasets_overview.md`, `docs/final_detection_dataset_summary.md`, `docs/segmentation_dataset_plan.md` |
| RGB-D контракт и оценка дистанции | `reports/ensemble_pipeline/benchmark.md` |
| Стадии системы | `DEV_ROADMAP.md` |

## Обучающие данные и веса

Датасет (доступ по запросу / collaborator): https://huggingface.co/datasets/AndreiPurits/strawBerry

| Задача | Путь в датасете | Разметка |
|---|---|---|
| Детекция ягод, train v3 | `images/detection_v3/` | YOLO bbox, 1500 пар image+label |
| Классификация зрелости | `images/classification_v2/` | класс = папка (green / turning / ripe / rotten), 2700 кропов |
| Сегментация ягоды | `images/segmentation_yolo/` | YOLO-seg полигоны, 2300 пар |
| Плодоножка OBB v2.1 | `images/peduncle_obb_v21/` | ориентированный бокс + berry labels + masks, 143 пары |
| Production-веса | `models/` | detector, classifier, segmenter, peduncle OBB |

Веса в датасете:

- `models/detector_yolov8s_v3_lowdensity_best.pt`
- `models/classifier_efficientnet_b0_best.pt`
- `models/segmenter_yolov8n_seg_best.pt`
- `models/peduncle_obb_yolov8n_v21_best.pt`

## Аппаратная реализация

- NVIDIA Jetson Orin 8 GB
- RGB-D камера Orbbec Gemini 215
- LiDAR RPLidar C1
- манипулятор RoArm-M3
- шасси Arduino Mega, PWM 4× моторы
