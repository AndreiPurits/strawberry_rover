# Зафиксированные модели

Конвейер на роботе: детектор (bbox) → классификатор (зрелость кропа) → сегментатор (маска). Плодоножка — отдельная OBB-модель на close-up ROI.

Веса на Hugging Face: https://huggingface.co/datasets/AndreiPurits/strawBerry (`models/`).

## Детектор

- Архитектура: YOLOv8s
- Обучение: `data/yolo_detection_dataset_v3/` (1 класс `strawberry`)
- Чекпоинт: `models/detector_yolov8s_v3_lowdensity_best.pt`
- Локальный путь обучения: `runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt`

## Классификатор зрелости

Запускается после детекции, на кропе bbox.

- Архитектура: EfficientNet-B0
- Обучение: `data/classification_dataset_v2/`
- Классы: green, turning, ripe, rotten
- Чекпоинт: `models/classifier_efficientnet_b0_best.pt`
- Локальный путь: `runs/classification_benchmark_v2/efficientnet_b0/best.pt`

## Сегментация

1 класс (`strawberry`). Зрелость в обучение сегментации не входит.

- Архитектура: YOLOv8n-seg (onboard)
- Обучение: `data/yolo_segmentation_dataset/`
- Чекпоинт: `models/segmenter_yolov8n_seg_best.pt`
- Локальный путь: `runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt`

Для более точной маски (grasp / offline) использовался YOLOv8s-seg (`runs/segment_benchmark/yolov8s_seg_benchmark/weights/best.pt`).

## Плодоножка (OBB)

- Архитектура: YOLOv8n-OBB
- Обучение: `data/peduncle_obb_berry_anchored_v21/`
- Чекпоинт: `models/peduncle_obb_yolov8n_v21_best.pt`
- Локальный путь: `runs/peduncle_obb/yolov8n_peduncle_v21_mask_calyx/weights/best.pt`
