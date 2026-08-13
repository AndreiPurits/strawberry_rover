# Реализованный контур

## Восприятие

Конвейер на борту: детектор → классификатор зрелости → сегментация → расстояние по depth внутри маски.

Код: `pipelines/strawberry_ensemble.py`  
Веса: `docs/model_selection.md`  
Датасет весов: https://huggingface.co/datasets/AndreiPurits/strawBerry (`models/`)

Production-набор:

- детектор YOLOv8s v3
- классификатор EfficientNet-B0
- сегментатор YOLOv8n-seg
- плодоножка YOLOv8n-OBB v2.1

На Jetson Orin (group 02, TensorRT, preset `very_fast`): около 38 FPS на holdout RGB.

## Захват по плодоножке

Код: `pipelines/peduncle_grasp/`  
Конфиг: `config/peduncle_grasp_v1.yaml`

Последовательность: оценка чашечки → close-up ROI → кандидаты OBB → ассоциация стебель–ягода → точка реза → проверка по кадрам.

## RGB-D

Камера Orbbec Gemini 215. Дистанция — медиана depth по пикселям маски. Контракт: `reports/ensemble_pipeline/benchmark.md`.
