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

## Фиксация цели

Код: `pipelines/arm_approach/`  
Конфиг: `config/arm_approach_v1.yaml`

После выбора ягоды цель удерживается в кадре камеры на манипуляторе: повторный вывод в ROI вокруг последней позиции, удержание координат при кратковременной потере детекции, отбраковка срабатываний на клешнях захвата.

## Подход манипулятора

Код: `pipelines/arm_approach/`  
Геометрия: `scripts/arm_approach_geometry.py`  
Описание: `docs/manipulator_approach.md`

Последовательность: lock выбранной ягоды → ближайшее успешное демо в пространстве суставов и изображения → коррекция остатка локальным якобианом → одно движение → проверка рабочей дистанции (standoff). Вариант осуществления — декартово визуальное серво с онлайн-оценкой отображения tip→cam.
