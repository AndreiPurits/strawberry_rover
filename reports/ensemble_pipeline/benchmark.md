# Ensemble: RGB-D и дистанция

Onboard: три зафиксированные модели + RGB-D Orbbec Gemini 215.

- Детектор: YOLOv8s v3
- Классификатор: EfficientNet-B0
- Сегментатор: YOLOv8n-seg
- Код: `pipelines/strawberry_ensemble.py`

## Вход камеры

RGB: `/camera/color/image_raw` (`bgr8`).  
Depth: `/camera/depth/image_raw` (`16UC1` в мм или `32FC1` в метрах).

Синхронизация: approximate time по timestamp заголовков, \(|t_{rgb}-t_{depth}| \le \mathrm{sync\_slop}\).  
Если разрешение depth не совпадает с RGB, дистанция не считается (`distance=None`). Драйвер камеры выравнивает depth на RGB.

## Оценка расстояния

Depth агрегируется **только внутри маски сегментации**. Метрика: медиана валидных пикселей. Пустая/невалидная глубина → `distance=None`.

## Ограничения реализации

- классификация и сегментация — по каждому детекции (без батча);
- нет трекинга id объектов в этом контуре.
