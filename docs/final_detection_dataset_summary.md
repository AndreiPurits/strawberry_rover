# Исходный набор детекции (канон)

Путь: `data/final_detection_dataset/` (на HF обучение детектора — `images/detection_v3/`, 1500 кадров, 1 класс).

Схема классов в канонических bbox (зрелость):

- `0` green
- `1` turning
- `2` ripe
- `3` rotten_or_overripe

## Объём канона

- Кадры / файлы меток: 675
- Bounding boxes: 2294
  - green: 1347
  - turning: 266
  - ripe: 681
  - rotten_or_overripe: 0
