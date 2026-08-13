# Датасеты

Детекция, классификация и сегментация — отдельные задачи. Обучающие выгрузки: https://huggingface.co/datasets/AndreiPurits/strawBerry

## Детекция

Задача: найти ягоды на полном кадре (bounding box).

Обучение (1 класс `0 = strawberry`): `images/detection_v3/` на HF, локально `data/yolo_detection_dataset_v3/` — 1500 кадров (train 1050 / val 225 / test 225).

Формат YOLO, координаты в \([0, 1]\):

`class_id x_center y_center width height`

Исходные канонические метки (`data/final_detection_dataset/`) содержат id зрелости в bbox:

- `0` green, `1` turning, `2` ripe, `3` rotten_or_overripe

Для обучения детектора id сводятся к одному классу `strawberry`. Классификация зрелости — отдельная модель по кропу.

## Классификация зрелости

Задача: класс одной ягоды (кроп, не сцена).

На HF: `images/classification_v2/` — 2700 кропов в папках `train|val|test` × `green|turning|ripe|rotten`.

Конвейер: кадр → bbox → кроп → классификатор.

## Сегментация

Задача: маска ягоды, 1 класс `strawberry`. Зрелость в классы сегментации не входит.

На HF: `images/segmentation_yolo/` — 2300 пар image+polygon label (train 1800 / val 250 / test 250).

## Плодоножка

Ориентированный бокс стебля, привязанный к ягоде: `images/peduncle_obb_v21/` (143 кадра train/val).
