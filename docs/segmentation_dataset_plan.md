# Сегментация

Модель учится на маске ягоды, один класс `strawberry`. Зрелость определяет отдельный классификатор.

Источник разметки — COCO-полигоны (категории `Strawberries`, `ripe`, `unripe`) сводятся к одному классу:

- `Strawberries` / `ripe` / `unripe` → `strawberry` (`0`)

Поднабор для обучения (YOLO-seg): train 1800, val 250, test 250.  
Выгрузка: `images/segmentation_yolo/` на https://huggingface.co/datasets/AndreiPurits/strawBerry
