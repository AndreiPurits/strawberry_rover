## Dataset normalization summary

- **Отчёт**: `/home/andrei/project/strawberry_rover_ws/data/normalized/dataset_report.txt`
- **Выходная структура**: `/home/andrei/project/strawberry_rover_ws/data/normalized`

## Что было сделано

- Все изображения собраны в `data/normalized/<dataset>/images/` (без изменения `data/raw/`).
- Labels (если распознаны как YOLO bbox или YOLO segmentation polygon) приведены к **YOLO-bbox** и сохранены в `labels/`.
- Для изображений с labels созданы визуализации bbox в `bbox_vis/`.

## По датасетам

### `strawberry_ds`
- **Найдено изображений**: 247
- **Найдено label-файлов**: 248
- **Формат labels (авто-детект)**: `yolo_bbox`
- **Скопировано изображений в normalized**: 247
- **Сконвертировано/записано YOLO labels**: 247
- **Пар image+label**: 247
- **Изображений без labels**: 0
- **Labels без изображений**: 1

### `strawberry_turkey`
- **Найдено изображений**: 566
- **Найдено label-файлов**: 566
- **Формат labels (авто-детект)**: `yolo_seg_polygon`
- **Скопировано изображений в normalized**: 566
- **Сконвертировано/записано YOLO labels**: 566
- **Пар image+label**: 566
- **Изображений без labels**: 0
- **Labels без изображений**: 0

### `my_data`
- **Найдено изображений**: 15
- **Найдено label-файлов**: 0
- **Формат labels (авто-детект)**: `none`
- **Скопировано изображений в normalized**: 15
- **Сконвертировано/записано YOLO labels**: 0
- **Пар image+label**: 0
- **Изображений без labels**: 15
- **Labels без изображений**: 0

## Что проверить вручную

- Если какой-то датасет определился как `pascal_voc_xml` или `coco_json`: сейчас конвертация не реализована, нужно решить целевой пайплайн/классы и добавить конвертер.
- Просмотреть несколько файлов в `bbox_vis/` на предмет корректности рамок.
- Проверить, что ID классов соответствуют ожидаемым названиям классов (например Roboflow `data.yaml`).
