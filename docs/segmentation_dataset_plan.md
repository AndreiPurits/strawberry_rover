## Segmentation dataset plan (strawberry_rover)

### Цель
- **Segmentation**: обучаем модель на **маске ягоды** и **только на 1 классе**.
- **Ripeness (ripe/unripe)**: определяется **отдельной моделью-классификатором** и **не используется** как класс для segmentation training.

### Источник данных
- `data/roboflow_downloads/`
  - `train/`, `valid/`, `test/`
  - `train/_annotations.coco.json`, `valid/_annotations.coco.json`, `test/_annotations.coco.json`
  - COCO-аннотации содержат `segmentation` (полигон) и `bbox`
  - Категории в источнике: `Strawberries`, `ripe`, `unripe`

### Что делаем для segmentation
Формируем новый датасет:

`data/segmentation_project_dataset/`
- `train/`, `val/`, `test/` — поднабор изображений для первого этапа
- `annotations/` — COCO аннотации для поднабора
- `reserve/` — все неиспользованные изображения + аннотации (не удаляем)
- `reports/` — отчёты

#### 1-class унификация
В целевых COCO аннотациях все berry-like категории сводим в одну:
- `Strawberries` → `strawberry`
- `ripe` → `strawberry`
- `unripe` → `strawberry`

Итоговая категория:
- `0 = strawberry`

#### Поднабор на первый benchmark
Целевые размеры (примерно):
- train: ~1800 images
- val: ~250 images
- test: ~250 images

Поднабор отбирается так, чтобы:
- сохранять разнообразие сцен и размеров ягод
- сохранять кадры с несколькими ягодами
- не брать только «простые» изображения
- по возможности иметь баланс сцен по наличию объектов `ripe`/`unripe` (в источнике), включая mixed scenes

Фактические числа и статистика записываются в:
- `data/segmentation_project_dataset/reports/subset_summary.txt`

### Что делаем для classifier UI (priority crops)
Отдельно извлекаем bbox-crop-ы для ручной проверки/разметки:
- `250` crop из объектов категории `ripe`
- `250` crop из объектов категории `unripe`

Требования к crop:
- crop делается **по bbox**, не по маске
- padding: `+15%` по каждой стороне bbox (с клиппингом по границе картинки)
- минимальный размер crop после padding: `>= 40 px` по ширине и высоте

Сохраняем в:
- `data/classifier_priority_queue/ripe/`
- `data/classifier_priority_queue/unripe/`
- `data/classifier_priority_queue/reports/priority_crops_report.txt`

### Интеграция в UI
В classifier UI очередь строится так:
1. **priority queue** из `data/classifier_priority_queue/{ripe,unripe}/`
2. затем обычные candidate crop из `data/classifier_candidates/{all,review_small}/`

Priority crop помечаются в UI как `priority/ripe` или `priority/unripe`.

