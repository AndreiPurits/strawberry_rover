# Патентный пакет — Strawberry Rover

Ветка `Патент` содержит **только** файлы для заявки. Полный репозиторий (разметка, train, ROS tooling): ветка `main`.


Ветка: `Патент`  
Код: https://github.com/AndreiPurits/strawberry_rover  
Размеченные кадры и веса: https://huggingface.co/datasets/AndreiPurits/strawBerry

Этот файл — оглавление того, что считать своим вкладом для заявки. Сторонние SDK (Orbbec, RPLidar, ROS `image_pipeline`) сюда не входят.

## 1. Что заявляется по сути

Бортовая система сбора клубники на Jetson Orin:

1. Найти ягоды на RGB-кадре (детектор).
2. Оценить зрелость кропа (классификатор).
3. Получить маску ягоды (сегментация).
4. Оценить расстояние по depth **только внутри маски**.
5. Построить close-up ROI у чашечки, найти ориентированные боксы плодоножки, **ассоциировать стебель с конкретной ягодой**, выдать **точку реза**, проверить стабильность по нескольким кадрам.
6. Передать цель манипулятору (RoArm-M3) и шасси (Arduino PWM + ROS2).

## 2. Картинки и разметка (Hugging Face)

Датасет (private): **https://huggingface.co/datasets/AndreiPurits/strawBerry**

Ожидаемая раскладка после выгрузки:

| Задача | Путь в датасете | Локальный источник |
|---|---|---|
| Детекция ягод (bbox + зрелость 0–3) | `images/detection_final/` | `data/final_detection_dataset/` |
| Детекция, train v3 (1 класс) | `images/detection_v3/` | `data/yolo_detection_dataset_v3/` |
| Классификация зрелости | `images/classification_v2/` | `data/classification_dataset_v2/` |
| Ручная классификация | `images/classification_manual/` | `data/classification_manual/` |
| Сегментация (полигоны) | `images/segmentation_yolo/` | `data/yolo_segmentation_dataset/` |
| Сегментация COCO-поднабор | `images/segmentation_project/` | `data/segmentation_project_dataset/` |
| Плодоножка approved | `images/peduncle_approved/` | `data/плодоножки апрувд/` |
| Плодоножка OBB berry-anchored v2.1 | `images/peduncle_obb_v21/` | `data/peduncle_obb_berry_anchored_v21/` |
| Holdout FPS | `images/fps_holdout/` | `data/ФПС ДАТАСЕТ/` |
| Production-веса | `models/` | `runs/.../weights/best.pt` |

Выгрузка с Orin (нужен SSH-ключ в [настройках HF](https://huggingface.co/settings/keys)):

```bash
python3 tools/upload_patent_dataset_to_hf.py
```

## 3. Код изобретения (этот репозиторий)

| Компонент | Файлы |
|---|---|
| Ensemble detect → classify → seg → depth-in-mask | `pipelines/strawberry_ensemble.py` |
| Захват по плодоножке | `pipelines/peduncle_grasp/` |
| Конфиг ассоциации / реза | `config/peduncle_grasp_v1.yaml` |
| Геометрия ягода–стебель | `scripts/peduncle_berry_geometry.py` |
| Полевой демо-проход | `scripts/run_peduncle_grasp_field_demo.py` |
| Обучение OBB плодоножки | `scripts/train_peduncle_obb_v2.py`, `scripts/train_peduncle_obb_v1.py` |
| Тесты геометрии grasp | `tests/test_peduncle_grasp_v1.py` |
| ROS perception / fusion | `src/rover_perception/` |
| Навигация, FSM, bringup, web | `src/rover_navigation/`, `src/rover_fsm/`, `src/rover_bringup/`, `src/rover_web_interface/` |
| Мост к руке | `src/roarm_ros2_http/` |
| Прошивка шасси | `arduino/MEGA_Rover_4x_RC_PWM/` |
| Выбор моделей | `docs/model_selection.md` |
| Датасеты | `docs/datasets_overview.md`, `docs/final_detection_dataset_summary.md`, `docs/segmentation_dataset_plan.md` |
| Дорожная карта | `DEV_ROADMAP.md` |
| Бенчмарк ensemble + depth | `reports/ensemble_pipeline/benchmark.md` |
| FPS / TensorRT | `reports/fps_dataset_trt_comparison.md` |

## 4. Модели — да, их нужно выкладывать

Имеет смысл класть **свои** frozen-веса на тот же HF-датасет, не в git:

- Это доказательство, что пайплайн доведён до работающих чекпоинтов (reduction to practice).
- Заявка и экспертиза смогут воспроизвести вывод без переобучения.
- Объём небольшой (~десятки МБ на production-набор), в отличие от 15 ГБ картинок.
- В git их нет специально (`.gitignore`: `*.pt`).

Класть:

- detector: `runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt`
- classifier: `runs/classification_benchmark_v2/efficientnet_b0/best.pt`
- segmenter: `runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt`
- peduncle OBB: `runs/peduncle_obb/yolov8n_peduncle_v21_mask_calyx/weights/best.pt`

Не класть:

- чужие pretrained `yolov8m.pt` / `yolo11n.pt` с Ultralytics (не наши);
- TensorRT `.engine` (привязаны к конкретной сборке Orin);
- промежуточные `last.pt` и мусор `_trash_v1`.

Опционально (если нужен запасной контур): MobileNetV3-Small, YOLOv8s-seg.

## 5. Железо (контекст реализации)

- Jetson Orin 8GB, power cap 50W  
- Orbbec Gemini 215 RGB-D  
- RPLidar C1  
- RoArm-M3 (HTTP)  
- Arduino Mega, PWM 4× моторы  

## 6. Чего не считать своим датасетом

Папки `data/roboflow_downloads/`, `V2_Strawberry Object Detection.*`, `strawberry rotten.*`, `strawberry ripeness detection.*` — сторонние выгрузки Roboflow. Для патента использовать только свои канонические наборы из таблицы в §2.
