# Стадии системы

Система состоит из контура восприятия (датасеты → модели → ensemble → дистанция → вход grasp), контура подхода манипулятора и контура платформы (сенсоры → навигация → исполнительные устройства).

## 1. Датасеты

Детекция ягод, классификация зрелости, сегментация маски, OBB плодоножки.  
Документы: `docs/datasets_overview.md`, `docs/final_detection_dataset_summary.md`, `docs/segmentation_dataset_plan.md`.  
Данные: https://huggingface.co/datasets/AndreiPurits/strawBerry

## 2. Модели

Зафиксированные onboard-чекпоинты: детектор, классификатор, сегментатор, OBB плодоножки.  
Документ: `docs/model_selection.md`.

## 3. Ensemble

На кадре: bbox ягоды → класс зрелости → маска → расстояние по depth внутри маски.  
Код: `pipelines/strawberry_ensemble.py`.  
Контракт RGB-D: `reports/ensemble_pipeline/benchmark.md`.

## 4. RGB-D

Синхронизация RGB и depth, единицы mm/m, дистанция только по маске.

## 5. Захват

Ассоциация плодоножки с ягодой и точка реза.  
Код: `pipelines/peduncle_grasp/`.

## 6. Подход манипулятора

Фиксация выбранной ягоды, одношаговое планирование по демо-приору и локальному якобиану, проверка standoff.  
Код: `pipelines/arm_approach/`.  
Описание: `docs/manipulator_approach.md`.

## 7. Платформа

Манипулятор RoArm-M3, шасси, навигация, операторский слой — полный код в ветке `main`.
