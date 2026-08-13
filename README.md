# Strawberry Rover — патентный пакет

Ветка содержит код и описание конвейера восприятия и захвата клубники для заявки.

Полный репозиторий разработки: [`main`](https://github.com/AndreiPurits/strawberry_rover/tree/main).  
Размеченные кадры и веса моделей: https://huggingface.co/datasets/AndreiPurits/strawBerry

Сводка для заявки: [`PATENT.md`](PATENT.md)

## Состав ветки

| Путь | Назначение |
|---|---|
| `pipelines/strawberry_ensemble.py` | детекция → зрелость → маска → дистанция по depth в маске |
| `pipelines/peduncle_grasp/` | чашечка → ROI → OBB плодоножки → ассоциация → точка реза |
| `config/peduncle_grasp_v1.yaml` | пороги ассоциации и реза |
| `scripts/peduncle_berry_geometry.py` | геометрия ягода–стебель |
| `scripts/run_peduncle_grasp_field_demo.py` | полевой проход |
| `tests/test_peduncle_grasp_v1.py` | тесты геометрии |
| `reports/ensemble_pipeline/benchmark.md` | RGB-D синхронизация и метрика дистанции |
| `docs/model_selection.md` | зафиксированные веса |
| `docs/datasets_overview.md` | датасеты детекции, классификации, сегментации |
| `DEV_ROADMAP.md` | стадии системы |
