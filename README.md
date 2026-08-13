# Strawberry Rover — патентный пакет

Узкая ветка только с материалами для заявки. Полный код и tooling: ветка [`main`](https://github.com/AndreiPurits/strawberry_rover/tree/main).

**Размеченные кадры и frozen-веса (private):** https://huggingface.co/datasets/AndreiPurits/strawBerry

Оглавление: [`PATENT.md`](PATENT.md)

## Содержимое этой ветки

| Путь | Зачем |
|---|---|
| `pipelines/strawberry_ensemble.py` | detect → classify → seg → distance in mask |
| `pipelines/peduncle_grasp/` | calyx → ROI → OBB → association → cut → temporal |
| `config/peduncle_grasp_v1.yaml` | пороги ассоциации и реза |
| `scripts/peduncle_berry_geometry.py` | геометрия ягода–стебель |
| `scripts/run_peduncle_grasp_field_demo.py` | полевой демо-проход |
| `tests/test_peduncle_grasp_v1.py` | юнит-тесты геометрии |
| `reports/ensemble_pipeline/benchmark.md` | контракт RGB-D / distance |
| `docs/model_selection.md` | frozen production weights |
| `docs/datasets_overview.md` и соседние | датасеты детекции / классификации / сегментации |
| `DEV_ROADMAP.md`, `CURRENT_STATE.md` | стадии системы |

ROS-стек, Arduino, скрипты разметки/обучения — в `main`.
