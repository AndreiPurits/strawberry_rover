# FPS dataset: PyTorch vs TensorRT (8099 images)

Holdout: `data/ФПС ДАТАСЕТ/images` · Model group: `02_lightened_current` · Classifier: PyTorch (unchanged)

## Full pipeline FPS

| preset | det/seg | PT FPS | TRT FPS | буст | PT ms | TRT ms |
|--------|---------|--------|---------|------|-------|--------|
| baseline | 640/384 | 21.8 | **31.9** | **+47%** | 46.0 | 31.3 |
| fast | 512/320 | 23.3 | **31.5** | **+35%** | 42.9 | 31.8 |
| very_fast | 480/320 | 23.1 | **38.0** | **+64%** | 43.2 | 26.3 |
| ultra_low | 416/256 | 23.1 | **34.1** | **+48%** | 43.3 | 29.3 |

**Рекомендуемый пресет `very_fast`:** 23.1 → **38.0 FPS** (+14.9 FPS, +64%).

## Разбивка по стадиям (very_fast)

| стадия | PT ms | TRT ms | экономия |
|--------|-------|--------|----------|
| detector | 21.8 | 13.7 | −37% |
| classifier | 3.2 | 3.2 | — |
| segmentation | 17.2 | 8.2 | −52% |

## Вывод

TensorRT ускоряет detector и segmenter на **35–64%** по end-to-end FPS. Classifier остаётся на PyTorch (~3 ms) — следующий кандидат на оптимизацию.

Raw: `runs/fps_dataset_benchmark_matrix/` (PT), `runs/fps_dataset_benchmark_matrix_v2/` (TRT)
