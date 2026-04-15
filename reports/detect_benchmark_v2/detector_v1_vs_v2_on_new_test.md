# Detector v1 vs v2 on stable new test (dataset_v2)

- Generated (UTC): `2026-04-13T14:10:53Z`
- Dataset: `data/yolo_detection_dataset_v2/data.yaml`
- Test dir: `/home/andrei/project/strawberry_rover_ws/data/yolo_detection_dataset_v2/images/test`
- Preview root: `data/test_compare_preview` (same images for both)

## Metrics on test

| model | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 0.892692 | 0.754580 | 0.948162 | 0.821900 | 67.34 | 14.85 | 21.47 |
| v2 | 0.943819 | 0.776762 | 0.919436 | 0.860158 | 63.37 | 15.78 | 21.48 |

## Notes

- Both models evaluated on the same fixed test split from `data/yolo_detection_dataset_v2/`.
- Timing measured by `predict()` over the full test set.
