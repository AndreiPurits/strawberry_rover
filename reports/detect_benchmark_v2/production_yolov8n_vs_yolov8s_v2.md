# Production detector vs candidate on stable new test (dataset_v2)

- Generated (UTC): `2026-04-13T14:19:53Z`
- Dataset: `data/yolo_detection_dataset_v2/data.yaml`
- Test dir: `/home/andrei/project/strawberry_rover_ws/data/yolo_detection_dataset_v2/images/test`
- Preview root: `data/test_compare_preview` (same images for both)

## Metrics on test

| model | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n_prod | 0.896138 | 0.751145 | 0.941478 | 0.804749 | 55.83 | 17.91 | 5.96 |
| yolov8s_v2 | 0.943819 | 0.776762 | 0.919436 | 0.860158 | 67.00 | 14.93 | 21.48 |

## Delta (candidate minus production)

- Δ mAP50: 0.047681
- Δ mAP50-95: 0.025616
- Δ precision: -0.022042
- Δ recall: 0.055409
- Δ ms/img: 11.17
- Δ FPS: -2.99

## Quick conclusion (recall → speed → stability)

- Quality: `yolov8s_v2` has higher mAP and recall; `yolov8n_prod` has higher precision.
- Speed (measured here by predict() wall time): `yolov8n_prod` is faster.
- Recommendation: keep `yolov8n_prod` for rover production if speed margin is critical; use `yolov8s_v2` for auto-labeling due to higher recall.

## Notes

- Both models evaluated on the same fixed test split from `data/yolo_detection_dataset_v2/`.
- Timing measured by `predict()` over the full test set.
