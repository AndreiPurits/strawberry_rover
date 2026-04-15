## Conf sweep on stable test (dataset_v2/test)

- Generated (UTC): `2026-04-13T14:27:13Z`
- Dataset: `data/yolo_detection_dataset_v2/data.yaml`
- Test dir: `/home/andrei/project/strawberry_rover_ws/data/yolo_detection_dataset_v2/images/test`
- Images: `191`
- IOU: `0.70`

| model | conf | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov8n_prod | 0.25 | 0.903079 | 0.789591 | 0.941478 | 0.804749 | 56.32 | 17.76 | 5.96 |
| yolov8s_v2 | 0.25 | 0.934057 | 0.804372 | 0.919436 | 0.860158 | 66.77 | 14.98 | 21.48 |
| yolov8n_prod | 0.35 | 0.900171 | 0.789204 | 0.941478 | 0.804749 | 52.74 | 18.96 | 5.96 |
| yolov8s_v2 | 0.35 | 0.924602 | 0.802247 | 0.891247 | 0.886544 | 63.47 | 15.76 | 21.48 |
| yolov8n_prod | 0.45 | 0.896456 | 0.788332 | 0.941478 | 0.804749 | 52.69 | 18.98 | 5.96 |
| yolov8s_v2 | 0.45 | 0.917141 | 0.799228 | 0.913769 | 0.866755 | 63.83 | 15.67 | 21.48 |

Notes:
- This uses `ultralytics.YOLO.val(conf=...)` on `split=test` plus wall-time `predict()` timing at the same `conf`.
- mAP is generally less sensitive to a single conf threshold than P/R and deployment behavior; use P/R + latency for threshold selection.

