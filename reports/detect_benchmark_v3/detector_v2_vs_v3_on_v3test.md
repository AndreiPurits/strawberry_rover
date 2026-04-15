# Detector v2 vs v3 on v3 stable test

- Generated (UTC): `2026-04-14T07:07:11Z`
- Dataset: `data/yolo_detection_dataset_v3/data.yaml`
- Test dir: `/home/andrei/project/strawberry_rover_ws/data/yolo_detection_dataset_v3/images/test`
- conf: `0.25`, iou: `0.70`
- Preview root: `data/test_compare_preview` (same images for both)

## Metrics on v3 test

| model | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2_on_v3test | 0.920541 | 0.761849 | 0.888074 | 0.938269 | 59.00 | 16.95 | 21.48 |
| v3 | 0.958288 | 0.877517 | 0.940086 | 0.930591 | 54.86 | 18.23 | 21.49 |

## Delta (v3 minus v2)

- Δ mAP50: 0.037747
- Δ mAP50-95: 0.115668
- Δ precision: 0.052012
- Δ recall: -0.007678
- Δ ms/img: -4.14
- Δ FPS: 1.28

## Questions answered

- Did low-density hypothesis help? (judge by v3 vs v2 on v3 test): **see deltas above**.
- Better recall? **Δ recall**
- Better mAP50-95? **Δ mAP50-95**
- Slower? **Δ ms/img / Δ FPS**
- Replace auto-label v2? Depends on recall gain vs speed/precision trade-off.
- Use v3 as production candidate? Depends on speed vs current production requirements.

