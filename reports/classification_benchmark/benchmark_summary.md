## Classification benchmark summary (dataset_v2)

- Generated (UTC): `2026-04-15T18:41:58Z`
- Dataset: `data/classification_dataset_v2`

| model | status | acc | P_macro | R_macro | F1_macro | ms/img | FPS | size (MB) | best weights |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mobilenet_v3_small | ok | 0.8963 | 0.9066 | 0.9000 | 0.9027 | 1.19 | 840.19 | 5.92 | `runs/classification_benchmark_v2/mobilenet_v3_small/best.pt` |
| resnet18 | ok | 0.8741 | 0.8746 | 0.8786 | 0.8760 | 2.49 | 402.06 | 42.71 | `runs/classification_benchmark_v2/resnet18/best.pt` |
| efficientnet_b0 | ok | 0.9259 | 0.9302 | 0.9301 | 0.9295 | 5.26 | 190.17 | 15.57 | `runs/classification_benchmark_v2/efficientnet_b0/best.pt` |

Per-class metrics are stored in the JSON report.
