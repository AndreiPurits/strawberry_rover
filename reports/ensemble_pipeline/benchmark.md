## Ensemble pipeline benchmark (MVP v1)

Goal: real-time per-strawberry outputs on Jetson Orin using **3 frozen production models** + **Gemini 215 RGB+Depth**.

### Setup

- **Detector**: `runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt`
- **Classifier**: `runs/classification_benchmark_v2/efficientnet_b0/best.pt`
- **Segmenter**: `runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt`
- **Pipeline entrypoint**: `scripts/run_strawberry_ensemble.py`

### Camera IO (Gemini 215)

This MVP supports two capture modes:

- **ROS2 mode (recommended for Gemini 215)**:
  - subscribes to configurable topics:
    - RGB: `/camera/color/image_raw` (expected `bgr8`)
    - Depth: `/camera/depth/image_raw` (expected `16UC1` in **mm** or `32FC1` in **m**)
  - sync policy:
    - **software approximate time** using message header timestamps
    - requires \(|t_{rgb} - t_{depth}| \le \text{sync_slop}\)
    - if depth resolution differs from RGB, depth is ignored (`distance=None`)

- **V4L2 mode (RGB only)**:
  - `--source v4l2`
  - depth is unavailable → `distance=None`

### Distance estimation

- depth is aggregated **only inside the segmentation mask**
- metric: **median depth** over valid pixels
- invalid/empty depth → `distance=None` (pipeline must not crash)

### How to run

ROS2 (Gemini 215):

```bash
python3 scripts/run_strawberry_ensemble.py \
  --source ros2 \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --sync-slop 0.03
```

V4L2 (debug RGB only):

```bash
python3 scripts/run_strawberry_ensemble.py --source v4l2 --v4l2-index 0 --size 640x480 --fps 30
```

### Profiling results (fill after measurement)

Environment:

- Jetson Orin: TODO
- Power mode: TODO
- CUDA / Torch / Ultralytics versions: TODO
- Camera resolution / FPS: TODO

Measured (overlay HUD):

| stage | ms/frame |
|---|---:|
| detector | TODO |
| classifier (sum over detections) | TODO |
| segmentation (sum over detections) | TODO |
| depth fusion (sum over detections) | TODO |
| total | TODO |
| FPS (effective) | TODO |

### Known bottlenecks / limitations (MVP v1)

- segmentation/classification currently run **per detection** (no batching)
- ROS2 sync is approximate and assumes depth is already **aligned to RGB** by the camera driver
- depth frame is ignored if its resolution differs from RGB (alignment step not implemented here)
- no tracking / object IDs yet (intentionally out of scope for MVP)

