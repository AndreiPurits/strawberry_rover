## Model selection (frozen checkpoints)

This document records the **currently selected** model checkpoints for the Strawberry Rover stack.

### Detector (YOLO) — production vs auto-label

We use a two-stage architecture:

- **Fast detector** for real-time operation on Jetson Orin (low latency, stable throughput).
- **More accurate downstream models** (classifier / segmentation) to refine decisions after detection.

#### Production detector (on-rover inference)

- **Model**: YOLOv8s (v3 low-density)
- **Checkpoint**: `runs/detect_benchmark_v3/yolov8s_v3_lowdensity/weights/best.pt`
- **Notes**:
  - Trained on `data/yolo_detection_dataset_v3/` (low-density selection).
  - Benchmark + comparisons live in `reports/detect_benchmark_v3/` and `data/yolo_detection_dataset_v3/reports/`.

#### Auto-label detector (dataset generation / labeling pipeline)

- **Model**: YOLOv8s (v2 resplit, fast NMS)
- **Checkpoint**: `runs/detect_benchmark_v2/yolov8s_v2_resplit_fastnms/weights/best.pt`
- **Rationale** (new stable test `data/yolo_detection_dataset_v2/images/test`):
  - Higher recall (fewer missed objects), which is critical for auto-labeling
  - Higher mAP50 and mAP50-95

### Benchmark baselines (do not delete)

- Final production-vs-candidate comparison reports:
  - `reports/detect_benchmark_v2/production_yolov8n_vs_yolov8s_v2.md`
  - `reports/detect_benchmark_v2/production_yolov8n_vs_yolov8s_v2.json`

- v2 vs v3 comparison on v3 test:
  - `reports/detect_benchmark_v3/detector_v2_vs_v3_on_v3test.md`
  - `reports/detect_benchmark_v3/detector_v2_vs_v3_on_v3test.json`

### Segmentation (strawberry mask) — production vs high-quality

Segmentation is trained as **1-class** (`0 = strawberry`) on `data/segmentation_project_dataset/` (via YOLO-seg conversion).
Ripeness labels are handled by a separate classifier and are **not** part of segmentation training.

#### Production segmentation (on-rover inference)

- **Model**: YOLOv8n-seg
- **Checkpoint**: `runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt`
- **Reason**:
  - Best **speed/quality tradeoff**
  - ~**29 FPS** on Orin (low latency)
  - Compact weights size
  - Sufficient mask quality for onboard use

#### High-quality segmentation (stereo/grasp / offline / “quality mode”)

- **Model**: YOLOv8s-seg
- **Checkpoint**: `runs/segment_benchmark/yolov8s_seg_benchmark/weights/best.pt`
- **Reason**:
  - Best mask quality on the common test split
  - Higher recall (useful for downstream stereo/grasp pipeline)

#### Not selected (for now)

- **Model**: YOLOv8m-seg
- **Checkpoint**: `runs/segment_benchmark/yolov8m_seg_benchmark/weights/best.pt`
- **Reason**:
  - Too heavy and too slow for the current quality gain

### Classification (ripeness) — production vs fast fallback

Classifier runs **after detection**:

- detector → bbox
- crop → classifier
- classifier → ripeness class + confidence

#### Production classifier (on-rover)

- **Model**: EfficientNet-B0
- **Checkpoint**: `runs/classification_benchmark_v2/efficientnet_b0/best.pt`
- **Reason**:
  - Best accuracy / macro metrics on `data/classification_dataset_v2/test/`
  - Still fast on Orin
  - Best overall speed/quality trade-off for ripeness classification

#### Fast fallback classifier

- **Model**: MobileNetV3-Small
- **Checkpoint**: `runs/classification_benchmark_v2/mobilenet_v3_small/best.pt`
- **Reason**:
  - Maximum FPS and very small weights
  - Useful as a lightweight backup option

#### Not selected (default)

- **Model**: ResNet-18
- **Checkpoint**: `runs/classification_benchmark_v2/resnet18/best.pt`
- **Reason**:
  - Worse quality than EfficientNet-B0 and much heavier than MobileNetV3-Small

