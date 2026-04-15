## Segmentation benchmark (fair test split)

- test split: `/home/andrei/project/strawberry_rover_ws/data/yolo_segmentation_dataset/images/test`
- data.yaml: `/home/andrei/project/strawberry_rover_ws/data/yolo_segmentation_dataset/data.yaml`
- imgsz: `640`  device: `0`

| model | status | epochs | mask mAP50-95 | mask R | fps | ms/img | size (MB) | weights |
|---|---|---:|---:|---:|---:|---:|---:|---|
| yolov8n-seg | completed | 150 | 0.8188 | 0.8777 | 29.16 | 34.30 | 6.5 | `/home/andrei/project/strawberry_rover_ws/runs/segment_benchmark/yolov8n_seg_benchmark/weights/best.pt` |
| yolov8s-seg | partial | 138 | 0.8363 | 0.9047 | 19.23 | 52.00 | 22.8 | `/home/andrei/project/strawberry_rover_ws/runs/segment_benchmark/yolov8s_seg_benchmark/weights/best.pt` |
| yolov8m-seg | partial | 66 | 0.8236 | 0.9107 | 10.03 | 99.73 | 156.5 | `/home/andrei/project/strawberry_rover_ws/runs/segment_benchmark/yolov8m_seg_benchmark/weights/best.pt` |

### Answers
1. Best mask quality (mAP50-95, recall): **yolov8s-seg**
2. Best speed (FPS): **yolov8n-seg**
3. Best speed/quality tradeoff: **yolov8n-seg**

### Recommendations
- Production segmentation default: **yolov8n-seg** (balanced).
- Stereo/grasp pipeline candidate: **yolov8n-seg** (prefers recall + latency).

### Notes on incomplete runs
- yolov8s-seg: run_status=`partial` (epochs_completed=138), evaluated by `best.pt`.
- yolov8m-seg: run_status=`partial` (epochs_completed=66), evaluated by `best.pt`.

### Do we need to continue training s/m?
- If a partial model is already best or close (< ~0.5% mAP50-95 gap), further training may not help and can overfit; consider stopping early.
- If quality gap is significant and latency is acceptable, continue training with early stopping enabled.
