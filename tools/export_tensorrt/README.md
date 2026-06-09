# TensorRT export (Jetson Orin)

Export YOLO detector/segmenter from `.pt` → ONNX → `.engine` for peak FPS @ 50W.

**Export must run on target Orin** (TensorRT version must match runtime).

## Prerequisites

```bash
source scripts/jetson_gpu_env.sh
pip install ultralytics onnx
# TensorRT comes with JetPack
```

## Export detector

```bash
python3 tools/export_tensorrt/export_yolo.py \
  --weights models/model_groups/02_lightened_current/detector_best.pt \
  --task detect \
  --imgsz 512 \
  --half \
  --outdir runs/export_tensorrt/detector_yolov8s_512
```

## Export segmenter

```bash
python3 tools/export_tensorrt/export_yolo.py \
  --weights models/model_groups/02_lightened_current/segmenter_best.pt \
  --task segment \
  --imgsz 320 \
  --half \
  --outdir runs/export_tensorrt/segmenter_yolov8n_320
```

## Export all presets (group 02)

Builds detector @ 416/480/512/640 and segmenter @ 256/320/384 with 45–60s cooldown between builds (thermal safety on Orin):

```bash
COOLDOWN_SEC=60 bash tools/export_tensorrt/export_all_group02.sh
```

Canonical engines: `runs/export_tensorrt/group02/{detector,segmenter}_imgsz{N}.engine`

## Full pipeline benchmark (PT vs TRT, all resolutions)

```bash
python3 tools/benchmark_fps_dataset_matrix.py \
  --holdout "data/ФПС ДАТАСЕТ/images" \
  --backends pt,trt \
  --presets baseline,fast,very_fast,ultra_low
```

Results: `runs/fps_dataset_benchmark_matrix/summary_matrix.{md,csv,json}` and `reports/fps_dataset_trt_comparison.md`

Presets:

| preset | det | seg | notes |
|--------|-----|-----|-------|
| baseline | 640 | 384 | current defaults |
| fast | 512 | 320 | |
| very_fast | 480 | 320 | recommended runtime |
| ultra_low | 416 | 256 | lowest resolution |

Engines are gitignored. Export must run on target Orin (TensorRT ABI match).
