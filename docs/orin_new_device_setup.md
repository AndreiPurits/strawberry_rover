# Новый Jetson Orin — установка с нуля

Репозиторий: https://github.com/AndreiPurits/strawberry_rover

## 1. Клонирование

```bash
git clone git@github.com:AndreiPurits/strawberry_rover.git strawberry_rover_ws
cd strawberry_rover_ws
git submodule update --init --recursive
```

Submodules:
- `src/rplidar_ros` — LiDAR
- `src/OrbbecSDK_ROS2` — стереокамера Orbbec

## 2. Артефакты вне git (скопировать вручную)

| Что | Куда | Источник |
|-----|------|----------|
| Веса `best.pt` | `runs/` | Google Drive / старый Orin |
| Jetson torch wheel | `torch-2.1.0a0+41361538.nv23.06-*.whl` в корень | старый Orin |
| Датасет FPS holdout | `data/ФПС ДАТАСЕТ/` | старый Orin / Drive |
| `third_party/vision/` | опционально | vendored torchvision для Jetson |

Проверка весов:
```bash
python3 -c "import json; print(json.load(open('models/weights_manifest.json')))"
```

## 3. Bootstrap Python + CUDA

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
bash bootstrap/orin_bootstrap.sh
source .venv_cuda/bin/activate
source scripts/jetson_gpu_env.sh
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. TensorRT engines (на целевом Orin)

```bash
export PATH=/usr/src/tensorrt/bin:$PATH
COOLDOWN_SEC=60 bash tools/export_tensorrt/export_all_group02.sh
```

Engines → `runs/export_tensorrt/group02/` (gitignored).

## 5. ROS2 workspace

```bash
source /opt/ros/foxy/setup.bash   # или humble
colcon build
source install/setup.bash
```

Камеры:
- **Orbbec:** `ros2 launch orbbec_camera gemini210.launch.py`
- **RealSense D405:** `bash bootstrap/realsense_d405_orin.sh` → см. `docs/realsense_d405_orin.md`

## 6. Проверка vision pipeline

```bash
source .venv_cuda/bin/activate
source scripts/jetson_gpu_env.sh
python3 tools/benchmark_holdout_full_pipeline_fps.py \
  --holdout "data/ФПС ДАТАСЕТ/images" \
  --model-group 02_lightened_current \
  --preset very_fast \
  --limit 50
```

Ожидаемо PT ~23 FPS, TRT ~38 FPS (после export).

## 7. Production runtime (group 02 + TRT)

```bash
# детектор/сегментер — .engine из runs/export_tensorrt/group02/
python3 scripts/run_strawberry_ensemble.py \
  --detector runs/export_tensorrt/group02/detector_imgsz480.engine \
  --segmenter runs/export_tensorrt/group02/segmenter_imgsz320.engine
```

## Бенчмарк-результаты (старый Orin, 8099 кадров)

См. `reports/fps_dataset_trt_comparison.md`:
- PT very_fast: **23.1 FPS**
- TRT very_fast: **38.0 FPS** (+64%)
