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

## 3. Окружение (как на старом Orin)

| Компонент | Значение |
|-----------|----------|
| JetPack | R35.4.x |
| ROS2 | Foxy |
| Python | 3.8 + `.venv_cuda` |
| PyTorch | wheel `torch-2.1.0a0+41361538.nv23.06` в корне репо |
| TensorRT | 8.5.x (JetPack) |

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
bash bootstrap/orin_bootstrap.sh
source .venv_cuda/bin/activate
source scripts/jetson_gpu_env.sh
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import tensorrt; print(tensorrt.__version__)"
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

### Стереокамера Orbbec

```bash
ros2 launch orbbec_camera gemini210.launch.py camera_name:=camera \
  depth_registration:=true align_mode:=SW \
  color_width:=1280 color_height:=720 color_fps:=30 color_format:=MJPG \
  depth_width:=1280 depth_height:=800 depth_fps:=30 depth_format:=Y16
```

Топики: `/camera/color/image_raw`, `/camera/depth/image_raw`

GUI + ML: `./scripts/run_strawberry_gui_with_camera.sh` или `./scripts/run_all.sh`

### RealSense D405 (отдельно, Orbbec не трогать)

`bash bootstrap/realsense_d405_orin.sh` → `docs/realsense_d405_orin.md`

### LiDAR RPLidar C1

```bash
# udev при необходимости:
bash src/rplidar_ros/scripts/create_udev_rules.sh
ros2 launch rover_bringup bringup.launch.py use_fake_lidar:=false lidar_serial_port:=/dev/ttyUSB0
```

### RoArm-M3

- Без ROS: `python3 tools/roarm_local_gui/roarm_test_gui.py`
- ROS2: `ros2 launch roarm_ros2_http selftest.launch.py roarm_ip:=<IP> mode:=status`

### Arduino Mega

Прошить `arduino/MEGA_Rover_4x_RC_PWM/MEGA_Rover_4x_RC_PWM.ino`, serial 115200, команды `M FL=.. FR=.. RL=.. RR=..`

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
