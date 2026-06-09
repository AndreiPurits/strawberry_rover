# Intel RealSense D405 on Jetson Orin

Отдельная камера. **Orbbec не трогаем.**

| Камера | Пакет | Launch |
|--------|-------|--------|
| Orbbec Gemini 210/215 | `src/OrbbecSDK_ROS2/` | `ros2 launch orbbec_camera gemini210.launch.py` |
| RealSense D405 | `src/realsense-ros/` | `ros2 launch realsense2_camera rs_launch.py ...` |

## Установка

```bash
bash bootstrap/realsense_d405_orin.sh
source /opt/ros/foxy/setup.bash
cd ~/project/strawberry_rover_ws
colcon build --packages-select realsense2_camera_msgs realsense2_camera
source install/setup.bash
```

Проверка USB:

```bash
rs-enumerate-devices -s
```

## Запуск D405

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=realsense_d405 \
  enable_color:=true \
  enable_depth:=true \
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=1280x720x30
```

Топики по умолчанию:
- `/realsense_d405/color/image_raw`
- `/realsense_d405/depth/image_rect_raw`
- `/realsense_d405/color/camera_info`

## Ensemble pipeline с RealSense

`run_strawberry_ensemble.py` принимает топики через аргументы:

```bash
python3 scripts/run_strawberry_ensemble.py \
  --rgb-topic /realsense_d405/color/image_raw \
  --depth-topic /realsense_d405/depth/image_rect_raw \
  --camera-info-topic /realsense_d405/color/camera_info
```

Orbbec GUI-скрипт (`scripts/run_strawberry_gui_with_camera.sh`) **не меняется**.
