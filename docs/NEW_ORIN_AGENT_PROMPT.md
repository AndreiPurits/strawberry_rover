# Промпт для нового агента (Cursor на новом Orin)

Скопируй блок ниже в первое сообщение агенту:

---

Ты работаешь в репозитории **strawberry_rover** на Jetson Orin 8GB @ 50W.

**Repo:** https://github.com/AndreiPurits/strawberry_rover

**Прочитай первым:** `AGENTS.md` → `docs/orin_new_device_setup.md` → `CURRENT_STATE.md` → `reports/fps_dataset_trt_comparison.md`

---

## 1. Окружение (должно быть почти как на старом Orin)

Целевой стек (проверено на рабочем Orin):

| Компонент | Версия / путь |
|-----------|----------------|
| JetPack / L4T | R35.4.x (aarch64) |
| ROS2 | **Foxy** (`/opt/ros/foxy/setup.bash`) |
| Python | 3.8, venv `.venv_cuda` |
| PyTorch | `torch-2.1.0a0+41361538.nv23.06` — **только local wheel**, не `pip install torch` |
| TensorRT | 8.5.x (JetPack), `trtexec` в `/usr/src/tensorrt/bin/` |
| Ultralytics + OpenCV | через `bootstrap/requirements-orin.txt` |
| OpenBLAS | `libopenblas0-pthread` (apt) + `source scripts/jetson_gpu_env.sh` |
| PyQt5 | `python3-pyqt5` (apt), не pip |
| torchvision | опционально `third_party/vision/` (vendored Jetson build) |

**Установка с нуля (повторить порядок старого Orin):**

```bash
git clone git@github.com:AndreiPurits/strawberry_rover.git strawberry_rover_ws
cd strawberry_rover_ws
git submodule update --init --recursive   # Orbbec + rplidar_ros

# Скопировать с старого Orin (не в git):
#   torch-2.1.0a0+41361538.nv23.06-*.whl → корень репо
#   runs/ (все best.pt)
#   data/ФПС ДАТАСЕТ/ (опционально, для бенчмарка)

sudo nvpmodel -m 0 && sudo jetson_clocks
bash bootstrap/orin_bootstrap.sh
source .venv_cuda/bin/activate
source scripts/jetson_gpu_env.sh

python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import tensorrt; print('TRT', tensorrt.__version__)"

# ROS2 workspace
source /opt/ros/foxy/setup.bash
colcon build
source install/setup.bash

# TensorRT engines (на целевом Orin, ~1 ч)
export PATH=/usr/src/tensorrt/bin:$PATH
COOLDOWN_SEC=60 bash tools/export_tensorrt/export_all_group02.sh
```

**Проверка FPS (ожидаемо как на старом Orin):** PT very_fast ~23 FPS, TRT very_fast ~38 FPS на `data/ФПС ДАТАСЕТ/images`.

---

## 2. Стереокамера Orbbec Gemini

- Драйвер: submodule `src/OrbbecSDK_ROS2/` (не трогать при установке RealSense)
- Launch: `ros2 launch orbbec_camera gemini210.launch.py camera_name:=camera`
- Топики:
  - `/camera/color/image_raw`
  - `/camera/depth/image_raw`
  - `/camera/color/camera_info`

**Рабочий профиль (проверен в `scripts/run_all.sh`):**
- Color: 1280×720 @ 30 MJPG
- Depth: 1280×800 @ 30 Y16
- `depth_registration:=true`, `align_mode:=SW` (стабильный старт depth)

**GUI + ML overlay:**
```bash
./scripts/run_strawberry_gui_with_camera.sh
# или полный цикл: ./scripts/run_all.sh
```

Vision pipeline читает RGB+Depth из ROS: `scripts/run_strawberry_ensemble.py --rgb-topic /camera/color/image_raw --depth-topic /camera/depth/image_raw`

**RealSense D405** — отдельная камера, отдельный install: `bash bootstrap/realsense_d405_orin.sh`, см. `docs/realsense_d405_orin.md`. Orbbec launch не менять.

---

## 3. ROS2

- Дистрибутив: **Foxy**
- Workspace: корень репо (`colcon build` → `install/setup.bash`)
- Bringup всего стека (симуляция + сенсоры):
  ```bash
  source /opt/ros/foxy/setup.bash && source install/setup.bash
  ros2 launch rover_bringup bringup.launch.py
  ```
- Пакеты: `src/rover_*`, `src/roarm_ros2_http/`, `src/rplidar_ros/` (submodule), `src/OrbbecSDK_ROS2/` (submodule)
- Web UI: `ros2 launch rover_web_interface web_interface.launch.py`

Для **реального поля** (не симуляция): камера Orbbec отдельно (`gemini210.launch.py`), LiDAR через bringup с `use_fake_lidar:=false`.

---

## 4. LiDAR (RPLidar C1)

- Драйвер: submodule `src/rplidar_ros/`
- Launch: `rplidar_c1_launch.py` (включается из `rover_bringup`)
- По умолчанию: `lidar_serial_port:=/dev/ttyUSB0`, frame `lidar_link`
- Реальный LiDAR:
  ```bash
  ros2 launch rover_bringup bringup.launch.py use_fake_lidar:=false lidar_serial_port:=/dev/ttyUSB0
  ```
- udev: `src/rplidar_ros/scripts/create_udev_rules.sh` (если порт не виден)
- Проверка: `ros2 topic echo /scan --once`

---

## 5. Роболапа RoArm-M3

Два режима:

**A) Без ROS (диагностика):**
```bash
python3 tools/roarm_local_gui/roarm_test_gui.py
```
- HTTP к ESP32: `GET http://<ip>/js` и `GET http://<ip>/js?json={...}`
- IP по умолчанию в GUI: `192.168.1.87` (уточнить на железе)
- Док: `tools/roarm_local_gui/README.md`

**B) ROS2 bridge:**
```bash
colcon build --packages-select roarm_ros2_http
ros2 launch roarm_ros2_http selftest.launch.py roarm_ip:=<IP> mode:=status
```
- Топик цели: `/roarm/target_pose` (`geometry_msgs/PoseStamped`)
- Safety box: x∈[0.05,0.35], y∈[-0.20,0.20], z∈[0.02,0.30]
- Док: `src/roarm_ros2_http/README.md`

---

## 6. Arduino Mega (моторы шасси)

- Скетч: `arduino/MEGA_Rover_4x_RC_PWM/MEGA_Rover_4x_RC_PWM.ino`
- Прошить через Arduino IDE, порт обычно `/dev/ttyACM0`
- Serial **115200**, формат команды:
  ```
  M FL=1500 FR=1500 RL=1500 RR=1500
  ```
- PWM 1000..2000 µs на пинах D5–D8 (FL/FR/RL/RR)
- Failsafe: нейтраль 1500 µs при таймауте 500 ms
- Mega **не** крутит моторы напрямую — только RC-PWM входы контроллеров

---

## Vision / ML (кратко)

- Pipeline: detector → crop → classifier → segmenter → depth
- Production group 02: `models/model_groups/02_lightened_current/`
- Веса в `runs/` — **не в git**, manifest: `models/weights_manifest.json`
- TRT very_fast: **38 FPS** (det 480 + seg 320 engines в `runs/export_tensorrt/group02/`)

---

## Правила агента

- Не обучать и не менять датасеты без явного запроса
- Не перезаписывать старые `best.pt`
- Не коммитить `data/`, `runs/`, `*.pt`, `*.engine`
- **Не трогать Orbbec** при работе с RealSense
- Benchmark только на holdout `data/ФПС ДАТАСЕТ/`

**Текущая задача:** [опиши — например: «воспроизвести окружение старого Orin, поднять Orbbec + LiDAR + проверить TRT FPS»]

---
