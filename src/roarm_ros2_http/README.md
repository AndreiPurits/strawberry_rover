# roarm_ros2_http

Minimal ROS2 MVP package for Waveshare RoArm-M3 via HTTP API.

Current assumptions:
- target coordinates are already in arm base frame;
- camera is not integrated yet;
- `/roarm/target_pose` remains stable for future stereo/perception publisher.

## Why GET is used instead of POST

This hardware/firmware accepts commands through:
- `GET /js?json=...`

Working low-level example:
- `curl -g 'http://<ip>/js?json={"T":100}'`

`roarm_client.py` uses only curl transport:
- command: `curl -g 'http://<ip>/js?json={...}'`
- status: `curl 'http://<ip>/js'`

RoArm-M3 uses curl transport due to requests incompatibility with ESP32 server.

## What this package does

- `roarm_http_driver` subscribes to `/roarm/target_pose` (`PoseStamped`) and sends safe `move_xyz` commands.
- `safety.py` blocks out-of-range targets:
  - `x: [0.05, 0.35]`
  - `y: [-0.20, 0.20]`
  - `z: [0.02, 0.30]`
- `demo_pick_sequence` runs: `home -> pregrasp -> grasp -> close_gripper -> retreat -> home`.
- `roarm_selftest` supports:
  - `status` mode (3 status reads, no motion)
  - `transport` mode (`get_status`, `go_home`, `open_gripper`, `close_gripper`)
  - `axes` mode (small `+/-X, +/-Y, +/-Z` moves with status reads)

## Step-by-step

1. Build:
   ```bash
   colcon build --packages-select roarm_ros2_http
   ```
2. Source:
   ```bash
   source install/setup.bash
   ```
3. Launch self-test in safe order:
   - status first (no movement):
   ```bash
   ros2 launch roarm_ros2_http selftest.launch.py roarm_ip:=192.168.4.1 mode:=status
   ```
   - then transport:
   ```bash
   ros2 launch roarm_ros2_http selftest.launch.py roarm_ip:=192.168.4.1 mode:=transport
   ```
4. Manual target publish:
   ```bash
   ros2 topic pub --once /roarm/target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'roarm_base'}, pose: {position: {x: 0.16, y: 0.0, z: 0.12}, orientation: {w: 1.0}}}"
   ```
5. Axes test last:
   ```bash
   ros2 launch roarm_ros2_http selftest.launch.py roarm_ip:=192.168.4.1 mode:=axes
   ```

## Optional runs

- Driver + demo pick:
  ```bash
  ros2 launch roarm_ros2_http demo_pick.launch.py roarm_ip:=192.168.4.1
  ```
- Fake target stream:
  ```bash
  ros2 run roarm_ros2_http fake_target_publisher
  ```

## Минимальная проверка (без ROS2)

Поведение как в официальном Python-примере Waveshare:
- статус: `GET http://<ip>/js`
- команда: `GET http://<ip>/js?json={"T":100}`

Запуск:
```bash
python3 src/roarm_ros2_http/roarm_ros2_http/smoke_test.py
```
или через entrypoint:
```bash
ros2 run roarm_ros2_http smoke_test
```

