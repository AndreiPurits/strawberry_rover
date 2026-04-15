## rover_bringup

ROS 2 bringup package for the strawberry rover.

This package provides top-level launch files to orchestrate startup of rover components.

Current integrated launch:
- `bringup.launch.py` starts `robot_state_publisher` using `rover_description/urdf/rover.urdf`
- starts real `rplidar_ros/rplidar_node` (C1 profile) on `/dev/ttyUSB0` by default
- can switch to `rover_fake_lidar/fake_lidar_node` with `use_fake_lidar:=true`
- starts `rover_simulation/rover_pose_simulator` with `rover_simulation/config/rover_pose_simulator.yaml`
- starts `rover_fake_camera/fake_camera_node` with `rover_fake_camera/config/fake_camera.yaml`
- starts `rover_fake_stereo/fake_stereo_node` with `rover_fake_stereo/config/fake_stereo.yaml`
- optionally starts `rover_navigation/rover_navigation_node` with `rover_navigation/config/rover_navigation.yaml` when `enable_navigation:=true`
- optionally starts `rviz2` with `rover_bringup/config/field_sim_debug.rviz` when `use_rviz:=true`

Main outputs expected from integrated bringup:
- `/scan`
- `/sim/scene_markers`
- `/sim/rover_pose`
- `/sim/camera/front/image_raw`
- `/sim/camera/bottom_rgb/image_raw`
- `/sim/stereo/left/image_raw`
- `/sim/stereo/right/image_raw`
- `/sim/stereo/debug/combined`
- `/cmd_vel`
- `/debug/steering`
- `/debug/centerline`
- `/debug/heading`
- `/debug/row_confidence`
- `/tf_static` (and `/tf` interface availability in the running graph)

Launch examples:

```bash
ros2 launch rover_bringup bringup.launch.py
ros2 launch rover_bringup bringup.launch.py use_rviz:=true
ros2 launch rover_bringup bringup.launch.py use_rviz:=true enable_navigation:=true
ros2 launch rover_bringup bringup.launch.py lidar_serial_port:=/dev/ttyUSB0
ros2 launch rover_bringup bringup.launch.py use_fake_lidar:=true
```

