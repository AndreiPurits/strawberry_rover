#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"

pkill -f "ros2 launch rover_bringup bringup.launch.py" || true
pkill -f "robot_state_publisher|rover_navigation_node|rover_pose_simulator|fake_lidar_node|fake_camera_node|fake_stereo_node|rviz2" || true
sleep 1

source /opt/ros/foxy/setup.bash
colcon build --packages-select \
  rover_description \
  rover_fake_camera \
  rover_fake_stereo \
  rover_simulation \
  rover_navigation \
  rover_fake_lidar \
  rover_bringup
source install/setup.bash

exec ros2 launch rover_bringup bringup.launch.py
