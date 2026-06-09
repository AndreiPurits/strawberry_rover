#!/usr/bin/env bash
# Intel RealSense D405 on Jetson Orin (ROS2 Foxy).
# Does NOT modify Orbbec drivers/packages (src/OrbbecSDK_ROS2 stays untouched).
#
# Usage:
#   bash bootstrap/realsense_d405_orin.sh
#   source /opt/ros/foxy/setup.bash && colcon build --packages-select realsense2_camera_msgs realsense2_camera
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LIBRS_TAG="v2.55.1"
RS_ROS_BRANCH="ros2-master"

echo "==> RealSense D405 install (Orin)"
echo "    repo: $REPO_ROOT"
echo "    Orbbec path (NOT touched): $REPO_ROOT/src/OrbbecSDK_ROS2"

if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing build dependencies..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    git cmake build-essential \
    libssl-dev libusb-1.0-0-dev libudev-dev pkg-config \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-dev python3-pip \
    ros-foxy-diagnostic-updater \
    ros-foxy-image-transport \
    ros-foxy-cv-bridge \
    || true
fi

if command -v rs-enumerate-devices >/dev/null 2>&1; then
  echo "==> librealsense already installed:"
  rs-enumerate-devices -s 2>/dev/null | head -5 || true
else
  echo "==> Building librealsense $LIBRS_TAG from source..."
  BUILD_DIR="/tmp/librealsense_build"
  rm -rf "$BUILD_DIR"
  git clone --depth 1 --branch "$LIBRS_TAG" https://github.com/IntelRealSense/librealsense.git "$BUILD_DIR"
  cmake -S "$BUILD_DIR" -B "$BUILD_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_WITH_CUDA=false
  cmake --build "$BUILD_DIR/build" -j"$(nproc)"
  sudo cmake --install "$BUILD_DIR/build"
  sudo ldconfig
fi

RS_ROS_DIR="$REPO_ROOT/src/realsense-ros"
if [ ! -d "$RS_ROS_DIR/.git" ]; then
  echo "==> Cloning realsense-ros into src/realsense-ros ..."
  git clone --depth 1 --branch "$RS_ROS_BRANCH" \
    https://github.com/IntelRealSense/realsense-ros.git "$RS_ROS_DIR"
else
  echo "==> realsense-ros already present: $RS_ROS_DIR"
fi

if [ -f "$RS_ROS_DIR/realsense2_camera/scripts/installudevrules.sh" ]; then
  echo "==> Installing RealSense udev rules..."
  sudo "$RS_ROS_DIR/realsense2_camera/scripts/installudevrules.sh" || true
fi

echo ""
echo "==> Next: rs-enumerate-devices -s"
echo "    colcon build --packages-select realsense2_camera_msgs realsense2_camera"
echo "    Orbbec unchanged: ros2 launch orbbec_camera gemini210.launch.py"
