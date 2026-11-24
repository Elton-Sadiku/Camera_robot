#!/usr/bin/env bash
# One-time setup for this project on Ubuntu 22.04 (safe to re-run).
#   ROS 2 Humble (desktop: rclpy, rviz2, tf2) + realsense2_camera
#   RealSense udev rule (camera access without root)
#   pyrealsense2 + CPU torch for the system Python 3.10 that ROS uses
# Usage: bash scripts/install_ubuntu.sh
set -euo pipefail

# never let conda's Python leak into the ROS install
while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate 2>/dev/null || break; done
PY=/usr/bin/python3

echo "== 1/4 ROS 2 apt repository"
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common curl
  sudo add-apt-repository -y universe
  V=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -fsSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.$(. /etc/os-release && echo "$VERSION_CODENAME")_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi

echo "== 2/4 ROS 2 Humble + realsense2_camera + Python tools"
sudo apt-get update
sudo apt-get install -y ros-humble-desktop ros-humble-realsense2-camera ros-humble-realsense2-description \
  python3-pip python3-numpy python3-colcon-common-extensions

echo "== 3/4 RealSense udev rule"
sudo curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
  -o /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "== 4/4 Python packages for system Python 3.10 (user site, no sudo)"
# numpy comes from apt (1.21) on purpose: ROS Humble's compiled packages expect numpy 1.x
$PY -m pip install --user --upgrade pyrealsense2
$PY -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu

echo
echo "Done. Replug the camera, then in a NEW terminal:  source $(cd "$(dirname "$0")/.." && pwd)/env.sh"
