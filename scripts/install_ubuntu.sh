#!/usr/bin/env bash
# One-time setup for this project on Ubuntu 22.04 (safe to re-run).
#   ROS 2 Humble (desktop: rclpy, rviz2, tf2) + realsense2_camera
#   RealSense udev rule (camera access without root)
#   egg detection: CPU torch + ultralytics 8.4.6 for the system Python 3.10 that ROS uses,
#   yolo_ros (mgonzs13, tag 4.7.1) built in ~/ros2_ws
# Usage: bash scripts/install_ubuntu.sh
set -euo pipefail

# never let conda's Python leak into the ROS install
while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate 2>/dev/null || break; done
PY=/usr/bin/python3

echo "== 1/5 ROS 2 apt repository"
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common curl
  sudo add-apt-repository -y universe
  V=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -fsSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.$(. /etc/os-release && echo "$VERSION_CODENAME")_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi

echo "== 2/5 ROS 2 Humble + realsense2_camera + Python tools"
sudo apt-get update
sudo apt-get install -y ros-humble-desktop ros-humble-realsense2-camera ros-humble-realsense2-description \
  python3-pip python3-numpy python3-colcon-common-extensions

echo "== 3/5 RealSense udev rule"
sudo curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
  -o /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "== 4/5 Python packages for system Python 3.10 (user site, no sudo)"
# numpy must stay 1.x (ROS Humble); packaging>=24.2 is needed by the newer setuptools that builds yolo_ros
$PY -m pip install --user torch "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install --user "ultralytics==8.4.6" "lap>=0.5.12" "numpy<2" "packaging>=24.2" ftfy regex tqdm
# ultralytics may replace torchvision with a CUDA build -> put the matching CPU build back
$PY -m pip install --user --no-deps --force-reinstall "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cpu

echo "== 5/5 yolo_ros 4.7.1 (the last Python/ultralytics release; it has use_3d) in ~/ros2_ws"
mkdir -p "$HOME/ros2_ws/src"
[ -d "$HOME/ros2_ws/src/yolo_ros" ] || git clone --depth 1 --branch 4.7.1 https://github.com/mgonzs13/yolo_ros.git "$HOME/ros2_ws/src/yolo_ros"
(cd "$HOME/ros2_ws" && env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c 'source /opt/ros/humble/setup.bash && colcon build --symlink-install')

echo
echo "Done. Replug the camera, then in a NEW terminal:  source $(cd "$(dirname "$0")/.." && pwd)/env.sh"
