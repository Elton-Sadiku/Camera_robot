# source env.sh  - per-terminal environment for this project (camera, ROS 2 nodes, RViz).
# Leaves your conda setup alone; just steps out of it in THIS shell, because ROS 2 Humble
# needs the system Python 3.10 and conda's base is Python 3.14.
while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate 2>/dev/null || break; done
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v -E '/(mini|ana)conda[^/]*/' | paste -sd:)"; export PATH
hash -r
D455_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
# yolo_ros (mgonzs13, tag 4.7.1) built in ~/ros2_ws - egg detection
[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$D455_ROOT/config/fastdds_shm.xml"   # needed for 90 Hz depth
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
cd "$D455_ROOT"
echo "D455 env: $(python3 --version), ROS_DISTRO=$ROS_DISTRO, DDS profile set, cwd=$D455_ROOT"
