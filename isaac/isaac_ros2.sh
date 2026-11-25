#!/usr/bin/env bash
# Start Isaac Sim 5.1 with its internal ROS 2 Humble bridge enabled (no need to source /opt/ros).
#   isaac/isaac_ros2.sh                              -> Isaac Sim GUI (then open "Isaac SIm/meshari_d455.usd", press Play)
#   isaac/isaac_ros2.sh isaac/meshari_d455.py [...]  -> run a script with Isaac's python
# Uses the same DDS settings as env.sh, so ROS 2 terminals (source ./env.sh) see the topics.
ISAACSIM="${ISAACSIM:-$HOME/isaacsim}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Isaac Sim (Python 3.11) must NOT see the system ROS (Python 3.10) or conda: ~/.bashrc sources
# /opt/ros/humble/setup.bash and conda in every terminal, so strip both from this process only.
strip() { echo "$1" | tr ':' '\n' | grep -v -E '^/opt/ros/|/(mini|ana)conda[^/]*/' | grep -v '^$' | paste -sd: ; }
PATH="$(strip "$PATH")"; export PATH
LD_LIBRARY_PATH="$(strip "${LD_LIBRARY_PATH:-}")"
unset PYTHONPATH PYTHONHOME AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PYTHON_VERSION \
      CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE _CE_CONDA _CE_M
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ISAACSIM/exts/isaacsim.ros2.bridge/humble/lib"
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/config/fastdds_shm.xml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
cd "$ROOT"
if [ $# -eq 0 ]; then
    exec "$ISAACSIM/isaac-sim.sh"
else
    exec "$ISAACSIM/python.sh" "$@"
fi
