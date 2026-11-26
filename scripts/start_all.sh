#!/usr/bin/env bash
# One command for the whole project: real D455 + yolo_ros egg detection + egg X, Y, Z + image window
# (+ Isaac Sim with the egg twin). Ctrl+C (or closing this window) stops everything.
#
#   scripts/start_all.sh            # real camera + Isaac Sim twin (default)
#   scripts/start_all.sh real       # real camera only (with RViz)
#   scripts/start_all.sh sim        # Isaac Sim only (test egg, its own D455, yolo on /sim)
#
# Settings below: the desk test rig values. When the camera is on the robot, set MOUNT_ARGS="" (MOUNT_RPY is used)
# and TWIN_ARGS="--twin-real-z".
# Logs: /tmp/meshari_egg/*.log
set -u
MODE="${1:-twin}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT_ARGS="mount_pitch:=0.1169 mount_roll:=-0.0105"      # desk rig: measured 6.7 deg down, -0.6 deg roll
TWIN_ARGS="--twin-real-table 1.028"                         # desk rig: real table height in base_link
LOG=/tmp/meshari_egg
mkdir -p "$LOG"
PIDS=()

ros() {  # ros <name> <command...>: run in the ROS environment, in the background, logged
    local name=$1; shift
    (cd "$ROOT" && env -i HOME="$HOME" PATH=/usr/bin:/bin DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-}" \
        bash -c "source ./env.sh >/dev/null && exec $*") >"$LOG/$name.log" 2>&1 &
    PIDS+=($!)
    echo "  started $name  (log: $LOG/$name.log)"
}

stop_all() {
    echo; echo "stopping everything ..."
    for p in "${PIDS[@]}"; do pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
    pkill -f "[k]it/python/bin/python3 isaac/meshari_d455.py" 2>/dev/null
    sleep 2
    for p in "${PIDS[@]}"; do pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
    echo "done."
}
trap stop_all EXIT INT TERM HUP

echo "== meshari egg detection: mode '$MODE'"
case "$MODE" in
  real|twin)
    RVIZ=false; [ "$MODE" = real ] && RVIZ=true
    ros camera "ros2 launch ros2/d455.launch.py rviz:=$RVIZ $MOUNT_ARGS"
    sleep 6
    ros yolo "ros2 launch ros2/egg_yolo.launch.py"
    ros egg_xyz "python3 ros2/egg_xyz.py"
    sleep 4
    ros image "ros2 run rqt_image_view rqt_image_view /yolo/dbg_image"
    if [ "$MODE" = twin ]; then
        (cd "$ROOT" && isaac/isaac_ros2.sh isaac/meshari_d455.py --run --twin $TWIN_ARGS) >"$LOG/isaac.log" 2>&1 &
        PIDS+=($!)
        echo "  started isaac twin  (log: $LOG/isaac.log) - the Isaac Sim window opens in ~30 s"
    fi
    ;;
  sim)
    (cd "$ROOT" && isaac/isaac_ros2.sh isaac/meshari_d455.py --run --test-egg) >"$LOG/isaac.log" 2>&1 &
    PIDS+=($!)
    echo "  started isaac sim  (log: $LOG/isaac.log)"
    ros tf "ros2 launch ros2/d455.launch.py camera:=false"
    ros yolo "ros2 launch ros2/egg_yolo.launch.py source:=sim"
    ros egg_xyz "python3 ros2/egg_xyz.py --ros-args -r __ns:=/sim -p detections:=/sim/yolo/detections_3d"
    sleep 4
    ros image "ros2 run rqt_image_view rqt_image_view /sim/yolo/dbg_image"
    ;;
  *) echo "unknown mode '$MODE' (real | twin | sim)"; exit 1 ;;
esac

echo
echo "Running. Egg positions below (Ctrl+C or close this window to stop everything):"
echo
tail -n 0 -F "$LOG/egg_xyz.log" 2>/dev/null | grep --line-buffered -E "egg #|ignored" &
PIDS+=($!)
wait
