"""ros2 launch ros2/d455.launch.py [mount_pitch:=0.35 base_frame:=base_link rviz:=false camera:=false disparity_shift:=50]

Starts the D455 node + the two static TFs:
  base_frame -> camera_link                (mount defaults from shared.d455_model MOUNT_XYZ/MOUNT_RPY, same as Isaac)
  camera_link -> camera_depth_optical_frame (fixed rpy -90/0/-90 deg = shared.d455_model.LINK_TO_OPTICAL)
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as L
from launch_ros.actions import Node

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.d455_model import DISPARITY_SHIFT, MOUNT_RPY, MOUNT_XYZ  # noqa: E402
MOUNT = {f"mount_{k}": str(v) for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), (*MOUNT_XYZ, *MOUNT_RPY))}


def generate_launch_description():
    args = [DeclareLaunchArgument(k, default_value=v)
            for k, v in {**MOUNT, "base_frame": "base_link", "rviz": "true", "camera": "true",
                              "disparity_shift": str(DISPARITY_SHIFT)}.items()]
    tf = lambda a, parent, child: Node(package="tf2_ros", executable="static_transform_publisher",  # noqa: E731
                                       arguments=[*a, "--frame-id", parent, "--child-frame-id", child])
    return LaunchDescription(args + [
        tf(["--x", L("mount_x"), "--y", L("mount_y"), "--z", L("mount_z"),
            "--roll", L("mount_roll"), "--pitch", L("mount_pitch"), "--yaw", L("mount_yaw")],
           L("base_frame"), "camera_link"),
        tf(["--roll", "-1.5707963", "--pitch", "0", "--yaw", "-1.5707963"],
           "camera_link", "camera_depth_optical_frame"),
        # camera:=false -> only TFs + RViz (e.g. to view the Isaac Sim D455 on /sim/... without the real camera)
        ExecuteProcess(cmd=["python3", str(HERE / "d455_node.py"), "--ros-args",
                             "-p", ["disparity_shift:=", L("disparity_shift")]], output="screen",
                       condition=IfCondition(L("camera"))),
        # base_link fixed frame, D455 cloud (Best Effort QoS, colored by forward distance X), TF axes
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(HERE.parent / "config" / "d455.rviz")],
             condition=IfCondition(L("rviz"))),
    ])
