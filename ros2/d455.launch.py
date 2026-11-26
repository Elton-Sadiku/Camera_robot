"""ros2 launch ros2/d455.launch.py [camera:=true rviz:=true base_frame:=base_link mount_pitch:=0.2016 mount_roll:=0.0]

Real D455 (official realsense2_camera driver) + the robot -> camera TFs + RViz.

  camera:=true (default)  colour 640x480 + depth 424x240 @30 Hz, depth ALIGNED to colour (what yolo_ros use_3d
                          needs), temporal filter, point cloud. Topics:
        /camera/camera/color/image_raw                    rgb8
        /camera/camera/aligned_depth_to_color/image_raw   16UC1 millimetres, same pixels as the colour image
        /camera/camera/aligned_depth_to_color/camera_info
        /camera/camera/depth/image_rect_raw, /camera/camera/depth/color/points
  camera:=false           no real camera: only the TFs + RViz (for the Isaac Sim D455 on /sim/...)

TFs:
  base_frame -> camera_link    mount from shared/d455_model.py MOUNT_XYZ / MOUNT_RPY (same as Isaac Sim);
                               override with mount_x/y/z/roll/pitch/yaw:=... (radians)
  camera_link -> camera_depth_optical_frame / camera_color_optical_frame
        camera:=true  -> published by realsense2_camera (factory calibration of your unit)
        camera:=false -> static: depth rpy -90/0/-90, colour from config/sim_color_extrinsics.json (the sim camera)
"""
import json
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as L
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.d455_model import MOUNT_RPY, MOUNT_XYZ  # noqa: E402

MOUNT = {f"mount_{k}": str(v) for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), (*MOUNT_XYZ, *MOUNT_RPY))}
EXT = HERE.parent / "config" / "sim_color_extrinsics.json"   # written by isaac/meshari_d455.py
COLOR_EXT = json.loads(EXT.read_text()) if EXT.exists() else {"xyz": [0.0, -0.059, 0.0],
                                                             "quat_xyzw": [-0.5, 0.5, -0.5, 0.5]}


def generate_launch_description():
    args = [DeclareLaunchArgument(k, default_value=v)
            for k, v in {**MOUNT, "base_frame": "base_link", "rviz": "true", "camera": "true"}.items()]
    tf = lambda a, parent, child, **kw: Node(package="tf2_ros", executable="static_transform_publisher",  # noqa: E731
                                             arguments=[*a, "--frame-id", parent, "--child-frame-id", child], **kw)
    x, y, z = (str(v) for v in COLOR_EXT["xyz"])
    qx, qy, qz, qw = (str(v) for v in COLOR_EXT["quat_xyzw"])
    rs_launch = Path(get_package_share_directory("realsense2_camera")) / "launch" / "rs_launch.py"
    no_camera = UnlessCondition(L("camera"))
    return LaunchDescription(args + [
        tf(["--x", L("mount_x"), "--y", L("mount_y"), "--z", L("mount_z"),
            "--roll", L("mount_roll"), "--pitch", L("mount_pitch"), "--yaw", L("mount_yaw")],
           L("base_frame"), "camera_link"),
        tf(["--roll", "-1.5707963", "--pitch", "0", "--yaw", "-1.5707963"],
           "camera_link", "camera_depth_optical_frame", condition=no_camera),
        tf(["--x", x, "--y", y, "--z", z, "--qx", qx, "--qy", qy, "--qz", qz, "--qw", qw],
           "camera_link", "camera_color_optical_frame", condition=no_camera),
        GroupAction(forwarding=False, condition=IfCondition(L("camera")), actions=[  # don't hand our args to rs_launch
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(rs_launch)), launch_arguments={
                "camera_namespace": "camera", "camera_name": "camera",
                "enable_color": "true", "rgb_camera.color_profile": "640,480,30",
                "enable_depth": "true", "depth_module.depth_profile": "424,240,30",   # 424x240: min depth ~16 cm
                "align_depth.enable": "true", "enable_sync": "true",
                "temporal_filter.enable": "true", "pointcloud.enable": "true",
                "enable_infra1": "false", "enable_infra2": "false"}.items())]),
        # base_link fixed frame, D455 clouds, yolo egg boxes (MarkerArray), egg points, TF axes
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(HERE.parent / "config" / "d455.rviz")],
             condition=IfCondition(L("rviz"))),
    ])
