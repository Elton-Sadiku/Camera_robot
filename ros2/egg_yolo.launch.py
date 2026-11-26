"""Egg detection with yolo_ros (mgonzs13/yolo_ros 4.7.1, ultralytics) in 3D: RGB box + aligned depth -> X, Y, Z.

  ros2 launch ros2/egg_yolo.launch.py                   # real D455 (start ros2/d455.launch.py first)
  ros2 launch ros2/egg_yolo.launch.py source:=sim       # Isaac Sim D455 (meshari_d455.usd playing)
  options: threshold:=auto|0.2  device:=cpu|cuda:0  model:=<.pt>  model_type:=World|YOLO|YOLOE  target_frame:=base_link

Model: detection/yolo/egg_real_sim_yolov8s.pt = YOLOv8s trained on real D455 egg images (detection/record_egg_yolo.py,
label_egg_yolo.py, train_egg_yolo.py; + Isaac Sim images from gen_egg_rgb.py). Scores ~0.97 on unseen real frames.
The zero-shot fallback: model:=detection/yolo/yolov8s-worldv2-egg.pt model_type:=World threshold:=0.015.

Outputs (namespace /yolo for real, /sim/yolo for sim):
  detections      yolo_msgs/DetectionArray   2D boxes in the colour image
  tracking        yolo_msgs/DetectionArray   + stable ids (ByteTrack)
  detections_3d   yolo_msgs/DetectionArray   bbox3d.center = egg X, Y, Z in target_frame (metres)
  dbg_image       sensor_msgs/Image          colour image with the boxes
  dgb_bb_markers  visualization_msgs/MarkerArray   3D boxes for RViz
ros2/egg_xyz.py prints the eggs' X, Y, Z and distance and publishes egg/point (PointStamped).
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as L

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {  # colour image, depth aligned to it, its camera_info, depth units per metre, yolo namespace, threshold
    "real": ("/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw",
             "/camera/camera/aligned_depth_to_color/camera_info", "1000", "yolo", "0.5"),      # 16UC1 millimetres
    "sim": ("/sim/camera/camera/color/image_raw", "/sim/camera/camera/aligned_depth_to_color/image_raw",
            "/sim/camera/camera/aligned_depth_to_color/camera_info", "1", "sim/yolo", "0.5"),  # 32FC1 metres
}
# (0.5 for the trained egg model. Old notes, zero-shot yolov8s-worldv2 + "egg"): a real brown egg scores 0.02 (standing upright) to 0.85,
# the rendered sim egg only ~0.013 (clean CG looks less like the web photos it learned from); other objects in
# both scenes stay below 0.01. ros2/egg_xyz.py then also checks the egg's real size and its 20 cm - 1 m range. A YOLO trained on your own egg images would allow one higher threshold for both.


def yolo(context):
    src = L("source").perform(context)
    if src not in SOURCES:
        raise RuntimeError(f"source:={src} - use real or sim")
    image, depth, info, divisor, ns, default_thr = SOURCES[src]
    thr = L("threshold").perform(context)
    thr = default_thr if thr == "auto" else thr
    yolo_launch = Path(get_package_share_directory("yolo_bringup")) / "launch" / "yolo.launch.py"
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(str(yolo_launch)), launch_arguments={
        "model_type": L("model_type"), "model": L("model"), "device": L("device"), "threshold": thr,
        "input_image_topic": image, "input_depth_topic": depth, "input_depth_info_topic": info,
        "depth_image_units_divisor": divisor, "target_frame": L("target_frame"),
        "tracker": str(ROOT / "config" / "egg_bytetrack.yaml"),  # default ByteTrack drops scores < 0.25
        "use_3d": "True", "use_tracking": "True", "use_debug": "True", "namespace": ns,
        # best effort: works with both the RealSense driver and Isaac Sim publishers
        "image_reliability": "2", "depth_image_reliability": "2", "depth_info_reliability": "2",
    }.items())]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="real", description="real or sim"),
        DeclareLaunchArgument("model", default_value=str(ROOT / "detection" / "yolo" / "egg_real_sim_yolov8s.pt")),
        DeclareLaunchArgument("model_type", default_value="YOLO"),
        DeclareLaunchArgument("device", default_value="cpu", description="cpu (ROS python has CPU torch) or cuda:0"),
        DeclareLaunchArgument("threshold", default_value="auto", description="auto = 0.5 (trained egg model)"),
        DeclareLaunchArgument("target_frame", default_value="base_link"),
        OpaqueFunction(function=yolo),
    ])
