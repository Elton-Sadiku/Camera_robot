"""Isaac Sim 5.1: a simulated D455 publishing on the SAME ROS 2 topics / frames as the real one.

Use it to (1) compare sim vs real clouds side by side in RViz, (2) run ros2/policy_node.py against
the simulator before touching hardware. Full 424x240 is rendered here; the policy node decimates.
The camera uses the mount from shared.d455_model MOUNT_XYZ / MOUNT_RPY (meshari torso slot, from CAD);
the robot base is the world origin. The table below is a made-up example, not a measured table.

  export ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/isaacsim/exts/isaacsim.ros2.bridge/humble/lib  # only if ROS isn't sourced
  export ISAACLAB_ROOT=/path/to/IsaacLab
  "$ISAACLAB_ROOT/isaaclab.sh" -p /path/to/Camera_robot/isaac/sim_publish_d455.py [--headless --enable_cameras]
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
app.update()  # let the bridge register its writers before we use them

import omni.replicator.core as rep  # noqa: E402
import omni.syntheticdata  # noqa: E402
from omni.syntheticdata._syntheticdata import SensorType  # noqa: E402
import torch  # noqa: E402
from isaacsim.ros2.bridge import read_camera_info  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.d455_model import MOUNT_RPY, MOUNT_XYZ, intrinsics, mount_quat_ros  # noqa: E402

FRAME = "camera_depth_optical_frame"
NS, PREFIX = "camera/camera", "depth"   # -> /camera/camera/depth/...

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 90))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
# Test scene: HIGH table close to the robot (meshari arm reach, see detection/gen_egg_depth.py), along the
# mount's forward direction (yaw of MOUNT_RPY; -Y for meshari). Camera 1.15 m up, 5 deg down: a 1.0 m table
# top is visible from ~0.27 m ahead; the palm reaches ~0.45 m. Table 0.12-0.72 m ahead, egg 0.36 m ahead.
import math  # noqa: E402

fx_, fy_ = math.cos(MOUNT_RPY[2]), math.sin(MOUNT_RPY[2])          # forward unit vector in base_link
ahead = lambda d, side=0.0: (MOUNT_XYZ[0] + fx_ * d - fy_ * side, MOUNT_XYZ[1] + fy_ * d + fx_ * side)  # noqa: E731
table = sim_utils.CuboidCfg(size=(0.6, 1.1, 1.0) if abs(fx_) > abs(fy_) else (1.1, 0.6, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg())
table.func("/World/table", table, translation=(*ahead(0.42), 0.5))
egg = sim_utils.SphereCfg(radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.8)))
egg.func("/World/egg", egg, translation=(*ahead(0.36, 0.0), 1.025))

# camera: real full-res intrinsics, mount from shared (example values for now). Optical (ros) -> USD (opengl) quaternion.
K = intrinsics(decimation=1)
q_ros = torch.tensor([mount_quat_ros(*MOUNT_RPY)])
q_gl = convert_camera_frame_orientation_convention(q_ros, origin="ros", target="opengl")[0].tolist()
cam_cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K["K"], width=K["width"], height=K["height"],
                                                           clipping_range=(0.05, 20.0))
cam_cfg.func("/World/d455", cam_cfg, translation=MOUNT_XYZ, orientation=tuple(q_gl))

sim.reset()  # graphs must exist before render products / writers are attached
rp = rep.create.render_product("/World/d455", (K["width"], K["height"])).path
depth_rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(SensorType.DistanceToImagePlane.name)
common = dict(frameId=FRAME, nodeNamespace=NS, queueSize=1)
for writer_name, topic in [(depth_rv + "ROS2PublishImage", f"{PREFIX}/image_rect_raw"),          # 32FC1 metres
                           (depth_rv + "ROS2PublishPointCloud", f"{PREFIX}/color/points")]:
    w = rep.writers.get(writer_name)
    w.initialize(topicName=topic, **common)
    w.attach([rp])

info, _ = read_camera_info(render_product_path=rp)
w = rep.writers.get("ROS2PublishCameraInfo")
w.initialize(topicName=f"{PREFIX}/camera_info", width=info.width, height=info.height,
             projectionType=info.distortion_model, k=info.k, r=info.r, p=info.p,
             physicalDistortionModel=info.distortion_model,
             physicalDistortionCoefficients=info.d, **common)
w.attach([rp])
app.update()

while app.is_running():
    sim.step()
app.close()
