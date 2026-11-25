"""Isaac Sim 5.1: mount the D455 in the torso slot of the meshari robot and publish it on ROS 2.

What it does
  1. opens  "Isaac SIm/meshari 1.usd"   (your robot file; it is NOT modified)
  2. puts a D455 in the rectangular camera slot of the torso, as a child of base_link
     (so the camera moves with the robot), looking forward (-Y), pitch adjustable
  3. adds a depth camera with the REAL D455 intrinsics (config/d455_intrinsics.json, 424x240)
     at the left-imager position, where the real D455 has its depth origin
  4. adds an OmniGraph that, on Play, publishes the same message types / frame ids as the real
     camera (ros2/d455_node.py), under the namespace /sim so sim and real can run side by side:
        /sim/camera/camera/depth/image_rect_raw    sensor_msgs/Image 32FC1 (metres)
        /sim/camera/camera/depth/camera_info       sensor_msgs/CameraInfo
        /sim/camera/camera/depth/color/points      sensor_msgs/PointCloud2
        /sim/camera/camera/color/image_raw         sensor_msgs/Image rgb8 (sim extra, 640x360)
     the real D455 stays on /camera/camera/... ; --ns camera/camera puts the sim on the real topics
     (then run only one of them). frame ids: camera_depth_optical_frame / camera_color_optical_frame
     The robot's existing ActionGraph keeps publishing /joint_states and listening on /joint_command.
  5. saves everything as  "Isaac SIm/meshari_d455.usd"  -> open it in the GUI and press Play.

Run (from the Camera_robot folder). isaac/isaac_ros2.sh sets ROS_DISTRO + the bridge libs, no /opt/ros needed:
  isaac/isaac_ros2.sh isaac/meshari_d455.py                  # build + save meshari_d455.usd, then exit
  isaac/isaac_ros2.sh isaac/meshari_d455.py --run            # build, save, and publish (GUI window)
  isaac/isaac_ros2.sh isaac/meshari_d455.py --run --headless # same without a window
Check from another terminal:  source ./env.sh && ros2 topic hz /sim/camera/camera/depth/image_rect_raw
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Camera_robot/
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "Isaac SIm" / "meshari 1.usd"))
ap.add_argument("--out", default=str(ROOT / "Isaac SIm" / "meshari_d455.usd"))
ap.add_argument("--pitch-deg", type=float, default=5.0,
                help="camera tilt, positive = down (default 5, keep equal to shared/d455_model.py MOUNT_RPY pitch)")
ap.add_argument("--run", action="store_true", help="after saving, press Play and keep publishing")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--ns", default="sim/camera/camera",
                help="ROS 2 namespace of the sim camera (default sim/camera/camera; the real D455 uses camera/camera)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless or not args.run})

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
app.update()

import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# ============================================================================
# Mount (world = base_link frame of this robot: X right->left arm, -Y forward, Z up)
# Measured from the torso CAD by a depth scan: the camera window is 12 cm wide,
# x 0.071..0.190 m, centre z = 1.153 m, window rim at y = -0.043..-0.054 m.
# ============================================================================
ROBOT = "/World/Sim"
PARENT_LINK = ROBOT + "/base_link"         # torso is fixed to base_link
SLOT_CENTER = (0.1306, -0.043, 1.153)      # front-glass centre of the D455, metres
FORWARD_YAW_DEG = -90.0                    # D455 asset looks along +X; rotate it to look along -Y
MOUNT = PARENT_LINK + "/camera_link"       # = ROS camera_link (x fwd, y left, z up)
D455_ASSET = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
DEPTH_ORIGIN_IN_LINK = (0.0, 0.0475, 0.0)  # left imager: the real D455's depth origin
COLOR_RES = (640, 360)

NS = args.ns.strip("/")
DEPTH_FRAME, COLOR_FRAME = "camera_depth_optical_frame", "camera_color_optical_frame"
GRAPH = "/World/D455_ROS2"


def set_rt(prim, pos, rpy_deg):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy_deg))


def to_parent(stage, parent_path, world_pos):
    """World point -> parent-local point (base_link is at the origin here, but stay general)."""
    m = UsdGeom.Xformable(stage.GetPrimAtPath(parent_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return m.GetInverse().Transform(Gf.Vec3d(*world_pos))


def mount_camera(stage):
    link = UsdGeom.Xform.Define(stage, MOUNT).GetPrim()
    set_rt(link, to_parent(stage, PARENT_LINK, SLOT_CENTER), (0.0, args.pitch_deg, FORWARD_YAW_DEG))

    # visual D455 (official NVIDIA model, real size 124 x 29 x 26 mm, origin = front glass)
    body = stage.DefinePrim(MOUNT + "/D455", "Xform")
    body.GetReferences().ClearReferences()
    body.GetReferences().AddReference(get_assets_root_path() + D455_ASSET)
    # The asset ships its own rigid body (+ mass) and colliders. Nested inside the base_link rigid body
    # that is an error ("missing xformstack reset"), and only disabling it is not enough, so REMOVE the
    # physics schemas (a list-op delete in this file; the NVIDIA asset itself is untouched).
    # The camera is then simply part of base_link and moves with the robot.
    from pxr import PhysxSchema

    for p in Usd.PrimRange(body):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, PhysxSchema.PhysxRigidBodyAPI,
                    UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, PhysxSchema.PhysxCollisionAPI):
            if p.HasAPI(api):
                p.RemoveAPI(api)
    # The asset's IMU sensor needs that rigid body ("did not match any rigid bodies"); the IMU is not
    # used by this project, so deactivate it.
    for p in list(Usd.PrimRange(body)):
        if p.GetTypeName() == "IsaacImuSensor":
            p.SetActive(False)
    color = next((p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Camera) and "color" in p.GetName().lower()), None)

    # depth camera with the real calibration (Omniverse has no principal-point offset: cx, cy
    # are taken as the image centre; the real ones are 1.5 px off, which is negligible)
    cal = json.loads((ROOT / "config" / "d455_intrinsics.json").read_text())
    w, h, fx = cal["width"], cal["height"], cal["fx"]
    pix = 1.93 / fx                               # mm per pixel, keeps the asset's 1.93 mm focal length
    cam = UsdGeom.Camera.Define(stage, MOUNT + "/" + DEPTH_FRAME)
    cam.CreateFocalLengthAttr(fx * pix)
    cam.CreateHorizontalApertureAttr(w * pix)
    cam.CreateVerticalApertureAttr(h * pix)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    # camera_link (x fwd, y left, z up) -> USD camera (looks -Z, +Y up): rows = images of cam x, y, z
    rot = Gf.Matrix4d(0, -1, 0, 0,   0, 0, 1, 0,   -1, 0, 0, 0,   *DEPTH_ORIGIN_IN_LINK, 1)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(rot)
    hfov = math.degrees(2 * math.atan(w / 2 / fx))
    print(f"[d455] depth camera {w}x{h}, fx={fx:.2f} px, HFOV {hfov:.1f} deg -> {cam.GetPath()}")
    return cam.GetPath().pathString, (color.GetPath().pathString if color else None), (w, h)


def build_graph(depth_cam, color_cam, depth_res):
    keys = og.Controller.Keys
    nodes = [("Tick", "omni.graph.action.OnPlaybackTick"),
             ("Context", "isaacsim.ros2.bridge.ROS2Context"),
             ("DepthRP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
             ("Depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
             ("Points", "isaacsim.ros2.bridge.ROS2CameraHelper"),
             ("DepthInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
    values = [("DepthRP.inputs:cameraPrim", [Sdf.Path(depth_cam)]),
              ("DepthRP.inputs:width", depth_res[0]), ("DepthRP.inputs:height", depth_res[1]),
              ("Depth.inputs:type", "depth"), ("Depth.inputs:topicName", "depth/image_rect_raw"),
              ("Points.inputs:type", "depth_pcl"), ("Points.inputs:topicName", "depth/color/points"),
              ("DepthInfo.inputs:topicName", "depth/camera_info")]
    edges = [("Tick.outputs:tick", "DepthRP.inputs:execIn")]
    for n in ("Depth", "Points", "DepthInfo"):
        values += [(f"{n}.inputs:nodeNamespace", NS), (f"{n}.inputs:frameId", DEPTH_FRAME), (f"{n}.inputs:queueSize", 1)]
        edges += [("DepthRP.outputs:execOut", f"{n}.inputs:execIn"),
                  ("DepthRP.outputs:renderProductPath", f"{n}.inputs:renderProductPath"),
                  ("Context.outputs:context", f"{n}.inputs:context")]
    if color_cam:
        nodes += [("ColorRP", "isaacsim.core.nodes.IsaacCreateRenderProduct"), ("Color", "isaacsim.ros2.bridge.ROS2CameraHelper")]
        values += [("ColorRP.inputs:cameraPrim", [Sdf.Path(color_cam)]),
                   ("ColorRP.inputs:width", COLOR_RES[0]), ("ColorRP.inputs:height", COLOR_RES[1]),
                   ("Color.inputs:type", "rgb"), ("Color.inputs:topicName", "color/image_raw"),
                   ("Color.inputs:nodeNamespace", NS), ("Color.inputs:frameId", COLOR_FRAME), ("Color.inputs:queueSize", 1)]
        edges += [("Tick.outputs:tick", "ColorRP.inputs:execIn"), ("ColorRP.outputs:execOut", "Color.inputs:execIn"),
                  ("ColorRP.outputs:renderProductPath", "Color.inputs:renderProductPath"),
                  ("Context.outputs:context", "Color.inputs:context")]
    og.Controller.edit({"graph_path": GRAPH, "evaluator_name": "execution"},
                       {keys.CREATE_NODES: nodes, keys.SET_VALUES: values, keys.CONNECT: edges})


def report_mount(stage, depth_cam):
    """ROS camera_link pose in base_link = shared.d455_model MOUNT_XYZ / MOUNT_RPY.
    As in realsense-ros, ROS camera_link sits at the depth origin (left imager), not at the glass centre."""
    cam_m = UsdGeom.Xformable(stage.GetPrimAtPath(depth_cam)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    base_m = UsdGeom.Xformable(stage.GetPrimAtPath(PARENT_LINK)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = (cam_m * base_m.GetInverse()).ExtractTranslation()
    print(f"[d455] ROS camera_link (depth origin) in base_link: xyz=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m, "
          f"rpy=(0, {math.radians(args.pitch_deg):.4f}, {math.radians(FORWARD_YAW_DEG):.4f}) rad")
    look = UsdGeom.Xformable(stage.GetPrimAtPath(depth_cam)).ComputeLocalToWorldTransform(0).TransformDir(Gf.Vec3d(0, 0, -1))
    print(f"[d455] depth camera looks along world ({look[0]:.2f}, {look[1]:.2f}, {look[2]:.2f})")


omni.usd.get_context().open_stage(args.src)
for _ in range(5):
    app.update()
stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath(PARENT_LINK).IsValid():
    raise SystemExit(f"{PARENT_LINK} not found in {args.src}")
for old in (MOUNT, GRAPH):  # re-running replaces the previous camera / graph
    if stage.GetPrimAtPath(old).IsValid():
        stage.RemovePrim(old)


def ensure_floor_and_light(stage):
    """The source file has no floor and only a distant light: the camera would see empty space
    (depth = inf everywhere, black RGB). Add a floor at z = 0 and a dome light if missing."""
    from pxr import UsdLux

    if not stage.GetPrimAtPath("/World/GroundPlane").IsValid():
        from isaacsim.core.api.objects import GroundPlane

        GroundPlane("/World/GroundPlane", z_position=0.0)
        print("[d455] added floor /World/GroundPlane at z = 0")
    if not any(p.IsA(UsdLux.DomeLight) for p in stage.Traverse()):
        UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)
        print("[d455] added /World/DomeLight")


ensure_floor_and_light(stage)
depth_cam, color_cam, depth_res = mount_camera(stage)
build_graph(depth_cam, color_cam, depth_res)
report_mount(stage, depth_cam)
omni.usd.get_context().save_as_stage(args.out)
print(f"[d455] saved {args.out}  (source file untouched: {args.src})")

if args.run:
    import omni.timeline

    omni.timeline.get_timeline_interface().play()
    print(f"[d455] publishing on /{NS}/... - Ctrl+C or close the window to stop")
    while app.is_running():
        app.update()
app.close()
