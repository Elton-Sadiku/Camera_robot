import math
import random

# ----------------------------------------------------------------------------
# Detect whether we run inside an already-open Isaac Sim (Script Editor)
# or as a standalone python.sh script (then Kit has to be started first).
# ----------------------------------------------------------------------------
try:
    import omni.kit.app  # noqa: F401  (only importable once Kit is running)

    STANDALONE = False
except ImportError:
    STANDALONE = True
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# ============================================================================
# CONFIG  -  everything you are expected to tune lives here
# ============================================================================

# --- Robot ------------------------------------------------------------------
ROBOT_PRIM_PATH = "/World/BothArms_Sim_02_21_26_Step_File"  # root prim of the robot in the stage
ROBOT_USD = ""  # standalone only: path to the saved robot .usd ("" = use stage)
# World direction the robot faces (where the table goes), unit vector in XY.
# None = automatic: the horizontal axis perpendicular to the shoulder line, on the
# side the hands hang toward (elbows bend forward). Set e.g. (0, -1) to force it.
ROBOT_FORWARD = None
# Move the robot up/down so the table (at TABLE_TOP_HEIGHT) ends up at the right
# height below its shoulders. False = keep the robot where it is and move the table.
AUTO_PLACE_ROBOT = True

# --- Camera (Intel RealSense D455) ------------------------------------------
D455_ASSET = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"  # under Isaac assets root
# None = aim automatically at the egg. A number = fixed tilt in degrees
# (0 = straight ahead, positive = DOWN). The egg is below the camera, so 0 can't see it.
CAMERA_PITCH_DEG = None
CAMERA_YAW_DEG = 0.0  # extra yaw relative to robot forward
CAMERA_ROLL_DEG = 0.0
# The camera sits on the robot's "forehead": the front face of the central body
# between the two shoulders. Extra offset in metres, robot frame (forward, left, up).
CAMERA_MOUNT_OFFSET = (0.005, 0.0, 0.0)
CAMERA_RESOLUTION = (640, 480)

# --- Table ------------------------------------------------------------------
# Like a person standing at a table: the arms hang down just in front of the
# table edge, and the egg is in front of the hands, midway between them, where
# the hands reach it by bending the shoulders and elbows forward.
# The egg position is defined relative to the SHOULDER axis (top-bar centre):
EGG_FORWARD_OF_SHOULDER = 0.30  # m, horizontally in front of the shoulder axis
EGG_BELOW_SHOULDER = 0.40  # m, egg centre below the shoulder axis
TABLE_CLEARANCE_TO_HANDS = 0.05  # m, front of the hanging hands -> near table edge
TABLE_TOP_HEIGHT = 0.75  # world Z of the tabletop (used when AUTO_PLACE_ROBOT)
TABLE_DEPTH = 0.60  # along robot forward (m)
TABLE_WIDTH = None  # None = hand-to-hand distance + 0.40 m
TABLE_THICKNESS = 0.03
TABLE_LEG = 0.04

# --- Egg --------------------------------------------------------------------
EGG_LENGTH = 0.057  # m, typical large hen egg
EGG_DIAMETER = 0.044  # m
EGG_MASS = 0.058  # kg
EGG_STATIC_FRICTION = 0.8
EGG_DYNAMIC_FRICTION = 0.6
EGG_RESTITUTION = 0.05

# --- Episodes / randomization -----------------------------------------------
PHASE = 1  # 1 = egg at the fixed grasp spot, 2 = random (x, y) around it each reset
RANDOM_RANGE_FWD = 0.08  # m, +/- along robot forward around the grasp spot
RANDOM_RANGE_LAT = 0.15  # m, +/- sideways (toward either hand)
RANDOM_YAW = True  # also randomize egg heading in phase 2
SEED = 0
EPISODE_STEPS = 300  # standalone demo: steps per episode
NUM_EPISODES = 5

# Fixed prim paths for the things this script creates
CAMERA_MOUNT_PATH = "/World/CameraMount"
D455_PATH = CAMERA_MOUNT_PATH + "/D455"
TABLE_PATH = "/World/Table"
EGG_PATH = "/World/Egg"


# ============================================================================
# Helpers
# ============================================================================
_FRAME = np.array([1.0, 0.0, 0.0])  # robot forward, set by robot_geometry()


def robot_frame():
    """Unit forward/left/up vectors (world) for the robot."""
    fwd = _FRAME
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    return fwd, left, up


def set_pose(prim, pos, rpy_deg=(0.0, 0.0, 0.0), scale=None):
    """Clear existing xform ops and set translate / rotateXYZ / scale."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rpy_deg)))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def make_physics_material(stage, path, static_f, dynamic_f, restitution):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(static_f)
    pm.CreateDynamicFrictionAttr(dynamic_f)
    pm.CreateRestitutionAttr(restitution)
    return mat


def bind_physics_material(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def ensure_physics_and_ground(stage):
    if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        # small objects (egg) like more solver iterations / CCD
        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableCCDAttr(True)
    # The GUI's default "/Environment" already contains a ground plane.
    if not stage.GetPrimAtPath("/World/GroundPlane").IsValid() and not stage.GetPrimAtPath("/Environment").IsValid():
        from isaacsim.core.api.objects import GroundPlane

        GroundPlane("/World/GroundPlane", z_position=0.0)


# ============================================================================
# 1) Robot
# ============================================================================
def load_robot(stage):
    if ROBOT_USD:
        prim = stage.DefinePrim(ROBOT_PRIM_PATH, "Xform")
        prim.GetReferences().AddReference(ROBOT_USD)
    prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not prim.IsValid():
        roots = [p.GetPath().pathString for p in stage.GetPseudoRoot().GetChildren()]
        raise RuntimeError(
            f"Robot prim '{ROBOT_PRIM_PATH}' not found. Top-level prims: {roots}. "
            "Set ROBOT_PRIM_PATH (or ROBOT_USD)."
        )
    return prim


def _mesh_boxes(stage, robot_prim):
    """World-space bbox (metres) of every mesh in the robot (incl. instanced ones)."""
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    boxes = []
    for p in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty():
                boxes.append((np.array(r.GetMin()) * mpu, np.array(r.GetMax()) * mpu))
    if not boxes:
        raise RuntimeError("No meshes found under the robot prim.")
    return boxes


def _union(boxes):
    return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)


def _reach(boxes, direction):
    """Furthest extent of a set of boxes along a (unit) direction."""
    return max(float(np.dot((lo + hi) / 2, direction) + np.dot((hi - lo) / 2, np.abs(direction))) for lo, hi in boxes)


def _lift_op(robot_prim):
    """A dedicated translate op (first in the stack = pure world shift) used to raise the robot."""
    xf = UsdGeom.Xformable(robot_prim)
    ops = xf.GetOrderedXformOps()
    op = next((o for o in ops if o.GetOpName() == "xformOp:translate:lift"), None)
    if op is None:
        op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "lift")
        xf.SetXformOpOrder([op] + list(ops))
    return op


def robot_geometry(stage, robot_prim):
    """Measure the robot: forward direction, hands, forehead (all in metres, world frame)."""
    global _FRAME
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if AUTO_PLACE_ROBOT:
        _lift_op(robot_prim).Set(Gf.Vec3d(0, 0, 0))  # measure un-lifted (script may be re-run)

    boxes = _mesh_boxes(stage, robot_prim)
    lo, hi = _union(boxes)
    size, center = hi - lo, (lo + hi) / 2
    H = size[2]  # arms hang down -> height ~ shoulder-to-fingertip
    if H > 5.0 or H < 0.1:
        print(f"[scene] WARNING: robot height {H:.3f} m looks wrong - check units / scale!")

    # Shoulder line = the longer horizontal axis, depth = the shorter one.
    w_ax, d_ax = (0, 1) if size[0] >= size[1] else (1, 0)
    lat, dep = np.eye(3)[w_ax], np.eye(3)[d_ax]
    cz = lambda b: (b[0] + b[1]) / 2
    hands = [b for b in boxes if b[1][2] < lo[2] + 0.30 * H]
    shoulders = [b for b in boxes if b[0][2] > hi[2] - 0.30 * H and abs(np.dot(cz(b) - center, lat)) > 0.25 * size[w_ax]]
    head = [b for b in boxes if b[0][2] > hi[2] - 0.35 * H and abs(np.dot(cz(b) - center, lat)) < 0.20 * size[w_ax]]
    head = head or [(lo, hi)]

    if ROBOT_FORWARD is None:
        # For this robot (BothArms_Sim), the hanging hands sit slightly BEHIND the
        # shoulder centre, so forward is the opposite side (checked against the
        # GUI). If the table appears behind the robot, set ROBOT_FORWARD manually.
        hc = np.mean([cz(b) for b in hands], axis=0)
        sc = np.mean([cz(b) for b in (shoulders or head)], axis=0)
        fwd = dep * (-1.0 if np.dot(hc - sc, dep) >= 0 else 1.0)
    else:
        fwd = np.array([ROBOT_FORWARD[0], ROBOT_FORWARD[1], 0.0], dtype=float)
        fwd /= np.linalg.norm(fwd)
    _FRAME = fwd
    fwd, left, up = robot_frame()

    left_hand = [b for b in hands if np.dot(cz(b) - center, left) > 0]
    right_hand = [b for b in hands if np.dot(cz(b) - center, left) <= 0]
    lh_c, rh_c = cz(_union(left_hand)), cz(_union(right_hand))
    hands_mid = (lh_c + rh_c) / 2
    hands_span = _reach(left_hand, left) + _reach(right_hand, -left)  # outer edge to outer edge
    hands_front = _reach(hands, fwd)  # furthest forward point of the hands
    hands_bottom = lo[2]

    # Forehead = front of the horizontal top bar, centred between the shoulders,
    # at the bar's centre height. The bar is a cylinder, so its diameter equals
    # its front-to-back depth.
    h_lo, h_hi = _union(head)
    bar_diameter = float(np.dot(h_hi - h_lo, np.abs(fwd)))
    forehead = np.array([0.0, 0.0, hi[2] - bar_diameter / 2])
    forehead += lat * np.dot(center, lat) + fwd * _reach(head, fwd)

    shoulder_z = forehead[2]  # shoulder joints sit on the bar axis
    shoulder_axis = forehead - fwd * (bar_diameter / 2)  # bar centre line (horizontal)

    lift = 0.0
    if AUTO_PLACE_ROBOT:
        egg_z = TABLE_TOP_HEIGHT + EGG_DIAMETER / 2
        lift = (egg_z + EGG_BELOW_SHOULDER) - shoulder_z
        _lift_op(robot_prim).Set(Gf.Vec3d(0, 0, lift / mpu))
    dz = np.array([0, 0, lift])

    print(f"[scene] robot size (m): {size.round(3)}, forward = {fwd.round(2)}, lifted by {lift:.3f} m")
    print(f"[scene] hands: left {(lh_c + dz).round(3)}, right {(rh_c + dz).round(3)}, span {hands_span:.3f} m")
    print(f"[scene] forehead (camera mount) at {(forehead + dz).round(3)}, "
          f"robot top {hi[2] + lift:.3f} m, fingertips {hands_bottom + lift:.3f} m above ground")
    return dict(
        forehead=forehead + dz, hands_mid=hands_mid + dz, hands_span=hands_span,
        hands_front=hands_front, hands_bottom=hands_bottom + lift,
        shoulder_axis=shoulder_axis + dz, shoulder_z=shoulder_z + lift,
        arm_reach=shoulder_z - hands_bottom,  # shoulder axis -> fingertips
    )


# ============================================================================
# 2) Camera mount + D455
# ============================================================================
def mount_pose(geo):
    fwd, left, up = robot_frame()
    off = CAMERA_MOUNT_OFFSET
    pos = geo["forehead"] + fwd * off[0] + left * off[1] + up * off[2]
    yaw = math.degrees(math.atan2(fwd[1], fwd[0])) + CAMERA_YAW_DEG
    pitch = CAMERA_PITCH_DEG
    if pitch is None:  # aim at the egg's grasp spot
        to_egg = geo["egg_target"] - pos
        pitch = math.degrees(math.atan2(-to_egg[2], float(np.dot(to_egg, fwd))))
    geo["camera_pitch"] = pitch
    # Sensor assets look along their local +X. Rotating about +Y by +pitch
    # tilts +X toward -Z, i.e. positive pitch looks DOWN.
    return pos, (CAMERA_ROLL_DEG, pitch, yaw)


def add_camera(stage, geo):
    from isaacsim.storage.native import get_assets_root_path

    mount = UsdGeom.Xform.Define(stage, CAMERA_MOUNT_PATH).GetPrim()
    pos, rpy = mount_pose(geo)
    set_pose(mount, pos, rpy)

    d455 = stage.DefinePrim(D455_PATH, "Xform")
    d455.GetReferences().ClearReferences()
    d455.GetReferences().AddReference(get_assets_root_path() + D455_ASSET)
    # The D455 asset ships with a rigid body -> it would fall off the robot.
    # Make it kinematic: it follows the mount (and pitch changes) exactly.
    for p in Usd.PrimRange(d455):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)

    cams = [p for p in Usd.PrimRange(d455) if p.IsA(UsdGeom.Camera)]
    if not cams:
        print("[scene] WARNING: no camera prim found inside the D455 asset")
        return None
    color = next((c for c in cams if "color" in c.GetName().lower()), cams[0])
    print("[scene] D455 cameras:", [c.GetPath().pathString for c in cams])
    print("[scene] using colour camera:", color.GetPath())
    return color.GetPath().pathString


def set_camera_pitch(pitch_deg, geo):
    """Change the D455 pitch at any time (also while the sim is running)."""
    global CAMERA_PITCH_DEG
    CAMERA_PITCH_DEG = pitch_deg
    stage = omni.usd.get_context().get_stage()
    pos, rpy = mount_pose(geo)
    set_pose(stage.GetPrimAtPath(CAMERA_MOUNT_PATH), pos, rpy)


def report_camera_direction(cam_path, geo):
    """Print where the camera actually looks (USD cameras look along local -Z)."""
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    look = m.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    print(f"[scene] camera look direction (world): ({look[0]:.2f}, {look[1]:.2f}, {look[2]:.2f})"
          f"  -> robot forward {robot_frame()[0].round(2)} tilted down by {geo['camera_pitch']:.1f} deg")


# ============================================================================
# 3) Table: in front of the robot, centred between the hands
# ============================================================================
def table_layout(geo):
    fwd, left, up = robot_frame()
    top_z = geo["shoulder_z"] - EGG_BELOW_SHOULDER - EGG_DIAMETER / 2
    depth = TABLE_DEPTH
    width = TABLE_WIDTH if TABLE_WIDTH is not None else geo["hands_span"] + 0.40

    def ground(p):  # drop to z = 0
        return np.array([p[0], p[1], 0.0])

    # Egg grasp spot: midway between the hands sideways, in front of the shoulders.
    mid = ground(geo["hands_mid"])
    axis = ground(geo["shoulder_axis"])
    egg_xy = mid + fwd * (np.dot(axis, fwd) + EGG_FORWARD_OF_SHOULDER - np.dot(mid, fwd))

    # Table near edge just in front of the hanging hands (they hang beside the edge).
    near = np.dot(mid, fwd)
    near_edge = geo["hands_front"] + TABLE_CLEARANCE_TO_HANDS
    if np.dot(egg_xy, fwd) < near_edge + 0.05:
        print("[scene] WARNING: egg would be at/behind the table edge - increase EGG_FORWARD_OF_SHOULDER")
    center_xy = mid + fwd * (near_edge + depth / 2.0 - near)

    egg_target = np.array([egg_xy[0], egg_xy[1], top_z + EGG_DIAMETER / 2])
    dist = float(np.linalg.norm(egg_target - geo["shoulder_axis"]))
    print(f"[scene] shoulder -> egg distance {dist:.3f} m, arm length (shoulder -> fingertips) "
          f"{geo['arm_reach']:.3f} m ({100 * dist / geo['arm_reach']:.0f}% of full reach)")
    if dist > 0.9 * geo["arm_reach"]:
        print("[scene] WARNING: egg is near/beyond the arm's reach")
    return dict(center_xy=center_xy, top_z=top_z, depth=depth, width=width,
                egg_xy=egg_xy, egg_target=egg_target)


def add_table(stage, layout):
    fwd, left, up = robot_frame()
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    d, w, t, z = layout["depth"], layout["width"], TABLE_THICKNESS, layout["top_z"]

    root = UsdGeom.Xform.Define(stage, TABLE_PATH).GetPrim()
    set_pose(root, layout["center_xy"], (0, 0, yaw))  # local X = robot forward
    mat = make_physics_material(stage, TABLE_PATH + "/PhysicsMaterial", 0.7, 0.5, 0.0)

    def box(name, pos, size, color):
        cube = UsdGeom.Cube.Define(stage, f"{TABLE_PATH}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        set_pose(cube.GetPrim(), pos, scale=size)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # static collider (no rigid body)
        bind_physics_material(cube.GetPrim(), mat)

    box("Top", (0, 0, z - t / 2), (d, w, t), (0.55, 0.40, 0.28))
    leg_h = z - t
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        box(f"Leg{i}", (sx * (d / 2 - TABLE_LEG), sy * (w / 2 - TABLE_LEG), leg_h / 2),
            (TABLE_LEG, TABLE_LEG, leg_h), (0.3, 0.3, 0.3))
    print(f"[scene] table: centre={layout['center_xy'][:2].round(3)}, top z={z:.3f}, size={d:.2f}x{w:.2f} m")


# ============================================================================
# 4) Egg (procedural mesh + rigid body)
# ============================================================================
def egg_mesh_points(length, diameter, rings=24, segments=32, asym=0.12):
    """Egg profile along local Z: blunt end at +Z, pointy end at -Z."""
    pts, faces, counts = [], [], []
    for i in range(rings + 1):
        t = math.pi * i / rings
        z = 0.5 * length * math.cos(t)
        r = 0.5 * diameter * math.sin(t) * (1.0 + asym * math.cos(t)) / (1.0 + asym * 0.35)
        for j in range(segments):
            a = 2 * math.pi * j / segments
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            faces += [a, a + segments, b + segments, b]
            counts.append(4)
    return pts, counts, faces


def add_egg(stage):
    mesh = UsdGeom.Mesh.Define(stage, EGG_PATH)
    pts, counts, idx = egg_mesh_points(EGG_LENGTH, EGG_DIAMETER)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.93, 0.87, 0.75)])
    prim = mesh.GetPrim()

    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(EGG_MASS)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateEnableCCDAttr(True)  # small, light object -> avoid tunnelling
    rb.CreateAngularDampingAttr(0.5)  # eggs don't spin forever
    PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(0.002)
    mat = make_physics_material(stage, EGG_PATH + "_PhysicsMaterial",
                                EGG_STATIC_FRICTION, EGG_DYNAMIC_FRICTION, EGG_RESTITUTION)
    bind_physics_material(prim, mat)
    return prim


# ============================================================================
# 5) Task: reset / randomization
# ============================================================================
class EggTask:
    """Holds the scene layout and puts the egg back for each episode."""

    def __init__(self, layout, seed=SEED):
        self.layout = layout
        self.rng = random.Random(seed)
        self.fwd, self.left, _ = robot_frame()
        self.egg_rigid = None  # set in standalone after world.reset()

    def sample_egg_xy(self, randomize):
        """(x, y) in world. Phase 1: the grasp spot. Phase 2: uniform around it."""
        c = self.layout["egg_xy"]
        if not randomize:
            return c[0], c[1]
        u = self.rng.uniform(-RANDOM_RANGE_FWD, RANDOM_RANGE_FWD)
        v = self.rng.uniform(-RANDOM_RANGE_LAT, RANDOM_RANGE_LAT)
        p = c + self.fwd * u + self.left * v
        return p[0], p[1]

    def egg_pose(self, randomize):
        x, y = self.sample_egg_xy(randomize)
        # Lying on its side (long axis horizontal) + 5 mm drop to avoid initial penetration.
        z = self.layout["top_z"] + EGG_DIAMETER / 2 + 0.005
        yaw = self.rng.uniform(-180, 180) if (randomize and RANDOM_YAW) else 0.0
        return np.array([x, y, z]), (90.0, 0.0, yaw)

    def reset(self, randomize=None):
        randomize = (PHASE >= 2) if randomize is None else randomize
        pos, rpy = self.egg_pose(randomize)
        if self.egg_rigid is not None:  # physics running: teleport through PhysX
            from isaacsim.core.utils.rotations import euler_angles_to_quat

            q = euler_angles_to_quat(np.array(rpy), degrees=True, extrinsic=True)  # same order as USD rotateXYZ
            self.egg_rigid.set_world_pose(position=pos, orientation=q)
            self.egg_rigid.set_linear_velocity(np.zeros(3))
            self.egg_rigid.set_angular_velocity(np.zeros(3))
        else:  # just authoring USD (GUI, sim stopped)
            stage = omni.usd.get_context().get_stage()
            set_pose(stage.GetPrimAtPath(EGG_PATH), pos, rpy)
        print(f"[scene] egg reset -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), randomized={randomize}")
        return pos


# ============================================================================
# Build everything
# ============================================================================
def build_scene():
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    ensure_physics_and_ground(stage)

    robot = load_robot(stage)
    geo = robot_geometry(stage, robot)
    layout = table_layout(geo)
    geo["egg_target"] = layout["egg_target"]  # camera auto-aims here
    cam_path = add_camera(stage, geo)
    add_table(stage, layout)
    add_egg(stage)

    task = EggTask(layout)
    task.reset(randomize=False)
    if cam_path:
        report_camera_direction(cam_path, geo)
    return task, geo, cam_path


def run_standalone():
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.sensors.camera import Camera

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
    task, geo, cam_path = build_scene()

    camera = None
    if cam_path:
        camera = Camera(prim_path=cam_path, resolution=CAMERA_RESOLUTION)
    world.reset()
    task.egg_rigid = SingleRigidPrim(EGG_PATH, name="egg")
    task.egg_rigid.initialize()
    if camera:
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()  # depth in metres

    for ep in range(NUM_EPISODES):
        task.reset()
        for _ in range(EPISODE_STEPS):
            if not simulation_app.is_running():
                return
            world.step(render=True)
            # >>> your policy / data collection goes here, e.g.:
            # rgb = camera.get_rgba()[..., :3]
            # depth = camera.get_current_frame()["distance_to_image_plane"]
        print(f"[scene] episode {ep} done")

    while simulation_app.is_running():  # keep the window open
        world.step(render=True)


if STANDALONE:
    if ROBOT_USD == "":
        raise SystemExit("Standalone mode needs ROBOT_USD (save the stage with the robot first).")
    omni.usd.get_context().new_stage()
    try:
        run_standalone()
    finally:
        simulation_app.close()
else:
    TASK, GEO, CAM_PATH = build_scene()
    print("[scene] done. Press Play. In the Script Editor you can now call e.g.:\n"
          "   set_camera_pitch(25, GEO)      # fixed tilt 25 deg down (None = aim at egg)\n"
          "   TASK.reset(randomize=True)     # random egg position (sim stopped)")
