"""Isaac Sim: drop a real-size egg onto the test table with real physics (falls, bounces, rolls, settles).

Builds the test table + egg with isaac/add_test_egg.py, then makes them physical:
  egg   : rigid body, 60 g (hen egg), collision = convex hull of the 57 x 44 mm egg mesh, continuous collision
          detection (a small fast object must not tunnel through the table), eggshell-on-mat friction and a
          small bounce
  table : static collider (top + leg), same physics material
and lifts the egg DROP_HEIGHT above the table with a random tilt and a small push / spin, so every drop differs.

Use it
  standalone : isaac/isaac_ros2.sh isaac/meshari_d455.py --run --drop-egg      (GUI; publishes the D455 too)
  GUI        : open "Isaac SIm/meshari_d455.usd", STOP the simulation, Window -> Script Editor -> File -> Open
               -> this file -> Run, then press Play. Run again (sim stopped) for a new drop.
Watch it with the D455: ros2 launch ros2/egg_yolo.launch.py source:=sim  (the egg is detected once it lands).
Nothing is saved unless you use File -> Save.
"""
import math
import random
from pathlib import Path

import omni.usd
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

# ---- settings
DROP_HEIGHT = 0.15          # m above the table top (egg's lowest point); a real egg dropped from 15 cm would crack
DROP_X, DROP_Y = None, None # None = the add_test_egg.py position (EGG_X, EGG_Y); or set metres in base_link/world
RANDOM_POSE = True          # random tilt + push + spin for each drop
EGG_MASS = 0.060            # kg, a large hen egg is 55-65 g
STATIC_FRICTION = 0.6       # eggshell on a rubber/fabric mat
DYNAMIC_FRICTION = 0.5
RESTITUTION = 0.15          # how bouncy (0 = no bounce, 1 = perfectly elastic); an egg barely bounces
ANGULAR_DAMPING = 0.3       # rolling resistance, so it settles like a real egg instead of rolling forever
SEED = None                 # an int makes the drop repeatable

HERE = Path(globals().get("__file__", Path.home() / "Desktop/camera/Camera1/Camera_robot/isaac/drop_egg.py")).resolve().parent
ns = {}
exec(compile((HERE / "add_test_egg.py").read_text(), str(HERE / "add_test_egg.py"), "exec"), ns)   # table + egg
stage = omni.usd.get_context().get_stage()
rng = random.Random(SEED)

# a physics scene with gravity (the robot file normally has one)
if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

# physics material shared by egg and table
mat = UsdShade.Material.Define(stage, "/World/TestLooks/EggPhysics")
pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(STATIC_FRICTION)
pm.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
pm.CreateRestitutionAttr(RESTITUTION)


def bind_physics(prim):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


# table = static colliders (no rigid body -> it does not move)
for path in ("/World/TestTable/Top", "/World/TestTable/Leg"):
    p = stage.GetPrimAtPath(path)
    UsdPhysics.CollisionAPI.Apply(p)
    bind_physics(p)

# egg = dynamic rigid body
egg = stage.GetPrimAtPath("/World/TestEgg")
UsdPhysics.RigidBodyAPI.Apply(egg)
UsdPhysics.CollisionAPI.Apply(egg)
UsdPhysics.MeshCollisionAPI.Apply(egg).CreateApproximationAttr("convexHull")   # an egg is convex: exact enough
UsdPhysics.MassAPI.Apply(egg).CreateMassAttr(EGG_MASS)
rb = PhysxSchema.PhysxRigidBodyAPI.Apply(egg)
rb.CreateEnableCCDAttr(True)
rb.CreateAngularDampingAttr(ANGULAR_DAMPING)
col = PhysxSchema.PhysxCollisionAPI.Apply(egg)
col.CreateContactOffsetAttr(0.002)       # small object: tight contact distances (metres)
col.CreateRestOffsetAttr(0.0)
bind_physics(egg)

# start pose: above the table, random tilt; velocity: a small push + spin
x = DROP_X if DROP_X is not None else ns["EGG_X"]
y = DROP_Y if DROP_Y is not None else ns["EGG_Y"]
z = ns["TABLE_TOP"] + DROP_HEIGHT + ns["EGG_L"] / 2       # half the egg length: clears the table in any tilt
rpy = (rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360)) if RANDOM_POSE else (90.0, 0.0, 75.0)
xf = UsdGeom.Xformable(egg)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
xf.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
rbu = UsdPhysics.RigidBodyAPI(egg)
if RANDOM_POSE:
    rbu.CreateVelocityAttr(Gf.Vec3f(rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0))          # m/s
    rbu.CreateAngularVelocityAttr(Gf.Vec3f(*(rng.uniform(-90, 90) for _ in range(3))))              # deg/s
else:
    rbu.CreateVelocityAttr(Gf.Vec3f(0, 0, 0))
    rbu.CreateAngularVelocityAttr(Gf.Vec3f(0, 0, 0))
fall_s = math.sqrt(2 * DROP_HEIGHT / 9.81)
print(f"[drop egg] {1000 * EGG_MASS:.0f} g egg from {100 * DROP_HEIGHT:.0f} cm above the table at x={x:.3f}, y={y:.3f} "
      f"(hits the table after ~{fall_s:.2f} s) - press Play")
