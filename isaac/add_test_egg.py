"""Isaac Sim Script Editor: put a test table + egg in front of the meshari robot's D455.

In Isaac Sim (meshari_d455.usd open): Window -> Script Editor -> File -> Open -> this file -> Run.
Run it again after changing EGG_X / EGG_Y to move the egg (or move /World/TestEgg with the gizmo).
HIGH table close to the robot, so the egg is both seen and reachable: the camera (1.15 m up, 5 deg down,
looking along -Y) sees a 1.0 m table from y = -0.27 m, and the palm reaches up to y = -0.45 m at that height
(arm reach measured on meshari 1.usd). Keep EGG_Y between -0.30 and -0.42 m and EGG_X between -0.20 and 0.45 m.
Nothing is saved unless you use File -> Save.
"""
import math

import omni.usd
from pxr import Gf, UsdGeom

EGG_X, EGG_Y = 0.13, -0.36          # world position of the egg (robot centre is x = 0.13, forward is -Y)
EGG_YAW_DEG = 30.0                  # heading of the egg lying on its side
TABLE_TOP = 1.00                    # m, high table (arm reach)
EGG_L, EGG_D = 0.057, 0.044         # hen egg, same shape as the training data

stage = omni.usd.get_context().get_stage()


def xform(prim, pos, rpy=(0, 0, 0), scale=(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))


# table: top 1.1 x 0.6 m, 3 cm thick, near edge 12 cm in front of the robot (just past the hanging arms)
UsdGeom.Xform.Define(stage, "/World/TestTable")
top = UsdGeom.Cube.Define(stage, "/World/TestTable/Top")
top.CreateSizeAttr(1.0)
top.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.42, 0.30)])
xform(top.GetPrim(), (0.13, -0.42, TABLE_TOP - 0.015), scale=(1.1, 0.6, 0.03))
leg = UsdGeom.Cube.Define(stage, "/World/TestTable/Leg")
leg.CreateSizeAttr(1.0)
xform(leg.GetPrim(), (0.13, -0.55, (TABLE_TOP - 0.03) / 2), scale=(0.06, 0.06, TABLE_TOP - 0.03))

# egg mesh (blunt end +Z), lying on its side
pts, idx, cnt, rings, segs, asym = [], [], [], 24, 32, 0.12
for i in range(rings + 1):
    t = math.pi * i / rings
    z = 0.5 * EGG_L * math.cos(t)
    r = 0.5 * EGG_D * math.sin(t) * (1 + asym * math.cos(t)) / (1 + asym * 0.35)
    pts += [Gf.Vec3f(r * math.cos(2 * math.pi * j / segs), r * math.sin(2 * math.pi * j / segs), z) for j in range(segs)]
for i in range(rings):
    for j in range(segs):
        a, b = i * segs + j, i * segs + (j + 1) % segs
        idx += [a, a + segs, b + segs, b]
        cnt.append(4)
egg = UsdGeom.Mesh.Define(stage, "/World/TestEgg")
egg.CreatePointsAttr(pts)
egg.CreateFaceVertexCountsAttr(cnt)
egg.CreateFaceVertexIndicesAttr(idx)
egg.CreateSubdivisionSchemeAttr("none")
egg.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.90, 0.80)])
xform(egg.GetPrim(), (EGG_X, EGG_Y, TABLE_TOP + EGG_D / 2), (90.0, 0.0, EGG_YAW_DEG))
print(f"[test egg] table top {TABLE_TOP} m, egg at x={EGG_X}, y={EGG_Y} -> watch /sim/egg/point")
