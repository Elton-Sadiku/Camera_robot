"""Isaac Sim Script Editor: put a test table + egg in front of the meshari robot's D455.

In Isaac Sim (meshari_d455.usd open): Window -> Script Editor -> File -> Open -> this file -> Run.
Run it again after changing EGG_X / EGG_Y to move the egg (or move /World/TestEgg with the gizmo).
Table as on the real robot: top 20 cm BELOW the camera (camera_link z = 1.153 m -> top 0.953 m) and its near
edge 20 cm in FRONT of the camera (camera y = -0.043 -> edge y = -0.243; forward is -Y). With the real colour
camera (79 x 64 deg, 11.55 deg down) the whole table is visible from its near edge (egg ~27 cm away); the palm reaches ~y = -0.43 m at this
height (arm reach measured on meshari 1.usd). Keep EGG_Y between -0.25 and -0.43 m, EGG_X between -0.20 and 0.45 m.
Nothing is saved unless you use File -> Save.
"""
import math

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade

EGG_X, EGG_Y = 0.13, -0.36          # world position of the egg (robot centre is x = 0.13, forward is -Y)
EGG_YAW_DEG = 75.0                  # heading of the egg lying on its side (90 = side-on to the camera, 0 = end-on)
TABLE_TOP = 1.153 - 0.20            # m, 20 cm below the camera
TABLE_NEAR_Y = -0.043 - 0.20        # near edge 20 cm in front of the camera
TABLE_D = 0.60                      # table depth (along -Y)
EGG_L, EGG_D = 0.057, 0.044         # hen egg, 57 x 44 mm

stage = omni.usd.get_context().get_stage()


def material(name, rgb, roughness):
    """UsdPreviewSurface (displayColor alone renders washed-out white under the dome light, and YOLO-World
    then sees a 'ball'): brown egg on a dark mat, like the real test scene."""
    mat = UsdShade.Material.Define(stage, f"/World/TestLooks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/TestLooks/{name}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def xform(prim, pos, rpy=(0, 0, 0), scale=(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))


# table: top 1.1 x 0.6 m, 3 cm thick
UsdGeom.Xform.Define(stage, "/World/TestTable")
top = UsdGeom.Cube.Define(stage, "/World/TestTable/Top")
top.CreateSizeAttr(1.0)
top.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.42, 0.30)])
xform(top.GetPrim(), (0.13, TABLE_NEAR_Y - TABLE_D / 2, TABLE_TOP - 0.015), scale=(1.1, TABLE_D, 0.03))
UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(material("DarkMat", (0.06, 0.07, 0.07), 0.8))
leg = UsdGeom.Cube.Define(stage, "/World/TestTable/Leg")
leg.CreateSizeAttr(1.0)
xform(leg.GetPrim(), (0.13, TABLE_NEAR_Y - TABLE_D / 2, (TABLE_TOP - 0.03) / 2), scale=(0.06, 0.06, TABLE_TOP - 0.03))

# egg mesh (blunt end +Z), lying on its side
pts, idx, cnt, rings, segs, asym = [], [], [], 24, 32, 0.12
for i in range(rings + 1):
    t = math.pi * i / rings
    z = 0.5 * EGG_L * math.cos(t)
    r = 0.5 * EGG_D * math.sin(t) * (1 + asym * math.cos(t)) / (1 + asym * 0.35)
    pts += [Gf.Vec3f(r * math.cos(2 * math.pi * j / segs), r * math.sin(2 * math.pi * j / segs), z) for j in range(segs)]
_k = EGG_D / (2 * max(math.hypot(p[0], p[1]) for p in pts))   # exact 44 mm width (the profile peaks below D/2)
pts = [Gf.Vec3f(p[0] * _k, p[1] * _k, p[2]) for p in pts]
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
egg.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.36, 0.22)])
UsdShade.MaterialBindingAPI.Apply(egg.GetPrim()).Bind(material("BrownEgg", (0.62, 0.36, 0.22), 0.55))
xform(egg.GetPrim(), (EGG_X, EGG_Y, TABLE_TOP + EGG_D / 2), (90.0, 0.0, EGG_YAW_DEG))
print(f"[test egg] table top {TABLE_TOP:.3f} m, near edge y={TABLE_NEAR_Y:.3f}, egg at x={EGG_X}, y={EGG_Y} -> watch /sim/egg/point (ros2/egg_xyz.py)")
