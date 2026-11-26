"""Isaac Sim 5.1: synthetic RGB egg dataset (YOLO format) from the meshari robot's D455 COLOUR camera.

  isaac/isaac_ros2.sh detection/gen_egg_rgb.py --frames 3000            # headless
  -> detection/data/yolo_egg_sim/{images,labels}/{train,val}  + preview_*.jpg (boxes drawn)

Scene, as on the real robot: table top 20 cm BELOW the camera, near edge 20 cm in FRONT of it (+-1-2 cm jitter).
Eggs are REAL size (57 x 44 mm hen egg, +-5 %), lying / standing / tilted, brown / cream / white, 1-3 per frame
(10 % of frames none), plus non-egg distractors (spool-like discs, boxes, cans, balls, cones). Lighting, table
colour and camera mount (+-1 cm, +-1 deg) are randomised every frame. The colour camera uses the real D455
calibration (meshari_d455.py, config/d455_color_intrinsics.json: 640x480, fx 385.7), so an egg has the same size
in pixels as on the real camera. Each label is checked: box width ~ fx * egg size / distance.
"""
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=str(ROOT / "Isaac SIm" / "meshari_d455.usd"))
ap.add_argument("--out", default=str(ROOT / "detection" / "data" / "yolo_egg_sim"))
ap.add_argument("--frames", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--preview", type=int, default=24)
ap.add_argument("--balls", action="store_true",
                help="add ping-pong / golf-size balls (38-45 mm) labelled as class 1 'ball' so the model learns ball != egg")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import json  # noqa: E402

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

CAL = json.loads((ROOT / "config" / "d455_color_intrinsics.json").read_text())
W, H = CAL["width"], CAL["height"]
CAM_LINK = "/World/Sim/base_link/camera_link"
CAM_POS, CAM_RPY = (0.1306, -0.043, 1.153), (0.0, 11.55, -90.0)   # as built by isaac/meshari_d455.py (measured pitch, no roll)
ROBOT_X = 0.13                       # robot centre line; forward is -Y
TABLE_BELOW, TABLE_AHEAD = 0.20, 0.20  # table top below / near edge ahead of the camera
REACH_FWD = 0.45                     # palm reaches ~0.43-0.45 m ahead at this table height (meshari 1.usd)
REACH_X = (-0.20, 0.45)
EGG_L, EGG_D = 0.057, 0.044          # real hen egg
rng = random.Random(args.seed)

omni.usd.get_context().open_stage(args.usd)
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()
for g in ("/World/D455_ROS2", "/World/ActionGraph"):    # no ROS publishing while generating
    if stage.GetPrimAtPath(g).IsValid():
        stage.GetPrimAtPath(g).SetActive(False)
cam_link = stage.GetPrimAtPath(CAM_LINK)
color_cam = next((p for p in Usd.PrimRange(cam_link) if p.IsA(UsdGeom.Camera) and "color" in p.GetName().lower()), None)
assert color_cam, "build meshari_d455.usd first (isaac/isaac_ros2.sh isaac/meshari_d455.py)"
hap = UsdGeom.Camera(color_cam).GetHorizontalApertureAttr().Get()
fl = UsdGeom.Camera(color_cam).GetFocalLengthAttr().Get()
FX = W * fl / hap
assert abs(FX - CAL["fx"]) < 2, f"colour camera fx {FX:.1f} != real {CAL['fx']:.1f}: rebuild meshari_d455.usd"


def set_xf(prim, pos, rpy=(0, 0, 0), scale=(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rpy)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def material(path):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    col = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    rough = sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, col, rough


def bind(prim, path):
    m = material(path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(m[0])
    return m


def egg_mesh(path, rings=24, segs=32, asym=0.12):
    """Hen egg along local Z (blunt end +Z): 57 mm long, 44 mm wide."""
    pts, idx, cnt = [], [], []
    for i in range(rings + 1):
        t = math.pi * i / rings
        z = 0.5 * EGG_L * math.cos(t)
        r = 0.5 * EGG_D * math.sin(t) * (1 + asym * math.cos(t)) / (1 + asym * 0.35)
        pts += [Gf.Vec3f(r * math.cos(2 * math.pi * j / segs), r * math.sin(2 * math.pi * j / segs), z) for j in range(segs)]
    k = EGG_D / (2 * max(math.hypot(p[0], p[1]) for p in pts))     # exact 44 mm width (the profile peaks below D/2)
    pts = [Gf.Vec3f(p[0] * k, p[1] * k, p[2]) for p in pts]
    for i in range(rings):
        for j in range(segs):
            a, b = i * segs + j, i * segs + (j + 1) % segs
            idx += [a, a + segs, b + segs, b]
            cnt.append(4)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none")   # subdivision would shrink it
    return m.GetPrim()


# check the mesh really is 57 x 44 mm
_e = egg_mesh("/World/_EggCheck")
_r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(_e).ComputeAlignedRange().GetSize()
print(f"[gen] egg mesh size {1000 * _r[0]:.1f} x {1000 * _r[1]:.1f} x {1000 * _r[2]:.1f} mm", flush=True)
stage.RemovePrim("/World/_EggCheck")

# ---- scene objects, created once and re-posed every frame
UsdGeom.Xform.Define(stage, "/World/EggScene")
top = UsdGeom.Cube.Define(stage, "/World/EggScene/TableTop"); top.CreateSizeAttr(1.0)
top_mat = bind(top.GetPrim(), "/World/EggScene/Looks/Table")
mat2 = UsdGeom.Cube.Define(stage, "/World/EggScene/Mat"); mat2.CreateSizeAttr(1.0)   # a mat on the table (like the real one)
mat2_mat = bind(mat2.GetPrim(), "/World/EggScene/Looks/Mat")
leg = UsdGeom.Cube.Define(stage, "/World/EggScene/Leg"); leg.CreateSizeAttr(1.0)
eggs = []
for k in range(3):
    e = egg_mesh(f"/World/EggScene/Egg{k}")
    add_labels(e, ["egg"], instance_name="class")
    eggs.append((e, bind(e, f"/World/EggScene/Looks/Egg{k}")))
balls = []   # labelled class 'ball' (--balls): ping-pong / golf size, some on a small holder like the real one
for k in range(3):
    b = UsdGeom.Sphere.Define(stage, f"/World/EggScene/Ball{k}"); b.CreateRadiusAttr(1.0)
    add_labels(b.GetPrim(), ["ball"], instance_name="class")
    hold = UsdGeom.Cylinder.Define(stage, f"/World/EggScene/BallHolder{k}"); hold.CreateRadiusAttr(1.0); hold.CreateHeightAttr(1.0)
    balls.append((b.GetPrim(), bind(b.GetPrim(), f"/World/EggScene/Looks/Ball{k}"),
                  hold.GetPrim(), bind(hold.GetPrim(), f"/World/EggScene/Looks/BallHolder{k}")))
pool = []
for i in range(5):
    for kind, cls in (("cube", UsdGeom.Cube), ("cyl", UsdGeom.Cylinder), ("sph", UsdGeom.Sphere), ("cone", UsdGeom.Cone)):
        p = cls.Define(stage, f"/World/EggScene/D{i}_{kind}")
        if kind == "cube":
            p.CreateSizeAttr(1.0)
        elif kind == "sph":
            p.CreateRadiusAttr(1.0)
        else:
            p.CreateRadiusAttr(1.0); p.CreateHeightAttr(1.0)
        pool.append((kind, p.GetPrim(), bind(p.GetPrim(), f"/World/EggScene/Looks/D{i}_{kind}")))
dome = UsdLux.DomeLight.Define(stage, "/World/EggScene/Dome")
sun = UsdLux.DistantLight.Define(stage, "/World/EggScene/Sun")
for p in stage.Traverse():   # the file's own lights would add uncontrolled light
    if p.HasAPI(UsdLux.LightAPI) and not str(p.GetPath()).startswith("/World/EggScene"):
        UsdGeom.Imageable(p).MakeInvisible()


def show(prim, on):
    UsdGeom.Imageable(prim).MakeVisible() if on else UsdGeom.Imageable(prim).MakeInvisible()


def paint(m, rgb, rough):
    m[1].Set(Gf.Vec3f(*map(float, rgb))); m[2].Set(float(rough))


BALL_COLOURS = [lambda: (rng.uniform(.9, 1.0), rng.uniform(.35, .55), rng.uniform(.05, .15)),   # orange ping-pong
                lambda: (rng.uniform(.9, 1.0), rng.uniform(.9, 1.0), rng.uniform(.88, 1.0)),    # white
                lambda: (rng.uniform(.9, 1.0), rng.uniform(.8, .95), rng.uniform(.1, .3)),      # yellow
                lambda: (rng.uniform(.5, .75), rng.uniform(.28, .45), rng.uniform(.15, .28))]   # egg-brown: shape decides
EGG_COLOURS = [lambda: (rng.uniform(.50, .75), rng.uniform(.28, .45), rng.uniform(.15, .28)),   # brown
               lambda: (rng.uniform(.75, .90), rng.uniform(.60, .72), rng.uniform(.45, .58)),   # cream / light brown
               lambda: (rng.uniform(.85, .97), rng.uniform(.83, .95), rng.uniform(.78, .90))]   # white


def randomize():
    j = lambda s: rng.uniform(-s, s)  # noqa: E731
    cam = (CAM_POS[0] + j(.01), CAM_POS[1] + j(.01), CAM_POS[2] + j(.01))
    set_xf(cam_link, cam, (CAM_RPY[0] + j(1.0), CAM_RPY[1] + j(1.0), CAM_RPY[2] + j(1.0)))
    ttop = CAM_POS[2] - TABLE_BELOW + j(0.01)
    near = CAM_POS[1] - TABLE_AHEAD + j(0.02)
    depth, width = rng.uniform(0.6, 0.9), rng.uniform(1.0, 1.5)
    cx, cy = ROBOT_X + j(0.08), near - depth / 2
    set_xf(top.GetPrim(), (cx, cy, ttop - 0.015), (0, 0, j(3)), (width, depth, 0.03))
    set_xf(leg.GetPrim(), (cx, cy, (ttop - 0.03) / 2), (0, 0, 0), (0.06, 0.06, ttop - 0.03))
    paint(top_mat, rng.choice([(.9, .9, .88), (.75, .75, .72), (.55, .42, .30), (.35, .25, .18)]) if rng.random() < .6
          else [rng.uniform(.05, .95)] * 3, rng.uniform(.3, .9))
    has_mat = rng.random() < 0.6           # dark mat on part of the table, like the real setup
    if has_mat:
        mw, md = rng.uniform(0.4, width), rng.uniform(0.3, depth)
        set_xf(mat2.GetPrim(), (cx + j((width - mw) / 2), cy + j((depth - md) / 2), ttop + 0.001), (0, 0, j(5)),
               (mw, md, 0.003))
        g = rng.uniform(.03, .15)
        paint(mat2_mat, (g, g * rng.uniform(.95, 1.1), g), rng.uniform(.5, .95))
    show(mat2.GetPrim(), has_mat)
    zt = ttop + (0.003 if has_mat else 0.0)
    # visible table: lowest view ray = tilt + half the vertical FOV (~32 deg)
    vfov2 = math.degrees(math.atan(H / 2 / CAL["fy"]))
    near_view = CAM_POS[1] - TABLE_BELOW / math.tan(math.radians(CAM_RPY[1] + vfov2)) + 0.01
    x_lo, x_hi = cx - width / 2 + 0.05, cx + width / 2 - 0.05
    y_lo, y_hi = cy - depth / 2 + 0.05, min(cy + depth / 2 - 0.03, near_view)
    placed = []

    def spot(r, area):
        ax0, ax1, ay0, ay1 = area
        for _ in range(30):
            x, y = rng.uniform(ax0, ax1), rng.uniform(ay0, ay1)
            if all(math.hypot(x - px, y - py) > r + pr + 0.005 for px, py, pr in placed):
                placed.append((x, y, r))
                return x, y
        return None

    n_egg = 0 if rng.random() < 0.10 else rng.choice([1, 1, 1, 2, 2, 3])
    for k, (e, m) in enumerate(eggs):
        xy = None
        if k < n_egg:
            far = rng.random() < 0.3      # 70 % in the reachable zone, 30 % anywhere on the table (far views)
            area = (max(x_lo, REACH_X[0]), min(x_hi, REACH_X[1]), CAM_POS[1] - REACH_FWD, y_hi) if not far else \
                   (x_lo, x_hi, y_lo, y_hi)
            xy = spot(0.035, area)
        if xy:
            s = rng.uniform(0.95, 1.05)                     # real egg sizes vary ~ +-5 %
            pose = rng.random()
            if pose < 0.6:      # lying on its side
                z, rpy = zt + EGG_D / 2 * s * 0.97, (90, 0, rng.uniform(-180, 180))
            elif pose < 0.85:   # standing (blunt end up or down)
                z, rpy = zt + EGG_L / 2 * s, (rng.choice([0, 180]), 0, rng.uniform(-180, 180))
            else:               # tilted (leaning on something / in a holder)
                t = rng.uniform(20, 60)
                z, rpy = zt + 0.5 * s * (EGG_L * math.sin(math.radians(t)) + EGG_D * math.cos(math.radians(t))) * 0.95, \
                    (90 - t, 0, rng.uniform(-180, 180))
            set_xf(e, (xy[0], xy[1], z), rpy, (s, s, s))
            paint(m, rng.choice(EGG_COLOURS)(), rng.uniform(.35, .8))
        show(e, bool(xy))
    n_ball = rng.choice([0, 1, 1, 2, 3]) if args.balls else 0
    for k, (b, bm, hold, hm) in enumerate(balls):
        near = (max(x_lo, REACH_X[0]), min(x_hi, REACH_X[1]), CAM_POS[1] - REACH_FWD, y_hi)   # where the eggs are
        xy = spot(0.03, near if rng.random() < 0.6 else (x_lo, x_hi, y_lo, y_hi)) if k < n_ball else None
        on_holder = bool(xy) and rng.random() < 0.4
        if xy:
            rad = rng.uniform(0.019, 0.0225)                       # ping-pong 40 mm, golf 43 mm
            hz = rng.uniform(0.01, 0.025) if on_holder else 0.0
            if on_holder:
                set_xf(hold, (xy[0], xy[1], zt + hz / 2), (0, 0, 0), (rad * rng.uniform(0.9, 1.3),) * 2 + (hz,))
                paint(hm, [rng.uniform(.1, .8) for _ in range(3)], rng.uniform(.3, .8))
            set_xf(b, (xy[0], xy[1], zt + hz + rad * (0.85 if on_holder else 1.0)), (0, 0, 0), (rad, rad, rad))
            paint(bm, rng.choice(BALL_COLOURS)(), rng.uniform(.2, .6))
        show(b, bool(xy))
        show(hold, on_holder)
    for _, p, _m in pool:
        show(p, False)
    for kind, p, m in rng.sample(pool, rng.randint(0, 6)):
        if kind == "cube":
            a, b, c = rng.uniform(.03, .15), rng.uniform(.03, .15), rng.uniform(.02, .12)
            r, sc, z, rpy = max(a, b) * .7, (a, b, c), zt + c / 2, (0, 0, rng.uniform(0, 90))
        elif kind == "sph":   # balls clearly smaller / larger than an egg (an egg-sized brown ball IS an egg from above)
            rad = rng.choice([rng.uniform(0.010, 0.016), rng.uniform(0.035, 0.06)])
            r, sc, z, rpy = rad, (rad, rad, rad), zt + rad, (0, 0, 0)
        elif kind == "cyl" and rng.random() < 0.4:   # spool-like wide disc
            rad, h = rng.uniform(.08, .15), rng.uniform(.03, .08)
            r, sc, z, rpy = rad, (rad, rad, h), zt + h / 2, (0, 0, 0)
        else:
            rad, h = rng.uniform(.015, .05), rng.uniform(.03, .15)
            r, sc, z, rpy = rad, (rad, rad, h), zt + h / 2, (0, 0, 0)
        xy = spot(r, (x_lo, x_hi, y_lo, y_hi))
        if xy:
            set_xf(p, (xy[0], xy[1], z), rpy, sc)
            paint(m, [rng.uniform(.05, .95) for _ in range(3)], rng.uniform(.2, .9))
            show(p, True)
    dome.CreateIntensityAttr(rng.uniform(300, 1500))
    dome.CreateColorAttr(Gf.Vec3f(1.0, rng.uniform(.9, 1.0), rng.uniform(.8, 1.0)))
    sun.CreateIntensityAttr(rng.uniform(0, 3000))
    sun.CreateAngleAttr(rng.uniform(0.5, 5))
    set_xf(sun.GetPrim(), (0, 0, 0), (rng.uniform(-60, 60), rng.uniform(-60, 60), 0))
    return cam


rp = rep.create.render_product(str(color_cam.GetPath()), (W, H))
rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
bbox_a = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
depth_a = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
for a in (rgb_a, bbox_a, depth_a):
    a.attach(rp)


def egg_boxes(bb, depth):
    """Egg (class 0) and ball (class 1) boxes, visible >= 30 %, as YOLO rows; ok=False if one is mostly hidden
    (ambiguous: drop the frame). sizes = egg sizes only."""
    data, info = bb["data"], bb["info"]
    cls_of = {}
    for k, v in info.get("idToLabels", {}).items():
        v = str(v)
        cls_of[int(k)] = 0 if "egg" in v else 1 if "ball" in v else None
    rows, ok, sizes = [], True, []
    for row in data:
        c = cls_of.get(int(row["semanticId"]))
        if c is None:
            continue
        x0, y0, x1, y1 = (float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        if row["occlusionRatio"] > 0.7 or (x1 - x0) * (y1 - y0) < 30:
            ok = ok and row["occlusionRatio"] > 0.97    # (almost) fully hidden is fine, half-hidden is not
            continue
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
        rows.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        boxes.append((x0, y0, x1, y1))
        if c:
            continue
        d = depth[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            sizes.append(max(x1 - x0, y1 - y0) * float(np.percentile(d, 10)) / FX)   # real size from pixels+depth
    return rows, ok, sizes


out = Path(args.out)
for s in ("train", "val"):
    (out / "images" / s).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s).mkdir(parents=True, exist_ok=True)
for _ in range(10):
    rep.orchestrator.step(rt_subframes=4)
made, n_eggs, dropped, all_sizes = 0, 0, 0, []
import time  # noqa: E402

t0 = time.time()
while made < args.frames:
    randomize()
    rep.orchestrator.step(rt_subframes=16)      # enough subframes for a sharp RTX image (no smear from the last pose)
    rgb = np.asarray(rgb_a.get_data())[..., :3]
    depth = np.asarray(depth_a.get_data(), dtype=np.float32)
    boxes = []
    rows, ok, sizes = egg_boxes(bbox_a.get_data(), depth)
    if not ok or rgb.size == 0:
        dropped += 1
        continue
    split = "val" if made % 10 == 0 else "train"
    name = f"sim_{args.seed:02d}_{made:05d}"
    Image.fromarray(rgb).save(out / "images" / split / f"{name}.jpg", quality=92)
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    made += 1
    n_eggs += sum(r.startswith("0 ") for r in rows)
    all_sizes += sizes
    if made <= args.preview:
        im = Image.fromarray(rgb.copy()); dr = ImageDraw.Draw(im)
        for b, r in zip(boxes, rows):
            egg_box = r.startswith("0 ")
            dr.rectangle(b, outline=(0, 255, 0) if egg_box else (255, 60, 60), width=2)
            dr.text((b[0], b[1] - 12), "egg" if egg_box else "ball", fill=(0, 255, 0) if egg_box else (255, 60, 60))
        im.save(out / f"preview_{made:03d}.jpg")
    if made % 250 == 0 or made == args.frames:
        sz = np.array(all_sizes)
        print(f"[gen] {made}/{args.frames} frames, {n_eggs} eggs, dropped {dropped}, {made / (time.time() - t0):.1f} fps, "
              f"egg size from pixels+depth: median {100 * np.median(sz):.1f} cm (real 4.4-5.7)", flush=True)
(out / "egg_sim.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: egg\n  1: ball\n")
app.close()
