"""Isaac Sim 5.1: synthetic DEPTH dataset for egg detection, rendered by the D455 mounted in meshari.

Uses "Isaac SIm/meshari_d455.usd" (built by isaac/meshari_d455.py): the D455 depth camera in the torso slot,
with the REAL intrinsics (424x240) and the shared mount tilt (MOUNT_RPY). Every frame randomizes:
  - table: HIGH (0.97-1.03 m) and close to the robot (arm reach), size, position, small yaw
  - egg: 57x44 mm (+-10 %), lying on its side, where the 5 deg camera sees it AND a hand reaches it (from ~16 cm to ~45 cm from the camera)
  - 0-6 distractors (cubes, cylinders, cones, spheres NOT egg-sized) so "any blob" is not an egg
  - camera mount jitter +-1 cm, +-1.5 deg (same idea as D455_MOUNT_EVENT)
  - ~10 % frames without an egg (negatives)
Saved CLEAN (no noise): depth in uint16 millimetres (0 = no return), plus the egg box from Replicator's
bounding_box_2d_tight. The D455 stereo noise (shared.d455_model.stereo_noise) is added at TRAINING time,
fresh every epoch, so sim/real use the exact same noise + to_obs code.

  isaac/isaac_ros2.sh detection/gen_egg_depth.py --frames 6000          # headless, ~minutes
  -> detection/data/egg_depth/shard_XXX.npz   (depth uint16 [N,240,424], box float32 [N,5]=has,x0,y0,x1,y1)
"""
import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=str(ROOT / "Isaac SIm" / "meshari_d455.usd"))
ap.add_argument("--out", default=str(ROOT / "detection" / "data" / "egg_depth"))
ap.add_argument("--frames", type=int, default=6000)
ap.add_argument("--shard", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--preview", type=int, default=0, help="also save N preview PNGs (depth + box)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

CAM_LINK = "/World/Sim/base_link/camera_link"
DEPTH_CAM = CAM_LINK + "/camera_depth_optical_frame"
W, H = 424, 240
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from shared.d455_model import MOUNT_RPY  # noqa: E402

TABLE_TOP = (0.97, 1.08)     # m, high table: see randomize(); up to 1.08 m so the 5 deg camera sees eggs from ~16 cm
MIN_CAM_DIST = 0.18          # m camera -> egg centre: the egg's front is then ~15.6 cm, the D455's closest depth
REACH_FWD = 0.45             # m ahead of the robot base: farthest the palm gets at a ~1 m table
REACH_X = (-0.20, 0.45)      # m: right arm reaches x -0.55..+0.09, left arm mirrored about the centre x 0.13

# camera_link pose as built by isaac/meshari_d455.py; the tilt follows shared.d455_model MOUNT_RPY (now 5 deg)
CAM_POS, CAM_RPY = (0.1306, -0.043, 1.153), (0.0, math.degrees(MOUNT_RPY[1]), -90.0)
ROBOT_X = 0.13                                                     # robot centre line (x); forward is -Y
EGG_L, EGG_D = 0.057, 0.044
rng = random.Random(args.seed)

omni.usd.get_context().open_stage(args.usd)
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()
for g in ("/World/D455_ROS2", "/World/ActionGraph"):   # no ROS publishing while generating
    if stage.GetPrimAtPath(g).IsValid():
        stage.GetPrimAtPath(g).SetActive(False)
assert stage.GetPrimAtPath(DEPTH_CAM).IsValid(), "build meshari_d455.usd first (isaac/meshari_d455.py)"


def set_xf(prim, pos, rpy=(0, 0, 0), scale=(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rpy)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def egg_mesh(path, rings=24, segs=32, asym=0.12):
    """Egg profile along local Z (blunt end +Z), 57 x 44 mm."""
    pts, idx, cnt = [], [], []
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
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none")
    return m.GetPrim()


# ---- scene objects (created once, re-posed every frame) ----
root = UsdGeom.Xform.Define(stage, "/World/DetScene").GetPrim()
table_top = UsdGeom.Cube.Define(stage, "/World/DetScene/TableTop")
table_top.CreateSizeAttr(1.0)
legs = [UsdGeom.Cube.Define(stage, f"/World/DetScene/Leg{i}") for i in range(4)]
for c in legs:
    c.CreateSizeAttr(1.0)
egg = egg_mesh("/World/DetScene/Egg")
add_labels(egg, ["egg"], instance_name="class")
pool = []
for i in range(6):
    for kind, cls in (("cube", UsdGeom.Cube), ("cyl", UsdGeom.Cylinder), ("sph", UsdGeom.Sphere), ("cone", UsdGeom.Cone)):
        p = cls.Define(stage, f"/World/DetScene/D{i}_{kind}")
        if kind == "cube":
            p.CreateSizeAttr(1.0)
        elif kind == "sph":
            p.CreateRadiusAttr(1.0)
        else:
            p.CreateRadiusAttr(1.0); p.CreateHeightAttr(1.0)
        pool.append((kind, p.GetPrim()))


def show(prim, on):
    UsdGeom.Imageable(prim).MakeVisible() if on else UsdGeom.Imageable(prim).MakeInvisible()


rp = rep.create.render_product(DEPTH_CAM, (W, H))
depth_a = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
bbox_a = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
depth_a.attach(rp)
bbox_a.attach(rp)
cam_link = stage.GetPrimAtPath(CAM_LINK)


def randomize():
    # camera mount jitter
    j = lambda s: rng.uniform(-s, s)  # noqa: E731
    set_xf(cam_link, (CAM_POS[0] + j(.01), CAM_POS[1] + j(.01), CAM_POS[2] + j(.01)),
           (CAM_RPY[0] + j(1.0), CAM_RPY[1] + j(1.5), CAM_RPY[2] + j(1.0)))
    # HIGH table close to the robot (forward = -Y), so the egg is both REACHABLE and SEEN by the 5 deg camera.
    # Arm reach measured on meshari 1.usd (7,020 joint poses): the palm reaches 0.43-0.45 m ahead at a 0.95-1.0 m
    # table but only 0.24 m at 0.75 m; the 5 deg camera sees a 1.0 m table from ~0.27 m ahead.
    top = rng.uniform(TABLE_TOP[0], TABLE_TOP[1])
    # near edge just in front of the hanging arms (they hang at y ~ -0.02..+0.09, through the table height)
    near, depth, width = rng.uniform(-0.11, -0.18), rng.uniform(0.5, 0.8), rng.uniform(0.9, 1.4)
    cx, cy, yaw = ROBOT_X + j(0.10), near - depth / 2, j(5.0)
    set_xf(table_top.GetPrim(), (cx, cy, top - 0.015), (0, 0, yaw), (width, depth, 0.03))
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        a = math.radians(yaw)
        lx, ly = sx * (width / 2 - 0.03), sy * (depth / 2 - 0.03)
        set_xf(legs[k].GetPrim(), (cx + lx * math.cos(a) - ly * math.sin(a), cy + lx * math.sin(a) + ly * math.cos(a), (top - 0.03) / 2),
               (0, 0, yaw), (0.04, 0.04, top - 0.03))
    # usable area on the table that the camera sees: its lowest view ray (tilt + half the 59 deg vertical FOV)
    # meets the table top this far ahead (20 deg tilt -> ~0.36 m, 5 deg -> ~0.63 m); farthest point still in range
    near_view = CAM_POS[1] - (CAM_POS[2] - top) / math.tan(math.radians(CAM_RPY[1] + 29.5)) - 0.03
    y_lo, y_hi = max(cy - depth / 2 + 0.05, -1.05), min(cy + depth / 2 - 0.05, near_view)
    x_lo, x_hi = cx - width / 2 + 0.05, cx + width / 2 - 0.05
    # the egg goes where the camera sees it AND a hand can reach it; distractors anywhere the camera sees
    egg_area = (max(x_lo, REACH_X[0]), min(x_hi, REACH_X[1]), max(y_lo, -REACH_FWD), y_hi)
    if rng.random() < 0.5:   # half the eggs in the nearest 10 cm of the visible table: close views (16-25 cm)
        egg_area = (egg_area[0], egg_area[1], max(egg_area[2], y_hi - 0.10), y_hi)
    placed = []

    def spot(r, area=None, z=None):
        ax0, ax1, ay0, ay1 = area or (x_lo, x_hi, y_lo, y_hi)
        if ax1 <= ax0 or ay1 <= ay0:
            return None
        for _ in range(30):
            x, y = rng.uniform(ax0, ax1), rng.uniform(ay0, ay1)
            if z is not None and math.dist((x, y, z), CAM_POS) < MIN_CAM_DIST:   # closer than the D455 can measure
                continue
            if all(math.hypot(x - px, y - py) > r + pr + 0.01 for px, py, pr in placed):
                placed.append((x, y, r))
                return x, y
        return None

    has_egg = rng.random() > 0.10
    s = rng.uniform(0.9, 1.1)
    xy = spot(0.035 * s, egg_area, top + EGG_D / 2 * s) if has_egg else None
    if xy:
        set_xf(egg, (xy[0], xy[1], top + EGG_D / 2 * s), (90, 0, rng.uniform(-180, 180)), (s, s, s))
    show(egg, bool(xy))
    # distractors
    n = rng.randint(0, 6)
    for kind, p in pool:
        show(p, False)
    chosen = rng.sample(pool, n)
    for kind, p in chosen:
        if kind == "cube":
            a = rng.uniform(0.03, 0.10); r = a * 0.7; sc, z, rpy = (a, a, a), top + a / 2, (0, 0, rng.uniform(0, 90))
        elif kind == "sph":
            rad = rng.choice([rng.uniform(0.012, 0.018), rng.uniform(0.033, 0.05)])  # skip egg-like 19-32 mm
            r = rad; sc, z, rpy = (rad, rad, rad), top + rad, (0, 0, 0)
        else:
            rad, h = rng.uniform(0.015, 0.05), rng.uniform(0.03, 0.15)
            r = rad; sc, z, rpy = (rad, rad, h), top + h / 2, (0, 0, 0)
        xy2 = spot(r)
        if xy2:
            set_xf(p, (xy2[0], xy2[1], z), rpy, sc)
            show(p, True)
    return bool(xy)


def egg_box(bb):
    data, info = bb["data"], bb["info"]
    egg_ids = [int(k) for k, v in info.get("idToLabels", {}).items() if "egg" in str(v)]
    for row in data:
        if int(row["semanticId"]) in egg_ids and row["occlusionRatio"] < 0.8:
            x0, y0, x1, y1 = float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])
            if (x1 - x0) * (y1 - y0) >= 12:
                return [1.0, x0, y0, x1, y1]
    return [0.0, 0.0, 0.0, 0.0, 0.0]


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for _ in range(5):
    rep.orchestrator.step(rt_subframes=1)
buf_d, buf_b, shard, made, pos = [], [], 0, 0, 0
import time  # noqa: E402

t0 = time.time()
while made < args.frames:
    randomize()
    rep.orchestrator.step(rt_subframes=1)
    d = np.asarray(depth_a.get_data(), dtype=np.float32)
    box = egg_box(bbox_a.get_data())
    d_mm = np.where(np.isfinite(d) & (d > 0) & (d < 60.0), np.round(d * 1000.0), 0).astype(np.uint16)
    buf_d.append(d_mm); buf_b.append(box); made += 1; pos += int(box[0])
    if made <= args.preview:
        from PIL import Image, ImageDraw

        v = np.clip(d_mm / 1500.0, 0, 1); img = Image.fromarray((255 * (1 - v) * (d_mm > 0)).astype(np.uint8)).convert("RGB")
        if box[0]:
            ImageDraw.Draw(img).rectangle(box[1:], outline=(255, 0, 0))
        img.resize((W * 2, H * 2)).save(out / f"preview_{made:03d}.png")
    if len(buf_d) == args.shard or made == args.frames:
        np.savez_compressed(out / f"shard_{shard:03d}.npz", depth=np.stack(buf_d), box=np.array(buf_b, np.float32))
        print(f"[gen] shard {shard}: {len(buf_d)} frames, total {made}/{args.frames}, with egg {pos}, "
              f"{made / (time.time() - t0):.1f} fps", flush=True)
        buf_d, buf_b, shard = [], [], shard + 1
app.close()
