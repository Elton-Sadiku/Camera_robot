"""Record REAL D455 depth frames WITH an egg, labelled automatically by comparing with the empty table.

How: the camera stays still. First it records the scene WITHOUT the egg (background). Then you place the egg
at N spots, one after another; at each spot it records a few frames. Pixels that became 1.2-12 cm CLOSER
than the background form the egg; their bounding box is the label. Frames where that blob is missing,
egg-sized wrong (a hand), or split in several big pieces are skipped. Timed, no keyboard needed:

  0-10 s     clear the egg from view, keep the camera still
  10-13 s    background is recorded
  then for each spot:  12 s to place the egg (hands out of view!) -> 4 s recording

  source ./env.sh && ros2 launch ros2/d455.launch.py
  source ./env.sh && python3 detection/record_real_eggs.py --spots 10
  -> detection/data/real_eggs/shard_XXX.npz  (depth uint16 mm, box [has, x0, y0, x1, y1]) + preview_XXX.png
Run it again after moving the camera / table for more variety (each run makes a new shard).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--topic", default="/camera/camera/depth/image_rect_raw")
ap.add_argument("--out", default=str(ROOT / "detection" / "data" / "real_eggs"))
ap.add_argument("--spots", type=int, default=10)
ap.add_argument("--place-s", type=float, default=12.0)
ap.add_argument("--record-s", type=float, default=4.0)
ap.add_argument("--every", type=float, default=0.2)
ap.add_argument("--fx", type=float, default=217.184)
ap.add_argument("--max-range", type=float, default=1.2, help="only look for the egg closer than this (m)")
ap.add_argument("--relabel", default=None, help="re-label a saved raw_XXX.npz without recording")
args = ap.parse_args()

rclpy.init()
node = Node("record_real_eggs")
latest = {}
node.create_subscription(Image, args.topic, lambda m: latest.__setitem__("m", m), qos_profile_sensor_data)


def frame():
    m = latest["m"]
    return np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.width).copy()


def spin_for(seconds, grab=False, every=args.every):
    out, t0, last = [], time.time(), 0.0
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.02)
        if grab and "m" in latest and time.time() - last >= every:
            out.append(frame())
            last = time.time()
    return out


def countdown(seconds, text):
    print(f"\a\n>>> {text}", flush=True)
    t_end = time.time() + seconds
    while time.time() < t_end:
        left = int(round(t_end - time.time()))
        print(f"    {left:2d} s", end="\r", flush=True)
        spin_for(min(1.0, max(0.0, t_end - time.time())))
    print("    go!   ", flush=True)


def background(bg_frames):
    """Median empty-scene depth + a per-pixel 'changed' threshold from its own noise (far pixels are noisier)."""
    ok = (bg_frames > 0).mean(0) >= 0.6
    with np.errstate(all="ignore"):
        bg = np.nan_to_num(np.where(ok, np.nanmedian(np.where(bg_frames > 0, bg_frames, np.nan), 0), 0))
        sd = np.nan_to_num(np.nanstd(np.where(bg_frames > 0, bg_frames, np.nan), 0))
    ok &= (bg > 0) & (bg < args.max_range)          # the workspace only: the far wall is too noisy
    thr = np.maximum(0.012, 4 * sd + 0.004).astype(np.float32)
    return bg.astype(np.float32), ok, thr


def label(depth, bg, bg_ok, why=None, thr=0.012):
    """Egg box from the background difference, or None (the reason is counted in `why`)."""
    why = why if why is not None else {}
    no = lambda k: (why.__setitem__(k, why.get(k, 0) + 1), None)[1]  # noqa: E731
    cur = depth.astype(np.float32) / 1000.0
    diff = bg - cur
    m = ((cur > 0) & bg_ok & (diff > thr) & (diff < 0.12)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # on a holey table only PIECES of the egg differ from the background: merge pieces that lie close together
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return no("nothing changed")
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    big = stats[order[0]]
    if big[cv2.CC_STAT_AREA] < 15:
        return no("change too small")
    if n > 2 and stats[order[1], cv2.CC_STAT_AREA] > 0.3 * big[cv2.CC_STAT_AREA] and \
            np.linalg.norm(cents[order[1]] - cents[order[0]]) > 40:        # a second big change elsewhere: ambiguous
        return no("several changes")
    # grow the egg: valid pixels near the change at the same depth (the whole visible egg, not only the changed part)
    seed = (lab == order[0]).astype(np.uint8)
    z = float(np.median(cur[(seed > 0) & (cur > 0)]))
    near = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    egg = near & (cur > 0) & (np.abs(cur - z) < 0.035)
    ys, xs = np.nonzero(egg)
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    size_w, size_h = w * z / args.fx, h * z / args.fx
    if not (0.02 <= min(size_w, size_h) and max(size_w, size_h) <= 0.09):     # not egg-sized (a hand, a shadow)
        return no(f"not egg-sized ({100 * size_w:.0f}x{100 * size_h:.0f} cm)" if max(size_w, size_h) < 1 else "not egg-sized")
    return [1.0, float(x), float(y), float(x + w), float(y + h)]


def write_outputs(frames, boxes, previews, shard):
    if not frames:
        print("no frames could be labelled - was the camera still and the egg in view?")
        return
    path = out / f"shard_{shard:03d}.npz"
    np.savez_compressed(path, depth=np.stack(frames), box=np.array(boxes, np.float32))
    tiles = []
    for d, b in previews:
        z = d.astype(np.float32) / 1000.0
        v = np.where(z > 0, 255 - np.clip((z - 0.15) / 1.35, 0, 1) * 200, 0).astype(np.uint8)
        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(v, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])), (0, 140, 255), 1)
        tiles.append(v)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out / f"preview_{shard:03d}.png"), grid)
    print(f"saved {len(frames)} labelled egg frames to {path} (preview: {out / f'preview_{shard:03d}.png'})")


out = Path(args.out)
if args.relabel:   # offline: label a saved recording again (same shard number as its raw file)
    raw = np.load(args.relabel)
    bg, bg_ok, thr = background(raw["background"].astype(np.float32) / 1000.0)
    frames, boxes, previews = [], [], []
    for k in sorted(set(raw["spot"].tolist())):
        ok, why = 0, {}
        for d in raw["depth"][raw["spot"] == k]:
            b = label(d, bg, bg_ok, why, thr)
            if b:
                frames.append(d); boxes.append(b); ok += 1
        print(f"spot {k}: {ok} labelled" + (f"  (skipped: {why})" if why else ""))
        if ok:
            previews.append((frames[-1], boxes[-1]))
    write_outputs(frames, boxes, previews, int(Path(args.relabel).stem.split("_")[1]))
    raise SystemExit

t0 = time.time()
while "m" not in latest and time.time() - t0 < 10:
    spin_for(0.5)
if "m" not in latest:
    raise SystemExit(f"no messages on {args.topic} - is ros2/d455.launch.py running?")

out.mkdir(parents=True, exist_ok=True)
shard = len(list(out.glob("shard_*.npz")))
countdown(10, "REMOVE THE EGG from view and keep the CAMERA STILL - recording the empty scene next")
bg_frames = np.stack(spin_for(3.0, grab=True)).astype(np.float32) / 1000.0
bg, bg_ok, thr = background(bg_frames)
print(f"    background: {len(bg_frames)} frames, {100 * bg_ok.mean():.0f}% valid", flush=True)

frames, boxes, previews = [], [], []
raw, raw_spot = [], []   # every recorded frame, kept so the labelling can be re-tuned offline
for k in range(1, args.spots + 1):
    countdown(args.place_s, f"spot {k}/{args.spots}: PLACE THE EGG somewhere new (turn it too), then HANDS OUT OF VIEW")
    got = spin_for(args.record_s, grab=True)
    raw += got
    raw_spot += [k] * len(got)
    ok, why = 0, {}
    for d in got:
        b = label(d, bg, bg_ok, why, thr)
        if b:
            frames.append(d)
            boxes.append(b)
            ok += 1
    print(f"    spot {k}: {ok}/{len(got)} frames labelled" + (f"  (skipped: {why})" if why else ""), flush=True)
    if ok:
        previews.append((frames[-1], boxes[-1]))
print("\a\n>>> done - you can take the egg away", flush=True)

np.savez_compressed(out / f"raw_{shard:03d}.npz", background=(bg_frames * 1000).astype(np.uint16),
                    depth=np.stack(raw), spot=np.array(raw_spot))
print(f"raw frames saved to {out / f'raw_{shard:03d}.npz'}")
write_outputs(frames, boxes, previews, shard)
node.destroy_node()
rclpy.shutdown()
