"""Guided recording of REAL D455 colour (+ aligned depth) frames for training the egg YOLO.

Needs the real camera running (ros2 launch ros2/d455.launch.py). Timed, no keyboard: for each placement it says
where to put the egg (MOVE, ~7 s), then records while you keep your hands OUT of view (HOLD, ~4 s).
The last placements are WITHOUT the egg (empty scenes = negatives).

  source ./env.sh && python3 detection/record_egg_yolo.py            # -> detection/data/yolo_egg/session_XXX.npz
  python3 detection/record_egg_yolo.py --set balls                   # ping-pong ball (NO egg): teaches "ball, not egg"
  python3 detection/record_egg_yolo.py --move-s 9 --hold-s 4         # more time to move

Then: python3 detection/label_egg_yolo.py  (auto boxes)  and  isaac/isaac_ros2.sh ... train (see README).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parent.parent
EGG_PLACES = [  # (egg?, where)
    (True, "lying on its side, ~20 cm in front, CENTRE"),
    (True, "lying, ~20 cm, LEFT"),
    (True, "lying, ~25 cm, RIGHT, turned the other way"),
    (True, "STANDING upright, ~25 cm, centre"),
    (True, "lying, ~35 cm, next to the spool"),
    (True, "STANDING, ~35 cm, left"),
    (True, "lying, ~45 cm, right"),
    (True, "ON TOP of the spool"),
    (True, "STANDING, behind the spool (~45 cm)"),
    (True, "lying, ~55 cm, left, on the far mat"),
    (True, "lying, ~65 cm, centre, far mat edge"),
    (True, "STANDING, ~65 cm, right"),
    (True, "lying on the WHITE table area (off the mat)"),
    (True, "lying, ~30 cm, pointed at the camera (end-on)"),
    (True, "anywhere you like, tilted / on a paper or object"),
    (False, "REMOVE the egg - empty table"),
    (False, "NO egg - move the spool / put other objects in view"),
    (False, "NO egg - your hand resting on the table, in view"),
]
BALL_PLACES = [  # the ping-pong ball (or another round thing), NO egg anywhere in view
    ("ball", "BALL on its holder, ~30 cm, LEFT  (egg out of view!)"),
    ("ball", "BALL on its holder, ~30 cm, CENTRE"),
    ("ball", "BALL on its holder, ~45 cm, RIGHT"),
    ("ball", "BALL lying on the mat, ~25 cm, centre"),
    ("ball", "BALL lying on the mat, ~40 cm, left"),
    ("ball", "BALL on the white table / notebook"),
    ("ball", "BALL far, ~60 cm, on the far mat"),
    ("ball", "BALL next to the headphones / other objects"),
    ("ball", "BALL, holder turned / tilted, ~35 cm"),
    ("ball", "BALL anywhere you like (a different spot)"),
    ("none", "REMOVE the ball - empty table"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--move-s", type=float, default=7.0)
ap.add_argument("--hold-s", type=float, default=4.0)
ap.add_argument("--every", type=int, default=3, help="keep every Nth 30 Hz frame while holding")
ap.add_argument("--out", default=str(ROOT / "detection" / "data" / "yolo_egg"))
ap.add_argument("--set", choices=["eggs", "balls"], default="eggs", help="eggs: 18 egg placements, balls: ping-pong ball")
args = ap.parse_args()
# (kind, where): kind True / "egg" = egg in view, "ball" = ball only (labelled class 'ball'), False / "none" = empty
PLACES = EGG_PLACES if args.set == "eggs" else BALL_PLACES
IS_EGG = [k is True or k == "egg" for k, _ in PLACES]
IS_BALL = [k == "ball" for k, _ in PLACES]

rclpy.init()
node = rclpy.create_node("record_egg_yolo")
br = CvBridge()
state = {"rec": False, "place": -1, "k": 0}
col, dep, place = [], [], []


def cb(c, d):
    if not state["rec"]:
        return
    state["k"] += 1
    if state["k"] % args.every:
        return
    col.append(br.imgmsg_to_cv2(c, "bgr8"))
    dep.append(br.imgmsg_to_cv2(d, "passthrough"))
    place.append(state["place"])


qos = qos_profile_sensor_data
sync = ApproximateTimeSynchronizer([Subscriber(node, Image, "/camera/camera/color/image_raw", qos_profile=qos),
                                    Subscriber(node, Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                               qos_profile=qos)], 5, 0.02)
sync.registerCallback(cb)


def wait(sec, label):
    t0 = time.time()
    shown = -1
    while time.time() - t0 < sec:
        rclpy.spin_once(node, timeout_sec=0.02)
        left = int(sec - (time.time() - t0)) + 1
        if left != shown:
            print(f"\r   {label} {left:2d} s ", end="", flush=True)
            shown = left
    print()


print("Waiting for the camera ...", flush=True)
t0 = time.time()
state["rec"] = True
while not col and time.time() - t0 < 10:
    rclpy.spin_once(node, timeout_sec=0.1)
state["rec"] = False
if not col:
    sys.exit("no frames on /camera/camera/color/image_raw + aligned_depth_to_color - start ros2/d455.launch.py")
col.clear(), dep.clear(), place.clear()

print(f"\n{len(PLACES)} placements, {args.move_s:.0f} s to move + {args.hold_s:.0f} s recording each.\n", flush=True)
for i, (kind, where) in enumerate(PLACES):
    print(f"\a\n[{i + 1}/{len(PLACES)}] MOVE: {where}", flush=True)
    wait(args.move_s, "move ...")
    print("\a   HOLD - hand stays in view" if "hand" in where else "\a   HOLD - hands out of view!", flush=True)
    state.update(rec=True, place=i)
    wait(args.hold_s, "recording")
    state["rec"] = False
print("\a\n>>> done - saving", flush=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
n = len(list(out.glob("session_*.npz")))
path = out / f"session_{n:03d}.npz"
np.savez_compressed(path, color=np.array(col), depth=np.array(dep), place=np.array(place),
                    has_egg=np.array([IS_EGG[p] for p in place]), has_ball=np.array([IS_BALL[p] for p in place]),
                    where=np.array([w for _, w in PLACES]))
print(f"saved {len(col)} frames ({sum(IS_EGG[p] for p in place)} with egg, {sum(IS_BALL[p] for p in place)} with ball) -> {path}",
      flush=True)
