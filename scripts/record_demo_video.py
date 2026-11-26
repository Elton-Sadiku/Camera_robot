"""Record a demo video of the live egg detection: yolo_ros debug image + the egg's X, Y, Z and distance.

Needs the pipeline running (ros2/d455.launch.py, ros2/egg_yolo.launch.py). Usage:
  source ./env.sh && python3 scripts/record_demo_video.py --seconds 30 --out Documentation/egg_demo.mp4
  (sim: --ns /sim)
The MP4 is written with OpenCV (mp4v) and, if ffmpeg is installed, re-encoded to H.264 so it plays everywhere
(phones, WhatsApp, Teams, PowerPoint).
"""
import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
from yolo_msgs.msg import DetectionArray

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--out", default="Documentation/egg_demo.mp4")
ap.add_argument("--ns", default="", help="'' for the real camera, /sim for Isaac Sim")
ap.add_argument("--fps", type=float, default=15.0)
args = ap.parse_args()

rclpy.init()
node = rclpy.create_node("record_demo_video")
br = CvBridge()
tf = Buffer()
TransformListener(tf, node)
state = {"img": None, "egg": None, "t_egg": 0.0}


def on_img(m):
    state["img"] = br.imgmsg_to_cv2(m, "bgr8")


def on_det(m):
    eggs = sorted((d for d in m.detections if d.class_name == "egg" and d.bbox3d.frame_id), key=lambda d: -d.score)
    if not eggs:
        return
    d = eggs[0]
    c = d.bbox3d.center.position
    dist = float("nan")
    try:
        t = tf.lookup_transform(d.bbox3d.frame_id, "camera_link", Time()).transform.translation
        dist = math.dist((c.x, c.y, c.z), (t.x, t.y, t.z))
    except Exception:
        pass
    state["egg"] = (d.score, c.x, c.y, c.z, dist, d.bbox3d.frame_id)
    state["t_egg"] = time.time()


node.create_subscription(Image, f"{args.ns}/yolo/dbg_image", on_img, qos_profile_sensor_data)
node.create_subscription(DetectionArray, f"{args.ns}/yolo/detections_3d", on_det, 10)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
raw = out.with_suffix(".raw.mp4")
writer = None
t0 = time.time()
next_frame = t0
n = 0
print(f"recording {args.seconds:.0f} s from {args.ns}/yolo/dbg_image ...", flush=True)
while time.time() - t0 < args.seconds:
    rclpy.spin_once(node, timeout_sec=0.005)
    if state["img"] is None or time.time() < next_frame:
        continue
    next_frame = max(next_frame + 1.0 / args.fps, time.time())   # steady pace, no burst after a late start
    im = state["img"].copy()
    h, w = im.shape[:2]
    if writer is None:
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h + 70))
    bar = cv2.copyMakeBorder(im, 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    fresh = state["egg"] is not None and time.time() - state["t_egg"] < 0.5
    if fresh:
        s, x, y, z, dist, frame = state["egg"]
        l1 = f"EGG  score {s:.2f}   distance {100 * dist:.1f} cm"
        l2 = f"X {x:+.3f}  Y {y:+.3f}  Z {z:+.3f} m  ({frame})"
        col = (80, 220, 80)
    else:
        l1, l2, col = "no egg", "", (120, 120, 255)
    cv2.putText(bar, l1, (12, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    cv2.putText(bar, l2, (12, h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(bar, "D455 + yolo_ros 3D  |  meshari egg detection", (w - 420, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    writer.write(bar)
    n += 1
    if n % int(args.fps * 5) == 0:
        print(f"  {time.time() - t0:4.0f} s", flush=True)
if writer is None:
    raise SystemExit(f"no images on {args.ns}/yolo/dbg_image - is egg_yolo.launch.py running?")
writer.release()
if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
else:
    raw.rename(out)
print(f"saved {n} frames ({n / args.fps:.0f} s) -> {out}", flush=True)
