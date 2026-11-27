"""Measure real-time behaviour of the camera -> detection -> Isaac Sim twin chain (delay, rate, position lag).

Run while the twin runs (scripts/start_all.sh, any detector), and move the egg around for part of the time:
  source ./env.sh && python3 scripts/measure_sync.py --seconds 60
  -> printed summary + Documentation/tests/sync_<date_time>.json

Every message carries the timestamp of the camera image it came from, so (receive time - stamp) = delay since the
camera took the picture:
  camera        /camera/camera/color/image_raw    rate (fps) and driver delay
  detection     /egg/points (PoseArray)           delay camera -> egg X, Y, Z (yolo_ros + egg_xyz, or free_detector)
  Isaac Sim     /twin/egg_poses (PoseArray)       delay camera -> sim egg moved (first sim frame showing that image)
Position: the sim egg vs the real detection of the same image. X, Y should match (difference = the twin's smoothing,
larger while the egg moves); Z differs by a constant on the desk rig (real table height != sim table), so the
spread of the Z difference is reported, not its value.
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--prefix", default="", help="/sim for the sim-only mode")
args = ap.parse_args()


def st(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Sync(Node):
    def __init__(self):
        super().__init__("measure_sync")
        p = args.prefix
        self.cam, self.det, self.twin, self.err = [], [], [], []
        self.real_by_stamp = {}
        self.held = 0
        self.seen_twin = set()
        self.create_subscription(Image, f"{p}/camera/camera/color/image_raw", self.on_cam, qos_profile_sensor_data)
        self.create_subscription(PoseArray, f"{p}/egg/points", self.on_det, 10)
        self.create_subscription(PoseArray, "/twin/egg_poses", self.on_twin, 10)

    def now(self):
        return time.time()

    def on_cam(self, m):
        self.cam.append((self.now(), st(m.header.stamp)))

    def on_det(self, m):
        if m.poses:
            s = st(m.header.stamp)
            self.det.append((self.now(), s))
            self.real_by_stamp[s] = [(q.position.x, q.position.y, q.position.z) for q in m.poses]

    def on_twin(self, m):
        s = st(m.header.stamp)
        if not m.poses or s in self.seen_twin:
            return
        self.seen_twin.add(s)
        self.twin.append((self.now(), s))
        real = self.real_by_stamp.get(s)
        if real:
            for q in m.poses:
                t = (q.position.x, q.position.y, q.position.z)
                r = min(real, key=lambda r: math.dist(r[:2], t[:2]))
                if math.dist(r[:2], t[:2]) <= 0.10:      # same egg (the twin's MATCH_M); others = eggs the twin
                    self.err.append((math.dist(r[:2], t[:2]), t[2] - r[2]))   # still holds from earlier frames
                else:
                    self.held += 1


def stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def rate(rows):
    if len(rows) < 2:
        return 0.0
    t = [r[0] for r in rows]
    return (len(t) - 1) / (t[-1] - t[0])


rclpy.init()
n = Sync()
print(f"measuring {args.seconds:.0f} s - move the egg around for part of the time ...", flush=True)
t0 = time.time()
while time.time() - t0 < args.seconds:
    rclpy.spin_once(n, timeout_sec=0.05)

res = {
    "date": datetime.now().isoformat(timespec="seconds"), "seconds": args.seconds,
    "camera_fps": rate(n.cam), "detection_hz": rate(n.det), "twin_updates_hz": rate(n.twin),
    "delay_camera_driver_ms": stats([1000 * (r - s) for r, s in n.cam]),
    "delay_camera_to_detection_ms": stats([1000 * (r - s) for r, s in n.det]),
    "delay_camera_to_isaac_ms": stats([1000 * (r - s) for r, s in n.twin]),
    "twin_vs_real_xy_cm": stats([100 * e[0] for e in n.err]),
    "twin_eggs_held_from_earlier_frames": n.held,
    "twin_vs_real_z_spread_cm": (lambda z: stats(100 * np.abs(z - np.median(z))) if len(z) else None)(
        np.array([e[1] for e in n.err])),
}
out = ROOT / "Documentation" / "tests"
out.mkdir(parents=True, exist_ok=True)
f = out / f"sync_{datetime.now():%Y%m%d_%H%M%S}.json"
f.write_text(json.dumps(res, indent=2))


def line(name, s, unit):
    return f"  {name:34s} " + (f"mean {s['mean']:7.1f} {unit}  median {s['median']:7.1f}  95% {s['p95']:7.1f}  max {s['max']:7.1f}  (n {s['n']})"
                               if s else "no data")


print(f"\ncamera {res['camera_fps']:.1f} fps   detections {res['detection_hz']:.1f} Hz   Isaac Sim twin updates "
      f"{res['twin_updates_hz']:.1f} Hz")
print(line("delay camera driver", res["delay_camera_driver_ms"], "ms"))
print(line("delay camera -> egg X, Y, Z", res["delay_camera_to_detection_ms"], "ms"))
print(line("delay camera -> Isaac Sim egg", res["delay_camera_to_isaac_ms"], "ms"))
print(line("sim egg vs real egg, X/Y", res["twin_vs_real_xy_cm"], "cm"))
print(line("sim egg vs real egg, Z spread", res["twin_vs_real_z_spread_cm"], "cm"))
print(f"  sim eggs kept from earlier frames (other egg not in this image): {n.held}")
print(f"\nsaved {f}")
n.destroy_node()
