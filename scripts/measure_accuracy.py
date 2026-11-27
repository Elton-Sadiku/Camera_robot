"""Position accuracy test with a ruler: does the measured egg position move exactly as far as the real egg?

Put a ruler (or tape) flat on the table. Place the egg at a mark, type the mark in cm and press Enter; the script
averages egg/point for 2 s. Repeat at 4-6 marks (e.g. 0, 10, 20, 30, 40 cm), then press Enter on an empty line.
Run with the detection running (scripts/start_all.sh real|twin, any detector):
  source ./env.sh && python3 scripts/measure_accuracy.py
  -> summary + Documentation/tests/accuracy_<date_time>.json

Why distances along a ruler: the absolute base_link position of a desk test rig is hard to measure by hand, but a
ruler gives exact DIFFERENCES. Each measured distance between two marks is compared with the ruler distance.
Reported:
  distance error   |measured distance - ruler distance| for every pair of marks (cm)
  straightness     how far the measured points are from a straight line (the ruler is straight) (cm)
  height spread    the egg lies on a flat table, so its Z should not change (cm)
  jitter           standard deviation of the 2 s of measurements at one mark (repeatability) (mm)
  distance from camera_link at every mark, for reference
"""
import itertools
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

ROOT = Path(__file__).resolve().parent.parent
TOPIC = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "/egg/point"


class Acc(Node):
    def __init__(self):
        super().__init__("measure_accuracy")
        self.buf, self.lock = [], threading.Lock()
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(PointStamped, TOPIC, self.cb, 10)

    def cb(self, m):
        with self.lock:
            self.buf.append((time.time(), (m.point.x, m.point.y, m.point.z)))

    def grab(self, secs=2.0):
        t0 = time.time()
        time.sleep(secs)
        with self.lock:
            pts = [p for t, p in self.buf if t >= t0]
        return np.array(pts)


rclpy.init()
n = Acc()
threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
time.sleep(1.0)
try:
    cam = n.tf.lookup_transform("base_link", "camera_link", Time()).transform.translation
    cam = np.array([cam.x, cam.y, cam.z])
except Exception:
    cam = None
print(f"listening on {TOPIC}. Place the egg at a ruler mark, type the mark in cm, Enter. Empty line = done.\n")
marks = []
while True:
    s = input("ruler mark (cm), or Enter to finish: ").strip().replace(",", ".")
    if not s:
        break
    try:
        r = float(s)
    except ValueError:
        print("  a number please")
        continue
    pts = n.grab()
    if len(pts) < 5:
        print(f"  only {len(pts)} measurements in 2 s - is the egg detected (20 cm - 1 m)? try again")
        continue
    p = np.median(pts, axis=0)
    jit = 1000 * float(np.linalg.norm(pts.std(axis=0)))
    d = float(np.linalg.norm(p - cam)) * 100 if cam is not None else float("nan")
    marks.append({"ruler_cm": r, "xyz_m": p.tolist(), "n": len(pts), "jitter_mm": jit, "dist_from_camera_cm": d})
    print(f"  {r:5.1f} cm: X {p[0]:+.3f} Y {p[1]:+.3f} Z {p[2]:+.3f} m   {d:5.1f} cm from camera   jitter {jit:.1f} mm"
          f"   ({len(pts)} values)")

if len(marks) < 2:
    sys.exit("need at least 2 marks")
P = np.array([m["xyz_m"] for m in marks])
R = np.array([m["ruler_cm"] for m in marks])
pair_err = [abs(100 * np.linalg.norm(P[i] - P[j]) - abs(R[i] - R[j])) for i, j in itertools.combinations(range(len(P)), 2)]
c = P.mean(0)                                                            # best straight line through the points
u = np.linalg.svd(P - c)[2][0]
straight = [100 * float(np.linalg.norm((p - c) - np.dot(p - c, u) * u)) for p in P]
span_meas = 100 * float(np.ptp((P - c) @ u))
res = {"date": datetime.now().isoformat(timespec="seconds"), "topic": TOPIC, "marks": marks,
       "distance_error_cm": {"mean": float(np.mean(pair_err)), "max": float(np.max(pair_err))},
       "scale_measured_over_ruler": span_meas / float(np.ptp(R)) if np.ptp(R) else None,
       "straightness_cm": {"mean": float(np.mean(straight)), "max": float(np.max(straight))},
       "height_spread_cm": 100 * float(np.ptp(P[:, 2])),
       "jitter_mm_mean": float(np.mean([m["jitter_mm"] for m in marks]))}
out = ROOT / "Documentation" / "tests"
out.mkdir(parents=True, exist_ok=True)
f = out / f"accuracy_{datetime.now():%Y%m%d_%H%M%S}.json"
f.write_text(json.dumps(res, indent=2))
print(f"\n{len(marks)} marks over {np.ptp(R):.0f} cm of ruler")
print(f"  distance error   mean {res['distance_error_cm']['mean']:.1f} cm   max {res['distance_error_cm']['max']:.1f} cm")
print(f"  scale            measured / ruler = {res['scale_measured_over_ruler']:.3f}  (1.000 = perfect)")
print(f"  straightness     mean {res['straightness_cm']['mean']:.1f} cm   max {res['straightness_cm']['max']:.1f} cm")
print(f"  height spread    {res['height_spread_cm']:.1f} cm (egg on a flat table: should be ~0)")
print(f"  jitter           {res['jitter_mm_mean']:.1f} mm (2 s at one mark)")
print(f"\nsaved {f}")
