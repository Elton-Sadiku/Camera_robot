"""Measure the camera's tilt (pitch, roll) and height above the table from the D455 depth - for MOUNT_RPY.

The table is assumed level. Clear the table (no egg, no hands), start the camera (scripts/start_all.sh real, or
ros2 launch ros2/d455.launch.py), then:
  source ./env.sh && python3 scripts/measure_mount.py
  -> pitch (down +), roll, camera height above the table, the launch arguments, Documentation/tests/mount_<time>.json

How: 15 aligned-depth frames are averaged; the lower 60 % of the image (where the table is) is turned into 3D
points with the camera intrinsics; RANSAC finds the biggest plane = the table; its normal is "up". A level camera
sees up as (0, -1, 0) in the optical frame (x right, y down, z forward); a camera tilted down by p sees
(0, -cos p, -sin p), a roll r moves it sideways. The plane's distance from the camera = height above the table.
Check: put the result into the launch (mount_pitch / mount_roll) - a lying egg's Z should then not change when it is
moved around the table.
"""
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

ROOT = Path(__file__).resolve().parent.parent
TOPIC = "/camera/camera/aligned_depth_to_color"


class Grab(Node):
    def __init__(self):
        super().__init__("measure_mount")
        self.br, self.frames, self.k = CvBridge(), [], None
        self.create_subscription(Image, f"{TOPIC}/image_raw", self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f"{TOPIC}/camera_info", self.on_info, qos_profile_sensor_data)

    def on_depth(self, m):
        d = self.br.imgmsg_to_cv2(m).astype(np.float32)
        self.frames.append(d * (0.001 if m.encoding == "16UC1" else 1.0))

    def on_info(self, m):
        self.k = np.array(m.k).reshape(3, 3)


rclpy.init()
n = Grab()
t0 = time.time()
while (len(n.frames) < 15 or n.k is None) and time.time() - t0 < 15:
    rclpy.spin_once(n, timeout_sec=0.1)
if len(n.frames) < 5 or n.k is None:
    raise SystemExit(f"no depth on {TOPIC} - start the camera first (scripts/start_all.sh real)")
stack = np.stack(n.frames)
stack[stack <= 0] = np.nan
depth = np.nanmedian(stack, axis=0)                       # median of the frames: less noise, no holes
h, w = depth.shape
fx, fy, cx, cy = n.k[0, 0], n.k[1, 1], n.k[0, 2], n.k[1, 2]
v, u = np.mgrid[int(0.4 * h):h:2, 0:w:2]                  # lower 60 % of the image, every 2nd pixel
z = depth[v, u]
ok = np.isfinite(z) & (z > 0.15) & (z < 2.0)
P = np.stack([(u[ok] - cx) / fx * z[ok], (v[ok] - cy) / fy * z[ok], z[ok]], 1)
if len(P) < 500:
    raise SystemExit("too few depth points on the table")

rng = np.random.default_rng(0)
best, best_in = None, None
for _ in range(400):                                        # RANSAC plane, 5 mm tolerance
    a, b, c = P[rng.choice(len(P), 3, replace=False)]
    nrm = np.cross(b - a, c - a)
    if np.linalg.norm(nrm) < 1e-9:
        continue
    nrm /= np.linalg.norm(nrm)
    inl = np.abs((P - a) @ nrm) < 0.005
    if best_in is None or inl.sum() > best_in.sum():
        best, best_in = nrm, inl
Q = P[best_in]                                              # refine on the inliers (least squares)
c = Q.mean(0)
nrm = np.linalg.svd(Q - c)[2][2]
if nrm[1] > 0:                                              # point the normal up (y is down in the optical frame)
    nrm = -nrm
height = float(abs(c @ nrm))
pitch = math.atan2(-nrm[2], -nrm[1])                        # + = looking down
roll = math.atan2(nrm[0], -nrm[1])
flat = 1000 * float(np.std((Q - c) @ nrm))
res = {"date": datetime.now().isoformat(timespec="seconds"), "frames": len(n.frames), "table_points": int(len(Q)),
       "share_of_points_on_table": float(best_in.mean()), "pitch_deg": math.degrees(pitch),
       "roll_deg": math.degrees(roll), "pitch_rad": pitch, "roll_rad": roll, "height_above_table_m": height,
       "table_flatness_mm": flat}
out = ROOT / "Documentation" / "tests"
out.mkdir(parents=True, exist_ok=True)
f = out / f"mount_{datetime.now():%Y%m%d_%H%M%S}.json"
f.write_text(json.dumps(res, indent=2))
print(f"table plane from {len(Q)} points ({100 * best_in.mean():.0f} % of the lower image), flatness {flat:.1f} mm")
print(f"  pitch  {math.degrees(pitch):6.2f} deg down")
print(f"  roll   {math.degrees(roll):6.2f} deg")
print(f"  camera {100 * height:5.1f} cm above the table")
print(f"\nlaunch:    ros2 launch ros2/d455.launch.py mount_pitch:={pitch:.4f} mount_roll:={roll:.4f}")
print(f"start_all: MOUNT_ARGS=\"mount_pitch:={pitch:.4f} mount_roll:={roll:.4f}\"")
print(f"robot:     shared/d455_model.py MOUNT_RPY = ({roll:.4f}, {pitch:.4f}, -1.5707963)  (+ rebuild the sim with "
      f"--pitch-deg {math.degrees(pitch):.2f} --roll-deg {math.degrees(roll):.2f})")
print(f"saved {f}")
n.destroy_node()
