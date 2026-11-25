"""Egg detection by GEOMETRY (no neural network): the egg is an egg-sized bump on the table surface.

Per depth frame:
  1. depth -> points (camera optical frame), after the same masked Gaussian the learned detector uses
  2. table plane by RANSAC on the points 0.16-1.2 m away (the largest flat surface), smoothed over frames
  3. points 1-7 cm ABOVE the plane = things standing on the table
  4. group them in the image (connected components) and keep a group whose real size is an egg's:
     footprint 3-8 cm, height 2.5-7 cm, enough points; the best match to 57 x 44 mm wins
  5. same tracker as the learned detector: start after 3 of 5 frames, hold 0.5 s, smooth the position

Runs next to ros2/egg_detector_node.py (learned) so the two can be compared, on its own topics:
  egg_geo/point        geometry_msgs/PointStamped   egg centre (top of the bump minus the egg radius), metres
  egg_geo/detection    std_msgs/Float32MultiArray   [score, x0, y0, x1, y1, X, Y, Z, height_cm]  (empty = none)
  egg_geo/distance_cm  std_msgs/Float32
  egg_geo/debug_image  sensor_msgs/Image bgr8       depth, table points green, bump points red, egg box orange

  python3 ros2/egg_geometry_node.py                                         # real D455
  python3 ros2/egg_geometry_node.py --ros-args -r __ns:=/sim -p depth_topic:=/sim/camera/camera/depth/image_rect_raw
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float32MultiArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.d455_model import Z_RANGE, denoise, intrinsics, to_obs  # noqa: E402

EGG_L, EGG_D = 0.057, 0.044


class EggGeometry(Node):
    def __init__(self):
        super().__init__("egg_geometry")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        depth_topic = p("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.max_range = float(p("max_range_m", 1.2))
        self.plane_tol = float(p("plane_tolerance_m", 0.008))     # table inlier distance
        self.h_min, self.h_max = float(p("min_height_m", 0.01)), float(p("max_height_m", 0.07))
        self.foot_min, self.foot_max = float(p("min_footprint_m", 0.03)), float(p("max_footprint_m", 0.08))
        self.top_min = float(p("min_top_m", 0.025))                 # egg lying: ~44 mm tall, standing: ~57 mm
        self.min_points = int(p("min_points", 25))
        self.hits, self.window, self.hold_s = int(p("confirm_hits", 3)), int(p("confirm_window", 5)), float(p("hold_s", 0.5))
        self.smooth = float(p("smooth", 0.5))
        k = intrinsics(decimation=1)
        self.K = (k["fx"], k["fy"], k["cx"], k["cy"])
        self.plane = None          # (n, d): n . x + d = 0, n pointing towards the camera side (up from the table)
        self.track, self.last_seen, self.recent = None, 0.0, []
        self.rng = np.random.default_rng(0)
        torch.set_num_threads(2)   # share the CPU with the camera node and the learned detector
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, depth_topic.rsplit("/", 1)[0] + "/camera_info", self.on_info, qos_profile_sensor_data)
        self.pub_pt = self.create_publisher(PointStamped, "egg_geo/point", 1)
        self.pub_det = self.create_publisher(Float32MultiArray, "egg_geo/detection", 1)
        self.pub_dist = self.create_publisher(Float32, "egg_geo/distance_cm", 1)
        self.pub_img = self.create_publisher(Image, "egg_geo/debug_image", 1)
        self.get_logger().info(f"geometric egg detector on {depth_topic}")

    def on_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    # ------------------------------------------------------------------ geometry
    def points(self, z):
        fx, fy, cx, cy = self.K
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return np.dstack(((u - cx) * z / fx, (v - cy) * z / fy, z))          # (H, W, 3), optical frame

    def fit_plane(self, P):
        """RANSAC plane on (N, 3) points -> (n, d) with n towards the camera, or None."""
        if len(P) < 200:
            return None
        S = P[self.rng.choice(len(P), min(len(P), 4000), replace=False)]
        # all 200 plane hypotheses at once: one (N x 200) matrix product instead of a Python loop (145 ms -> ~3 ms)
        a, b, c = (S[i] for i in self.rng.integers(0, len(S), (3, 200)))
        N = np.cross(b - a, c - a)
        norm = np.linalg.norm(N, axis=1)
        ok = norm > 1e-9
        if not ok.any():
            return None
        N, a = N[ok] / norm[ok, None], a[ok]
        D = -(N * a).sum(1)
        counts = (np.abs(S @ N.T + D) < self.plane_tol).sum(0)
        k = int(np.argmax(counts))
        if counts[k] < 0.15 * len(S):
            return None
        n, d = N[k], D[k]
        inl = S[np.abs(S @ n + d) < self.plane_tol]                          # least-squares refit on the inliers
        c = inl.mean(0)
        n = np.linalg.eigh(np.cov((inl - c).T))[1][:, 0]                     # normal = smallest-variance axis
        d = -n @ c
        if d < 0:                                                            # camera (origin) on the positive side
            n, d = -n, -d
        return n, d

    @torch.inference_mode()
    def on_depth(self, msg: Image):
        dtype = {"16UC1": np.uint16, "32FC1": np.float32}[msg.encoding]
        raw = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.width)
        z = torch.from_numpy(raw.astype(np.float32) * (0.001 if dtype == np.uint16 else 1.0))
        z = denoise(to_obs(z)).numpy()                                       # same masked Gaussian as the learned detector
        z = np.where(z < self.max_range, z, 0.0)
        P = self.points(z)
        valid = z > 0
        plane = self.fit_plane(P[valid])
        if plane is not None:                                                # the table is static: smooth it over frames
            if self.plane is not None and plane[0] @ self.plane[0] > 0.95:
                n = 0.8 * self.plane[0] + 0.2 * plane[0]
                n /= np.linalg.norm(n)
                plane = (n, 0.8 * self.plane[1] + 0.2 * plane[1])
            self.plane = plane
        out, height_map = [], None
        if self.plane is not None:
            n, d = self.plane
            height_map = np.where(valid, P @ n + d, 0.0)                     # metres above the table
            bump = valid & (height_map > self.h_min) & (height_map < self.h_max)
            m = cv2.morphologyEx(bump.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cnt, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            cands = []
            for i in range(1, cnt):
                sel = (lab == i) & bump
                if np.count_nonzero(sel) < self.min_points:
                    continue
                pts, hh = P[sel], height_map[sel]
                flat = pts - np.outer(pts @ n + d, n)                        # project onto the table
                c = flat.mean(0)
                ev, evec = np.linalg.eigh(np.cov((flat - c).T))
                ax = flat - c
                extent = sorted([np.ptp(ax @ evec[:, k]) for k in (1, 2)])    # footprint: short, long side
                top = float(np.percentile(hh, 95))
                if not (self.foot_min <= extent[1] <= self.foot_max and extent[0] >= 0.6 * self.foot_min
                        and self.top_min <= top <= self.h_max):
                    continue
                # score: how close to an egg (footprint 57 x 44 lying, 44 x 44 standing; height 44 or 57 mm)
                err = abs(extent[1] - EGG_L) / EGG_L + abs(extent[0] - EGG_D) / EGG_D + min(abs(top - EGG_D), abs(top - EGG_L)) / EGG_D
                score = float(np.exp(-err))
                # centre: the bump's footprint centre, lifted to half the egg's height above the table
                centre = c + n * (top / 2)
                x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
                cands.append([score, x, y, x + w, y + h, *centre.tolist(), 100 * top])
            cands.sort(key=lambda c: -c[0])
            out = self.update_track(cands)
        if out:
            s, x0, y0, x1, y1, X, Y, Z, top_cm = out
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x, pt.point.y, pt.point.z = X, Y, Z
            self.pub_pt.publish(pt)
            dist = 100.0 * float(np.sqrt(X * X + Y * Y + Z * Z))
            self.pub_dist.publish(Float32(data=dist))
            self.get_logger().info(f"egg (geometry) {dist:5.1f} cm  (X {100 * X:+5.1f}, Y {100 * Y:+5.1f}, Z {100 * Z:5.1f} cm, "
                                   f"{top_cm:.1f} cm tall, score {s:.2f})", throttle_duration_sec=0.5)
        self.pub_det.publish(Float32MultiArray(data=[float(v) for v in out]))
        if self.pub_img.get_subscription_count():
            self.pub_img.publish(self.debug_image(msg, z, height_map, out))

    def update_track(self, cands):
        now = self.get_clock().now().nanoseconds * 1e-9
        near = lambda a, b: np.linalg.norm(np.subtract(a[5:8], b[5:8])) < 0.04  # noqa: E731
        if self.track is not None:
            match = next((c for c in cands if near(c, self.track)), None)
            if match:
                a = self.smooth
                self.track = [match[0]] + [a * m + (1 - a) * t for m, t in zip(match[1:], self.track[1:])]
                self.last_seen = now
            elif now - self.last_seen > self.hold_s:
                self.track = None
        if self.track is None:
            cand = cands[0] if cands else None
            self.recent = (self.recent + [cand])[-self.window:]
            if cand and sum(1 for r in self.recent if r and near(r, cand)) >= self.hits:
                self.track, self.last_seen, self.recent = list(cand), now, []
        return list(self.track) if self.track is not None else []

    def debug_image(self, msg, z, hmap, det):
        g = np.where(z > 0, 255 - np.clip((z - Z_RANGE[0]) / (self.max_range - Z_RANGE[0]), 0, 1) * 200, 0).astype(np.uint8)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if hmap is not None:
            on_table = (z > 0) & (np.abs(hmap) < self.plane_tol)
            above = (z > 0) & (hmap > self.h_min) & (hmap < self.h_max)
            img[on_table] = (0.6 * img[on_table] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
            img[above] = (40, 40, 230)
        big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        if det:
            x0, y0, x1, y1 = [int(round(2 * v)) for v in det[1:5]]
            dist = 100.0 * float(np.sqrt(det[5] ** 2 + det[6] ** 2 + det[7] ** 2))
            cv2.rectangle(big, (x0, y0), (x1, y1), (0, 140, 255), 2)
            cv2.putText(big, f"egg {dist:.1f} cm", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        m = Image(header=msg.header, height=big.shape[0], width=big.shape[1], encoding="bgr8", step=big.shape[1] * 3)
        m.data = big.tobytes()
        return m


def main():
    rclpy.init()
    node = EggGeometry()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
