"""Egg detection on D455 depth: depth topic -> shared.d455_model.real_obs -> egg detector -> egg position.

Same node for the real camera and the simulator (16UC1 from the real D455, 32FC1 from Isaac Sim):
  real:  python3 ros2/egg_detector_node.py
  sim :  python3 ros2/egg_detector_node.py --ros-args -r __ns:=/sim -p depth_topic:=/sim/camera/camera/depth/image_rect_raw
Run both at once to compare sim and real side by side (/egg/... and /sim/egg/...).

Publishes (relative to the node namespace):
  egg/point        geometry_msgs/PointStamped   egg centre in the camera's depth optical frame (metres)
  egg/detection    std_msgs/Float32MultiArray   [score, x0, y0, x1, y1, X, Y, Z]  (empty = no egg)
  egg/distance_cm  std_msgs/Float32             straight-line distance camera -> egg centre, in cm
  egg/debug_image  sensor_msgs/Image bgr8       depth (2x) with the box and the distance in cm (rqt_image_view)
The distance is also printed in this terminal twice per second.

An egg is reported only if it passes every check (all are ROS parameters, e.g. -p score_threshold:=0.5):
  score_threshold 0.35         network confidence needed to START tracking an egg
  track_threshold 0.15         lower confidence that KEEPS tracking it at the same spot (the real score jumps)
  hold_s 0.5 / smooth 0.5      keep the last position through short misses; smooth the position
  min_valid_fraction 0.6       the box centre is a solid surface (real D455 speckle is not)
  min_size_m / max_size_m      0.02 / 0.09: real size of the box at its depth (egg: 44-57 mm)
  max_distance_m 1.2           trained with eggs 0.5-0.95 m away
  confirm_hits / confirm_window / confirm_distance_m   3 / 5 / 0.04: same spot in 3 of the last 5 frames
Model: detection/models/egg_detector.ts.pt (train with detection/train_egg_detector.py).
"""
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float32MultiArray

try:
    import cv2  # only for drawing the debug image (ships with ROS 2 Humble)
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detection.egg_model import decode  # noqa: E402
from shared.d455_model import Z_RANGE, detector_input, intrinsics, real_obs  # noqa: E402

EGG_RADIUS = 0.022  # m, half the egg's short axis: visible surface -> egg centre


class EggDetector(Node):
    def __init__(self):
        super().__init__("egg_detector")
        model = self.declare_parameter("model", str(ROOT / "detection" / "models" / "egg_detector.ts.pt")).value
        depth_topic = self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw").value
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        # hysteresis tracker: the real D455's noise makes the score jump (about 0.2-0.85 on a still egg), so
        # START tracking at score_threshold (confirmed over several frames), KEEP tracking down to track_threshold
        # near the same spot, HOLD the last position for hold_s through short misses, and smooth the position.
        self.thresh = float(p("score_threshold", 0.35))
        self.track_thresh = float(p("track_threshold", 0.15))
        self.hold_s = float(p("hold_s", 0.5))
        self.smooth = float(p("smooth", 0.5))   # weight of the newest position (1 = no smoothing)
        self.track = None                        # [score, x0, y0, x1, y1, X, Y, Z] of the tracked egg
        self.last_seen = 0.0
        # trained with eggs 0.5-0.95 m away; beyond ~1.2 m the depth is near the 1.5 m cut-off and detections are unreliable
        self.max_dist = float(p("max_distance_m", 1.2))
        # plausibility checks against things that are not eggs (real D455 speckle, hands, large objects)
        self.min_valid = float(p("min_valid_fraction", 0.6))   # solid surface: valid depth in the box centre
        self.size_min, self.size_max = float(p("min_size_m", 0.02)), float(p("max_size_m", 0.09))  # egg: 44-57 mm
        # temporal check: report only when seen in >= confirm_hits of the last confirm_window frames, same place
        self.hits, self.window = int(p("confirm_hits", 3)), int(p("confirm_window", 5))
        self.confirm_dist = float(p("confirm_distance_m", 0.04))
        self.recent = []
        # masked Gaussian (shared.d455_model.detector_input): ON for models trained with it (the default model);
        # set false for the old egg_detector_sim_only model, which was trained without the filter
        self.use_filter = bool(p("denoise", True))
        torch.set_num_threads(2)
        self.net = torch.jit.load(model, map_location="cpu").eval()
        k = intrinsics(decimation=1)  # fallback until camera_info arrives
        self.K = (k["fx"], k["fy"], k["cx"], k["cy"])
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, depth_topic.rsplit("/", 1)[0] + "/camera_info", self.on_info, qos_profile_sensor_data)
        self.pub_pt = self.create_publisher(PointStamped, "egg/point", 1)
        self.pub_det = self.create_publisher(Float32MultiArray, "egg/detection", 1)
        self.pub_dist = self.create_publisher(Float32, "egg/distance_cm", 1)
        self.pub_img = self.create_publisher(Image, "egg/debug_image", 1)
        self.get_logger().info(f"egg detector on {depth_topic} (model {model})")

    def on_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    @torch.inference_mode()
    def on_depth(self, msg: Image):
        dtype = {"16UC1": np.uint16, "32FC1": np.float32}[msg.encoding]
        depth = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.width)
        # (1, H, W): metres / z_max, invalid = 0, masked Gaussian applied - exactly the training input
        obs = detector_input(real_obs(depth, stride=1) * Z_RANGE[1]) if self.use_filter else real_obs(depth, stride=1)
        dets = decode(self.net(obs[None]), k=3, thresh=min(self.thresh, self.track_thresh))[0]
        z_m = obs[0].numpy() * Z_RANGE[1]
        cands = [c for c in (self.check(d, z_m) for d in dets.tolist()) if c]   # plausible eggs, best score first
        now = self.get_clock().now().nanoseconds * 1e-9
        near = lambda c, ref: np.linalg.norm(np.subtract(c[5:8], ref[5:8])) < self.confirm_dist  # noqa: E731
        if self.track is not None:
            # keep: any plausible candidate near the tracked egg, even with a low score
            match = next((c for c in cands if near(c, self.track)), None)
            if match:
                a = self.smooth   # smooth the position (and box) so the distance does not jump
                self.track = [match[0]] + [a * m + (1 - a) * t for m, t in zip(match[1:], self.track[1:])]
                self.last_seen = now
            elif now - self.last_seen > self.hold_s:
                self.track = None
        if self.track is None:
            # start: a strong candidate (>= score_threshold) at the same spot in >= hits of the last `window` frames
            cand = next((c for c in cands if c[0] >= self.thresh), None)
            self.recent = (self.recent + [cand if cand else None])[-self.window:]
            if cand and sum(1 for r in self.recent if r and near(r, cand)) >= self.hits:
                self.track, self.last_seen, self.recent = list(cand), now, []
        out = []
        if self.track is not None:
            s, x0, y0, x1, y1, X, Y, z = self.track
            cand = self.track
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x, pt.point.y, pt.point.z = X, Y, z
            self.pub_pt.publish(pt)
            out = cand
            dist_cm = 100.0 * float(np.sqrt(X * X + Y * Y + z * z))   # camera -> egg centre, straight line
            self.pub_dist.publish(Float32(data=dist_cm))
            self.get_logger().info(f"egg {dist_cm:5.1f} cm from the camera  (X {100 * X:+5.1f}, Y {100 * Y:+5.1f}, "
                                   f"Z {100 * z:5.1f} cm, score {s:.2f})", throttle_duration_sec=0.5)
        self.pub_det.publish(Float32MultiArray(data=[float(v) for v in out]))
        if self.pub_img.get_subscription_count():
            self.pub_img.publish(self.debug_image(msg, z_m, out))

    def check(self, det, z_m):
        """One network box -> [score, x0, y0, x1, y1, X, Y, Z] if it is a plausible egg, else None."""
        s, x0, y0, x1, y1 = det
        # depth of the egg's visible surface: median of valid pixels in the central half of the box
        cx0, cx1 = int(x0 + (x1 - x0) / 4), int(np.ceil(x1 - (x1 - x0) / 4))
        cy0, cy1 = int(y0 + (y1 - y0) / 4), int(np.ceil(y1 - (y1 - y0) / 4))
        patch = z_m[max(cy0, 0):cy1 + 1, max(cx0, 0):cx1 + 1]
        if patch.size == 0:
            return None
        valid = patch[patch > 0]
        if valid.size / patch.size < self.min_valid:          # scattered speckle, not a solid surface
            return None
        z = float(np.median(valid)) + EGG_RADIUS
        if z > self.max_dist:
            return None
        fx, fy, cx, cy = self.K
        w_m, h_m = (x1 - x0) * z / fx, (y1 - y0) * z / fy       # real size of the box at that distance
        if not (self.size_min <= min(w_m, h_m) and max(w_m, h_m) <= self.size_max):
            return None
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        return [s, x0, y0, x1, y1, (u - cx) * z / fx, (v - cy) * z / fy, z]

    def debug_image(self, msg, z_m, det):
        img = np.where(z_m > 0, 255 - np.clip(z_m / Z_RANGE[1], 0, 1) * 200, 0).astype(np.uint8)
        if cv2 is not None:  # colour, 2x larger, box + distance in cm
            big = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            if det:
                x0, y0, x1, y1 = [int(round(2 * v)) for v in det[1:5]]
                dist_cm = 100.0 * float(np.sqrt(det[5] ** 2 + det[6] ** 2 + det[7] ** 2))
                cv2.rectangle(big, (x0, y0), (x1, y1), (0, 140, 255), 2)
                label = f"egg {dist_cm:.1f} cm"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = y0 - 8 if y0 - 8 - th > 0 else y1 + th + 8
                cv2.rectangle(big, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), (0, 140, 255), -1)
                cv2.putText(big, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            m = Image(header=msg.header, height=big.shape[0], width=big.shape[1], encoding="bgr8", step=big.shape[1] * 3)
            m.data = big.tobytes()
            return m
        if det:
            x0, y0, x1, y1 = [int(round(v)) for v in det[1:5]]
            x0, x1 = np.clip([x0, x1], 0, img.shape[1] - 1); y0, y1 = np.clip([y0, y1], 0, img.shape[0] - 1)
            img[y0, x0:x1 + 1] = img[y1, x0:x1 + 1] = 0
            img[y0:y1 + 1, x0] = img[y0:y1 + 1, x1] = 0
        m = Image(header=msg.header, height=img.shape[0], width=img.shape[1], encoding="mono8", step=img.shape[1])
        m.data = img.tobytes()
        return m


def main():
    rclpy.init()
    node = EggDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():  # a real error; otherwise a publish raced with shutdown (e.g. SIGTERM)
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
