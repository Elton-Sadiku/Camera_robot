"""License-free egg detection node: D-FINE (ONNX, onnxruntime) + aligned depth -> egg X, Y, Z in base_link.

Replaces yolo_ros (GPL-3.0) + ultralytics (AGPL-3.0) + ros2/egg_xyz.py. Uses only ROS 2 (Apache-2.0), numpy/OpenCV
(BSD/Apache-2.0) and onnxruntime (MIT). Written for this project; no code from yolo_ros.

  source ./env.sh && python3 free_detector/egg_detect_node.py                  # real D455 (d455.launch.py running)
  python3 free_detector/egg_detect_node.py --ros-args -p source:=sim           # Isaac Sim (/sim/... topics)
  options: -p model:=<model.onnx> -p threshold:=0.5 -p keep_threshold:=0.3 -p threads:=4

Subscribes (best effort): <cam>/color/image_raw, <cam>/aligned_depth_to_color/image_raw (+ camera_info)
                          cam = /camera/camera (real) or /sim/camera/camera (sim)
Publishes (same as egg_xyz.py, so RViz and the Isaac Sim twin work unchanged; /sim prefix for source:=sim):
  egg/point     geometry_msgs/PointStamped  best egg centre in base_link (m)
  egg/points    geometry_msgs/PoseArray     every accepted egg (empty = none)
  free/dbg_image  sensor_msgs/Image         colour image with the boxes (egg green, ball orange)
  free/markers    visualization_msgs/MarkerArray  egg spheres for RViz

3D: the depth pixels in the middle half of the box (the egg itself, not the table around it) are sorted and the
nearest 40 % taken -> the egg's front surface; the centre is surface_offset_m (egg radius, 22 mm) further along the
viewing ray. Pixel + depth -> camera optical frame with the camera_info intrinsics -> base_link via TF.
Filters as before: centre 20 cm - 1 m from camera_link, real size 2-10 cm.
Anti-blink: a new object needs score >= threshold (0.5); an object seen in the previous frames stays while its
score >= keep_threshold (0.3) and its box overlaps the old one.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/egg_free_pkgs"))        # onnxruntime (installed with --target there)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray  # noqa: E402
from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"egg": (0, 200, 0), "ball": (0, 140, 255)}   # BGR


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


class EggDetect(Node):
    def __init__(self):
        super().__init__("egg_detect")
        src = self.declare_parameter("source", "real").value
        model = self.declare_parameter("model", str(HERE / "models" / "dfine_s_egg_ball" / "model.onnx")).value
        self.thr = float(self.declare_parameter("threshold", 0.5).value)
        self.keep = float(self.declare_parameter("keep_threshold", 0.3).value)
        self.target = self.declare_parameter("target_frame", "base_link").value
        self.cam_frame = self.declare_parameter("camera_frame", "camera_link").value
        self.d_min = float(self.declare_parameter("min_distance_m", 0.20).value)
        self.d_max = float(self.declare_parameter("max_distance_m", 1.0).value)
        self.s_min = float(self.declare_parameter("min_size_m", 0.02).value)
        self.s_max = float(self.declare_parameter("max_size_m", 0.10).value)
        self.offset = float(self.declare_parameter("surface_offset_m", 0.022).value)
        threads = int(self.declare_parameter("threads", 4).value)

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        lj = Path(model).with_name("labels.json")
        self.labels = json.loads(lj.read_text()) if lj.exists() else ["egg", "ball"]
        self.size = self.sess.get_inputs()[0].shape[2]

        cam = "/camera/camera" if src == "real" else "/sim/camera/camera"
        pre = "" if src == "real" else "/sim/"
        self.br = CvBridge()
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub_pt = self.create_publisher(PointStamped, pre + "egg/point", 1)
        self.pub_all = self.create_publisher(PoseArray, pre + "egg/points", 1)
        self.pub_img = self.create_publisher(Image, pre + "free/dbg_image", 1)
        self.pub_mk = self.create_publisher(MarkerArray, pre + "free/markers", 1)
        # newest image only: when the detector is slower than the camera, old frames are dropped instead of queued
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        subs = [Subscriber(self, Image, f"{cam}/color/image_raw", qos_profile=qos),
                Subscriber(self, Image, f"{cam}/aligned_depth_to_color/image_raw", qos_profile=qos),
                Subscriber(self, CameraInfo, f"{cam}/aligned_depth_to_color/camera_info", qos_profile=qos)]
        self.sync = ApproximateTimeSynchronizer(subs, 2, 0.05)
        self.sync.registerCallback(self.cb)
        self.tracks = []          # [(label, box xyxy, id)] from the previous frame
        self.next_id = 1
        self.t_inf = []
        self.get_logger().info(f"D-FINE {Path(model).name} ({self.labels}) on {cam}, threshold {self.thr} "
                               f"(keep {self.keep}), eggs {100 * self.d_min:.0f}-{100 * self.d_max:.0f} cm")

    # ------------------------------------------------------------------ detection
    def detect(self, bgr):
        h, w = bgr.shape[:2]
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        t = time.time()
        scores, boxes = self.sess.run(None, {"images": x})
        self.t_inf = (self.t_inf + [time.time() - t])[-30:]
        sc, cl = scores[0].max(1), scores[0].argmax(1)
        keep = sc >= self.keep
        out = []
        for s, c, b in zip(sc[keep], cl[keep], boxes[0][keep]):
            cx, cy, bw, bh = b * [w, h, w, h]
            out.append([float(s), self.labels[int(c)], [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]])
        out.sort(key=lambda d: -d[0])
        dets = []                                     # one box per object (drop overlapping duplicates)
        for d in out:
            if all(iou(d[2], e[2]) < 0.6 for e in dets):
                dets.append(d)
        # anti-blink: >= threshold to start, >= keep_threshold to continue an object from the last frame
        final, tracks = [], []
        for s, lab, box in dets:
            old = max(((iou(box, t[1]), t) for t in self.tracks if t[0] == lab), default=(0, None), key=lambda a: a[0])
            if old[0] > 0.3 or s >= self.thr:
                tid = old[1][2] if old[0] > 0.3 else self.next_id
                if tid == self.next_id:
                    self.next_id += 1
                final.append((s, lab, box, tid))
                tracks.append((lab, box, tid))
        self.tracks = tracks
        return final

    # ------------------------------------------------------------------ 3D
    def to_3d(self, depth, k, box, scale):
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) / 4, (y1 - y0) / 4                     # middle half of the box
        u0, u1 = int(max(x0 + mx, 0)), int(min(x1 - mx, depth.shape[1]))
        v0, v1 = int(max(y0 + my, 0)), int(min(y1 - my, depth.shape[0]))
        if u1 <= u0 or v1 <= v0:
            return None
        z = depth[v0:v1, u0:u1].astype(np.float32).ravel() * scale
        z = np.sort(z[np.isfinite(z) & (z > 0.05)])
        if z.size < 5:
            return None
        surf = float(np.median(z[: max(3, int(0.4 * z.size))]))   # front surface of the egg
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        p = ray * surf
        p = p + ray / np.linalg.norm(ray) * self.offset              # surface -> centre
        size = ((x1 - x0) * surf / fx, (y1 - y0) * surf / fy)
        return p, size

    def cb(self, img_msg, depth_msg, info_msg):
        bgr = self.br.imgmsg_to_cv2(img_msg, "bgr8")
        depth = self.br.imgmsg_to_cv2(depth_msg)
        scale = 0.001 if depth.dtype == np.uint16 else 1.0          # real: mm, Isaac Sim: m
        dets = self.detect(bgr)
        try:
            tr = self.tf.lookup_transform(self.target, info_msg.header.frame_id, Time())
            cam = self.tf.lookup_transform(self.target, self.cam_frame, Time()).transform.translation
        except Exception:
            tr = None
        allp = PoseArray()
        allp.header.stamp, allp.header.frame_id = img_msg.header.stamp, self.target
        markers = MarkerArray()
        mk = Marker(action=Marker.DELETEALL)
        markers.markers.append(mk)
        best = True
        for s, lab, box, tid in dets:
            col = COLORS.get(lab, (255, 255, 255))
            text = f"{lab} #{tid} {s:.2f}"
            if lab == "egg" and tr is not None:
                r = self.to_3d(depth, info_msg.k, box, scale)
                if r is not None:
                    p_cam, size = r
                    q, t = tr.transform.rotation, tr.transform.translation
                    p = quat_rotate((q.x, q.y, q.z, q.w), p_cam) + [t.x, t.y, t.z]
                    dist = math.dist(p, (cam.x, cam.y, cam.z))
                    if not self.d_min <= dist <= self.d_max:
                        self.get_logger().info(f"egg at {100 * dist:.1f} cm ignored (range {100 * self.d_min:.0f}-"
                                               f"{100 * self.d_max:.0f} cm)", throttle_duration_sec=1.0)
                    elif not self.s_min <= max(size) <= self.s_max:
                        self.get_logger().info(f"box {100 * size[0]:.1f} x {100 * size[1]:.1f} cm at {100 * dist:.1f}"
                                               " cm ignored (not egg-sized)", throttle_duration_sec=1.0)
                    else:
                        self.get_logger().info(f"egg #{tid} score {s:.2f}: X {p[0]:+.3f} Y {p[1]:+.3f} Z {p[2]:+.3f} m "
                                               f"in {self.target}, {100 * dist:5.1f} cm from the camera")
                        text += f"  {100 * dist:.0f} cm"
                        pose = Pose()
                        pose.position.x, pose.position.y, pose.position.z = map(float, p)
                        pose.orientation.w = 1.0
                        allp.poses.append(pose)
                        m = Marker(type=Marker.SPHERE, action=Marker.ADD, id=tid, ns="egg")
                        m.header = allp.header
                        m.pose = pose
                        m.scale.x = m.scale.y = 0.044
                        m.scale.z = 0.057
                        m.color.g, m.color.a = 0.8, 0.9
                        markers.markers.append(m)
                        if best:
                            best = False
                            ps = PointStamped()
                            ps.header = allp.header
                            ps.point = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                            self.pub_pt.publish(ps)
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), col, 2)
            cv2.putText(bgr, text, (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        ms = 1000 * float(np.mean(self.t_inf)) if self.t_inf else 0
        cv2.putText(bgr, f"D-FINE (Apache-2.0)  {ms:.0f} ms", (8, bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        self.pub_all.publish(allp)          # also when empty: the twin then knows there is no egg
        self.pub_mk.publish(markers)
        out = self.br.cv2_to_imgmsg(bgr, "bgr8")
        out.header = img_msg.header
        self.pub_img.publish(out)


def main():
    rclpy.init()
    node = EggDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == "__main__":
    main()
