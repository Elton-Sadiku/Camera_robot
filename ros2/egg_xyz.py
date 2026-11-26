"""Print the 3D position of each egg found by yolo_ros (ros2/egg_yolo.launch.py) and publish the best one.

  python3 ros2/egg_xyz.py                                    # real: /yolo/detections_3d
  python3 ros2/egg_xyz.py --ros-args -r __ns:=/sim -p detections:=/sim/yolo/detections_3d

Publishes egg/point (geometry_msgs/PointStamped, egg centre in base_link, metres) for the highest-score egg,
egg/points (geometry_msgs/PoseArray) with EVERY accepted egg (several eggs; empty = none, used by the Isaac Sim twin),
and logs X, Y, Z plus the straight-line distance from the camera (camera_link) in cm.
Only eggs whose centre is 20 cm - 1 m from the camera count (-p min_distance_m:=0.20 -p max_distance_m:=1.0);
the others are ignored (closer than ~16 cm the D455 has no depth, past 1 m the egg is out of the arms' reach).
Also ignored: boxes whose real size (bbox3d.size from the depth) is not egg-like, 2-10 cm (egg: 44 x 57 mm).
"""
import math

import rclpy
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from yolo_msgs.msg import DetectionArray


class EggXYZ(Node):
    def __init__(self):
        super().__init__("egg_xyz")
        topic = self.declare_parameter("detections", "/yolo/detections_3d").value
        self.cam_frame = self.declare_parameter("camera_frame", "camera_link").value
        self.d_min = float(self.declare_parameter("min_distance_m", 0.20).value)
        self.d_max = float(self.declare_parameter("max_distance_m", 1.0).value)
        self.s_min = float(self.declare_parameter("min_size_m", 0.02).value)
        self.s_max = float(self.declare_parameter("max_size_m", 0.10).value)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub = self.create_publisher(PointStamped, "egg/point", 1)
        self.pub_all = self.create_publisher(PoseArray, "egg/points", 1)
        self.create_subscription(DetectionArray, topic, self.cb, 10)
        self.get_logger().info(f"listening on {topic}, eggs {100 * self.d_min:.0f}-{100 * self.d_max:.0f} cm "
                               f"from {self.cam_frame}")

    def cb(self, msg: DetectionArray):
        eggs = sorted((d for d in msg.detections if d.class_name == "egg" and d.bbox3d.frame_id),
                      key=lambda d: -d.score)
        best = True
        allp = PoseArray()
        allp.header.stamp, allp.header.frame_id = msg.header.stamp, "base_link"
        for d in eggs:
            c, frame = d.bbox3d.center.position, d.bbox3d.frame_id
            try:
                t = self.tf.lookup_transform(frame, self.cam_frame, Time()).transform.translation
            except Exception:  # TF not there yet: distance unknown, can't check the range
                continue
            dist = math.dist((c.x, c.y, c.z), (t.x, t.y, t.z))
            if not self.d_min <= dist <= self.d_max:
                self.get_logger().info(f"egg at {100 * dist:.1f} cm ignored (range {100 * self.d_min:.0f}-"
                                       f"{100 * self.d_max:.0f} cm)", throttle_duration_sec=1.0)
                continue
            w, h = d.bbox3d.size.x, d.bbox3d.size.y
            if not (self.s_min <= max(w, h) <= self.s_max):
                self.get_logger().info(f"box {100 * w:.1f} x {100 * h:.1f} cm at {100 * dist:.1f} cm ignored (not egg-sized)",
                                       throttle_duration_sec=1.0)
                continue
            self.get_logger().info(f"egg{d.id and ' #' + d.id} score {d.score:.2f}: X {c.x:+.3f} Y {c.y:+.3f} "
                                   f"Z {c.z:+.3f} m in {frame}, {100 * dist:5.1f} cm from the camera")
            q = Pose()
            q.position.x, q.position.y, q.position.z = c.x, c.y, c.z
            q.orientation.w = 1.0
            allp.poses.append(q)
            allp.header.frame_id = frame
            if best:
                best = False
                p = PointStamped()
                p.header.stamp, p.header.frame_id = msg.header.stamp, frame
                p.point.x, p.point.y, p.point.z = c.x, c.y, c.z
                self.pub.publish(p)
        self.pub_all.publish(allp)   # also when empty: the twin then knows there is no egg


def main():
    rclpy.init()
    node = EggXYZ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == "__main__":
    main()
