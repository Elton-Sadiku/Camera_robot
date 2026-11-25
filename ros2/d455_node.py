"""ROS 2 Humble publisher for the D455 (wraps D455PointCloud).

Topics use the realsense-ros naming, so everything downstream works unchanged with this node,
the official realsense2_camera driver, or the Isaac Sim publisher (isaac/sim_publish_d455.py):
  /camera/camera/depth/image_rect_raw   sensor_msgs/Image       16UC1 mm, full res  (~73 MB/s @90 Hz)
  /camera/camera/depth/camera_info      sensor_msgs/CameraInfo
  /camera/camera/depth/color/points     sensor_msgs/PointCloud2 organised xyz float32, NaN = invalid
                                        (424x240 full res, ~110 MB/s; cloud_decimation:=2 -> 212x120)
All frames are stamped with the sensor capture time and frame_id camera_depth_optical_frame.

Run (system Python 3.10, ROS sourced, NOT conda):  python3 ros2/d455_node.py --ros-args -p disparity_shift:=50
"""
import array
import signal
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pointcloud"), str(ROOT)]
from d455_pointcloud import D455PointCloud  # noqa: E402
from shared.d455_model import DEPTH_PROFILE, DISPARITY_SHIFT, FAR_LIMIT_M  # noqa: E402

XYZ_FIELDS = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1) for k, n in enumerate("xyz")]


class D455Node(Node):
    def __init__(self):
        super().__init__("d455")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.frame_id = p("frame_id", "camera_depth_optical_frame")
        self.cloud_dec = p("cloud_decimation", 1)          # 1 = full 424x240 cloud, 0 = don't publish it
        ns = p("topic_prefix", "/camera/camera/depth")

        w, h, fps = DEPTH_PROFILE  # one source of truth, shared with the policy node and Isaac Lab
        # RealSense temporal filter (~0.1 ms): averages each pixel over recent frames and keeps a pixel that was
        # valid in at least 1 of the last 8 frames (holes_fill 6). Measured on this D455 at a low table angle:
        # table depth 24.5% -> 33.5% valid, noise 1.5 -> 0.9 mm, flicker 4.5% -> 0.7%. temporal_filter:=false = raw.
        # (A stronger laser or longer exposure measured worse here; holes_fill 8 never forgets = ghosts when objects move.)
        filters = []
        if p("temporal_filter", True):
            import pyrealsense2 as rs
            tf = rs.temporal_filter()
            tf.set_option(rs.option.holes_fill, float(p("holes_fill", 6)))
            filters.append(tf)
        self.cam = D455PointCloud(width=p("width", w), height=p("height", h), fps=p("fps", fps), decimation=1,
                                  disparity_shift=p("disparity_shift", DISPARITY_SHIFT),
                                  far_limit_m=p("far_limit_m", FAR_LIMIT_M),
                                  queue_size=p("queue_size", 2), filters=filters)
        # make sure no visual preset left over from other tools (e.g. "high density") is active: it can blind the
        # camera below ~1.6 m. Default preset = what every measurement and recording in this project used.
        sensor = self.cam.profile.get_device().first_depth_sensor()
        import pyrealsense2 as rs
        if sensor.supports(rs.option.visual_preset):
            sensor.set_option(rs.option.visual_preset, 1)   # 1 = Default
        self.get_logger().info(f"D455 {self.cam.intrinsics.width}x{self.cam.intrinsics.height}@{fps}, "
                               f"measuring {self.cam.z_range[0]:.2f}-{self.cam.z_range[1]:.2f} m")
        self.cloud_rays = D455PointCloud._ray_table(self.cam.intrinsics, max(self.cloud_dec, 1))

        qos = qos_profile_sensor_data  # best effort, keep last 5: never blocks the camera on a slow subscriber
        self.pub_depth = self.create_publisher(Image, f"{ns}/image_rect_raw", qos)
        self.pub_info = self.create_publisher(CameraInfo, f"{ns}/camera_info", qos)
        self.pub_cloud = self.create_publisher(PointCloud2, f"{ns}/color/points", qos) if self.cloud_dec else None
        self.info = self._camera_info()

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)  # capture never waits on the executor
        self.thread.start()
        self.create_timer(5.0, lambda: self.get_logger().info(f"dropped frames so far: {self.cam.dropped}"))

    def _camera_info(self):
        i, m = self.cam.intrinsics, CameraInfo()
        m.width, m.height, m.distortion_model = i.width, i.height, "plumb_bob"
        m.d = [float(c) for c in i.coeffs]
        m.k = [i.fx, 0.0, i.ppx, 0.0, i.fy, i.ppy, 0.0, 0.0, 1.0]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        m.p = [i.fx, 0.0, i.ppx, 0.0, 0.0, i.fy, i.ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
        m.header.frame_id = self.frame_id
        return m

    def _loop(self):
        while self.running and rclpy.ok():
            try:
                self._publish_once()
            except RuntimeError:  # frame timeout
                continue
            except Exception:  # Ctrl+C: rclpy shut the context down before destroy_node() stopped us
                if rclpy.ok():
                    raise
                break

    def _publish_once(self):
        depth, stamp_ms, _ = self.cam.read_depth(timeout_ms=200)
        # D455 timestamps are in the global (host-synchronised) time domain -> capture time
        sec, ms = divmod(stamp_ms, 1000.0)
        stamp = Time(seconds=int(sec), nanoseconds=int(ms * 1e6)).to_msg()

        img = Image(height=depth.shape[0], width=depth.shape[1], encoding="16UC1",
                    step=depth.shape[1] * 2, is_bigendian=0)
        img.header.stamp, img.header.frame_id = stamp, self.frame_id
        img.data = array.array("B", depth.tobytes())  # array.array is the fast path in Humble rclpy
        self.info.header.stamp = stamp
        self.pub_depth.publish(img)
        self.pub_info.publish(self.info)

        if self.pub_cloud and self.pub_cloud.get_subscription_count():
            d = self.cloud_dec
            z = depth[::d, ::d].astype(np.float32) * self.cam.depth_scale
            z[(z < self.cam.z_range[0]) | (z > self.cam.z_range[1])] = np.nan
            pts = self.cloud_rays * z[..., None]
            c = PointCloud2(height=pts.shape[0], width=pts.shape[1], fields=XYZ_FIELDS, is_bigendian=False,
                            point_step=12, row_step=12 * pts.shape[1], is_dense=False)
            c.header.stamp, c.header.frame_id = stamp, self.frame_id
            c.data = array.array("B", pts.tobytes())
            self.pub_cloud.publish(c)

    def destroy_node(self):
        self.running = False
        self.thread.join()  # returns within one frame-wait (<=200 ms); must finish before the camera closes
        self.cam.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = D455Node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ros2 launch may send a 2nd SIGINT; interrupting the shutdown would kill the capture
        # thread inside librealsense ("FATAL: exception not rethrown"), so ignore it from here on.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
