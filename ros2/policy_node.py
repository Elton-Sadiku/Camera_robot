"""Deployment: depth topic -> shared.d455_model.real_obs -> exported Isaac Lab policy -> action topic.

Accepts 16UC1 (real D455 / realsense2_camera) and 32FC1 (Isaac Sim bridge), so the exact same node
can be validated against the simulator first, then pointed at the real camera.

  python3 ros2/policy_node.py --ros-args -p policy:=/path/to/exported/policy.pt

`policy.pt` is the TorchScript file from Isaac Lab's play.py (logs/rsl_rl/<task>/<run>/exported/policy.pt).
If your policy also uses proprioception, append it in build_obs() in the SAME order as the env's
ObservationsCfg policy group.
"""
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.d455_model import real_obs  # noqa: E402


class PolicyNode(Node):
    def __init__(self):
        super().__init__("d455_policy")
        self.policy = torch.jit.load(self.declare_parameter("policy", "policy.pt").value).eval()
        depth_topic = self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw").value
        # keep-last-1: always act on the newest frame, never on a backlog
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.pub = self.create_publisher(Float32MultiArray, "policy/action", 1)

    def build_obs(self, depth_obs):
        return depth_obs.flatten()[None]  # + torch.cat([...proprio...]) if your policy group has more terms

    @torch.inference_mode()
    def on_depth(self, msg: Image):
        dtype = {"16UC1": np.uint16, "32FC1": np.float32}[msg.encoding]
        depth = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.width)
        action = self.policy(self.build_obs(real_obs(depth)))
        self.pub.publish(Float32MultiArray(data=action.squeeze(0).tolist()))


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
