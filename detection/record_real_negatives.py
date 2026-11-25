"""Record REAL D455 depth frames with NO egg in view, as "no egg" training examples.

The sim-trained detector has never seen the real camera's speckle, holes and the objects in your room.
These frames teach it what is NOT an egg. Labels are automatic (every frame = no egg), so:
  -> make sure NO egg is visible while recording; move the camera / objects / hands around for variety.

  source ./env.sh && ros2 launch ros2/d455.launch.py            (real camera running)
  source ./env.sh && python3 detection/record_real_negatives.py --seconds 90
  -> detection/data/real_negatives/shard_XXX.npz  (depth uint16 mm [N,240,424], box zeros [N,5])
Then retrain: ~/isaacsim/python.sh detection/train_egg_detector.py --real-neg detection/data/real_negatives --finetune
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--topic", default="/camera/camera/depth/image_rect_raw")
ap.add_argument("--out", default=str(ROOT / "detection" / "data" / "real_negatives"))
ap.add_argument("--seconds", type=float, default=90.0)
ap.add_argument("--every", type=float, default=0.2, help="save one frame every N seconds (5 per second)")
ap.add_argument("--no-wait", action="store_true", help="start without waiting for Enter")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
shard = len(list(out.glob("shard_*.npz")))   # never overwrite earlier recordings
frames, last = [], 0.0

rclpy.init()
node = Node("record_real_negatives")


def on_depth(m):
    global last
    now = time.time()
    if now - last < args.every:
        return
    last = now
    d = np.frombuffer(bytes(m.data), np.uint16 if m.encoding == "16UC1" else np.float32).reshape(m.height, m.width)
    if d.dtype != np.uint16:   # 32FC1 metres -> mm
        d = np.where(np.isfinite(d), np.round(d * 1000), 0).clip(0, 65535).astype(np.uint16)
    frames.append(d.copy())


# wait for the camera first
got = []
probe = node.create_subscription(Image, args.topic, lambda m: got.append(1), qos_profile_sensor_data)
t0 = time.time()
while not got and time.time() - t0 < 10:
    rclpy.spin_once(node, timeout_sec=0.5)
if not got:
    raise SystemExit(f"no messages on {args.topic} - is ros2/d455.launch.py running?")
node.destroy_subscription(probe)

if not args.no_wait:
    input("\nREMOVE THE EGG from the camera's view, then press Enter to start recording... ")
node.create_subscription(Image, args.topic, on_depth, qos_profile_sensor_data)
print(f"recording {args.seconds:.0f} s - move the camera, your hands and other objects around (NO egg)")
t0 = time.time()
next_print = t0
while time.time() - t0 < args.seconds:
    rclpy.spin_once(node, timeout_sec=0.1)
    if time.time() >= next_print:
        print(f"  {time.time() - t0:5.0f} s  {len(frames)} frames", flush=True)
        next_print += 10
path = out / f"shard_{shard:03d}.npz"
np.savez_compressed(path, depth=np.stack(frames), box=np.zeros((len(frames), 5), np.float32))
print(f"saved {len(frames)} no-egg frames to {path}")
node.destroy_node()
rclpy.shutdown()
