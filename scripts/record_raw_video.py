"""Record a CLEAN video from the D455 colour camera (no boxes, no text, no detection) for labelling in Label Studio.

Same image the egg detector sees: 640x480 colour at 30 fps (the realsense2_camera profile in ros2/d455.launch.py),
so boxes drawn on these frames train exactly what runs live.

  python3 scripts/record_raw_video.py                      # 60 s, preview window, frames every 10th image
  python3 scripts/record_raw_video.py --seconds 120 --frames-every 15
  python3 scripts/record_raw_video.py --source ros         # if the camera driver is already running (ROS topic)
  -> detection/data/label_studio/rec_<date_time>/
        video.mp4        H.264, plays everywhere; import into Label Studio as a video (Video Object Tracking)
        frames/*.jpg     single images (every Nth frame); import into Label Studio for "Object Detection with
                         Bounding Boxes" -> export "YOLO" -> train with detection/train_egg_yolo.py

Stop early with q in the preview window or Ctrl+C. Nothing is drawn on the saved images; the red REC dot and the
timer exist only in the preview.
Source 'camera' (default) opens the D455 directly with pyrealsense2 (no ROS needed, the driver must NOT be running);
source 'ros' records /camera/camera/color/image_raw while ros2/d455.launch.py runs.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:                     # started with conda's python (no pyrealsense2 / cv2)? -> re-run with the system python 3.10
    import cv2
    import numpy as np
    import pyrealsense2  # noqa: F401
except ImportError:
    if sys.executable != "/usr/bin/python3" and os.path.exists("/usr/bin/python3"):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        os.execve("/usr/bin/python3", ["/usr/bin/python3", __file__, *sys.argv[1:]], env)
    raise

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60.0, help="recording length (q / Ctrl+C stops earlier)")
ap.add_argument("--frames-every", type=int, default=10, help="also save every Nth frame as JPG (0 = no frames)")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--countdown", type=int, default=3, help="seconds before recording starts")
ap.add_argument("--source", choices=["camera", "ros"], default="camera")
ap.add_argument("--topic", default="/camera/camera/color/image_raw")
ap.add_argument("--no-preview", action="store_true")
ap.add_argument("--out", default=None, help="output folder (default detection/data/label_studio/rec_<date_time>)")
args = ap.parse_args()

out = Path(args.out) if args.out else ROOT / "detection" / "data" / "label_studio" / datetime.now().strftime("rec_%Y%m%d_%H%M%S")
(out / "frames").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- frame source
class CameraSource:
    def __init__(self):
        import pyrealsense2 as rs

        if not rs.context().query_devices():
            raise SystemExit("no RealSense camera found - plug in the D455 (USB 3) and try again")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        try:
            self.pipe.start(cfg)
        except RuntimeError as e:
            raise SystemExit(f"cannot open the D455 ({e}). Is the ROS driver running? Stop it, or use --source ros")
        for _ in range(15):                      # let auto-exposure settle
            self.pipe.wait_for_frames()

    def read(self):
        f = self.pipe.wait_for_frames(2000).get_color_frame()
        return np.asanyarray(f.get_data()).copy() if f else None

    def close(self):
        self.pipe.stop()


class RosSource:
    def __init__(self):
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        rclpy.init()
        self.rclpy, self.node, self.br, self.last = rclpy, rclpy.create_node("record_raw_video"), CvBridge(), None
        self.node.create_subscription(Image, args.topic, self._cb, qos_profile_sensor_data)
        t0 = time.time()
        while self.last is None and time.time() - t0 < 10:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.last is None:
            raise SystemExit(f"no images on {args.topic} - start ros2/d455.launch.py first (source ./env.sh)")

    def _cb(self, m):
        self.last = self.br.imgmsg_to_cv2(m, "bgr8")
        self.new = True

    def read(self):
        self.new = False
        t0 = time.time()
        while not self.new and time.time() - t0 < 2:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.last.copy() if self.new else None

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


src = CameraSource() if args.source == "camera" else RosSource()
preview = not args.no_preview


def show(img, text, rec):
    if not preview:
        return True
    p = img.copy()
    if rec:
        cv2.circle(p, (22, 22), 9, (0, 0, 255), -1)
    cv2.putText(p, text, (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if rec else (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(p, "preview only - saved video is clean   |   q = stop", (10, p.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("D455 raw recording (for Label Studio)", p)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


# ---------------------------------------------------------------- countdown + record
print(f"recording to {out}")
t0 = time.time()
while time.time() - t0 < args.countdown:
    img = src.read()
    if img is not None and not show(img, f"starting in {args.countdown - int(time.time() - t0)} ...", False):
        raise SystemExit("stopped")

raw = out / "video_raw.mp4"
writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
n = saved = 0
t0 = time.time()
print(f"REC  {args.seconds:.0f} s  (q in the window or Ctrl+C to stop early)", flush=True)
try:
    while time.time() - t0 < args.seconds:
        img = src.read()
        if img is None:
            continue
        if img.shape[1] != args.width or img.shape[0] != args.height:
            img = cv2.resize(img, (args.width, args.height))
        writer.write(img)                                   # the clean image, nothing drawn on it
        if args.frames_every and n % args.frames_every == 0:
            cv2.imwrite(str(out / "frames" / f"{out.name}_{n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        n += 1
        if not show(img, f"REC {time.time() - t0:5.1f} s   frames {saved}", True):
            break
except KeyboardInterrupt:
    pass
finally:
    writer.release()
    src.close()
    if preview:
        cv2.destroyAllWindows()

dur = time.time() - t0
video = out / "video.mp4"
if shutil.which("ffmpeg"):   # H.264 so Label Studio / browsers can play it
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(video)], check=True)
    raw.unlink()
else:
    raw.rename(video)
print(f"\nsaved {n} frames ({dur:.0f} s, {n / max(dur, 1e-6):.1f} fps) -> {video}")
print(f"      {saved} JPG frames -> {out / 'frames'}")
