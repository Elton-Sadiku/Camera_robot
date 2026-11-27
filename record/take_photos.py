"""Take photos with the D455 by pressing SPACE - in the same format the egg detector uses (640x480 colour, JPG).

  cd ~/Desktop/camera/Camera1/Camera_robot/record
  python3 take_photos.py                 # photos are saved here, in record/
  python3 take_photos.py --out somewhere # or in another folder

  SPACE = take a photo     q / Esc = quit

The saved photo is clean (nothing drawn on it); the text and the white flash exist only in the preview window.
The photos can be labelled in Label Studio ("Object Detection with Bounding Boxes", labels egg and ball), exported
as "YOLO with Images" and imported with detection/import_label_studio.py.
The ROS camera driver must NOT be running (the D455 can be used by only one program at a time).
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:                     # started with conda's python (no pyrealsense2 / cv2)? -> re-run with the system python 3.10
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError:
    if sys.executable != "/usr/bin/python3" and os.path.exists("/usr/bin/python3"):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        os.execve("/usr/bin/python3", ["/usr/bin/python3", __file__, *sys.argv[1:]], env)
    raise

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="folder for the photos (default: record/)")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

if not rs.context().query_devices():
    raise SystemExit("no RealSense camera found - plug in the D455 (USB 3) and try again")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
try:
    pipe.start(cfg)
except RuntimeError as e:
    raise SystemExit(f"cannot open the D455 ({e}). Is the detection (start_all.sh / d455.launch.py) running? Stop it.")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
n = 0
flash = 0.0
win = "D455 photos  -  SPACE = photo, q = quit"
print(f"photos -> {out}\nSPACE = take a photo, q / Esc = quit", flush=True)
try:
    while True:
        f = pipe.wait_for_frames(2000).get_color_frame()
        if not f:
            continue
        img = np.asanyarray(f.get_data()).copy()
        view = img.copy()
        if time.time() - flash < 0.15:                      # short white flash after a photo
            view = cv2.addWeighted(view, 0.4, np.full_like(view, 255), 0.6, 0)
        cv2.putText(view, f"SPACE = photo   q = quit   photos: {n}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            n += 1
            p = out / f"photo_{stamp}_{n:04d}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])   # the clean image, nothing drawn on it
            flash = time.time()
            print(f"  saved {p.name}", flush=True)
        elif k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    pass
finally:
    pipe.stop()
    cv2.destroyAllWindows()
print(f"{n} photos saved in {out}")
