"""Train the egg YOLO on the labelled real frames (detection/label_egg_yolo.py), on the GPU.

Isaac Sim's python has CUDA torch; ultralytics is kept in a separate folder so Isaac Sim stays untouched:
  PYTHONPATH=~/egg_train_pkgs ~/isaacsim/python.sh detection/train_egg_yolo.py
  -> detection/yolo/egg_yolov8s.pt   (use: ros2 launch ros2/egg_yolo.launch.py model_type:=YOLO model:=...)
  real + Isaac Sim, starting from the real model:
  PYTHONPATH=~/egg_train_pkgs ~/isaacsim/python.sh detection/train_egg_yolo.py --data detection/data/egg_real_sim.yaml \
      --base detection/yolo/egg_yolov8s.pt --name egg_real_sim --epochs 40
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "detection" / "data" / "yolo_egg" / "dataset" / "egg.yaml"))
ap.add_argument("--base", default=str(ROOT / "detection" / "yolo" / "yolov8s.pt"))
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--name", default="egg", help="run name; the model is saved as detection/yolo/<name>_yolov8s.pt")
args = ap.parse_args()

model = YOLO(args.base)
res = model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch, device=0, workers=4,
                  project=str(ROOT / "detection" / "yolo" / "runs"), name=args.name, exist_ok=True,
                  # few placements -> lean on augmentation: colour, scale, flips, mosaic, a little rotation
                  hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=10, scale=0.6, fliplr=0.5, mosaic=1.0, mixup=0.1,
                  patience=30, plots=False, verbose=False)   # plots=True crashes in 8.4.6 (UnboundLocalError)
best = Path(res.save_dir) / "weights" / "best.pt"
out = ROOT / "detection" / "yolo" / f"{args.name}_yolov8s.pt"
shutil.copy2(best, out)
print(f"[train] best model -> {out}")
