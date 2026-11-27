"""Compare the license-free D-FINE model (ONNX) with the YOLO model on the same held-out frames.

  python3 free_detector/eval_compare.py
  python3 free_detector/eval_compare.py --onnx free_detector/models/<name>/model.onnx --yolo ''   # D-FINE only

Frame sets (never used for training): real D455 validation frames, Label Studio validation frames, Isaac Sim
validation frames. Counts at score 0.5: frames whose eggs were all found, ball-only frames wrongly called egg.
The YOLO column needs ultralytics (only for this comparison; the robot does not need it).
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/egg_free_pkgs"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "detection" / "data"
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=str(ROOT / "free_detector" / "models" / "dfine_s_egg_ball" / "model.onnx"))
ap.add_argument("--yolo", default=str(ROOT / "detection" / "yolo" / "egg_ball_ls_yolov8s.pt"))
ap.add_argument("--conf", type=float, default=0.5)
args = ap.parse_args()

SETS = {
    "real D455 val": sorted(glob.glob(str(D / "yolo_egg/dataset/images/val/*.jpg"))),
    "Label Studio val": sorted(glob.glob(str(D / "label_studio/dataset/images/val/*.jpg"))),
    "Isaac Sim val": sorted(glob.glob(str(D / "yolo_egg_sim_ball/images/val/*.jpg")))[:300],
}


def gt(p):
    lp = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
    rows = [r.split()[0] for r in lp.read_text().splitlines() if r.strip()] if lp.exists() else []
    return rows.count("0"), rows.count("1")


class Dfine:
    def __init__(self, path):
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.labels = json.loads(Path(path).with_name("labels.json").read_text())

    def eggs(self, p):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = (cv2.resize(im, (640, 640)).astype(np.float32) / 255).transpose(2, 0, 1)[None]
        sc, _ = self.s.run(None, {"images": x})
        s, c = sc[0].max(1), sc[0].argmax(1)
        return int(((c == self.labels.index("egg")) & (s >= args.conf)).sum())


class Yolo:
    def __init__(self, path):
        from ultralytics import YOLO
        self.m = YOLO(path)

    def eggs(self, p):
        b = self.m.predict(p, conf=args.conf, device="cpu", verbose=False)[0].boxes
        return int((b.cls == 0).sum())


models = {"D-FINE (free)": Dfine(args.onnx)}
if args.yolo:
    models["YOLO egg_ball_ls"] = Yolo(args.yolo)
for name, m in models.items():
    print(f"== {name}")
    for s, ims in SETS.items():
        ef = eo = bf = bb = 0
        t = time.time()
        for p in ims:
            ge, gb = gt(p)
            n = m.eggs(p)
            if ge:
                ef += 1; eo += n >= ge
            elif gb:
                bf += 1; bb += n > 0
        print(f"   {s:17s} eggs found {eo:3d}/{ef:<3d}  ball-only called egg {bb:2d}/{bf:<3d}"
              f"  ({1000 * (time.time() - t) / max(len(ims), 1):.0f} ms/image)")
