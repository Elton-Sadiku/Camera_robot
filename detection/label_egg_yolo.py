"""Auto-label the guided recordings (classes: 0 egg, 1 ball) (detection/record_egg_yolo.py) and write a YOLO dataset.

The egg does not move during a placement, so one box per placement is enough: YOLO-World ("egg") is run on
every frame at a very low confidence, the boxes that agree (IoU) are merged into one robust box, and the box is
checked against the aligned depth (real size 2-10 cm). Frames of placements without an egg get no label
(negatives). Placements whose box fails a check are skipped and listed - look at review_*.jpg.

  python3 detection/label_egg_yolo.py      # -> detection/data/yolo_egg/dataset/{images,labels}/{train,val} + egg.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "detection" / "data" / "yolo_egg"))
ap.add_argument("--world", default=str(ROOT / "detection" / "yolo" / "yolov8s-worldv2-egg.pt"))
ap.add_argument("--skip", default="", help="placements to drop after review, e.g. 0:3,0:7 (session:place)")
ap.add_argument("--egg", default="", help="placements recorded as 'no egg' that DO show the egg, e.g. 0:15,0:16")
ap.add_argument("--detector", default=str(ROOT / "detection" / "yolo" / "egg_real_sim_yolov8s.pt"),
                help="trained egg model: in BALL placements (record_egg_yolo.py --set balls) its false 'egg' boxes "
                     "become the 'ball' labels")
args = ap.parse_args()

from ultralytics import YOLO, YOLOWorld  # noqa: E402

world = YOLOWorld(args.world)
detector = YOLO(args.detector) if Path(args.detector).exists() else None
data = Path(args.data)
ds = data / "dataset"
skip = {tuple(int(v) for v in s.split(":")) for s in args.skip.split(",") if s}
force_egg = {tuple(int(v) for v in s.split(":")) for s in args.egg.split(",") if s}
FX = 386.0  # D455 colour fx at 640x480 (approx.; only used for the size check)


def egg_size(depth, b):
    """(real size m, distance m) of box b from the aligned depth in its central half."""
    x0, y0, x1, y1 = (int(v) for v in b)
    cx, cy, w, h = (x0 + x1) // 2, (y0 + y1) // 2, x1 - x0, y1 - y0
    v = depth[cy - h // 4:cy + h // 4 + 1, cx - w // 4:cx + w // 4 + 1]
    v = v[v > 0]
    if not v.size:
        return 0.0, 0.0
    zm = float(np.median(v)) / 1000
    return max(w, h) * zm / FX, zm


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy + 1e-9)


counts = {"train": 0, "val": 0, "neg": 0, "skipped": []}
for split in ("train", "val"):
    (ds / "images" / split).mkdir(parents=True, exist_ok=True)
    (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

for sess in sorted(data.glob("session_*.npz")):
    sid = int(sess.stem.split("_")[1])
    z = np.load(sess)
    C, D, P, E, W = z["color"], z["depth"], z["place"], z["has_egg"], z["where"]
    B = z["has_ball"] if "has_ball" in z.files else np.zeros_like(E)
    review = []
    for p in sorted(set(P.tolist())):
        idx = np.nonzero(P == p)[0]
        split = "val" if p % 5 == 2 else "train"        # whole placements go to val, so val is truly unseen
        box, cls = None, 0
        if (sid, p) in skip:
            counts["skipped"].append(f"{sid}:{p} ({W[p]}) dropped by --skip")
            continue
        is_ball = bool(B[idx[0]]) and not (sid, p) in force_egg
        if E[idx[0]] or (sid, p) in force_egg or is_ball:
            cls = 1 if is_ball else 0
            model, conf = (detector, 0.05) if is_ball and detector else (world, 0.001)
            cands = []
            for i in idx:   # best EGG-SIZED box of each frame (the top box can be the spool)
                r = model.predict(C[i], conf=conf, verbose=False, device="cpu")[0]
                ok = [(float(c), b) for c, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
                      if 0.02 <= egg_size(D[i], b)[0] <= 0.10]
                if ok:
                    cands.append(max(ok))
            best = max(cands, default=None)
            if best:
                agree = [b for s, b in cands if iou(b, best[1]) > 0.5]
                box = np.median(np.array(agree), 0) if len(agree) >= 0.5 * len(cands) else None
            note = f"{len(cands)}/{len(idx)} frames with an egg-sized box"
            if box is None and is_ball:      # the model does not fire on this ball: a plain negative frame is fine
                counts["skipped"].append(f"{sid}:{p} ({W[p]}) no box -> kept as negative ({note})")
            elif box is None:
                counts["skipped"].append(f"{sid}:{p} ({W[p]}) {note or 'no agreeing box'}")
                continue
        im = C[idx[len(idx) // 2]].copy()
        if box is not None:
            cv2.rectangle(im, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]),
                          (0, 0, 255) if cls else (0, 255, 0), 3)
        cv2.putText(im, f"{sid}:{p} {split}{' BALL' if cls else ''}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        review.append(cv2.resize(im, (320, 240)))
        for i in idx:
            name = f"s{sid:03d}_p{p:02d}_{i:04d}"
            h, w = C[i].shape[:2]
            cv2.imwrite(str(ds / "images" / split / f"{name}.jpg"), C[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(ds / "labels" / split / f"{name}.txt", "w") as f:
                if box is not None:
                    x0, y0, x1, y1 = box
                    f.write(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
            counts[split] += 1
            counts["neg"] += box is None
    while len(review) % 6:
        review.append(np.zeros_like(review[0]))
    sheet = cv2.vconcat([cv2.hconcat(review[r:r + 6]) for r in range(0, len(review), 6)])
    cv2.imwrite(str(data / f"review_{sid:03d}.jpg"), sheet)

(ds / "egg.yaml").write_text(f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n  0: egg\n  1: ball\n")
print(f"train {counts['train']} / val {counts['val']} images ({counts['neg']} without egg) -> {ds}")
print("skipped placements:", *counts["skipped"], sep="\n  ") if counts["skipped"] else print("skipped: none")
print(f"review the boxes: {data}/review_*.jpg")
