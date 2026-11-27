"""Train the license-free egg/ball detector: D-FINE-S (Apache-2.0) with PyTorch + Hugging Face transformers.

No Ultralytics code or weights: the start model is ustc-community/dfine-small-coco (Apache-2.0, pretrained on COCO),
the labels are our own (YOLO .txt files: class cx cy w h, normalised - exactly what D-FINE's loss expects).

Runs on the GPU with Isaac Sim's python (CUDA torch 2.7 + transformers 4.57 already installed there):
  ~/isaacsim/python.sh free_detector/train_dfine.py                       # ~40 min on the RTX 5070
  ~/isaacsim/python.sh free_detector/train_dfine.py --epochs 5 --name test   # quick check
  -> free_detector/models/<name>/   (Hugging Face folder: model.safetensors + config.json + log.txt)

Data: the same image lists as the YOLO model egg_ball_ls (detection/data/egg_ball_ls2_train.txt / _val.txt):
real D455 frames, Label Studio frames (x3), Isaac Sim eggs and balls. Classes 0 egg, 1 ball.
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from transformers import DFineForObjectDetection

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "detection" / "data"
BASE = "ustc-community/dfine-small-coco"
CLASSES = ["egg", "ball"]
SIZE = 640

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=str(DATA / "egg_ball_ls2_train.txt"))
ap.add_argument("--val", default=str(DATA / "egg_ball_ls2_val.txt"))
ap.add_argument("--base", default=BASE, help="Hugging Face id or folder of the start model")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--name", default="dfine_s_egg_ball")
args = ap.parse_args()
OUT = Path(__file__).resolve().parent / "models" / args.name
OUT.mkdir(parents=True, exist_ok=True)


def label_path(img):
    return Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")


class YoloList(Dataset):
    """Images from a list file + their YOLO labels (class cx cy w h, normalised)."""

    def __init__(self, list_file, train):
        self.ims = [l.strip() for l in open(list_file) if l.strip()]
        self.train = train
        aug = [T.RandomPhotometricDistort(p=0.5), T.RandomZoomOut(fill=114, side_range=(1.0, 1.6), p=0.3),
               T.RandomIoUCrop(), T.RandomHorizontalFlip()] if train else []
        self.tf = T.Compose(aug + [T.Resize((SIZE, SIZE)), T.SanitizeBoundingBoxes(min_size=2),
                                   T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        img = Image.open(self.ims[i]).convert("RGB")
        w, h = img.size
        rows = []
        lp = label_path(self.ims[i])
        if lp.exists():
            rows = [list(map(float, r.split())) for r in lp.read_text().splitlines() if r.strip()]
        cls = torch.tensor([int(r[0]) for r in rows], dtype=torch.int64)
        b = torch.tensor([r[1:5] for r in rows], dtype=torch.float32).reshape(-1, 4)
        xyxy = torch.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                            (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)
        boxes = tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(h, w))
        tgt = {"boxes": boxes, "labels": cls}
        if self.train and len(rows) == 0:          # RandomIoUCrop needs boxes; empty images: no crop
            img, tgt = T.Compose([T.RandomPhotometricDistort(p=0.5), T.RandomHorizontalFlip(),
                                  T.Resize((SIZE, SIZE)), T.ToImage(),
                                  T.ToDtype(torch.float32, scale=True)])(img, tgt)
        else:
            img, tgt = self.tf(img, tgt)
        cls = tgt["labels"]
        bx = tgt["boxes"].as_subclass(torch.Tensor) / SIZE          # -> normalised cx cy w h
        cxcywh = torch.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2,
                              bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], 1).clamp(0, 1)
        return img, {"class_labels": cls, "boxes": cxcywh}


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def evaluate(model, loader, dev):
    """Simple robot-oriented check at score 0.5: eggs found, ball-only frames called egg, empty frames called egg."""
    model.eval()
    egg_f = egg_ok = ball_f = ball_bad = none_f = none_bad = 0
    for x, ys in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(pixel_values=x.to(dev))
        prob = o.logits.float().sigmoid()                         # (B, queries, classes)
        for p, y in zip(prob, ys):
            score, c = p.max(-1)
            n_egg = int(((c == 0) & (score >= 0.5)).sum())
            ge, gb = int((y["class_labels"] == 0).sum()), int((y["class_labels"] == 1).sum())
            if ge:
                egg_f += 1; egg_ok += n_egg >= ge
            elif gb:
                ball_f += 1; ball_bad += n_egg > 0
            else:
                none_f += 1; none_bad += n_egg > 0
    model.train()
    return egg_ok / max(egg_f, 1), ball_bad / max(ball_f, 1), none_bad / max(none_f, 1), (egg_f, ball_f, none_f)


def main():
    random.seed(0); torch.manual_seed(0)
    dev = "cuda"
    model = DFineForObjectDetection.from_pretrained(
        args.base, num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)}, ignore_mismatched_sizes=True).to(dev)
    tr = DataLoader(YoloList(args.train, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=True)
    va = DataLoader(YoloList(args.val, False), batch_size=16, num_workers=args.workers, collate_fn=collate)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
                            weight_decay=1e-4)
    steps = args.epochs * len(tr)
    warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    log = open(OUT / "log.txt", "a")
    say = lambda s: (print(s, flush=True), log.write(s + "\n"), log.flush())
    say(f"start {time.ctime()}: base {args.base}, {len(tr.dataset)} train / {len(va.dataset)} val images, "
        f"{args.epochs} epochs x {len(tr)} steps")
    best = -1.0
    model.train()
    for ep in range(1, args.epochs + 1):
        t0, tot = time.time(), 0.0
        for i, (x, ys) in enumerate(tr):
            ys = [{k: v.to(dev) for k, v in y.items()} for y in ys]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=x.to(dev), labels=ys).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step()
            tot += loss.item()
        egg, ball_bad, none_bad, n = evaluate(model, va, dev)
        score = egg - ball_bad - none_bad
        say(f"epoch {ep:3d}/{args.epochs}  loss {tot / len(tr):.3f}  val: eggs found {egg:.1%}  "
            f"ball-only called egg {ball_bad:.1%}  empty called egg {none_bad:.1%}  (frames {n})  {time.time() - t0:.0f} s")
        if score >= best and ep >= args.epochs // 3:     # ties -> the later (more trained) model; skip the first epochs
            best = score
            model.save_pretrained(OUT)
            say(f"   best so far -> {OUT}")
        model.save_pretrained(OUT / "last")
    say(f"done {time.ctime()}")


if __name__ == "__main__":
    main()
