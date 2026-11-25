"""Train the depth egg detector on the Isaac Sim dataset (detection/gen_egg_depth.py).

Every batch gets FRESH D455 noise from shared.d455_model (stereo_noise with random subpixel 0.05-0.12,
then to_obs and / Z_RANGE[1]) - the same functions the real camera path uses via real_obs(), so the
network sees in training what it will see from the real D455.

  ~/isaacsim/python.sh detection/train_egg_detector.py            # GPU (torch 2.7 + CUDA in Isaac Sim's python)
  -> detection/models/egg_detector.pt     state dict + metadata
     detection/models/egg_detector.ts.pt  TorchScript (raw maps; decode with detection/egg_model.decode)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detection.egg_model import SIZE_SCALE, STRIDE, EggNet, decode  # noqa: E402
from shared.d455_model import Z_RANGE, detector_input, intrinsics, stereo_noise, stereo_shadow, to_obs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=[str(ROOT / "detection" / "data" / "egg_depth")],
                help="one or more sim dataset dirs (e.g. the 20 deg and the 5 deg set)")
ap.add_argument("--out", default=str(ROOT / "detection" / "models"))
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--real-neg", default=None, help="dir of REAL no-egg frames (detection/record_real_negatives.py)")
ap.add_argument("--real-pos", default=None, help="dir of REAL egg frames with boxes (detection/record_real_eggs.py)")
ap.add_argument("--real-frac", type=float, default=0.25, help="share of each batch taken from REAL frames (half egg, half no-egg)")
ap.add_argument("--finetune", action="store_true", help="start from detection/models/egg_detector.pt")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
FX = intrinsics(decimation=1)["fx"]  # same fx as isaac/d455_cfg.py feeds stereo_noise

shards = sorted(s for d in args.data for s in Path(d).glob("shard_*.npz"))
assert shards, f"no shards in {args.data}"
# mm as int16 (anything past 32 m is far outside Z_RANGE anyway) - halves memory vs int32
D = torch.from_numpy(np.minimum(np.concatenate([np.load(s)["depth"] for s in shards]), 32000).astype(np.int16))
B = torch.from_numpy(np.concatenate([np.load(s)["box"] for s in shards]))
N, H, W = D.shape
n_val = int(N * args.val_frac)
idx = torch.randperm(N)
val_i, tr_i = idx[:n_val], idx[n_val:]
print(f"[train] {N} frames ({int(B[:, 0].sum())} with egg), train {len(tr_i)}, val {n_val}, device {dev}", flush=True)
h4, w4 = H // STRIDE, W // STRIDE

# REAL frames already carry the real sensor's noise and shadows, so they skip stereo_noise/stereo_shadow
def load_real(d, kind):
    rs = sorted(Path(d).glob("shard_*.npz"))
    assert rs, f"no shards in {d}"
    dep = torch.from_numpy(np.minimum(np.concatenate([np.load(s)["depth"] for s in rs]), 32000).astype(np.int16))
    box = torch.from_numpy(np.concatenate([np.load(s)["box"] for s in rs]).astype(np.float32))
    blk = torch.arange(len(dep)) // 5   # hold out every 5th block of 5 frames (neighbours are near duplicates)
    va, tr = torch.nonzero(blk % 5 == 0).flatten(), torch.nonzero(blk % 5 != 0).flatten()
    print(f"[train] + {len(dep)} REAL {kind} frames: train {len(tr)}, held out {len(va)}", flush=True)
    return dep, box, tr, va


R = RB = RP = RPB = None
if args.real_neg:
    R, RB, r_tr, r_val = load_real(args.real_neg, "no-egg")
if args.real_pos:
    RP, RPB, rp_tr, rp_val = load_real(args.real_pos, "egg")


def make_obs(depth_mm, train, real=False):
    """Detector input, identical to ros2/egg_detector_node.py: detector_input() = to_obs + masked Gaussian."""
    z = depth_mm.to(dev).float() / 1000.0
    if not real:                                        # sim: add the D455's noise and stereo shadow first
        z = stereo_noise(z, FX, subpixel=float(np.random.uniform(0.05, 0.12)) if train else 0.08)
        z = stereo_shadow(z, FX)
    return detector_input(z)[:, None]


def targets(boxes, flip):
    """CenterNet targets at stride 4: gaussian heatmap, size / offset at the centre cell."""
    n = boxes.shape[0]
    heat = torch.zeros(n, 1, h4, w4, device=dev)
    reg = torch.zeros(n, 4, h4, w4, device=dev)
    mask = torch.zeros(n, 1, h4, w4, device=dev)
    yy, xx = torch.meshgrid(torch.arange(h4, device=dev).float(), torch.arange(w4, device=dev).float(), indexing="ij")
    for i in range(n):
        has, x0, y0, x1, y1 = boxes[i].tolist()
        if not has:
            continue
        if flip[i]:
            x0, x1 = W - x1, W - x0
        cx, cy, bw, bh = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE, x1 - x0, y1 - y0
        sig = max(0.6, 0.15 * math.hypot(bw, bh) / STRIDE)
        heat[i, 0] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
        ix, iy = min(int(cx), w4 - 1), min(int(cy), h4 - 1)
        heat[i, 0, iy, ix] = 1.0
        reg[i, :, iy, ix] = torch.tensor([bw / SIZE_SCALE, bh / SIZE_SCALE, cx - ix, cy - iy], device=dev)
        mask[i, 0, iy, ix] = 1.0
    return heat, reg, mask


def loss_fn(out, heat, reg, mask):
    p = torch.sigmoid(out[:, 0:1]).clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    neg_w = (1 - heat) ** 4
    focal = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * neg_w * p ** 2 * torch.log(1 - p)).sum() / pos.sum().clamp(min=1)
    l1 = (F.l1_loss(out[:, 1:5], reg, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    return focal + l1, focal.item(), l1.item()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


@torch.no_grad()
def evaluate(model, thresh=0.4):
    model.eval()
    tp = fp = fn = 0; cerr = []
    for s in range(0, n_val, 64):
        ii = val_i[s:s + 64]
        dets = decode(model(make_obs(D[ii], False)), k=3, thresh=thresh)
        for d, b in zip(dets, B[ii].tolist()):
            if b[0]:
                if len(d) and iou(d[0, 1:].tolist(), b[1:]) >= 0.5:
                    tp += 1
                    cerr.append(math.hypot((d[0, 1] + d[0, 3]).item() / 2 - (b[1] + b[3]) / 2, (d[0, 2] + d[0, 4]).item() / 2 - (b[2] + b[4]) / 2))
                    fp += len(d) - 1
                else:
                    fn += 1; fp += len(d)
            else:
                fp += len(d)
    model.train()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return prec, rec, (float(np.median(cerr)) if cerr else float("nan"))


@torch.no_grad()
def real_fp_rate(model, thresh=0.5):
    """Share of held-out REAL no-egg frames where the raw network still fires (before the node's checks)."""
    if R is None:
        return float("nan")
    model.eval()
    fired = 0
    for s in range(0, len(r_val), 64):
        dets = decode(model(make_obs(R[r_val[s:s + 64]], False, real=True)), k=1, thresh=thresh)
        fired += sum(1 for d in dets if len(d))
    model.train()
    return fired / len(r_val)


@torch.no_grad()
def real_recall(model, thresh=0.5):
    """Share of held-out REAL egg frames where the best box overlaps the auto label (IoU >= 0.3)."""
    if RP is None:
        return float("nan")
    model.eval()
    hit = 0
    for s in range(0, len(rp_val), 64):
        ii = rp_val[s:s + 64]
        dets = decode(model(make_obs(RP[ii], False, real=True)), k=1, thresh=thresh)
        hit += sum(1 for d, b in zip(dets, RPB[ii].tolist()) if len(d) and iou(d[0, 1:].tolist(), b[1:]) >= 0.3)
    model.train()
    return hit / len(rp_val)


model = EggNet().to(dev)
if args.finetune:
    model.load_state_dict(torch.load(Path(args.out) / "egg_detector.pt", map_location=dev)["state_dict"])
    print(f"[train] fine-tuning from {Path(args.out) / 'egg_detector.pt'}", flush=True)
print(f"[train] before: real no-egg firing {100 * real_fp_rate(model):.1f}%, real egg found {100 * real_recall(model):.1f}%",
      flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps = args.epochs * math.ceil(len(tr_i) / args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
t0 = time.time()
for ep in range(args.epochs):
    perm = tr_i[torch.randperm(len(tr_i))]
    tot = 0.0
    for s in range(0, len(perm), args.batch):
        ii = perm[s:s + args.batch]
        x, bx = make_obs(D[ii], True), B[ii]
        nr = max(1, int(round(len(ii) * args.real_frac)))
        for dep, box, tr in ((R, RB, r_tr if R is not None else None), (RP, RPB, rp_tr if RP is not None else None)):
            if dep is None:
                continue          # mix in real frames: no-egg (label: none) and egg (auto box)
            n_this = nr if (R is None or RP is None) else max(1, nr // 2)
            jj = tr[torch.randint(len(tr), (n_this,))]
            x = torch.cat([x, make_obs(dep[jj], True, real=True)])
            bx = torch.cat([bx, box[jj]])
        # no left-right flip: the D455 stereo shadow is always on the LEFT of objects
        flip = [False] * len(x)
        heat, reg, mask = targets(bx, flip)
        loss, lf, lr_ = loss_fn(model(x), heat, reg, mask)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        tot += loss.item() * len(ii)
    if ep % 3 == 2 or ep == args.epochs - 1:
        p, r, ce = evaluate(model)
        fp_real = (f"  real no-egg firing {100 * real_fp_rate(model):.1f}%" if R is not None else "") + \
                  (f"  real egg found {100 * real_recall(model):.1f}%" if RP is not None else "")
        print(f"[train] epoch {ep + 1}/{args.epochs} loss {tot / len(tr_i):.3f}  val precision {p:.3f} recall {r:.3f} "
              f"centre err {ce:.2f}px{fp_real}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        print(f"[train] epoch {ep + 1}/{args.epochs} loss {tot / len(tr_i):.3f}  ({time.time() - t0:.0f}s)", flush=True)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
import shutil  # noqa: E402

for f in ("egg_detector.pt", "egg_detector.ts.pt"):   # keep the first (sim-only) model once, for comparison / rollback
    if (out / f).exists() and not (out / f.replace("egg_detector", "egg_detector_sim_only")).exists():
        shutil.copy2(out / f, out / f.replace("egg_detector", "egg_detector_sim_only"))
model.eval().cpu()
torch.save({"state_dict": model.state_dict(), "z_range": Z_RANGE, "input": [1, 1, H, W], "stride": STRIDE,
            "size_scale": SIZE_SCALE}, out / "egg_detector.pt")
torch.jit.trace(model, torch.zeros(1, 1, H, W)).save(str(out / "egg_detector.ts.pt"))
print(f"[train] saved {out / 'egg_detector.pt'} and {out / 'egg_detector.ts.pt'}")
