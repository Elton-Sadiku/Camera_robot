"""Egg detector on D455 depth: a small CenterNet-style network (pure PyTorch, no extra packages).

Input : (N, 1, 240, 424) = shared.d455_model.real_obs(depth) - metres, invalid -> 0, divided by Z_RANGE[1].
        Exactly what the real camera gives after real_obs(), and what training builds from sim depth.
Output: stride-4 maps (60 x 106): heatmap logit (egg centre), box size (w, h) / SIZE_SCALE, sub-cell offset.
decode() turns them into boxes [score, x0, y0, x1, y1] in input pixels.

Used by detection/train_egg_detector.py (Isaac Sim python, GPU) and ros2/egg_detector_node.py (ROS, CPU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
SIZE_SCALE = 16.0  # predicted size * 16 = pixels


def _cbr(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EggNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.s2 = nn.Sequential(_cbr(1, w, 2), _cbr(w, w))                 # 1/2
        self.s4 = nn.Sequential(_cbr(w, 2 * w, 2), _cbr(2 * w, 2 * w))     # 1/4
        self.s8 = nn.Sequential(_cbr(2 * w, 4 * w, 2), _cbr(4 * w, 4 * w), _cbr(4 * w, 4 * w))   # 1/8
        self.s16 = nn.Sequential(_cbr(4 * w, 4 * w, 2), _cbr(4 * w, 4 * w))                     # 1/16, context
        self.up16 = _cbr(4 * w, 4 * w)
        self.up8 = _cbr(8 * w, 2 * w)
        self.fuse4 = nn.Sequential(_cbr(4 * w, 2 * w), _cbr(2 * w, 2 * w))
        self.heat = nn.Conv2d(2 * w, 1, 1)
        self.size = nn.Conv2d(2 * w, 2, 1)
        self.off = nn.Conv2d(2 * w, 2, 1)
        nn.init.constant_(self.heat.bias, -2.19)  # prior p = 0.1 (CenterNet init)

    def forward(self, x):
        f4 = self.s4(self.s2(x))
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        u8 = torch.cat([f8, F.interpolate(self.up16(f16), size=f8.shape[-2:], mode="nearest")], 1)
        u4 = torch.cat([f4, F.interpolate(self.up8(u8), size=f4.shape[-2:], mode="nearest")], 1)
        y = self.fuse4(u4)
        return torch.cat([self.heat(y), self.size(y), self.off(y)], 1)   # (N, 5, H/4, W/4)


@torch.no_grad()
def decode(out, k=5, thresh=0.3):
    """(N,5,h,w) -> list (per image) of tensors (M,5) [score, x0, y0, x1, y1] in input pixels, best first."""
    heat = torch.sigmoid(out[:, 0:1])
    peak = heat * (F.max_pool2d(heat, 3, 1, 1) == heat).float()
    n, _, h, w = heat.shape
    scores, idx = peak.view(n, -1).topk(min(k, h * w))
    res = []
    for i in range(n):
        keep = scores[i] >= thresh
        s, j = scores[i][keep], idx[i][keep]
        ys, xs = (j // w).float(), (j % w).float()
        ox, oy = out[i, 3].view(-1)[j], out[i, 4].view(-1)[j]
        bw, bh = out[i, 1].view(-1)[j] * SIZE_SCALE, out[i, 2].view(-1)[j] * SIZE_SCALE
        cx, cy = (xs + ox) * STRIDE, (ys + oy) * STRIDE
        res.append(torch.stack([s, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1))
    return res
