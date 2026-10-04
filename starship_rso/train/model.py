"""Temporal point detector: a U-Net over K background-registered frames.

Input channels: the current frame, K-1 past frames warped into the current
frame's coordinates, and the K-1 differences (current - past). Outputs a
centre heatmap (logits) and a sub-pixel offset at full input resolution, so
targets only a few pixels across are not lost to striding. This is the
CenterNet formulation (Zhou et al., 2019) without a stride, adapted to
unresolved points; ``frames=1`` gives the single-frame baseline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def input_channels(frames: int) -> int:
    return frames + max(frames - 1, 0)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TemporalUNet(nn.Module):
    def __init__(self, frames: int = 3, base: int = 32, depth: int = 4, prior: float = 0.01):
        super().__init__()
        self.frames = frames
        self.depth = depth
        cin = input_channels(frames)
        chans = [base * 2**i for i in range(depth)]
        self.enc = nn.ModuleList()
        c = cin
        for ch in chans:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.bottleneck = ConvBlock(chans[-1], chans[-1] * 2)
        self.dec = nn.ModuleList()
        c = chans[-1] * 2
        for ch in reversed(chans):
            self.dec.append(ConvBlock(c + ch, ch))
            c = ch
        self.heat = nn.Conv2d(c, 1, 1)
        self.offset = nn.Conv2d(c, 2, 1)
        nn.init.constant_(self.heat.bias, -math.log((1 - prior) / prior))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        m = 2**self.depth
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for blk, sk in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, sk], 1))
        heat, off = self.heat(x), self.offset(x)
        return heat[..., :h, :w], off[..., :h, :w]


def build_input(cur: torch.Tensor, past: list[torch.Tensor]) -> torch.Tensor:
    """Stack current, warped past frames and differences; tensors are (B, 1, H, W) in [0, 255]."""
    cur_n = (cur - 127.5) / 64.0
    chans = [cur_n]
    diffs = []
    for p in past:
        pn = (p - 127.5) / 64.0
        chans.append(pn)
        diffs.append(cur_n - pn)
    return torch.cat(chans + diffs, 1)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
    """Penalty-reduced pixel-wise focal loss (CenterNet) with an ignore/weight map."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos * weight
    neg_loss = -torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg * weight
    n_pos = pos.mul(weight).sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def offset_loss(pred: torch.Tensor, target: torch.Tensor, pos: torch.Tensor):
    n = pos.sum().clamp(min=1.0)
    return (F.l1_loss(pred, target, reduction="none") * pos).sum() / (2 * n)


@torch.no_grad()
def decode(heat_logits: torch.Tensor, offset: torch.Tensor, threshold: float, nms: int = 3, max_det: int = 400):
    """Peaks of the sigmoid heatmap -> list (per batch item) of (x, y, prob) with sub-pixel offsets."""
    p = torch.sigmoid(heat_logits)
    k = 2 * nms + 1
    mx = F.max_pool2d(p, k, stride=1, padding=nms)
    keep = (p == mx) & (p >= threshold)
    out = []
    for b in range(p.shape[0]):
        ys, xs = torch.nonzero(keep[b, 0], as_tuple=True)
        sc = p[b, 0, ys, xs]
        if sc.numel() > max_det:
            top = torch.topk(sc, max_det).indices
            ys, xs, sc = ys[top], xs[top], sc[top]
        dx = offset[b, 0, ys, xs]
        dy = offset[b, 1, ys, xs]
        out.append(torch.stack([xs.float() + dx, ys.float() + dy, sc], 1).cpu())
    return out
