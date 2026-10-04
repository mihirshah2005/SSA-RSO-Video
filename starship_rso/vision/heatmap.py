"""Learned temporal heatmap detector for the live pipeline (+ fusion with the classical one).

Builds exactly the training input: the current frame and K-1 past frames
warped into the current frame with the pipeline's own registration chain.
The first K-1 frames of every shot produce no learned detections (the model
never saw incomplete windows).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..config import HeatmapCfg
from ..types import Detection
from .classical import estimate_noise, logistic_conf, remeasure

log = logging.getLogger(__name__)


class HeatmapDetector:
    def __init__(self, cfg: HeatmapCfg, shape: tuple[int, int]):
        import torch

        from ..train.model import TemporalUNet
        from ..train.train_heatmap import pick_device

        if not cfg.weights:
            raise ValueError("detector.heatmap.weights is not set (train a model first; see docs/MANUAL_STEPS.md)")
        self.cfg = cfg
        self._n = 0
        self._noise: float | None = None
        self._rng = np.random.default_rng(0)
        self.shape = shape
        self.torch = torch
        self.device = pick_device(cfg.device)
        ck = torch.load(cfg.weights, map_location="cpu", weights_only=False)
        meta = ck.get("meta", {})
        self.frames = int(meta.get("frames", cfg.frames))
        if self.frames != cfg.frames:
            log.warning("checkpoint was trained with %d frames; using that instead of config %d", self.frames, cfg.frames)
        self.model = TemporalUNet(self.frames, meta.get("base", 32), meta.get("depth", 4))
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(self.device)
        self.half = cfg.half and self.device.type == "cuda"
        if self.half:
            self.model.half()

    def reset(self) -> None:
        self._noise = None  # a new shot may have a different noise level
        self._n = 0

    def restrict_history(self, valid) -> None:
        pass  # uses only the shared History (restricted by the pipeline) and the current mask

    def _infer(self, stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        from ..train.model import build_input

        with torch.no_grad():
            t = torch.from_numpy(stack[None]).to(self.device)
            x = build_input(t[:, -1:], [t[:, q : q + 1] for q in range(t.shape[1] - 2, -1, -1)])
            if self.half:
                x = x.half()
            hl, of = self.model(x)
            return torch.sigmoid(hl.float())[0, 0].cpu().numpy(), of.float()[0].cpu().numpy()

    def heat(self, stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        k, h, w = stack.shape
        if c.tile <= 0 or (h <= c.tile and w <= c.tile):
            return self._infer(stack)
        prob = np.zeros((h, w), np.float32)
        off = np.zeros((2, h, w), np.float32)
        ov = 64
        step = c.tile - 2 * ov
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                ya, xa = max(0, min(y0 - ov, h - c.tile)), max(0, min(x0 - ov, w - c.tile))
                p, o = self._infer(np.ascontiguousarray(stack[:, ya : ya + c.tile, xa : xa + c.tile]))
                # keep the central part of each tile
                yb0, xb0 = (0 if ya == 0 else ov), (0 if xa == 0 else ov)
                prob[ya + yb0 : ya + c.tile, xa + xb0 : xa + c.tile] = p[yb0:, xb0:]
                off[:, ya + yb0 : ya + c.tile, xa + xb0 : xa + c.tile] = o[:, yb0:, xb0:]
        return prob, off

    def detect(self, gray, history, valid, frame_index=-1, t=0.0, want_debug=False, static_background=False,
               report=True):
        c = self.cfg
        k = self.frames
        if not report or len(history) < k - 1 or not all(history[q][2] for q in range(k - 1)):
            return []
        self._n += 1
        if self._noise is None or self._n % 30 == 1:  # refreshed whether or not anything is found
            self._noise = estimate_noise(gray, self._rng)
        h, w = gray.shape
        past = [cv2.warpPerspective(history[q][0], history[q][1], (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE) for q in range(k - 2, -1, -1)]
        stack = np.stack(past + [gray], 0).astype(np.float32)
        prob, off = self.heat(stack)
        pm = np.where(valid, prob, 0.0).astype(np.float32)
        mx = cv2.dilate(pm, np.ones((7, 7), np.uint8))
        ys, xs = np.nonzero((pm >= mx) & (pm >= c.low_threshold))
        if len(ys) == 0:
            return []
        dets = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            p = float(prob[y, x])
            # map the validated operating point (threshold) to the tracker's 0.5 birth level,
            # matching the classical detector's "0.5 = just passes"
            dets.append(Detection(
                x=x + float(off[0, y, x]), y=y + float(off[1, y, x]), score=p / c.threshold,
                confidence=float(logistic_conf(p / c.threshold)), sigma=1.0, residual=0.0,
                pos_sigma=float(np.clip(0.5 / max(p, 0.1), 0.2, 3.0)), polarity=1, source="heatmap",
                frame_index=frame_index, t=t,
            ))
        # size, flux and shape measured exactly as for classical detections, so track features
        # (and the category rules or model trained on them) do not depend on the detector
        return remeasure(gray, dets, self._noise)  # starts from the network's sub-pixel position


class FusedDetector:
    """Union of classical and learned detections; coincident pairs keep the learned position."""

    def __init__(self, classical, heatmap, radius: float = 3.0):
        self.classical = classical
        self.heatmap = heatmap
        self.radius = radius

    def reset(self) -> None:
        self.classical.reset()
        self.heatmap.reset()

    def restrict_history(self, valid) -> None:
        self.classical.restrict_history(valid)
        self.heatmap.restrict_history(valid)

    def detect(self, gray, history, valid, frame_index=-1, t=0.0, want_debug=False, static_background=False,
               report=True):
        a = self.classical.detect(gray, history, valid, frame_index, t, want_debug, static_background, report)
        b = self.heatmap.detect(gray, history, valid, frame_index, t, want_debug, static_background, report)
        if not a or not b:
            return a + b
        tree = cKDTree(np.array([[d.x, d.y] for d in b]))
        used = set()
        out = list(b)
        for d in a:
            dist, j = tree.query([d.x, d.y], distance_upper_bound=self.radius)
            if np.isfinite(dist) and j not in used:
                used.add(j)
                bj = out[j]
                bj.confidence = max(bj.confidence, d.confidence)
                bj.residual = d.residual
                bj.source = "fused"
            else:
                out.append(d)
        return out
