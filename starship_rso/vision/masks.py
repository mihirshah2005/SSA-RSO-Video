"""Masks: broadcast graphics (HUD), frame border, and camera-fixed vehicle structure.

For a camera mounted on the ship, the ship's own structure is fixed in the
image while the Earth moves. ``StaticStructureMask`` decides this per pixel
from two exponential moving averages at quarter resolution:

* ``du`` -- mean |I_t - I_{t-1}| without registration (small on the vehicle),
* ``dr`` -- mean |I_t - warp(I_{t-1})| after background registration (small on
  the Earth, large on textured vehicle structure because the warp moves it).

A pixel is vehicle structure when it is textured, ``du`` is small and
registration makes the difference clearly *worse* (``dr > ratio * du + delta``).
Smooth regions are ambiguous by construction and stay valid. When the
background has no texture (black sky, registration unavailable) the fallback
is "static and bright or textured" from the temporal standard deviation;
black sky is static but dark and flat, so it stays valid. Small static blobs
(a hovering object) are left unmasked by an area threshold.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import MaskCfg, StaticMaskCfg
from .ops import scale_matrix


def hud_mask(shape: tuple[int, int], rects: list[list[float]], border_px: int = 0) -> np.ndarray:
    """Boolean mask, True where HUD graphics or the border make pixels unusable."""
    h, w = shape
    m = np.zeros((h, w), bool)
    for r in rects:
        x0, y0, x1, y1 = r
        xa, xb = int(np.floor(x0 * w)), int(np.ceil(x1 * w))
        ya, yb = int(np.floor(y0 * h)), int(np.ceil(y1 * h))
        m[max(0, ya) : min(h, yb), max(0, xa) : min(w, xb)] = True
    if border_px > 0:
        b = int(border_px)
        m[:b, :] = True
        m[-b:, :] = True
        m[:, :b] = True
        m[:, -b:] = True
    return m


def _scaled_h(H: np.ndarray, sx: float, sy: float) -> np.ndarray:
    S = scale_matrix(sx, sy)
    return S @ H @ np.linalg.inv(S)


class StaticStructureMask:
    ratio = 2.0
    delta = 2.0
    du_max = 4.0
    vehicle_frac = 0.6  # fraction of recent registered frames with structure evidence

    def __init__(self, cfg: StaticMaskCfg, shape: tuple[int, int], downsample: int = 4):
        self.cfg = cfg
        self.shape = shape
        self.ds = max(1, int(downsample))
        h, w = shape
        self._small = (max(1, w // self.ds), max(1, h // self.ds))
        self._sx, self._sy = self._small[0] / w, self._small[1] / h
        self.reset()

    def reset(self) -> None:
        self._mean = self._sq = self._du = self._grad = None
        self._prev_small: np.ndarray | None = None
        self._t0 = self._t_last = None
        self._n = 0
        self._n_reg = 0
        self._reg_frac = 0.0
        self._mask = np.zeros(self.shape, bool)
        self._computed = False

    @property
    def ready(self) -> bool:
        """True once the vehicle mask has been estimated for this shot (always true when disabled)."""
        return (not self.cfg.enabled) or self._computed

    def update(self, gray: np.ndarray, t: float, H_prev_to_cur: np.ndarray | None = None,
               reg_valid: bool = False) -> np.ndarray:
        if not self.cfg.enabled:
            return self._mask
        small = cv2.resize(gray, self._small, interpolation=cv2.INTER_AREA).astype(np.float32)
        last = self._t_last if self._t_last is not None else t
        dt = max(t - last, 1e-3)
        # cumulative average while young, exponential memory of tau_s afterwards
        a = max(float(1.0 - np.exp(-dt / max(self.cfg.tau_s, 1e-3))), 1.0 / (self._n + 1))
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy) / 4.0
        if self._mean is None:
            self._mean, self._sq, self._grad = small.copy(), small * small, grad
            self._t0 = t
        else:
            self._mean += a * (small - self._mean)
            self._sq += a * (small * small - self._sq)
            self._grad += a * (grad - self._grad)
            self._reg_frac += a * ((1.0 if reg_valid else 0.0) - self._reg_frac)
            if reg_valid and H_prev_to_cur is not None and self._prev_small is not None:
                Hs = _scaled_h(H_prev_to_cur, self._sx, self._sy)
                warped = cv2.warpPerspective(self._prev_small, Hs, self._small, flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE)
                du = cv2.blur(np.abs(small - self._prev_small), (3, 3))
                dr = cv2.blur(np.abs(small - warped), (3, 3))
                # instantaneous evidence: textured, unchanged in camera coordinates, and made worse
                # by background registration. Its long-run frequency separates permanent structure
                # from objects that merely pass slowly through a pixel.
                ind = ((grad > self.cfg.grad_thresh) & (du < self.du_max) & (dr > self.ratio * du + self.delta))
                ind = ind.astype(np.float32)
                a_reg = max(float(1.0 - np.exp(-dt / max(self.cfg.tau_s, 1e-3))), 1.0 / (self._n_reg + 1))
                if self._du is None:
                    self._du = ind
                else:
                    self._du += a_reg * (ind - self._du)
                self._n_reg += 1
        self._prev_small = small
        self._t_last = t
        self._n += 1
        t0 = self._t0 if self._t0 is not None else t
        if (t - t0) < self.cfg.warmup_s:
            return self._mask
        if self._n % max(1, self.cfg.update_every) == 0 or not self._computed:
            self._mask = self._compute()
            self._computed = True
        return self._mask

    def _compute(self) -> np.ndarray:
        c = self.cfg
        textured = self._grad > c.grad_thresh
        if self._du is not None and self._n_reg >= 5 and self._reg_frac > 0.3:
            structure = self._du > self.vehicle_frac
        else:
            std = np.sqrt(np.maximum(self._sq - self._mean**2, 0.0))
            structure = (std < c.std_thresh) & ((self._mean > c.bright_thresh) | textured)
        m = structure.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        min_area_small = c.min_area_px / float(self.ds * self.ds)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_small
        m = keep[lab].astype(np.uint8)
        # fill small holes: smooth patches enclosed by vehicle structure belong to the vehicle.
        # Large enclosed regions (Earth seen between a flap and the body) stay valid.
        inv = 1 - m
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        hs, ws = m.shape
        max_hole = 0.02 * hs * ws
        fill = np.zeros(n2, bool)
        for k in range(1, n2):
            x, y, bw, bh, area = stats2[k]
            fill[k] = x > 0 and y > 0 and x + bw < ws and y + bh < hs and area <= max_hole
        m = (m.astype(bool) | fill[lab2]).astype(np.uint8)
        h, w = self.shape
        full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        if c.dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c.dilate_px + 1, 2 * c.dilate_px + 1))
            full = cv2.dilate(full, k)
        return full.astype(bool)

    @property
    def mask(self) -> np.ndarray:
        return self._mask


class MaskManager:
    """Combines the fixed HUD/border mask with the adaptive vehicle mask."""

    def __init__(self, cfg: MaskCfg, shape: tuple[int, int]):
        self.cfg = cfg
        self.shape = shape
        self.fixed = hud_mask(shape, cfg.hud_rects, cfg.border_px)
        self.static = StaticStructureMask(cfg.static, shape)
        self.invalid = self.fixed.copy()

    def reset(self) -> None:
        self.static.reset()
        self.invalid = self.fixed.copy()

    @property
    def valid(self) -> np.ndarray:
        return ~self.invalid

    @property
    def ready(self) -> bool:
        return self.static.ready

    def update(self, gray: np.ndarray, t: float, H_prev_to_cur: np.ndarray | None = None,
               reg_valid: bool = False) -> np.ndarray:
        """Update with the new frame and its registration; return the *valid* mask."""
        vm = self.static.update(gray, t, H_prev_to_cur, reg_valid)
        self.invalid = self.fixed | vm
        return ~self.invalid

    @property
    def masked_fraction(self) -> float:
        return float(self.invalid.mean())
