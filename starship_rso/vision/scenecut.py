"""Camera-cut detection on downscaled frames (histogram correlation + mean difference)."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import SceneCutCfg


class SceneCutDetector:
    def __init__(self, cfg: SceneCutCfg):
        self.cfg = cfg
        self._prev_small: np.ndarray | None = None
        self._prev_hist: np.ndarray | None = None
        self.shot_id = 0
        self.last_corr = 1.0
        self.last_mad = 0.0

    def _small(self, gray_u8: np.ndarray) -> np.ndarray:
        h, w = gray_u8.shape
        s = min(1.0, self.cfg.work_width / float(w))
        return cv2.resize(gray_u8, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

    def update(self, gray_u8: np.ndarray, valid: np.ndarray | None = None) -> bool:
        """Return True if this frame starts a new shot."""
        small = self._small(gray_u8)
        hist = cv2.calcHist([small], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            self.last_corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
            self.last_mad = float(np.mean(np.abs(small.astype(np.float32) - self._prev_small.astype(np.float32))))
            cut = self.last_corr < self.cfg.hist_corr_thresh or self.last_mad > self.cfg.mean_abs_thresh
        elif self._prev_small is not None:
            cut = True  # resolution change
        self._prev_small, self._prev_hist = small, hist
        if cut:
            self.shot_id += 1
        return cut
