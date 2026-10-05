"""Background (Earth) motion between consecutive frames.

Sparse corners on the unmasked background are tracked with pyramidal
Lucas-Kanade, filtered by forward-backward consistency, and a motion model is
fitted with RANSAC. A distant Earth surface is close to a plane, so a
homography is the most general model; it is used only when the corners
support it and fit clearly better than an affine or similarity model (see
``_fit_supported``). Poor fits are reported, not hidden. When the frame has
no usable texture and is dark (black sky) the background is flagged static
and the identity is returned, which is correct for a fixed black background;
a bright featureless field is reported as unmeasurable motion instead.

``History`` keeps the last few frames with their homographies to the current
frame so residual images can be formed against several past frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import RegistrationCfg
from .ops import apply_h, scale_homography_xy


@dataclass
class Registration:
    H: np.ndarray  # previous -> current (processing pixels)
    valid: bool
    n_points: int = 0
    n_inliers: int = 0
    rms: float = float("nan")
    dt: float = 0.0
    pts_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pts_cur: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    reason: str = ""
    static_background: bool = False  # background fixed in the image (black sky, or nothing moves)
    _Hinv: np.ndarray | None = field(default=None, repr=False)

    def flow_at(self, x: float, y: float) -> tuple[float, float] | None:
        """Background image velocity (px/s) at (x, y) in the *current* frame."""
        if not self.valid or self.dt <= 0:
            return None
        if self._Hinv is None:
            self._Hinv = np.linalg.inv(self.H)
        prev = apply_h(self._Hinv, np.array([[x, y]]))[0]
        return float((x - prev[0]) / self.dt), float((y - prev[1]) / self.dt)


class BackgroundRegistrar:
    textured_grad = 2.0  # grey levels per pixel (Sobel/4) for a pixel to count as textured
    black_sky_frac = 0.02  # textured fraction below which a corner-less frame is "black sky"...
    dark_level = 40.0  # ...provided it is also dark (median grey level)

    def __init__(self, cfg: RegistrationCfg, shape: tuple[int, int]):
        self.cfg = cfg
        self.shape = shape
        h, w = shape
        s = min(1.0, cfg.work_width / float(w)) if w > 0 else 1.0
        self.work_size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        self.sx = self.work_size[0] / float(w)
        self.sy = self.work_size[1] / float(h)
        self._low_qual: float | None = None  # noise-floor corner bar, reused for a few frames
        self._low_qual_age = 0

    def reset(self) -> None:
        """Forget per-shot state (call at a camera cut)."""
        self._low_qual, self._low_qual_age = None, 0

    def _work(self, gray_u8: np.ndarray) -> np.ndarray:
        if self.work_size == (self.shape[1], self.shape[0]):
            return gray_u8
        return cv2.resize(gray_u8, self.work_size, interpolation=cv2.INTER_AREA)

    def _to_full(self, pts: np.ndarray) -> np.ndarray:
        cx, cy = 0.5 * self.sx - 0.5, 0.5 * self.sy - 0.5
        return np.c_[(pts[:, 0] - cx) / self.sx, (pts[:, 1] - cy) / self.sy]

    def _fit(self, a: np.ndarray, b: np.ndarray, model: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        """RANSAC fit of ``model`` (homography | affine | similarity); returns (3x3, inlier mask)."""
        t = self.cfg.ransac_thresh_px
        if model == "homography":
            Hw, inl = cv2.findHomography(a, b, cv2.RANSAC, t)
        else:
            fit = cv2.estimateAffine2D if model == "affine" else cv2.estimateAffinePartial2D
            A, inl = fit(a, b, method=cv2.RANSAC, ransacReprojThreshold=t)
            Hw = np.vstack([A, [0, 0, 1]]) if A is not None else None
        if Hw is None or inl is None:
            return None, None
        return Hw, inl.ravel().astype(bool)

    def _fit_supported(self, a: np.ndarray, b: np.ndarray, usable_area: float):
        """Simplest model that explains the motion as well as the most general supported one.

        Over-fitted perspective terms extrapolate badly (several px) into image areas without
        corners, which then show up as spurious motion. Candidates run from similarity (shift,
        rotation, scale) through affine (linear flow gradient) to homography (plane-induced
        perspective, up to ``cfg.model``). The most general model is used only when the inliers
        support it (number and spread) AND it fits clearly better than the simpler ones on its
        own inlier set. Returns (H, inlier mask, model name).
        """
        c = self.cfg
        order = ["similarity", "affine", "homography"]
        allowed = order[: order.index(c.model) + 1] if c.model in order else order
        need = {"similarity": (0, 0.0), "affine": (c.affine_min_inliers, c.affine_min_spread),
                "homography": (c.full_model_min_inliers, c.full_model_min_spread)}
        fits = {}
        for m in allowed:
            Hw, inl = self._fit(a, b, m)
            if Hw is None:
                continue
            n_in = int(inl.sum())
            spread = cv2.contourArea(cv2.convexHull(a[inl].astype(np.float32))) / usable_area if n_in >= 3 else 0.0
            if m == "similarity" or (n_in >= need[m][0] and spread >= need[m][1]):
                fits[m] = (Hw, inl)
        if not fits:
            return None, None, allowed[0]
        general = [m for m in allowed if m in fits][-1]
        sel = fits[general][1]
        rms = {m: float(np.sqrt(np.mean(np.sum((apply_h(H, a[sel]) - b[sel]) ** 2, axis=1)))) for m, (H, _) in fits.items()}
        for m in allowed:
            if m in fits and rms[m] <= max(1.2 * rms[general], rms[general] + 0.05):
                return fits[m][0], fits[m][1], m
        return fits[general][0], fits[general][1], general

    def estimate(
        self, prev_u8: np.ndarray, cur_u8: np.ndarray, valid: np.ndarray | None, dt: float
    ) -> Registration:
        c = self.cfg
        p = self._work(prev_u8)
        q = self._work(cur_u8)
        mask = None
        if valid is not None:
            mask = cv2.resize(valid.astype(np.uint8) * 255, self.work_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        # texture: fraction of usable pixels with real gradient (a mean would let a large black
        # area hide a thin, moving Earth limb)
        ps = cv2.GaussianBlur(p, (0, 0), 1.5)  # suppress pixel noise: texture means structure
        gx = cv2.Sobel(ps, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ps, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) / 4.0
        sel = mask > 0 if mask is not None and (mask > 0).any() else np.ones(mag.shape, bool)
        tex_frac = float((mag[sel] > self.textured_grad).mean())
        ident = np.eye(3)
        pts = None
        reuse = self._low_qual is not None and self._low_qual_age < 10
        if tex_frac > 0.002:
            pts = cv2.goodFeaturesToTrack(
                p, maxCorners=c.max_corners, qualityLevel=self._low_qual if reuse else c.quality,
                minDistance=c.min_distance, mask=mask, blockSize=7
            )
            self._low_qual_age += 1
        if not reuse:
            self._low_qual = None
        if not reuse and pts is not None and len(pts) < max(c.min_inliers, c.max_corners // 3):
            # Few corners: the strongest corner sets the (relative) quality bar, so weak cloud
            # texture on a smooth ocean is ignored. Ask again with a bar just above the noise:
            # pixel noise alone makes "corners" that LK follows with zero motion, so the bar is
            # ten times the median corner response (most of a smooth scene is noise).
            eig = cv2.cornerMinEigenVal(p, 7)
            ev = eig[sel]
            top = float(ev.max()) if ev.size else 0.0
            if top > 0:
                qual = max(1e-4, 10.0 * float(np.median(ev)) / top)
                if qual < c.quality:
                    more = cv2.goodFeaturesToTrack(p, maxCorners=c.max_corners, qualityLevel=qual,
                                                   minDistance=c.min_distance, mask=mask, blockSize=7)
                    if more is not None and len(more) > len(pts):
                        pts = more
                        # the scene changes slowly: keep this bar for the next frames (refreshed every 10)
                        self._low_qual, self._low_qual_age = qual, 0
        # "black sky" needs both: no texture AND dark. A bright featureless field (smooth ocean,
        # overexposed cloud) may well be moving; its motion is simply unmeasurable here.
        level = float(np.median(p[sel])) if sel.any() else 0.0
        black_sky = tex_frac < self.black_sky_frac and level < self.dark_level

        def too_few(reason: str, n: int = 0) -> Registration:
            if black_sky:  # the few corners are objects crossing the sky, not the background
                return Registration(ident, False, n_points=n, dt=dt, reason=f"black sky (texture {tex_frac:.3f})",
                                    static_background=True)
            return Registration(ident, False, n_points=n, dt=dt, reason=f"{reason} (texture {tex_frac:.3f})")

        if pts is None or len(pts) < c.min_inliers:
            return too_few("unmeasurable motion")
        lk = dict(
            winSize=(c.lk_win, c.lk_win),
            maxLevel=c.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(p, q, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(q, p, nxt, None, **lk)
        fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < c.fb_thresh_px)
        a, b = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
        if len(a) < c.min_inliers:
            return too_few("too few consistent tracks", len(a))
        # Which corners belong to the background? Two candidate sets are tried:
        #  1. corners that clearly move (> static_px). Before the ship is masked, its camera-fixed
        #     corners would otherwise pull the fit toward "no motion".
        #  2. all corners. A slowly drifting Earth (well under static_px per frame) still has a
        #     measurable, coherent motion; it accumulates over the longer residual baselines.
        # A set counts as the moving background only if its inliers are numerous, cover a real part
        # of the image and the motion they imply is significant given their scatter; LK noise on
        # faint camera-fixed corners (a dark hull on black sky) implies no significant motion.
        # Otherwise the background is fixed in the image, or incoherent.
        disp = np.linalg.norm(b - a, axis=1)
        moving = disp > c.static_px
        usable = float(sel.sum()) if sel.any() else float(p.size)
        best = None
        candidates = []
        if moving.sum() >= c.min_inliers:
            candidates.append((a[moving], b[moving]))
        candidates.append((a, b))
        for am, bm in candidates:
            Hw, inl, model = self._fit_supported(am, bm, usable)
            if Hw is None:
                continue
            n_in = int(inl.sum())
            if n_in < max(c.min_inliers, 0.5 * len(am)):
                continue
            spread = cv2.contourArea(cv2.convexHull(am[inl].astype(np.float32))) / usable if n_in >= 3 else 0.0
            # Is the fitted motion significant? Compare the motion the model predicts at the inliers
            # with the model's own uncertainty (per-corner scatter / sqrt(inliers per parameter)).
            # Heavily compressed video gives ~0.3 px of scatter per corner, yet hundreds of corners
            # pin a 0.3 px/frame Earth drift down to a few hundredths of a pixel.
            pred = apply_h(Hw, am[inl]) - am[inl]
            m_pred = float(np.sqrt(np.mean(np.sum(pred**2, axis=1))))
            r_mod = float(np.sqrt(np.mean(np.sum((apply_h(Hw, am[inl]) - bm[inl]) ** 2, axis=1))))
            se = max(r_mod, 0.05) * np.sqrt(8.0 / n_in)
            if spread >= c.min_moving_spread and m_pred > max(3.0 * se, 0.05):
                best = (am, bm, Hw, inl)
                break
        if best is None:
            n_static = int((~moving).sum())
            if n_static >= c.min_inliers:
                return Registration(ident, True, n_points=len(a), n_inliers=n_static,
                                    rms=float(np.sqrt(np.mean(disp[~moving] ** 2)) / min(self.sx, self.sy)), dt=dt,
                                    reason="background fixed in the image", static_background=True)
            return Registration(ident, False, n_points=len(a), dt=dt, reason="incoherent motion")
        a, b, Hw, inl = best
        n_in = int(inl.sum())
        res = np.linalg.norm(apply_h(Hw, a[inl]) - b[inl], axis=1)
        H = scale_homography_xy(Hw, self.sx, self.sy)
        return Registration(
            H,
            True,
            n_points=len(a),
            n_inliers=n_in,
            rms=float(np.sqrt(np.mean(res**2)) / min(self.sx, self.sy)),
            dt=dt,
            pts_prev=self._to_full(a[inl]),
            pts_cur=self._to_full(b[inl]),
        )


class History:
    """Past processed frames with homographies mapping each into the current frame."""

    def __init__(self, maxlen: int):
        self.maxlen = max(0, int(maxlen))
        self._frames: deque = deque(maxlen=max(1, self.maxlen))  # (gray, H_to_current, chain_ok, valid_mask)

    def clear(self) -> None:
        self._frames.clear()

    def advance(self, reg: Registration) -> None:
        """Called after a new frame arrives: compose the new prev->current homography."""
        H = reg.H if reg.valid else np.eye(3)
        ok = reg.valid or reg.static_background
        self._frames = deque(
            ((g, H @ Hk, v and ok, m) for g, Hk, v, m in self._frames), maxlen=self._frames.maxlen
        )

    def push(self, gray: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Store a processed frame and its valid mask (pixels that were usable background)."""
        if self.maxlen > 0:
            self._frames.appendleft((gray, np.eye(3), True, valid))

    def restrict(self, valid: np.ndarray) -> None:
        """AND a camera-fixed mask (e.g. a newly learned vehicle mask) into every stored frame's mask.

        The vehicle does not move in the image, so the current mask applies to past frames as
        they are; without this, frames stored before the mask existed would make the background
        behind the ship's edge look available to the motion residual.
        """
        self._frames = deque(
            ((g, Hk, v, valid.copy() if m is None else (m & valid)) for g, Hk, v, m in self._frames),
            maxlen=self._frames.maxlen,
        )

    def items(self):
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)
