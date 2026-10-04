"""Classical point detector: multi-scale top-hat appearance + registered temporal residual.

Two kinds of evidence are combined:

* **Appearance** -- a white (or black) top-hat removes structures larger than
  the structuring element; its output divided by a robust local noise scale is
  ``snr_app``. Several scales (with downsampling) catch both 1-3 px points and
  large defocused discs from near-field particles.
* **Motion** -- the current value minus the max-filtered value of each
  background-registered past frame (``min`` over past frames) is ``snr_res``.
  Clouds that move with the Earth cancel; anything moving differently from the
  background, including an object *fixed* in the image while the Earth slides
  beneath it, survives.

In low-clutter regions (black sky, flat ocean) appearance alone is enough; in
cluttered regions (cloud fields) both are required, because a single cumulus
puff looks exactly like a dot in one frame. Clutter is the local mean absolute
high-pass of the image. Scores are in units of their threshold: ``u = 1``
means "just passes"; candidates down to ``low_factor`` are kept as
low-confidence detections for the tracker's second association stage.

For speed, the dense work is limited to the top-hat and its local maxima; the
temporal residual is evaluated only at appearance peaks, by mapping each peak
back into the past frames through the registration homographies.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..config import ClassicalCfg
from ..types import Detection
from .ops import apply_h, ellipse_kernel, peak_stats_map, robust_sigma, scale_matrix


def logistic_conf(u):
    return 1.0 / (1.0 + np.exp(-4.0 * (np.asarray(u, dtype=np.float64) - 1.0)))


def batch_centroids(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, halfwin: int) -> dict[str, np.ndarray]:
    """Vectorised intensity-weighted centroids/second moments of positive ``img`` around peaks."""
    h, w = img.shape
    r = np.arange(-halfwin, halfwin + 1)
    oy, ox = np.meshgrid(r, r, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()
    yy = ys[:, None] + oy[None, :]
    xx = xs[:, None] + ox[None, :]
    inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    vals = img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)].astype(np.float64)
    wts = np.where(inside, np.clip(vals, 0, None), 0.0)
    tot = wts.sum(1)
    tot_safe = np.where(tot > 0, tot, 1.0)
    cx = (wts * xx).sum(1) / tot_safe
    cy = (wts * yy).sum(1) / tot_safe
    dx, dy = xx - cx[:, None], yy - cy[:, None]
    ixx = (wts * dx * dx).sum(1) / tot_safe
    iyy = (wts * dy * dy).sum(1) / tot_safe
    ixy = (wts * dx * dy).sum(1) / tot_safe
    trace = ixx + iyy
    sigma = np.sqrt(np.maximum(trace / 2.0, 1e-6))
    ell = np.where(trace > 1e-9, np.sqrt((ixx - iyy) ** 2 + 4 * ixy**2) / np.maximum(trace, 1e-9), 0.0)
    peak = wts.max(1)
    area = (wts > 0.25 * peak[:, None]).sum(1)
    return {"x": cx, "y": cy, "flux": tot, "peak": peak, "sigma": sigma, "ellipticity": ell, "area": area, "ok": tot > 0}


def _neighbourhood_extreme(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, half: int, mode: str) -> np.ndarray:
    """Max (or min) of ``img`` over a (2*half+1)^2 window at integer positions (clipped)."""
    h, w = img.shape
    r = np.arange(-half, half + 1)
    oy, ox = np.meshgrid(r, r, indexing="ij")
    yy = np.clip(ys[:, None] + oy.ravel()[None, :], 0, h - 1)
    xx = np.clip(xs[:, None] + ox.ravel()[None, :], 0, w - 1)
    v = img[yy, xx]
    return v.max(1) if mode == "max" else v.min(1)


def _mask_of(item) -> np.ndarray | None:
    return item[3] if len(item) > 3 else None


def _row_median(a: np.ndarray) -> np.ndarray:
    """Median of each row ignoring NaN (0 for an all-NaN row); fast when no NaN is present."""
    nan = np.isnan(a)
    if not nan.any():
        return np.median(a, axis=1)
    out = np.median(np.where(nan, 0.0, a), axis=1)
    for i in np.nonzero(nan.any(axis=1))[0]:  # only windows that reach past the image edge
        row = a[i][~nan[i]]
        out[i] = float(np.median(row)) if row.size else 0.0
    return out


def _measure_group(gray: np.ndarray, cx: np.ndarray, cy: np.ndarray, pol: np.ndarray, hw: int,
                   noise: float) -> dict[str, np.ndarray]:
    """Measure several objects with the same window half-width ``hw`` at once.

    For each window: background = median of the window border; the object is the
    8-connected region above max(20% of its own peak, 2 x noise, 1.5 x the robust spread
    of the border) that contains the brightest point near the window centre (the
    border spread is the local texture, so a dot on clouds does not merge with them). In large windows, seed, threshold and
    region come from a k x k box-smoothed copy (k grows with the window, noise
    falls by k), so a faint defocused disc whose single pixels sit at 1-2 noise
    sigma is still found as a disc rather than as one noise spike; the box's own
    variance is removed from the moments. Moments of the region give position,
    size and shape; flux sums the raw signal over the connected region above the
    noise level. Neighbouring texture or another object does not count. Pixels
    outside the image are excluded (not repeated). Windows are labelled together
    as one mosaic separated by empty rows: two ``connectedComponents`` calls per group.
    """
    from scipy.ndimage import uniform_filter

    h, w = gray.shape
    n, S = len(cx), 2 * hw + 1
    r = np.arange(-hw, hw + 1)
    yr = np.rint(cy).astype(int)[:, None] + r[None]  # (n, S) unclipped
    xr = np.rint(cx).astype(int)[:, None] + r[None]
    inside = ((yr >= 0) & (yr < h))[:, :, None] & ((xr >= 0) & (xr < w))[:, None, :]  # (n, S, S)
    yy, xx = np.clip(yr, 0, h - 1), np.clip(xr, 0, w - 1)
    v = gray[yy[:, :, None], xx[:, None, :]].astype(np.float64) * pol[:, None, None]
    v = np.where(inside, v, np.nan)
    border = np.concatenate([v[:, 0], v[:, -1], v[:, 1:-1, 0], v[:, 1:-1, -1]], axis=1)
    bg = _row_median(border)
    v = np.nan_to_num(v - bg[:, None, None], nan=0.0)  # outside: background level
    k = 2 * (hw // 9) + 1  # 1 for point sources (hw <= 8), up to 5 for the largest windows
    vs = uniform_filter(v, size=(1, k, k), mode="nearest") if k > 1 else v
    nk = noise / k  # noise of a k x k mean
    r0 = max(1, min(hw - 1, hw // 3))
    c = np.where(inside, vs, -np.inf)[:, hw - r0 : hw + r0 + 1, hw - r0 : hw + r0 + 1].reshape(n, -1)
    j = np.argmax(c, axis=1)
    iy, ix = j // (2 * r0 + 1) + hw - r0, j % (2 * r0 + 1) + hw - r0
    ar = np.arange(n)
    peak_s = vs[ar, iy, ix]
    # background variation seen on the window border: noise on black sky, texture on clouds.
    # Requiring the object to stand above it stops a dot from merging with the clouds around it.
    bs = np.concatenate([vs[:, 0], vs[:, -1], vs[:, 1:-1, 0], vs[:, 1:-1, -1]], axis=1)
    bin_ = np.concatenate([inside[:, 0], inside[:, -1], inside[:, 1:-1, 0], inside[:, 1:-1, -1]], axis=1)
    bs = np.where(bin_, bs, np.nan)
    spread = 1.4826 * _row_median(np.abs(bs - _row_median(bs)[:, None]))
    thr = np.maximum.reduce([0.2 * peak_s, np.full(n, 2.0 * nk), 1.5 * spread])

    def region(mask: np.ndarray) -> np.ndarray:
        mosaic = np.zeros((n, S + 1, S), np.uint8)
        mosaic[:, :S] = mask & inside
        _, lab = cv2.connectedComponents(mosaic.reshape(n * (S + 1), S), connectivity=8)
        lab = lab.reshape(n, S + 1, S)[:, :S]
        seed = lab[ar, iy, ix]
        return (lab == seed[:, None, None]) & (seed[:, None, None] > 0)

    comp = region(vs > thr[:, None, None])
    wts = np.where(comp, vs - thr[:, None, None], 0.0)
    tot = wts.sum((1, 2))
    ok = (peak_s > thr) & (tot > 0)
    ts = np.where(ok, tot, 1.0)
    gx, gy = xx[:, None, :].astype(np.float64), yy[:, :, None].astype(np.float64)
    mx = (wts * gx).sum((1, 2)) / ts
    my = (wts * gy).sum((1, 2)) / ts
    dx, dy = gx - mx[:, None, None], gy - my[:, None, None]
    box_var = (k * k - 1) / 12.0  # variance added by the k x k box
    ixx = np.maximum((wts * dx * dx).sum((1, 2)) / ts - box_var, 0.0)
    iyy = np.maximum((wts * dy * dy).sum((1, 2)) / ts - box_var, 0.0)
    ixy = (wts * dx * dy).sum((1, 2)) / ts
    tr = ixx + iyy
    flux = np.where(region(vs > 2.0 * nk), v, 0.0).sum((1, 2))
    peak = np.where(comp, v, -np.inf).max((1, 2))
    return {
        "ok": ok,
        "x": mx,
        "y": my,
        # weights above a threshold under-estimate a Gaussian's width (about 0.7x); the bias is the
        # same whichever scale or detector found the object, and the category rules use this estimator
        "sigma": np.sqrt(np.maximum(tr / 2.0, 0.09)),
        "ellipticity": np.where(tr > 1e-9, np.sqrt((ixx - iyy) ** 2 + 4 * ixy**2) / np.maximum(tr, 1e-9), 0.0),
        "flux": flux,
        "peak": np.where(ok, peak, peak_s),
        "area": comp.sum((1, 2)),
    }


def estimate_noise(gray: np.ndarray, rng: np.random.Generator, n: int = 4000) -> float:
    """Per-pixel noise (grey levels) from a sparse Laplacian sample; robust to stars and texture edges."""
    h, w = gray.shape
    ys = rng.integers(1, h - 1, n)
    xs = rng.integers(1, w - 1, n)
    lap = gray[ys, xs] - 0.25 * (gray[ys - 1, xs] + gray[ys + 1, xs] + gray[ys, xs - 1] + gray[ys, xs + 1])
    return max(robust_sigma(lap, floor=0.3) / np.sqrt(1.25), 0.3)


def remeasure(gray: np.ndarray, dets: list[Detection], noise: float, iters: int = 6) -> list[Detection]:
    """Centroid, size and flux of detections on the full-resolution image.

    Uses :func:`_measure_group`. This makes size and flux comparable whichever
    scale or detector found the object (moments on a coarse top-hat are dominated
    by its positive noise floor). The window follows the measured size, so a
    detection triggered by one edge or corner of a resolved object converges on
    the whole object; a point source settles after one or two passes. Detections
    with nothing measurable above the noise are dropped: their placeholder size
    and flux would not be comparable with the others and would corrupt track features.
    Returns the measured detections (updated in place).
    """
    if not dets:
        return []

    bins = np.array([3, 5, 8, 12, 17, 24])  # few distinct window sizes: few vectorised groups

    def half_width(sig: np.ndarray) -> np.ndarray:
        want = np.clip(np.rint(3 * np.maximum(sig, 1.0)) + 2, 3, 24)
        return bins[np.searchsorted(bins, want)]

    x = np.array([d.x for d in dets])
    y = np.array([d.y for d in dets])
    sig = np.array([d.sigma for d in dets])
    pol = np.array([d.polarity for d in dets], float)
    res = {k: np.zeros(len(dets)) for k in ("ellipticity", "flux", "peak", "area")}
    done = np.zeros(len(dets), bool)  # measured at least once
    active = np.ones(len(dets), bool)
    for _ in range(iters):
        hws = half_width(sig)
        for hw in np.unique(hws[active]):
            idx = np.nonzero(active & (hws == hw))[0]
            m = _measure_group(gray, x[idx], y[idx], pol[idx], int(hw), noise)
            good = m["ok"]
            g = idx[good]
            moved = np.hypot(m["x"][good] - x[g], m["y"][good] - y[g])
            x[g], y[g], sig[g] = m["x"][good], m["y"][good], m["sigma"][good]
            for key in res:
                res[key][g] = m[key][good]
            done[g] = True
            # converged: the window no longer moves and already matches the measured size
            active[idx[~good]] = False
            active[g] = (moved >= 0.3) | (half_width(sig[g]) != hw)
        if not active.any():
            break
    out = []
    for k in np.nonzero(done)[0]:
        d = dets[k]
        d.x, d.y, d.sigma = float(x[k]), float(y[k]), float(sig[k])
        d.ellipticity, d.flux, d.peak = float(res["ellipticity"][k]), float(res["flux"][k]), float(res["peak"][k])
        d.area = int(res["area"][k])
        d.pos_sigma = float(np.clip(1.5 * d.sigma * noise / max(d.peak, noise), 0.1, 3.0))
        out.append(d)
    return out


class ClassicalDetector:
    def __init__(self, cfg: ClassicalCfg, shape: tuple[int, int]):
        self.cfg = cfg
        self.shape = shape
        self._frame = 0
        self._sigma_app: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._sigma_res: dict[tuple[int, int], tuple[float, float]] = {}
        self._clutter: np.ndarray | None = None
        # Coarse scales keep their own longer history of downsampled frames: a large object that is
        # nearly fixed in the image needs a long baseline before the background has moved past it.
        n_coarse = max(c for c in (cfg.coarse_history or [1])) if cfg.coarse_history else 0
        self._coarse: dict[int, deque] = {
            sc.downsample: deque(maxlen=max(1, n_coarse)) for sc in cfg.scales if sc.downsample > 1
        }
        h, w = shape
        # per-axis resize factors: w // ds need not divide exactly
        self._sxy = {sc.downsample: (max(1, w // sc.downsample) / w, max(1, h // sc.downsample) / h)
                     for sc in cfg.scales}
        self._noise_hp: float | None = None
        self._rng = np.random.default_rng(0)
        self.debug: dict[str, np.ndarray] = {}

    def restrict_history(self, valid: np.ndarray) -> None:
        """AND a camera-fixed mask into the coarse-scale history (see ``History.restrict``)."""
        for dq in self._coarse.values():
            for item in dq:
                item[3] = item[3] & self._down_mask(valid, item[0].shape)

    def reset(self) -> None:
        self._sigma_app.clear()
        self._sigma_res.clear()
        self._clutter = None
        self._noise_hp = None
        for d in self._coarse.values():
            d.clear()
        self._frame = 0

    # ------------------------------------------------------------------ API
    def detect(
        self,
        gray: np.ndarray,
        history: list[tuple[np.ndarray, np.ndarray, bool]],
        valid: np.ndarray,
        frame_index: int = -1,
        t: float = 0.0,
        want_debug: bool = False,
        static_background: bool = False,
        report: bool = True,
    ) -> list[Detection]:
        """Detect points in ``gray`` (float32, processing resolution).

        ``history`` items are ``(past_gray, H_past_to_current, chain_valid)``,
        most recent first, as produced by :class:`registration.History`; item 0
        is the previous frame, so its homography is the latest registration.
        Appearance-only detections (low-clutter regions) are allowed only when
        no temporal residual is available or the background is static (black
        sky); with a moving, registered background every detection needs motion
        evidence relative to that background. With ``report=False`` only the
        detector's own frame history is updated (cheap; used while the vehicle
        mask is still being learned) and nothing is returned.
        """
        c = self.cfg
        self._frame += 1
        recompute = ((self._frame - 1) % max(1, c.noise_every) == 0) or not self._sigma_app
        polarities = {"bright": [1], "dark": [-1], "both": [1, -1]}[c.polarity]
        usable = [(history[k - 1][0], history[k - 1][1], _mask_of(history[k - 1]))
                  for k in c.history if 0 < k <= len(history) and history[k - 1][2]]
        H_prev, ok_prev = (history[0][1], history[0][2]) if history else (None, False)
        levels_now = {ds: self._down(gray, ds) for ds in self._coarse}
        valid_now = {ds: self._down_mask(valid, lv.shape) for ds, lv in levels_now.items()}
        self._advance_coarse(H_prev, ok_prev)
        if not report:
            for ds, img in levels_now.items():
                self._coarse[ds].appendleft([img, np.eye(3), True, valid_now[ds]])
            return []
        if want_debug:
            self.debug = {}
        if recompute or self._clutter is None:
            self._clutter = self._clutter_map(gray)

        dets: list[Detection] = []
        for pol in polarities:
            for si, sc in enumerate(c.scales):
                ds = sc.downsample
                if ds == 1:
                    past, img_ds, vd = self._past_levels_full(usable), gray, valid
                else:
                    past, img_ds, vd = self._past_levels_coarse(ds), levels_now[ds], valid_now[ds]
                # appearance-only evidence only where the background itself is fixed (black sky)
                dets += self._detect_scale(img_ds, past, pol, si, sc.radius, ds, vd, recompute, want_debug,
                                           allow_clean=static_background)
        for ds, img in levels_now.items():
            self._coarse[ds].appendleft([img, np.eye(3), True, valid_now[ds]])
        dets = self._merge(dets)
        dets.sort(key=lambda d: -d.score)
        dets = dets[: c.max_detections]
        dets = self._absorb(self._remeasure(gray, dets, recompute))
        for d in dets:
            d.frame_index, d.t = frame_index, t
        if want_debug:
            self.debug["clutter"] = self._clutter
        return dets

    # ---------------------------------------------------------------- helpers
    def _clutter_map(self, gray: np.ndarray) -> np.ndarray:
        """Local *median* |high-pass| (grey levels): background texture, not the objects in it.

        A mean would let a single bright object raise its own neighbourhood's clutter and
        disqualify itself from appearance-only detection on black sky.
        """
        h, w = gray.shape
        small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        hp = np.abs(small - cv2.GaussianBlur(small, (0, 0), 2.0))
        win = max(3, self.cfg.clutter_window // 4) | 1
        q = np.clip(hp * 10.0, 0, 255).astype(np.uint8)  # 0.1 grey-level resolution for the median filter
        cl = cv2.medianBlur(q, win).astype(np.float32) / 10.0
        return cv2.resize(cl, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _down(g: np.ndarray, ds: int) -> np.ndarray:
        h, w = g.shape
        return cv2.resize(g, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _down_mask(valid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Coarse valid mask: a coarse pixel is valid only if all its fine pixels are."""
        a = cv2.resize(valid.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return a > 0.999

    def _S(self, ds: int) -> np.ndarray:
        return scale_matrix(*self._sxy[ds])

    def _advance_coarse(self, H_prev: np.ndarray | None, ok: bool) -> None:
        for ds, dq in self._coarse.items():
            if H_prev is None:
                dq.clear()
                continue
            S = self._S(ds)
            Hds = S @ H_prev @ np.linalg.inv(S)
            for item in dq:
                item[1] = Hds @ item[1]
                item[2] = item[2] and ok

    def _past_levels_full(self, usable):
        return [(g, np.linalg.inv(H), m) for g, H, m in usable]

    def _past_levels_coarse(self, ds: int):
        dq = self._coarse[ds]
        out = []
        for k in self.cfg.coarse_history:
            if 0 < k <= len(dq) and dq[k - 1][2]:
                out.append((dq[k - 1][0], np.linalg.inv(dq[k - 1][1]), dq[k - 1][3]))
        return out

    def _residual_at(self, im, past_levels, xs, ys, pol):
        """Residual (current - max-filtered registered past, min over past) at integer pixels.

        Returns (residual, available). ``past_levels`` are (past_image, H_cur_to_past,
        past_valid_mask) at the same resolution as ``im``. The residual is
        unavailable where the background point was hidden (masked) in the past
        frame: otherwise the edge of the ship, sliding over the moving Earth,
        would create a band of spurious motion.
        """
        c = self.cfg
        h, w = im.shape
        cur = im[ys, xs].astype(np.float64)
        half = max(0, c.residual_dilate // 2)
        res = np.full(len(xs), np.inf)
        avail = np.ones(len(xs), bool)
        pts = np.c_[xs, ys].astype(np.float64)
        for past, Hinv, pvalid in past_levels:
            q = apply_h(Hinv, pts)
            qx, qy = np.rint(q[:, 0]).astype(int), np.rint(q[:, 1]).astype(int)
            inside = (qx >= half + 1) & (qx < w - half - 1) & (qy >= half + 1) & (qy < h - half - 1)
            avail &= inside
            if pvalid is not None:
                avail &= _neighbourhood_extreme(pvalid.view(np.uint8), qx, qy, half + 1, "min") > 0
            if pol > 0:
                ref = _neighbourhood_extreme(past, qx, qy, half, "max")
            else:
                ref = 255.0 - _neighbourhood_extreme(past, qx, qy, half, "min")
            res = np.minimum(res, cur - ref)
        res[~avail] = 0.0
        return res, avail

    # ---------------------------------------------------------------- scale
    def _detect_scale(self, img_ds, levels, pol, si, radius, ds, valid, recompute, want_debug, allow_clean=True):
        c = self.cfg
        im = img_ds if pol > 0 else (255.0 - img_ds)
        vd = valid
        if ds > 1:
            clut = cv2.resize(self._clutter, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            clut = self._clutter
        opened = cv2.morphologyEx(im, cv2.MORPH_OPEN, ellipse_kernel(radius))
        tophat = cv2.subtract(im, opened)
        nms = max(1, int(round(c.nms_radius / ds)))
        kern = np.ones((2 * nms + 1, 2 * nms + 1), np.uint8)
        is_peak = (tophat >= cv2.dilate(tophat, kern)) & (tophat > 0) & vd
        ys, xs = np.nonzero(is_peak)
        if len(ys) == 0:
            return []
        pv = tophat[ys, xs]
        all_peaks = (ys, xs)
        key = (pol, si)
        stats = self._sigma_app.get(key)
        if recompute or stats is None or stats[1].shape != tophat.shape:
            stats = peak_stats_map(pv, ys, xs, tophat.shape, max(16, c.noise_block // ds))
            self._sigma_app[key] = stats
        sa = ((pv - stats[0][ys, xs]) / stats[1][ys, xs]).astype(np.float64)
        if want_debug and si == 0:
            self.debug[f"tophat_{pol}"] = tophat
        thr0 = c.low_factor * min(c.snr_app, c.snr_clean)
        sel = sa >= thr0
        ys, xs, sa = ys[sel], xs[sel], sa[sel]
        if len(ys) == 0:
            return []
        order = np.argsort(-sa)[: 20 * c.max_detections]
        ys, xs, sa = ys[order], xs[order], sa[order]

        lc = (clut[ys, xs] < c.clutter_thresh) & allow_clean
        u = np.where(lc, sa / c.snr_clean, 0.0)
        rr = np.zeros(len(xs))
        if levels:
            res, avail = self._residual_at(im, levels, xs, ys, pol)
            skey = (pol, ds)
            st = self._sigma_res.get(skey)
            if recompute or st is None or (not np.isscalar(st[0]) and st[0].shape != im.shape):
                st = self._sigma_res[skey] = self._residual_stats(im, levels, all_peaks, pol, ds)
            rmed, rsig = st
            if not np.isscalar(rmed):
                rmed, rsig = rmed[ys, xs], rsig[ys, xs]
            rr = np.where(avail, (res - rmed) / rsig, 0.0)
            # fraction of the object's own contrast that is new at this background location: about 1
            # for anything that moved there, small for a cloud puff whose brightness merely flickers
            # (heavy video compression) or that is slightly misregistered
            frac = res / np.maximum(tophat[ys, xs].astype(np.float64), 1e-3)
            u_motion = np.minimum.reduce([sa / c.snr_app, rr / c.snr_res, frac / max(c.min_residual_frac, 1e-3)])
            u = np.maximum(u, np.where(avail, u_motion, 0.0))
        keep = u >= c.low_factor
        if not keep.any():
            return []
        ys, xs, sa, u, rr = ys[keep], xs[keep], sa[keep], u[keep], rr[keep]
        order = np.argsort(-u)[: 2 * c.max_detections]
        ys, xs, sa, u, rr = ys[order], xs[order], sa[order], u[order], rr[order]

        hw = max(radius + 1, int(round(c.centroid_halfwin / ds)))
        m = batch_centroids(tophat, xs, ys, hw)
        cx, cy, sg = m["x"], m["y"], m["sigma"]
        if ds > 1:
            sx, sy = self._sxy[ds]
            cx, cy = (cx + 0.5) / sx - 0.5, (cy + 0.5) / sy - 0.5
            # provisional size: the structuring-element scale; re-measured at full resolution later
            sg = np.full_like(sg, 0.5 * radius * ds)
        pos_sigma = np.clip(2.0 * sg / np.maximum(sa, 1.0), 0.15, 3.0 * ds)
        conf = logistic_conf(u)
        out = []
        for i in np.nonzero(m["ok"])[0]:
            out.append(
                Detection(
                    x=float(cx[i]),
                    y=float(cy[i]),
                    score=float(u[i]),
                    confidence=float(conf[i]),
                    flux=float(m["flux"][i] * ds * ds),
                    peak=float(m["peak"][i]),
                    sigma=float(sg[i]),
                    ellipticity=float(m["ellipticity"][i]),
                    area=int(m["area"][i] * ds * ds),
                    residual=float(rr[i]),
                    pos_sigma=float(pos_sigma[i]),
                    polarity=pol,
                    source=f"classical/s{si}",
                )
            )
        return out

    def _residual_stats(self, im, levels, peaks, pol, ds: int = 1, n: int = 20000):
        """Robust location/scale *maps* of the residual at (a sample of) all appearance peaks.

        Most peaks are background texture moving with the Earth, so this measures
        the residual left by misregistration, compression and noise exactly where
        candidates are evaluated. It is local (blocks of ``residual_block`` px):
        heavily compressed, high-contrast clouds leave far larger residuals than
        open ocean or black sky, and one global scale either floods the clouds
        with detections or blinds the quiet areas. Blocks with too few peaks use
        the global value.
        """
        ys, xs = peaks
        if len(ys) == 0:
            return 0.0, 1.0
        idx = self._rng.choice(len(ys), size=min(n, len(ys)), replace=False)
        res, avail = self._residual_at(im, levels, xs[idx], ys[idx], pol)
        if avail.sum() < 10:
            return 0.0, 1.0
        block = max(16, int(self.cfg.residual_block) // max(1, ds))
        return peak_stats_map(res[avail], ys[idx][avail], xs[idx][avail], im.shape, block, floor=0.5)

    # ---------------------------------------------------------------- merge
    def _merge_radius(self, d: Detection) -> float:
        """Suppression radius from the scale that found the detection (stable, unlike moments)."""
        si = int(d.source.rsplit("s", 1)[-1]) if "/s" in d.source else 0
        sc = self.cfg.scales[min(si, len(self.cfg.scales) - 1)]
        return max(float(self.cfg.nms_radius), 0.75 * (sc.radius + 1) * sc.downsample if sc.downsample > 1 else 0.0)

    def _merge(self, dets: list[Detection]) -> list[Detection]:
        """Greedy cross-scale/cross-polarity NMS: keep the strongest within its scale's extent."""
        if len(dets) < 2:
            return dets
        dets = sorted(dets, key=lambda d: -d.score)
        pts = np.array([[d.x, d.y] for d in dets])
        rad = np.array([self._merge_radius(d) for d in dets])
        tree = cKDTree(pts)
        pairs = tree.query_pairs(float(rad.max()), output_type="ndarray")
        if len(pairs) == 0:
            return dets
        i, j = pairs[:, 0], pairs[:, 1]
        dist = np.hypot(*(pts[i] - pts[j]).T)
        close = dist <= np.maximum(rad[i], rad[j])
        i, j = i[close], j[close]
        lo, hi = np.minimum(i, j), np.maximum(i, j)  # lo has the higher score (sorted order)
        order = np.argsort(lo, kind="stable")
        lo, hi = lo[order], hi[order]
        alive = np.ones(len(dets), bool)
        starts = np.searchsorted(lo, np.arange(len(dets)))
        ends = np.searchsorted(lo, np.arange(len(dets)), side="right")
        for a in range(len(dets)):
            if alive[a] and ends[a] > starts[a]:
                alive[hi[starts[a] : ends[a]]] = False
        return [d for d, ok in zip(dets, alive) if ok]

    def _absorb(self, dets: list[Detection], extent: float = 2.5, min_rel_score: float = 0.5) -> list[Detection]:
        """One detection per resolved object, after full-resolution re-measurement.

        A large object (a payload tens of pixels across) also produces fine-scale peaks on its
        edges and glints; before re-measurement they can lie outside the coarse detection's
        merge radius, and each would start its own track. Here the largest objects go first:
        any detection inside ``extent`` x sigma of one (and not much stronger than it) is
        absorbed, and the object keeps the best score of its parts.
        """
        if len(dets) < 2:
            return dets
        rad = np.array([max(self._merge_radius(d), extent * d.sigma) for d in dets])
        pts = np.array([[d.x, d.y] for d in dets])
        tree = cKDTree(pts)
        alive = np.ones(len(dets), bool)
        for a in np.argsort(-rad, kind="stable"):
            if not alive[a]:
                continue
            big = dets[a]
            for b in tree.query_ball_point(pts[a], float(rad[a])):
                if b == a or not alive[b]:
                    continue
                same = np.hypot(*(pts[a] - pts[b])) < 1.0  # converged on the same object
                if not same and (dets[b].sigma > big.sigma or big.score < min_rel_score * dets[b].score):
                    continue  # a larger object, or a much stronger compact source beside a faint diffuse one
                alive[b] = False
                if dets[b].score > big.score:
                    big.score, big.confidence = dets[b].score, max(big.confidence, dets[b].confidence)
        return [d for d, ok in zip(dets, alive) if ok]

    # ------------------------------------------------------------ re-measure
    def _remeasure(self, gray: np.ndarray, dets: list[Detection], recompute: bool) -> list[Detection]:
        """Full-resolution centroid, size and flux of every kept detection (see :func:`remeasure`)."""
        if recompute or self._noise_hp is None:  # also on frames without detections: stays current
            self._noise_hp = estimate_noise(gray, self._rng)
        return remeasure(gray, dets, self._noise_hp)
