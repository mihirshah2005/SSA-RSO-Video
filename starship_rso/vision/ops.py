"""Small image operations shared by the detectors."""

from __future__ import annotations

import cv2
import numpy as np


def to_gray_f32(frame: np.ndarray) -> np.ndarray:
    """BGR/grey uint8 -> float32 grey in [0, 255]."""
    if frame.ndim == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame
    return g.astype(np.float32, copy=False)


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return frame
    h, w = frame.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, size, interpolation=interp)


def ellipse_kernel(radius: int) -> np.ndarray:
    d = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def block_robust_stats(
    img: np.ndarray, valid: np.ndarray | None, block: int, floor: float = 0.25, subsample: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Per-block robust location and scale (median, 1.4826 * MAD), bilinearly upsampled.

    Only ``valid`` pixels contribute; blocks without enough valid pixels
    inherit the global estimate. ``subsample`` strides inside blocks for speed.
    """
    h, w = img.shape
    b = max(8, int(block))
    nby, nbx = max(1, int(np.ceil(h / b))), max(1, int(np.ceil(w / b)))
    a = np.full((nby * b, nbx * b), np.nan, dtype=np.float32)
    a[:h, :w] = img
    if valid is not None:
        a[:h, :w][~valid] = np.nan
    s = max(1, int(subsample))
    blocks = a.reshape(nby, b, nbx, b)[:, ::s, :, ::s].transpose(0, 2, 1, 3).reshape(nby, nbx, -1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(blocks, axis=2)
        mad = np.nanmedian(np.abs(blocks - med[..., None]), axis=2)
        count = np.sum(np.isfinite(blocks), axis=2)
        vals = a[:h, :w][::s, ::s]
        gmed = np.nanmedian(vals)
        gmad = np.nanmedian(np.abs(vals - gmed))
    gmed = float(gmed) if np.isfinite(gmed) else 0.0
    gsig = 1.4826 * float(gmad) if np.isfinite(gmad) else 1.0
    sig = 1.4826 * mad
    bad = (count < max(8, blocks.shape[2] // 8)) | ~np.isfinite(sig) | ~np.isfinite(med)
    sig[bad] = gsig
    med[bad] = gmed
    sig = np.maximum(sig, floor).astype(np.float32)
    med = med.astype(np.float32)

    def up(m):
        if m.shape == (1, 1):
            return np.full((h, w), float(m[0, 0]), dtype=np.float32)
        return cv2.resize(m, (nbx * b, nby * b), interpolation=cv2.INTER_LINEAR)[:h, :w]

    return up(med), up(sig)


def peak_stats_map(
    values: np.ndarray, ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int], block: int,
    min_count: int = 12, floor: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust location/scale of *peak* values per block (median, 1.4826 * MAD), upsampled.

    Thresholding local maxima against the distribution of local maxima (not of
    all pixels) calibrates the false-alarm rate on exactly what is thresholded:
    the top-hat of pure noise is positively biased and its maxima sit far above
    the pixel-level noise scale.
    """
    h, w = shape
    b = max(8, int(block))
    nby, nbx = max(1, int(np.ceil(h / b))), max(1, int(np.ceil(w / b)))
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros(shape, np.float32), np.full(shape, floor, np.float32)
    gmed = float(np.median(v))
    gsig = max(1.4826 * float(np.median(np.abs(v - gmed))), floor)
    med = np.full((nby, nbx), gmed, np.float32)
    sig = np.full((nby, nbx), gsig, np.float32)
    bid = (ys // b) * nbx + (xs // b)
    nb = nby * nbx
    # vectorised per-block medians: sort by (block, value) and index the middle of each segment
    order = np.lexsort((v, bid))
    bid_s, v_s = bid[order], v[order]
    starts = np.searchsorted(bid_s, np.arange(nb))
    ends = np.searchsorted(bid_s, np.arange(nb), side="right")
    cnt = ends - starts
    ok = cnt >= min_count
    mid = np.clip((starts + ends - 1) // 2, 0, len(v_s) - 1)
    bmed = np.where(ok, v_s[mid], gmed)
    dev = np.abs(v_s - bmed[bid_s])
    order2 = np.lexsort((dev, bid_s))
    dev_s = dev[order2]
    bmad = np.where(ok, dev_s[mid], gsig / 1.4826)
    med.flat[:] = bmed.astype(np.float32)
    sig.flat[:] = np.maximum(1.4826 * bmad, floor).astype(np.float32)
    if (nby, nbx) == (1, 1):
        return np.full(shape, med[0, 0], np.float32), np.full(shape, sig[0, 0], np.float32)
    # smooth block maps (3x3) so single noisy blocks do not create seams
    med = cv2.blur(med, (3, 3), borderType=cv2.BORDER_REPLICATE)
    sig = cv2.blur(sig, (3, 3), borderType=cv2.BORDER_REPLICATE)
    up = lambda m: cv2.resize(m, (nbx * b, nby * b), interpolation=cv2.INTER_LINEAR)[:h, :w]  # noqa: E731
    return up(med), up(sig)


def block_robust_sigma(img, valid, block, floor: float = 0.25, subsample: int = 2) -> np.ndarray:
    return block_robust_stats(img, valid, block, floor, subsample)[1]


def robust_sigma(values: np.ndarray, floor: float = 0.25) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return floor
    med = np.median(v)
    return float(max(1.4826 * np.median(np.abs(v - med)), floor))


def local_std(img: np.ndarray, window: int) -> np.ndarray:
    k = (int(window), int(window))
    m = cv2.blur(img, k)
    m2 = cv2.blur(img * img, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def local_maxima(score: np.ndarray, radius: int, threshold: float, valid: np.ndarray | None = None):
    """Return (ys, xs) of local maxima of ``score`` >= threshold within a (2r+1)^2 window."""
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dil = cv2.dilate(score, k)
    m = (score >= dil) & (score >= threshold)
    if valid is not None:
        m &= valid
    ys, xs = np.nonzero(m)
    return ys, xs


def weighted_centroid(img: np.ndarray, x: int, y: int, halfwin: int) -> dict | None:
    """Intensity-weighted centroid and second moments of positive ``img`` around (x, y)."""
    h, w = img.shape
    x0, x1 = max(0, x - halfwin), min(w, x + halfwin + 1)
    y0, y1 = max(0, y - halfwin), min(h, y + halfwin + 1)
    patch = img[y0:y1, x0:x1].astype(np.float64)
    wts = np.clip(patch, 0, None)
    tot = wts.sum()
    if tot <= 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((wts * xx).sum() / tot)
    cy = float((wts * yy).sum() / tot)
    dx, dy = xx - cx, yy - cy
    ixx = float((wts * dx * dx).sum() / tot)
    iyy = float((wts * dy * dy).sum() / tot)
    ixy = float((wts * dx * dy).sum() / tot)
    tr = ixx + iyy
    sigma = float(np.sqrt(max(tr / 2.0, 1e-6)))
    ell = float(np.sqrt((ixx - iyy) ** 2 + 4 * ixy * ixy) / tr) if tr > 1e-9 else 0.0
    return {
        "x": cx,
        "y": cy,
        "flux": float(tot),
        "peak": float(patch.max()),
        "sigma": sigma,
        "ellipticity": ell,
        "area": int((wts > 0.25 * wts.max()).sum()),
    }


def warp_to(img: np.ndarray, H: np.ndarray, shape: tuple[int, int], border_value: float = 0.0) -> np.ndarray:
    """Warp ``img`` with homography ``H`` (source -> destination) into ``shape`` (h, w)."""
    h, w = shape
    if np.allclose(H, np.eye(3)):
        return img
    return cv2.warpPerspective(
        img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )


def warp_valid(H: np.ndarray, shape: tuple[int, int], margin: int = 2) -> np.ndarray:
    """Boolean mask of destination pixels that receive data from the source frame."""
    h, w = shape
    if np.allclose(H, np.eye(3)):
        return np.ones((h, w), bool)
    ones = np.ones((h, w), np.uint8)
    m = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if margin > 0:
        m = cv2.erode(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return m.astype(bool)


def scale_homography(H_work: np.ndarray, s: float) -> np.ndarray:
    """Convert a homography estimated on an image resized by ``s`` back to full resolution.

    Resize maps x_work = s*x + (s - 1)/2 (pixel-centre convention)."""
    c = 0.5 * s - 0.5
    S = np.array([[s, 0, c], [0, s, c], [0, 0, 1.0]])
    return np.linalg.inv(S) @ H_work @ S


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Pixel-centre resize map: x' = sx * x + (sx - 1) / 2 (and likewise for y)."""
    return np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])


def scale_homography_xy(H_work: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Homography estimated on an image resized by (sx, sy) -> full-resolution homography."""
    S = scale_matrix(sx, sy)
    return np.linalg.inv(S) @ H_work @ S


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.c_[pts, np.ones(len(pts))] @ H.T
    return hom[:, :2] / hom[:, 2:3]
