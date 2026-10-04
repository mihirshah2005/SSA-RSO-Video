"""Rendering primitives for synthetic frames: backgrounds, PSFs, defocused discs, compression."""

from __future__ import annotations

import cv2
import numpy as np


def fractal_noise(h: int, w: int, rng: np.random.Generator, beta: float = 3.0, cutoff: float = 0.0) -> np.ndarray:
    """Periodic (tileable) 1/f^(beta/2)-amplitude noise in [0, 1], made by FFT filtering white noise."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    if cutoff > 0:
        amp *= np.exp(-((f / cutoff) ** 2))
    amp[0, 0] = 0.0
    spec = np.fft.rfft2(rng.standard_normal((h, w))) * amp
    out = np.fft.irfft2(spec, s=(h, w)).astype(np.float32)
    out -= out.min()
    out /= max(float(out.max()), 1e-6)
    return out


def earth_texture(h: int, w: int, rng: np.random.Generator, cloud_cover: float = 0.45) -> np.ndarray:
    """BGR uint8 ocean-and-cumulus texture (tileable enough for scrolling with wrap)."""
    n = fractal_noise(h, w, rng, beta=3.2)
    fine = fractal_noise(h, w, rng, beta=2.2)
    clouds = np.clip((n * 0.6 + fine * 0.4 - (1 - cloud_cover)) * 4.0, 0, 1)
    clouds = cv2.GaussianBlur(clouds, (0, 0), 0.8)
    ocean = np.dstack([np.full((h, w), 150, np.float32), np.full((h, w), 70, np.float32), np.full((h, w), 25, np.float32)])
    ocean *= (0.85 + 0.3 * fractal_noise(h, w, rng, beta=4.0))[..., None]
    white = np.dstack([np.full((h, w), 245, np.float32)] * 3)
    img = ocean * (1 - clouds[..., None]) + white * clouds[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def vehicle_layer(h: int, w: int, polygon_norm: list[list[float]], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Static ship body (BGR) and its boolean mask: steel gradient + dark tile pattern."""
    mask = np.zeros((h, w), np.uint8)
    pts = (np.array(polygon_norm) * [w, h]).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    steel = 120 + 80 * np.sin(xx / w * 3.0 + yy / h * 1.5)
    img = np.dstack([steel] * 3)
    # hexagonal-ish tile grid on the lower part
    step = max(6, w // 40)
    tiles = ((xx % step) < 1.2) | ((yy % step) < 1.2)
    dark = (yy / h + xx / w) > 1.0
    img[dark] = 18
    img[dark & tiles] = 60
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), mask.astype(bool)


def add_gaussian_spot(img: np.ndarray, x: float, y: float, sigma: float, amp: float) -> None:
    h, w = img.shape[:2]
    r = int(np.ceil(4 * sigma)) + 1
    x0, x1 = int(np.floor(x)) - r, int(np.floor(x)) + r + 1
    y0, y1 = int(np.floor(y)) - r, int(np.floor(y)) + r + 1
    xa, xb, ya, yb = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if xa >= xb or ya >= yb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    g = amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    img[ya:yb, xa:xb] += g[..., None] if img.ndim == 3 else g


def add_disc(img: np.ndarray, x: float, y: float, radius: float, amp: float, ring: float = 0.35) -> None:
    """Defocused point source: soft-edged disc, slightly brighter rim (out-of-focus bokeh)."""
    h, w = img.shape[:2]
    r = int(np.ceil(radius + 2))
    xa, xb = max(0, int(x) - r), min(w, int(x) + r + 2)
    ya, yb = max(0, int(y) - r), min(h, int(y) + r + 2)
    if xa >= xb or ya >= yb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    edge = np.clip(radius + 0.5 - d, 0, 1)
    rim = 1 + ring * np.exp(-((d - radius * 0.85) ** 2) / (2 * max(radius * 0.12, 0.5) ** 2))
    g = amp * edge * rim
    img[ya:yb, xa:xb] += g[..., None] if img.ndim == 3 else g


def add_box(img: np.ndarray, x: float, y: float, half_w: float, half_h: float, amp: float, angle_deg: float = 0.0) -> None:
    """Resolved object (e.g. a payload a few metres away): filled rotated rectangle."""
    rect = ((float(x), float(y)), (max(1.0, 2 * half_w), max(1.0, 2 * half_h)), float(angle_deg))
    box = cv2.boxPoints(rect).astype(np.float32)
    layer = np.zeros(img.shape[:2], np.float32)
    cv2.fillConvexPoly(layer, np.round(box * 16).astype(np.int32), 1.0, lineType=cv2.LINE_AA, shift=4)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    img += (amp * layer)[..., None] if img.ndim == 3 else amp * layer


def compress(img_u8: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return img_u8
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img_u8


def mag_to_amp(mag: float, mag_ref: float = 0.0, amp_ref: float = 120.0, amp_max: float = 255.0) -> float:
    return float(min(amp_max, amp_ref * 10 ** (-0.4 * (mag - mag_ref))))
