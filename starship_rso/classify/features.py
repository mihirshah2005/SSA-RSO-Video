"""Causal per-track evidence for coarse physical categories.

Every feature is computed from observations up to the current frame only.
Features describe *image* behaviour (size, brightness variation, motion
relative to the camera and to the Earth background); they support categories,
never identities.
"""

from __future__ import annotations

import numpy as np

from ..types import TrackPoint

FEATURE_NAMES = [
    "n_obs",
    "duration_s",
    "speed_px_s",
    "rel_speed_px_s",
    "bg_speed_px_s",
    "has_bg",
    "curv_rms_px",
    "accel_px_s2",
    "sigma_med",
    "sigma_max",
    "ellip_med",
    "flux_med",
    "flux_cv",
    "flicker_power",
    "flicker_hz",
    "log_sigma_rate",
    "log_flux_rate",
    "residual_med",
    "bright_frac",
    "first_x_norm",
    "first_y_norm",
    "first_sigma",
    "door_dist_norm",
    "veh_dist_norm",
    "met_first",
    "fill_frac",
]


def _detrended_log_scatter(t: np.ndarray, flux: np.ndarray) -> float:
    """Robust scatter of log-flux about a straight line: fractional brightness variation.

    A smooth trend (an object receding or approaching) is removed first, so only
    erratic variation (tumbling, glints) counts.
    """
    lf = np.log(np.maximum(flux, 1e-6))
    if len(lf) < 4 or np.ptp(t) <= 1e-6:
        med = np.median(lf)
        return float(1.4826 * np.median(np.abs(lf - med))) if len(lf) > 1 else 0.0
    res = lf - np.polyval(np.polyfit(t - t.mean(), lf, 1), t - t.mean())
    return float(1.4826 * np.median(np.abs(res - np.median(res))))


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3 or np.ptp(t) <= 1e-6:
        return 0.0
    A = np.c_[t - t.mean(), np.ones_like(t)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def _robust_slope(t: np.ndarray, y: np.ndarray, max_n: int = 60) -> float:
    """Theil-Sen slope (median of pairwise slopes): a few blended or mis-measured frames
    (an object passing a vehicle edge or a particle) cannot fake a trend."""
    if len(t) < 3 or np.ptp(t) <= 1e-6:
        return 0.0
    if len(t) > max_n:
        idx = np.linspace(0, len(t) - 1, max_n).round().astype(int)
        t, y = t[idx], y[idx]
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 1e-6
    return float(np.median((y[j][ok] - y[i][ok]) / dt[ok])) if ok.any() else 0.0


def _flicker(t: np.ndarray, flux: np.ndarray, f_min: float = 0.5) -> tuple[float, float]:
    """Fraction of detrended log-flux variance in the strongest periodogram peak, and its frequency.

    Only frequencies >= max(f_min, 2 / duration) count, so slow trends are not mistaken for tumbling.
    """
    if len(t) < 16 or np.ptp(t) <= 0.3:
        return 0.0, 0.0
    tt = np.linspace(t[0], t[-1], len(t))
    f = np.interp(tt, t, np.log(np.maximum(flux, 1e-6)))
    f = f - np.polyval(np.polyfit(tt - tt.mean(), f, 2), tt - tt.mean())
    if np.var(f) <= 1e-12:
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(f * np.hanning(len(f)))) ** 2
    freqs = np.fft.rfftfreq(len(f), d=(tt[1] - tt[0]))
    total = max(spec[1:].sum(), 1e-12)
    band = freqs >= max(f_min, 2.0 / np.ptp(tt))
    if not band.any():
        return 0.0, 0.0
    k = int(np.argmax(np.where(band, spec, -1.0)))
    return float(spec[k] / total), float(freqs[k])


def track_features(
    points: list[TrackPoint],
    width: int,
    height: int,
    door_xy: list[float] | None = None,
    window: int = 90,
    first: TrackPoint | None = None,
    n_obs: int | None = None,
    vehicle_dist: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute features from the most recent ``window`` observed points of a track.

    ``first`` / ``n_obs`` come from the track itself so that first-sighting
    features stay correct after the stored history has been trimmed.
    """
    obs = [p for p in points if p.detection is not None]
    f = {k: 0.0 for k in FEATURE_NAMES}
    if not obs:
        return f
    first = first if first is not None else obs[0]
    rec = obs[-window:]
    t = np.array([p.t for p in rec])
    x = np.array([p.detection.x for p in rec])
    y = np.array([p.detection.y for p in rec])
    sig = np.array([p.detection.sigma for p in rec])
    flux = np.array([max(p.detection.flux, 1e-6) for p in rec])
    f["n_obs"] = float(n_obs if n_obs is not None else len(obs))
    f["duration_s"] = float(obs[-1].t - first.t)
    if points[0].seq >= 0:  # processed-frame counter: independent of dropped source frames
        span = points[-1].seq - points[0].seq + 1
    else:
        span = points[-1].frame_index - points[0].frame_index + 1
    f["fill_frac"] = float(len(obs) / max(span, 1))
    if len(rec) >= 2 and np.ptp(t) > 1e-6:
        vx, vy = _slope(t, x), _slope(t, y)
    else:
        vx, vy = rec[-1].vx, rec[-1].vy
    f["speed_px_s"] = float(np.hypot(vx, vy))
    flows = [p.bg_flow for p in rec if p.bg_flow is not None]
    if len(flows) >= max(2, len(rec) // 3):
        bx, by = np.median(np.array(flows), axis=0)
        f["has_bg"] = 1.0
        f["bg_speed_px_s"] = float(np.hypot(bx, by))
        f["rel_speed_px_s"] = float(np.hypot(vx - bx, vy - by))
    else:
        f["rel_speed_px_s"] = f["speed_px_s"]  # no background reference: camera-relative
    if len(rec) >= 6 and np.ptp(t) > 1e-6:
        tc = t - t.mean()
        res_lin = np.r_[x - np.polyval(np.polyfit(tc, x, 1), tc), y - np.polyval(np.polyfit(tc, y, 1), tc)]
        px, py = np.polyfit(tc, x, 2), np.polyfit(tc, y, 2)
        res_q = np.r_[x - np.polyval(px, tc), y - np.polyval(py, tc)]
        f["curv_rms_px"] = float(np.sqrt(max(np.mean(res_lin**2) - np.mean(res_q**2), 0.0)))
        f["accel_px_s2"] = float(2.0 * np.hypot(px[0], py[0]))
    f["sigma_med"] = float(np.median(sig))
    f["sigma_max"] = float(np.max(sig))
    f["ellip_med"] = float(np.median([p.detection.ellipticity for p in rec]))
    f["flux_med"] = float(np.median(flux))
    f["flux_cv"] = _detrended_log_scatter(t, flux)
    f["flicker_power"], f["flicker_hz"] = _flicker(t, flux)
    f["log_sigma_rate"] = _robust_slope(t, np.log(np.maximum(sig, 0.3)))
    f["log_flux_rate"] = _slope(t, np.log(flux))
    f["residual_med"] = float(np.median([p.detection.residual for p in rec]))
    f["bright_frac"] = float(np.mean([p.detection.polarity > 0 for p in rec]))
    f["first_x_norm"] = float(first.detection.x / max(width, 1))
    f["first_y_norm"] = float(first.detection.y / max(height, 1))
    f["first_sigma"] = float(first.detection.sigma)
    if door_xy is not None:
        f["door_dist_norm"] = float(np.hypot(f["first_x_norm"] - door_xy[0], f["first_y_norm"] - door_xy[1]))
    else:
        f["door_dist_norm"] = -1.0
    f["met_first"] = float(first.met) if first.met is not None else float("nan")
    # distance (normalised by the image width) from the ship's masked structure; 1.0 = unknown/far
    f["veh_dist_norm"] = 1.0
    if vehicle_dist is not None:
        hh, ww = vehicle_dist.shape
        xs = np.clip(np.rint(x[-15:] * ww / max(width, 1)).astype(int), 0, ww - 1)
        ys = np.clip(np.rint(y[-15:] * hh / max(height, 1)).astype(int), 0, hh - 1)
        f["veh_dist_norm"] = float(np.median(vehicle_dist[ys, xs]) / max(ww, 1))
    return f


def feature_vector(f: dict[str, float]) -> np.ndarray:
    return np.array([f.get(k, 0.0) for k in FEATURE_NAMES], dtype=float)
