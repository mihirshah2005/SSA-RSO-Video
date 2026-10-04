"""Self-calibration of camera pointing from the Earth itself.

Two independent constraints, both available in Earth-facing coast footage:

1. **Limb cone.** From geocentric distance ``r`` the Earth's limb is a circle
   of angular radius ``rho = asin(R_e / r)`` around nadir. Rays ``d_i``
   through limb pixels satisfy ``d_i . n = cos(rho)``. Fitting ``n`` (2 dof)
   and the focal length (1 dof) to clicked or extracted limb points gives the
   nadir direction in camera coordinates plus the field of view. (The visible
   horizon sits slightly above the solid-Earth limb because of the atmosphere;
   ``limb_height_km`` accounts for it.)
2. **Focus of expansion.** For a camera translating over the Earth, the
   background flow radiates from the image of the ground-relative velocity
   direction. Each flow vector gives a ray pair ``(a_i, b_i)`` with
   ``t . (a_i x b_i) = 0``; the smallest singular vector of the stacked
   cross products is the translation direction ``t`` (sign fixed by
   requiring expansion away from ``t``).

Nadir (``-R`` in RIC) and velocity relative to the ground (computed in RIC
from the ship ephemeris) then give the full attitude via TRIAD. None of this
uses the objects to be identified, so calibration stays independent of the
identity test.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import PinholeCamera, triad


@dataclass
class LimbFit:
    nadir_cam: np.ndarray
    fx: float
    hfov_deg: float
    rms_deg: float
    n_points: int
    cov: np.ndarray | None


def fit_limb(
    points_px: np.ndarray,
    width: int,
    height: int,
    r_obs_km: float,
    init_hfov_deg: float = 90.0,
    fix_focal: bool = False,
    limb_height_km: float = 20.0,
    k1: float = 0.0,
) -> LimbFit:
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("need at least 4 limb points")
    rho = np.arcsin((6378.137 + limb_height_km) / r_obs_km)
    cam0 = PinholeCamera.from_hfov(width, height, init_hfov_deg, k1=k1)
    # initial nadir: direction to the centre of the limb arc, pushed by rho toward the Earth side
    rays0 = cam0.unproject(pts)
    m = rays0.mean(0)
    m /= np.linalg.norm(m)

    def unpack(p):
        th, ph = p[0], p[1]
        n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        f = cam0.fx if fix_focal else np.exp(p[2])
        return n, f

    def resid(p):
        n, f = unpack(p)
        cam = PinholeCamera(width, height, f, f, cam0.cx, cam0.cy, k1)
        d = cam.unproject(pts)
        return np.degrees(np.arccos(np.clip(d @ n, -1, 1)) - rho)

    best = None
    # try several starting nadirs around the mean ray (the Earth side is unknown a priori)
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        perp = np.cross(m, [0.0, 0.0, 1.0] if abs(m[2]) < 0.9 else [1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(m, perp)
        n0 = np.cos(rho) * m + np.sin(rho) * (np.cos(ang) * perp + np.sin(ang) * perp2)
        n0 /= np.linalg.norm(n0)
        p0 = [np.arccos(np.clip(n0[2], -1, 1)), np.arctan2(n0[1], n0[0])]
        lb, ub = [-np.inf, -np.inf], [np.inf, np.inf]
        if not fix_focal:
            # bound the field of view (20-170 deg): as f -> infinity every ray collapses onto the
            # optical axis and any cone of half-angle rho through it fits perfectly (degenerate)
            f_lo = 0.5 * width / np.tan(0.5 * np.deg2rad(170.0))
            f_hi = 0.5 * width / np.tan(0.5 * np.deg2rad(20.0))
            p0.append(float(np.clip(np.log(cam0.fx), np.log(f_lo) + 1e-6, np.log(f_hi) - 1e-6)))
            lb.append(np.log(f_lo))
            ub.append(np.log(f_hi))
        sol = least_squares(resid, p0, method="trf", bounds=(lb, ub))
        if best is None or sol.cost < best.cost:
            best = sol
    n, f = unpack(best.x)
    rms = float(np.sqrt(np.mean(best.fun**2)))
    cov = None
    try:
        J = best.jac
        dof = max(1, len(pts) - len(best.x))
        cov = np.linalg.inv(J.T @ J) * (2 * best.cost / dof)
    except np.linalg.LinAlgError:
        pass
    return LimbFit(n, float(f), float(np.rad2deg(2 * np.arctan(0.5 * width / f))), rms, len(pts), cov)


def extract_limb_points(gray: np.ndarray, invalid: np.ndarray | None = None, n_max: int = 400) -> np.ndarray:
    """Boundary between bright Earth and black space (Otsu + largest contour).

    Rough by construction (atmospheric glow, clouds at the limb); review the
    points before trusting the fit. Returns (N, 2) pixel coordinates.
    """
    g = gray if gray.dtype == np.uint8 else np.clip(gray, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invalid is not None:
        b[invalid] = 0
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    h, w = g.shape
    edge = (c[:, 0] > 3) & (c[:, 0] < w - 4) & (c[:, 1] > 3) & (c[:, 1] < h - 4)
    if invalid is not None:
        dil = cv2.dilate(invalid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ci = c.astype(int)
        edge &= ~dil[ci[:, 1], ci[:, 0]]
    c = c[edge]
    if len(c) > n_max:
        c = c[np.linspace(0, len(c) - 1, n_max).astype(int)]
    return c


def fit_foe(pts_prev: np.ndarray, pts_cur: np.ndarray, cam: PinholeCamera) -> tuple[np.ndarray, float]:
    """Translation direction (camera frame, unit) from background flow; returns (t, rms_residual).

    Assumes negligible camera rotation between the two frames (the ship holds
    attitude; use frames a fraction of a second apart). Sign: background
    points move *away* from the direction of travel.
    """
    a = cam.unproject(np.asarray(pts_prev, float))
    b = cam.unproject(np.asarray(pts_cur, float))
    A = np.cross(a, b)
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1e-9
    A = A[keep] / norms[keep, None]
    if len(A) < 3:
        raise ValueError("not enough flow vectors")
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    t = Vt[-1]
    # expansion check: angle to t grows from a to b for a forward-moving camera
    ang_a = np.arccos(np.clip(a[keep] @ t, -1, 1))
    ang_b = np.arccos(np.clip(b[keep] @ t, -1, 1))
    if np.median(ang_b - ang_a) < 0:
        t = -t
    rms = float(np.sqrt(np.mean((A @ t) ** 2)))
    return t / np.linalg.norm(t), rms


def attitude_from_nadir_and_velocity(nadir_cam: np.ndarray, vel_cam: np.ndarray, vel_ric: np.ndarray) -> np.ndarray:
    """R_cam_ric from nadir (trusted) and ground-relative velocity direction (both in camera frame)."""
    return triad(nadir_cam, vel_cam, np.array([-1.0, 0.0, 0.0]), vel_ric)
