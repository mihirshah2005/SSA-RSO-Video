"""Reference frames: TEME (quasi-inertial, SGP4 output), Earth-fixed, geodetic, RIC.

All relative geometry (ship -> object) is formed in TEME at a common instant,
so frame-tie errors cancel to first order. TEME <-> Earth-fixed uses GMST
only (no polar motion / equation of the equinoxes): errors of tens of metres
at the surface, negligible here.

RIC (a.k.a. RSW / LVLH-like) at the ship: R = radial (up), I = in-track,
C = cross-track (orbit normal). A camera attitude is stored relative to RIC
because the ship holds an attitude relative to its orbit during coast.
"""

from __future__ import annotations

import numpy as np

from .timeutil import gmst_rad

WGS84_A = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
OMEGA_EARTH = 7.2921150e-5  # rad/s
R_EARTH = 6378.137


def rot3(theta) -> np.ndarray:
    """Rotation matrices about z by ``theta`` (vectorised): frame rotation R3(theta)."""
    th = np.asarray(theta, dtype=np.float64)
    c, s = np.cos(th), np.sin(th)
    z, o = np.zeros_like(th), np.ones_like(th)
    return np.stack([np.stack([c, s, z], -1), np.stack([-s, c, z], -1), np.stack([z, z, o], -1)], -2)


def teme_to_ecef(r_teme: np.ndarray, utc) -> np.ndarray:
    """Position TEME -> Earth-fixed (pseudo-Earth-fixed, GMST rotation only)."""
    R = rot3(gmst_rad(utc))
    return np.einsum("...ij,...j->...i", R, r_teme)


def ecef_to_teme(r_ecef: np.ndarray, utc) -> np.ndarray:
    R = rot3(gmst_rad(utc))
    return np.einsum("...ji,...j->...i", R, r_ecef)


def teme_velocity_relative_to_ground(r_teme: np.ndarray, v_teme: np.ndarray) -> np.ndarray:
    """Velocity relative to the rotating Earth, expressed in TEME axes."""
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    return v_teme - np.cross(w, r_teme)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + alt_km) * np.cos(lat) * np.cos(lon),
            (n + alt_km) * np.cos(lat) * np.sin(lon),
            (n * (1 - WGS84_E2) + alt_km) * np.sin(lat),
        ]
    )


def ecef_to_geodetic(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Earth-fixed -> (lat_deg, lon_deg, alt_km), Bowring iteration (vectorised)."""
    r = np.asarray(r, dtype=np.float64)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.maximum(np.cos(lat), 1e-12) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.maximum(np.cos(lat), 1e-12) - n
    return np.rad2deg(lat), np.rad2deg(lon), alt


def ric_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows are the R, I, C unit vectors: ``v_ric = M @ v_teme`` (vectorised over leading dims)."""
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rh = r / np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)
    ch = h / np.linalg.norm(h, axis=-1, keepdims=True)
    ih = np.cross(ch, rh)
    return np.stack([rh, ih, ch], axis=-2)


def to_ric(vec_teme: np.ndarray, r_ship: np.ndarray, v_ship: np.ndarray) -> np.ndarray:
    M = ric_matrix(r_ship, v_ship)
    return np.einsum("...ij,...j->...i", M, vec_teme)


def earth_angular_radius(r_obs_norm: float, r_earth: float = R_EARTH) -> float:
    """Half-angle (rad) of the Earth disc seen from geocentric distance ``r_obs_norm``."""
    return float(np.arcsin(min(1.0, r_earth / r_obs_norm)))


def line_of_sight_clear(r_obs: np.ndarray, r_tgt: np.ndarray, r_block: float = R_EARTH + 80.0) -> np.ndarray:
    """True where the segment observer->target does not pass through a sphere of radius ``r_block``."""
    d = r_tgt - r_obs
    dd = np.einsum("...i,...i->...", d, d)
    s = np.clip(-np.einsum("...i,...i->...", r_obs, d) / np.maximum(dd, 1e-12), 0.0, 1.0)
    closest = r_obs + s[..., None] * d
    return np.linalg.norm(closest, axis=-1) > r_block
