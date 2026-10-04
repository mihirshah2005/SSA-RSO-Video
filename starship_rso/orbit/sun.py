"""Low-precision Sun direction and Earth-shadow tests.

Solar position from the Astronomical Almanac low-precision formulae (about
0.01 deg over 1950-2050), expressed in the mean equator/equinox of date,
which agrees with TEME to well within the needs of illumination checks.
"""

from __future__ import annotations

import numpy as np

from .frames import R_EARTH
from .timeutil import JD_J2000, utc_to_jd

AU_KM = 149597870.7


def sun_position_teme(utc) -> np.ndarray:
    jd, fr = utc_to_jd(utc)
    n = (jd - JD_J2000) + fr
    L = np.deg2rad(np.mod(280.460 + 0.9856474 * n, 360.0))
    g = np.deg2rad(np.mod(357.528 + 0.9856003 * n, 360.0))
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)
    r = (1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)) * AU_KM
    return np.stack([r * np.cos(lam), r * np.cos(eps) * np.sin(lam), r * np.sin(eps) * np.sin(lam)], -1)


def is_sunlit(r_obj: np.ndarray, r_sun: np.ndarray) -> np.ndarray:
    """Cylindrical Earth-shadow model (adequate for LEO illumination flags)."""
    s = r_sun / np.linalg.norm(r_sun, axis=-1, keepdims=True)
    proj = np.einsum("...i,...i->...", r_obj, s)
    perp = np.linalg.norm(r_obj - proj[..., None] * s, axis=-1)
    return ~((proj < 0) & (perp < R_EARTH))


def phase_angle(r_obs: np.ndarray, r_obj: np.ndarray, r_sun: np.ndarray) -> np.ndarray:
    """Sun-object-observer angle (rad): 0 = fully lit as seen by the observer."""
    a = r_sun - r_obj
    b = r_obs - r_obj
    cosang = np.einsum("...i,...i->...", a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.arccos(np.clip(cosang, -1.0, 1.0))
