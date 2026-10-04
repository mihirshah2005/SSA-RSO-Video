"""Two-body orbits with J2 secular drift (used for the nominal ship ephemeris and tests).

Never use these to propagate GP (SGP4 mean) elements: GP elements are not
osculating two-body elements. Use :mod:`orbit.propagate` for catalogue data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU = 398600.4418  # km^3/s^2
RE = 6378.137  # km
J2 = 1.08262668e-3


def solve_kepler(M, e, tol: float = 1e-12, maxiter: int = 50):
    M = np.mod(np.asarray(M, dtype=np.float64), 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        dE = -f / (1 - e * np.cos(E))
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def elements_to_state(a, e, i, raan, argp, M):
    """Classical elements (km, -, rad) -> inertial position/velocity (km, km/s). Vectorised over M."""
    M, raan, argp = np.broadcast_arrays(np.asarray(M, float), np.asarray(raan, float), np.asarray(argp, float))
    E = solve_kepler(M, e)
    cosE, sinE = np.cos(E), np.sin(E)
    r_pf = np.stack([a * (cosE - e), a * np.sqrt(1 - e * e) * sinE, np.zeros_like(E)], -1)
    rn = a * (1 - e * cosE)
    v_pf = np.stack([-np.sqrt(MU * a) / rn * sinE, np.sqrt(MU * a * (1 - e * e)) / rn * cosE, np.zeros_like(E)], -1)
    R = _pqw_to_inertial(raan, i, argp)
    return np.einsum("...ij,...j->...i", R, r_pf), np.einsum("...ij,...j->...i", R, v_pf)


def _pqw_to_inertial(raan, i, argp) -> np.ndarray:
    """Perifocal -> inertial rotation(s); broadcasts over array inputs, shape (..., 3, 3)."""
    raan, i, argp = np.broadcast_arrays(np.asarray(raan, float), np.asarray(i, float), np.asarray(argp, float))
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    rows = [
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ]
    return np.stack([np.stack(r, -1) for r in rows], -2)


def state_to_elements(r: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Osculating classical elements from a single state vector."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(n)
    evec = ((vn**2 - MU / rn) * r - np.dot(r, v) * v) / MU
    e = float(np.linalg.norm(evec))
    a = 1.0 / (2.0 / rn - vn**2 / MU)
    i = float(np.arccos(np.clip(h[2] / hn, -1, 1)))
    raan = float(np.arctan2(n[1], n[0])) if nn > 1e-12 else 0.0
    if e > 1e-9 and nn > 1e-12:
        argp = float(np.arctan2(np.dot(np.cross(n, evec), h) / hn, np.dot(n, evec)))
        nu = float(np.arctan2(np.dot(np.cross(evec, r), h) / hn, np.dot(evec, r)))
    else:  # circular: measure from the node
        argp = 0.0
        ref = n / nn if nn > 1e-12 else np.array([1.0, 0, 0])
        nu = float(np.arctan2(np.dot(np.cross(ref, r), h) / hn, np.dot(ref, r)))
    E = 2 * np.arctan(np.sqrt(max(1 - e, 0) / (1 + e)) * np.tan(nu / 2))
    M = float(np.mod(E - e * np.sin(E), 2 * np.pi))
    return {"a": float(a), "e": e, "i": i, "raan": np.mod(raan, 2 * np.pi), "argp": np.mod(argp, 2 * np.pi), "M": M}


def j2_rates(a: float, e: float, i: float) -> tuple[float, float, float]:
    """Secular rates (rad/s) of RAAN, argument of perigee and mean anomaly under J2."""
    n = np.sqrt(MU / a**3)
    p = a * (1 - e * e)
    k = 1.5 * J2 * (RE / p) ** 2 * n
    raan_dot = -k * np.cos(i)
    argp_dot = 0.5 * k * (5 * np.cos(i) ** 2 - 1)
    M_dot = n + 0.5 * k * np.sqrt(1 - e * e) * (3 * np.cos(i) ** 2 - 1)
    return float(raan_dot), float(argp_dot), float(M_dot)


@dataclass
class KeplerOrbit:
    epoch_utc: float
    a: float
    e: float
    i: float
    raan: float
    argp: float
    M0: float
    use_j2: bool = True

    def state(self, utc) -> tuple[np.ndarray, np.ndarray]:
        dt = np.asarray(utc, dtype=np.float64) - self.epoch_utc
        if self.use_j2:
            rd, wd, md = j2_rates(self.a, self.e, self.i)
        else:
            rd, wd, md = 0.0, 0.0, float(np.sqrt(MU / self.a**3))
        raan = self.raan + rd * dt
        argp = self.argp + wd * dt
        M = self.M0 + md * dt
        return elements_to_state(self.a, self.e, self.i, raan, argp, M)

    @property
    def period_s(self) -> float:
        return float(2 * np.pi * np.sqrt(self.a**3 / MU))
