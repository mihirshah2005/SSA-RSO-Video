"""Ship orbit fitted to the broadcast's own telemetry (HUD altitude and speed).

SpaceX publishes no Starship state vector, but the webcast overlay shows
altitude (1 km steps) and ground-relative speed (1 km/h steps) every frame.
Over half an hour of coast these pin down the in-plane orbit:

* altitude and speed versus time give the size, shape and perigee timing;
* two latitude-dependent effects give the along-track *phase*: altitude above
  the WGS84 ellipsoid differs from geocentric radius by up to 21 km x sin^2(lat),
  and the ground-relative speed carries the Earth's rotation, |w x r|, which
  shrinks with cos(lat). Both change by several units of the display resolution
  between the equator and 30 deg latitude.

The orbital *plane* comes from the deployed payload group (released from the
ship, in the same plane; in-plane orbit raising leaves the plane alone), or
from the launch site and inclination when no group exists.

The motion model integrates two-body + J2 numerically in TEME, so short-period
J2 terms (kilometres in radius) are not mistaken for orbit shape. Drag over
half an hour at 270 km is metres and is ignored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .frames import OMEGA_EARTH, ecef_to_geodetic, teme_to_ecef
from .kepler import J2, MU, elements_to_state
from .ship import ShipEphemeris

R_J2 = 6378.137


def _accel(r: np.ndarray) -> np.ndarray:
    x, y, z = r
    rn2 = x * x + y * y + z * z
    rn = np.sqrt(rn2)
    k = 1.5 * J2 * MU * R_J2**2 / rn**5
    zz = 5.0 * z * z / rn2
    return -MU * r / rn**3 + k * np.array([x * (zz - 1.0), y * (zz - 1.0), z * (zz - 3.0)])


def propagate_j2(r0: np.ndarray, v0: np.ndarray, t0: float, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-body + J2 states (TEME, km, km/s) at ``times`` (POSIX s), integrated from (r0, v0) at t0."""
    from scipy.integrate import solve_ivp

    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    out_r = np.empty((times.size, 3))
    out_v = np.empty((times.size, 3))

    def f(_t, y):
        return np.concatenate([y[3:], _accel(y[:3])])

    y0 = np.concatenate([r0, v0])
    for sign in (1.0, -1.0):
        sel = (times - t0) * sign >= 0.0
        if not sel.any():
            continue
        dt = times[sel] - t0
        span = (0.0, float(dt.max() if sign > 0 else dt.min()))
        if span[1] == 0.0:
            out_r[sel], out_v[sel] = r0, v0
            continue
        sol = solve_ivp(f, span, y0, method="DOP853", rtol=1e-11, atol=1e-9, dense_output=True)
        y = sol.sol(dt)
        out_r[sel], out_v[sel] = y[:3].T, y[3:].T
    return out_r, out_v


@dataclass
class HudSeries:
    met: np.ndarray
    alt_km: np.ndarray  # NaN where unread
    speed_kmh: np.ndarray  # NaN where unread


def load_hud_series(csv_path: str | Path, timemap, every_s: float = 1.0) -> HudSeries:
    """HUD readings mapped to MET, cleaned of OCR misreads, one sample per ``every_s``.

    Misreads (a dropped leading digit, a misread digit) are removed against a running median:
    the true values change by well under 1 km and 1 km/h per second.
    """
    import csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            met, _ = timemap.met(float(r["video_s"]))
            if met is None:
                continue

            def num(key):
                v = r.get(key, "")
                return float(v) if v not in ("", None) else np.nan

            rows.append((met, num("alt_km"), num("speed_kmh")))
    a = np.array(rows, dtype=np.float64)
    a = a[np.argsort(a[:, 0])]
    met, alt, spd = a[:, 0], a[:, 1], a[:, 2]

    def clean(x, tol, win=41):
        good = np.isfinite(x)
        med = np.full_like(x, np.nan)
        idx = np.nonzero(good)[0]
        for k, i in enumerate(idx):
            lo, hi = max(0, k - win // 2), min(len(idx), k + win // 2 + 1)
            med[i] = np.median(x[idx[lo:hi]])
        return np.where(good & (np.abs(x - med) <= tol), x, np.nan)

    # physically plausible ranges first (a dropped leading digit turns 271 into 71 or 1), then a
    # running median over ~30 s of readings: misreads can persist for several seconds
    alt = np.where((alt > 120.0) & (alt < 2000.0), alt, np.nan)
    spd = np.where((spd > 20000.0) & (spd < 30000.0), spd, np.nan)
    alt = clean(alt, 2.0, win=151)
    spd = clean(spd, 5.0, win=151)
    # one sample per bin: the median of the clean readings inside it
    edges = np.arange(met.min(), met.max() + every_s, every_s)
    b = np.digitize(met, edges)
    out_t, out_a, out_s = [], [], []
    for k in np.unique(b):
        m = b == k
        with np.errstate(all="ignore"):
            out_t.append(float(np.mean(met[m])))
            out_a.append(float(np.nanmedian(alt[m])) if np.isfinite(alt[m]).any() else np.nan)
            out_s.append(float(np.nanmedian(spd[m])) if np.isfinite(spd[m]).any() else np.nan)
    return HudSeries(np.array(out_t), np.array(out_a), np.array(out_s))


def plane_from_group(records, utc: float) -> tuple[float, float, int]:
    """Inclination and RAAN (rad, TEME at ``utc``) of the payload group: median orbit normal."""
    from .propagate import Propagator

    r, v, ok = Propagator(records).states(np.array([utc]))
    h = np.cross(r[:, 0], v[:, 0])
    good = ok[:, 0] & np.all(np.isfinite(h), axis=1)
    if not good.any():
        raise ValueError("no group object propagates to the requested time")
    hn = h[good] / np.linalg.norm(h[good], axis=1, keepdims=True)
    n = np.median(hn, axis=0)
    n /= np.linalg.norm(n)
    inc = float(np.arccos(np.clip(n[2], -1, 1)))
    raan = float(np.arctan2(n[0], -n[1]))
    return inc, raan, int(good.sum())


def plane_from_site(liftoff_utc: float, lat_deg: float, lon_deg: float, inc_deg: float, launch_pass: str):
    """Plane containing the launch site at liftoff with the given inclination (rad, TEME)."""
    from .frames import ecef_to_teme, geodetic_to_ecef

    i = np.deg2rad(inc_deg)
    site = ecef_to_teme(geodetic_to_ecef(lat_deg, lon_deg, 0.0), liftoff_utc)
    lat_gc = np.arcsin(site[2] / np.linalg.norm(site))
    u0 = np.arcsin(np.clip(np.sin(lat_gc) / np.sin(i), -1, 1))
    if launch_pass == "descending":
        u0 = np.pi - u0
    alpha = np.arctan2(site[1], site[0])
    raan = alpha - np.arctan2(np.cos(i) * np.sin(u0), np.cos(u0))
    return float(i), float(raan)


def _state(params, inc, raan):
    a, ec, es, u0 = params
    e = float(np.hypot(ec, es))
    argp = float(np.arctan2(es, ec)) if e > 1e-9 else 0.0
    nu = u0 - argp
    E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    r, v = elements_to_state(a, e, inc, raan, argp, M)
    return np.asarray(r, dtype=np.float64).reshape(3), np.asarray(v, dtype=np.float64).reshape(3)


def predict_hud(r, v, utc, alt_mode: str):
    """Altitude (km) and ground-relative speed (km/h) as the overlay would show them."""
    if alt_mode == "geodetic":
        _, _, alt = ecef_to_geodetic(teme_to_ecef(r, utc))
    elif alt_mode == "spherical":
        alt = np.linalg.norm(r, axis=1) - 6371.0
    elif alt_mode == "equatorial":
        alt = np.linalg.norm(r, axis=1) - R_J2
    else:
        raise ValueError(alt_mode)
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    vg = v - np.cross(w, r)
    return alt, np.linalg.norm(vg, axis=1) * 3600.0


@dataclass
class ShipFit:
    epoch_utc: float
    r_teme: list[float]
    v_teme: list[float]
    alt_mode: str
    rms_alt_km: float
    rms_speed_kmh: float
    n_alt: int
    n_speed: int
    perigee_km: float
    apogee_km: float
    inc_deg: float
    raan_deg: float
    u_deg: float  # argument of latitude at the epoch
    u_sigma_deg: float  # 1-sigma along-track phase from the fit covariance
    position_sigma_km: float
    plane_source: str
    alternatives: dict = field(default_factory=dict)  # rms per altitude convention

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"type": "fitted_ship", **asdict(self)}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ShipFit":
        d = json.loads(Path(path).read_text())
        d.pop("type", None)
        return cls(**d)


def fit_ship(series: HudSeries, liftoff_utc: float, inc: float, raan: float, plane_source: str,
             epoch_met: float | None = None, u_guess_rad: float | None = None,
             alt_sigma_km: float = 0.35, speed_sigma_kmh: float = 0.35, prior=None) -> ShipFit:
    """Least-squares fit of (a, e cos w, e sin w, u0) to the HUD series, for each altitude convention.

    The display quantises to 1 km and 1 km/h, i.e. about 0.29 of a unit (1/sqrt(12)) of noise; the
    sigmas default slightly above that. The convention with the lowest normalised residual wins.

    The telemetry sees latitude only through sin^2 and cos^2, so it fixes the phase *modulo 180 deg*
    (the same profile half an orbit later, in the other hemisphere). ``prior`` (any ShipEphemeris,
    normally the nominal one from the launch site and time) picks the branch: its phase is uncertain
    by degrees, not by half an orbit.
    """
    from scipy.optimize import least_squares

    ok_a = np.isfinite(series.alt_km)
    ok_s = np.isfinite(series.speed_kmh)
    met = series.met
    if epoch_met is None:
        epoch_met = float(np.median(met))
    t0 = liftoff_utc + epoch_met
    utc = liftoff_utc + met
    spd = np.nanmedian(series.speed_kmh)
    alt = np.nanmedian(series.alt_km)
    a0 = 6371.0 + alt
    u_starts = [u_guess_rad] if u_guess_rad is not None else list(np.linspace(0, 2 * np.pi, 12, endpoint=False))

    def residuals(p, mode):
        r0, v0 = _state(p, inc, raan)
        r, v = propagate_j2(r0, v0, t0, utc)
        pa, ps = predict_hud(r, v, utc, mode)
        return np.concatenate([(pa[ok_a] - series.alt_km[ok_a]) / alt_sigma_km,
                               (ps[ok_s] - series.speed_kmh[ok_s]) / speed_sigma_kmh])

    results = {}
    for mode in ("geodetic", "spherical", "equatorial"):
        best = None
        for u in u_starts:
            x0 = np.array([a0, 0.0005, 0.0005, u])
            try:
                sol = least_squares(residuals, x0, args=(mode,), x_scale=[1.0, 1e-3, 1e-3, 0.01],
                                    bounds=([6500.0, -0.05, -0.05, -np.inf], [7000.0, 0.05, 0.05, np.inf]))
            except Exception:  # noqa: BLE001 - a failed start is just skipped
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is not None:
            results[mode] = best
    if not results:
        raise RuntimeError("ship fit failed for every altitude convention")
    mode = min(results, key=lambda m: results[m].cost)
    sol = results[mode]
    if prior is not None:
        # the mirror branch: same radius profile and |latitude| history, half an orbit away
        x_alt = sol.x.copy()
        x_alt[1:3] *= -1.0
        x_alt[3] += np.pi
        rp, _ = prior.state(t0)
        d = [np.linalg.norm(_state(x, inc, raan)[0] - rp) for x in (sol.x, x_alt)]
        if d[1] < d[0]:
            sol = least_squares(residuals, x_alt, args=(mode,), x_scale=[1.0, 1e-3, 1e-3, 0.01],
                                bounds=([6500.0, -0.05, -0.05, -np.inf], [7000.0, 0.05, 0.05, np.inf]))
    J = sol.jac
    dof = max(1, len(sol.fun) - len(sol.x))
    s2 = max(1.0, 2.0 * sol.cost / dof)  # inflate when residuals exceed the assumed noise
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.nan)
    a, ec, es, u0 = sol.x
    e = float(np.hypot(ec, es))
    r0, v0 = _state(sol.x, inc, raan)
    res = sol.fun
    na, ns = int(ok_a.sum()), int(ok_s.sum())
    u_sig = float(np.sqrt(max(cov[3, 3], 0.0))) if np.isfinite(cov[3, 3]) else float("nan")
    # along-track uncertainty dominates; add the radial scatter of the altitude residuals
    pos_sig = float(np.hypot(a * u_sig, np.std(res[:na]) * alt_sigma_km if na else 0.0))
    alts = {m: {"rms_alt_km": float(np.sqrt(np.mean(results[m].fun[:na] ** 2)) * alt_sigma_km) if na else None,
                "rms_speed_kmh": float(np.sqrt(np.mean(results[m].fun[na:] ** 2)) * speed_sigma_kmh) if ns else None}
            for m in results}
    return ShipFit(
        epoch_utc=t0, r_teme=r0.tolist(), v_teme=v0.tolist(), alt_mode=mode,
        rms_alt_km=alts[mode]["rms_alt_km"], rms_speed_kmh=alts[mode]["rms_speed_kmh"], n_alt=na, n_speed=ns,
        perigee_km=float(a * (1 - e) - R_J2), apogee_km=float(a * (1 + e) - R_J2),
        inc_deg=float(np.rad2deg(inc)), raan_deg=float(np.rad2deg(raan) % 360), u_deg=float(np.rad2deg(u0) % 360),
        u_sigma_deg=float(np.rad2deg(u_sig)), position_sigma_km=pos_sig, plane_source=plane_source,
        alternatives=alts,
    )


class FittedShipEphemeris(ShipEphemeris):
    """Ship trajectory from :func:`fit_ship` (two-body + J2 from the fitted epoch state)."""

    independent = True  # derived from the broadcast telemetry, not from the payloads' along-track positions

    def __init__(self, fit: ShipFit):
        self.fit = fit
        self.r0 = np.array(fit.r_teme)
        self.v0 = np.array(fit.v_teme)
        self.position_sigma_km = fit.position_sigma_km
        self.description = (f"fitted to HUD telemetry ({fit.alt_mode} altitude, {fit.perigee_km:.0f}x{fit.apogee_km:.0f} km, "
                            f"phase +/-{fit.u_sigma_deg:.2f} deg)")
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def state(self, utc):
        t = np.atleast_1d(np.asarray(utc, dtype=np.float64))
        r, v = propagate_j2(self.r0, self.v0, self.fit.epoch_utc, t)
        if np.ndim(utc) == 0:
            return r[0], v[0]
        return r, v

    @classmethod
    def load(cls, path: str | Path) -> "FittedShipEphemeris":
        return cls(ShipFit.load(path))
