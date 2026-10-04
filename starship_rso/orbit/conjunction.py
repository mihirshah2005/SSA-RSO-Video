"""Which catalogued objects came close enough to the ship to be visible at all?

Coarse-to-fine screen: propagate every record on a coarse grid, keep objects
whose minimum range could fall inside ``range_km`` (with a relative-speed
margin), then refine those on a fine grid. For each hit the closest approach,
illumination, Earth occultation, estimated magnitude and apparent size are
reported. With a nominal (phase-uncertain) ship ephemeris, pass several
phase-offset ephemerides; the screen keeps the minimum over them, so it is
conservative: an object absent from the result could not have been seen under
any of the plausible ship positions.

A *negative* result -- no foreign object within visible range -- is itself
a finding for the report: it bounds which dots can be catalogued satellites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .frames import line_of_sight_clear
from .photometry import VisibilityModel
from .propagate import Propagator
from .ship import ShipEphemeris
from .sun import is_sunlit, phase_angle, sun_position_teme


@dataclass
class ScreenHit:
    norad_id: str
    name: str
    t_closest_utc: float
    range_min_km: float
    rel_speed_km_s: float
    sunlit: bool
    los_clear: bool
    mag_est: float
    size_px_est: float
    visible_est: bool
    ephemeris: str

    def to_dict(self) -> dict:
        return asdict(self)


def screen(
    prop: Propagator,
    ships: list[ShipEphemeris],
    t0_utc: float,
    t1_utc: float,
    range_km: float = 100.0,
    coarse_dt: float = 10.0,
    fine_dt: float = 0.5,
    ifov_rad: float = np.deg2rad(90.0) / 1920,
    vis: VisibilityModel | None = None,
    extra_radius_km: float = 0.0,
) -> list[ScreenHit]:
    """``extra_radius_km`` widens the screen when ``ships`` are discrete samples of an uncertain
    ephemeris (half the along-track spacing between hypotheses), keeping the result conservative."""
    vis = vis or VisibilityModel()
    limit = range_km + extra_radius_km
    tc = np.arange(t0_utc, t1_utc + coarse_dt, coarse_dt)
    r, v, ok = prop.states(tc)  # (n, T, 3)
    best: dict[int, ScreenHit] = {}
    for ship in ships:
        rs, vs = ship.state(tc)
        d = np.linalg.norm(r - rs[None], axis=-1)
        dv = np.linalg.norm(v - vs[None], axis=-1)
        margin = 0.5 * coarse_dt * np.nanmax(dv, axis=1)
        dmin = np.nanmin(np.where(ok, d, np.inf), axis=1)
        cand = np.nonzero(dmin <= limit + margin)[0]
        if len(cand) == 0:
            continue
        tf = np.arange(t0_utc, t1_utc + fine_dt, fine_dt)
        sub = Propagator([prop.records[i] for i in cand])
        rf, vf, okf = sub.states(tf)
        rsf, vsf = ship.state(tf)
        df = np.where(okf, np.linalg.norm(rf - rsf[None], axis=-1), np.inf)
        k = np.argmin(df, axis=1)
        sun = sun_position_teme(tf)
        for j, i in enumerate(cand):
            tk = k[j]
            if not np.isfinite(df[j, tk]):
                continue
            # closest approach between grid samples (straight-line relative motion over +-dt/2)
            dr, dvv = rf[j, tk] - rsf[tk], vf[j, tk] - vsf[tk]
            tau = float(np.clip(-np.dot(dr, dvv) / max(np.dot(dvv, dvv), 1e-12), -0.5 * fine_dt, 0.5 * fine_dt))
            rmin = float(np.linalg.norm(dr + dvv * tau))
            if rmin > limit:
                continue
            ro, so = rf[j, tk] + vf[j, tk] * tau, rsf[tk] + vsf[tk] * tau
            lit = bool(is_sunlit(ro, sun[tk]))
            ph = float(phase_angle(so, ro, sun[tk]))
            nadir_dot = float(np.dot((ro - so) / max(rmin, 1e-9), -so / np.linalg.norm(so)))
            earth_bg = nadir_dot > np.cos(np.arcsin(6378.137 / np.linalg.norm(so)))
            a = vis.assess(np.array([rmin]), np.array([ph]), np.array([lit]), ifov_rad, np.array([earth_bg]))
            hit = ScreenHit(
                norad_id=prop.ids[i],
                name=prop.names[i],
                t_closest_utc=float(tf[tk] + tau),
                range_min_km=rmin,
                rel_speed_km_s=float(np.linalg.norm(vf[j, tk] - vsf[tk])),
                sunlit=lit,
                los_clear=bool(line_of_sight_clear(so, ro)),
                mag_est=float(a["mag"][0]),
                size_px_est=float(a["px"][0]),
                visible_est=bool(a["visible"][0]),
                ephemeris=ship.description,
            )
            prev = best.get(i)
            if prev is None or hit.range_min_km < prev.range_min_km:
                best[i] = hit
    return sorted(best.values(), key=lambda h: h.range_min_km)


def phase_sweep(ship, range_km: float, sigmas: float = 3.0, max_n: int = 2001) -> tuple[list, float]:
    """Nominal-ephemeris copies spanning +-``sigmas`` of the along-track phase uncertainty.

    Hypotheses are spaced at most ``range_km`` apart along track, so no object that came within
    ``range_km`` of the true ship can fall between them. Returns (ships, spacing_km).
    """
    if not hasattr(ship, "with_phase_offset") or ship.phase_sigma_deg <= 0:
        return [ship], 0.0
    a = float(ship.orbit.a)
    step_deg = np.rad2deg(range_km / a)
    span = 2 * sigmas * ship.phase_sigma_deg
    n = int(min(max_n, np.ceil(span / step_deg) + 1))
    offs = np.linspace(-sigmas, sigmas, n) * ship.phase_sigma_deg
    spacing_km = float(np.deg2rad(span / max(n - 1, 1)) * a)
    return [ship.with_phase_offset(float(o)) for o in offs], spacing_km
