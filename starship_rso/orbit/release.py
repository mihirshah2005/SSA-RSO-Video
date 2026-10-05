"""Estimate when each deployed object left the ship, from its element sets.

Each payload was co-located with the ship at its release time ``t_i``. Given
element sets fitted *after* deployment, back-propagate every object over the
deployment window and find the time of closest approach to a ship reference
(an independent ship ephemeris if available, otherwise the group centroid).

This is only informative when element-set errors are small compared with the
separation the objects build up. With ~0.5 m/s separation speeds the
separation after one release interval (~70 s) is tens of metres, while GP
errors for a just-deployed LEO group are typically kilometres. The Monte-Carlo
spread returned here makes that limitation explicit: if ``t_sigma`` is
comparable to the release spacing, the mapping DEPLOY-k -> NORAD id is not
identifiable from these data and the system must say so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .omm import OMMRecord
from .propagate import Propagator
from .ship import ShipEphemeris


@dataclass
class ReleaseEstimate:
    norad_id: str
    name: str
    t_release_utc: float
    t_sigma_s: float
    min_sep_km: float
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


def _closest_times(r_obj: np.ndarray, r_ref: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(r_obj - r_ref[None], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    k = np.argmin(d, axis=1)
    # parabolic refinement around the discrete minimum
    tk = t[k].astype(float)
    for j, kk in enumerate(k):
        if 0 < kk < len(t) - 1:
            y0, y1, y2 = d[j, kk - 1], d[j, kk], d[j, kk + 1]
            den = y0 - 2 * y1 + y2
            if den > 0:
                tk[j] = t[kk] + 0.5 * (y0 - y2) / den * (t[1] - t[0])
    return tk, d[np.arange(len(k)), k]


def estimate_release_times(
    records: list[OMMRecord],
    t0_utc: float,
    t1_utc: float,
    ship: ShipEphemeris | None = None,
    dt: float = 1.0,
    along_track_sigma_km: float = 1.0,
    n_mc: int = 64,
    seed: int = 0,
) -> list[ReleaseEstimate]:
    t = np.arange(t0_utc, t1_utc + dt, dt)
    prop = Propagator(records)
    r, v, ok = prop.states(t)
    method = "ship" if ship is not None else "group-centroid"

    def reference(rr: np.ndarray) -> np.ndarray:
        if ship is not None:
            return ship.state(t)[0]
        with np.errstate(all="ignore"):
            return np.nanmedian(rr, axis=0)

    t_hat, dmin = _closest_times(r, reference(r), t)
    # Monte Carlo: shift each object along its velocity by N(0, sigma) (dominant GP error mode)
    rng = np.random.default_rng(seed)
    vhat = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    samples = np.empty((n_mc, len(records)))
    for m in range(n_mc):
        shift = rng.normal(0.0, along_track_sigma_km, size=(len(records), 1, 1))
        rp = r + shift * vhat
        samples[m], _ = _closest_times(rp, reference(rp), t)
    t_sig = samples.std(axis=0)
    # a minimum on the window edge means the closest approach was not found: not identifiable
    edge = (np.abs(t_hat - t[0]) < 1.5 * dt) | (np.abs(t_hat - t[-1]) < 1.5 * dt)
    # an object released from the ship passes within metres of it; if the propagated elements never
    # come within a few sigma of the ship, they are not valid back to the release (orbit raising,
    # drag or a stale epoch) and the "closest approach" time means nothing
    ship_sig = float(getattr(ship, "position_sigma_km", np.nan)) if ship is not None else np.nan
    reach = max(5.0, 3.0 * float(np.hypot(along_track_sigma_km, ship_sig if np.isfinite(ship_sig) else 0.0)))
    t_sig = np.where(edge | (dmin > reach), np.inf, t_sig)
    return [
        ReleaseEstimate(prop.ids[i], prop.names[i], float(t_hat[i]), float(t_sig[i]), float(dmin[i]), method)
        for i in range(len(records))
    ]
