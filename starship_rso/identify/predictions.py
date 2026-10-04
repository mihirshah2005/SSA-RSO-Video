"""Cached ship-relative predictions for candidate catalogue objects.

Built once per clip (never per frame): every candidate is propagated on a
regular UTC grid together with the ship ephemeris, and the relative position
is stored in the ship's RIC frame. Queries interpolate linearly; the grid
step is chosen so interpolation error stays far below the localisation budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..orbit.frames import ric_matrix
from ..orbit.omm import OMMRecord
from ..orbit.propagate import Propagator
from ..orbit.ship import ShipEphemeris
from ..orbit.sun import is_sunlit, sun_position_teme


@dataclass
class PredictionCache:
    t: np.ndarray  # (T,) utc grid
    ids: list[str]
    names: list[str]
    rel_ric: np.ndarray  # (N, T, 3) km, object minus ship, ship RIC axes
    sunlit: np.ndarray  # (N, T) bool
    ship_r: np.ndarray  # (T, 3) TEME
    ship_v: np.ndarray  # (T, 3)
    ship_description: str
    ship_sigma_km: float
    retrieved_note: str = ""

    @classmethod
    def build(
        cls,
        records: list[OMMRecord],
        ship: ShipEphemeris,
        t0_utc: float,
        t1_utc: float,
        dt: float = 0.25,
        note: str = "",
    ) -> "PredictionCache":
        t = np.arange(t0_utc, t1_utc + dt, dt)
        prop = Propagator(records)
        r, v, ok = prop.states(t)
        rs, vs = ship.state(t)
        M = ric_matrix(rs, vs)  # (T, 3, 3)
        rel = np.einsum("tij,ntj->nti", M, r - rs[None])
        sun = sun_position_teme(t)
        lit = is_sunlit(r, sun[None]) & ok
        return cls(t, prop.ids, prop.names, rel, lit, rs, vs, ship.description,
                   getattr(ship, "position_sigma_km", float("nan")), note)

    def __len__(self) -> int:
        return len(self.ids)

    def covers(self, utc: float) -> bool:
        return len(self.t) > 0 and self.t[0] <= utc <= self.t[-1]

    def rel_at(self, utc) -> np.ndarray:
        """Relative positions (N, len(utc), 3) at arbitrary times (linear interpolation)."""
        q = np.atleast_1d(np.asarray(utc, dtype=np.float64))
        if len(self.t) < 2:
            return np.repeat(self.rel_ric[:, :1], len(q), axis=1)
        x = (q - self.t[0]) / (self.t[1] - self.t[0])
        i = np.clip(np.floor(x).astype(int), 0, len(self.t) - 2)
        w = np.clip(x - i, 0.0, 1.0)[None, :, None]
        return self.rel_ric[:, i] * (1 - w) + self.rel_ric[:, i + 1] * w

    def sunlit_at(self, utc) -> np.ndarray:
        q = np.atleast_1d(np.asarray(utc, dtype=np.float64))
        i = np.clip(np.searchsorted(self.t, q), 0, len(self.t) - 1)
        return self.sunlit[:, i]

    def ground_velocity_ric(self, utc: float) -> np.ndarray:
        """Ship velocity relative to the rotating Earth, in RIC (for FOE-based attitude)."""
        from ..orbit.frames import teme_velocity_relative_to_ground

        i = int(np.clip(np.searchsorted(self.t, utc), 0, len(self.t) - 1))
        vg = teme_velocity_relative_to_ground(self.ship_r[i], self.ship_v[i])
        return ric_matrix(self.ship_r[i], self.ship_v[i]) @ vg
