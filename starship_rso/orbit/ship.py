"""Where was the camera? Ship ephemerides with explicit uncertainty.

SpaceX does not publish Starship state vectors, so three estimates exist,
in decreasing order of accuracy:

1. ``SGP4ShipEphemeris`` -- an OMM for the ship itself or a proxy object
   (e.g. the earliest element set of a payload released from it, valid near
   its release epoch).
2. ``GroupCentroidEphemeris`` -- the robust centroid of a freshly deployed
   payload group. During deployment the ship sits inside the group, within the
   group's spread (km-level, growing with time since release).
3. ``NominalShipEphemeris`` -- built from published mission facts: launch
   site and time, inclination, perigee/apogee, and an assumed downrange angle
   at SECO. The orbital *plane* is well constrained (the site is in it at
   liftoff), the along-track *phase* is not: ``phase_sigma_deg`` of 5 deg is
   ~600 km along track. Use it for coarse screening only.

All return TEME position/velocity (km, km/s) at POSIX times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import R_EARTH, ecef_to_teme, geodetic_to_ecef
from .kepler import MU, KeplerOrbit
from .omm import OMMRecord
from .propagate import Propagator


class ShipEphemeris:
    description = "abstract"
    position_sigma_km = float("nan")
    # True when the estimate does not come from the payloads' own along-track positions, so it can
    # serve as the reference for release-time estimates
    independent = False

    def state(self, utc) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError


class SGP4ShipEphemeris(ShipEphemeris):
    independent = True

    def __init__(self, record: OMMRecord, position_sigma_km: float = 2.0):
        self.record = record
        self._p = Propagator([record])
        self.description = f"SGP4 proxy {record.name} ({record.norad_id}, epoch {record.epoch_utc:.0f})"
        self.position_sigma_km = position_sigma_km

    def state(self, utc):
        r, v, _ = self._p.states(utc)
        if np.ndim(utc) == 0:
            return r[0, 0], v[0, 0]
        return r[0], v[0]


class GroupCentroidEphemeris(ShipEphemeris):
    def __init__(self, records: list[OMMRecord], position_sigma_km: float = 5.0):
        if not records:
            raise ValueError("empty group")
        self._p = Propagator(records)
        self.description = f"centroid of {len(records)} deployed objects"
        self.position_sigma_km = position_sigma_km

    def state(self, utc):
        r, v, ok = self._p.states(utc)
        with np.errstate(all="ignore"):
            rm, vm = np.nanmedian(r, axis=0), np.nanmedian(v, axis=0)
        if np.ndim(utc) == 0:
            return rm[0], vm[0]
        return rm, vm


@dataclass
class NominalParams:
    liftoff_utc: float
    site_lat_deg: float
    site_lon_deg: float
    inc_deg: float
    perigee_km: float
    apogee_km: float
    launch_pass: str = "descending"
    seco_met_s: float = 491.0
    downrange_deg_at_seco: float = 14.0
    phase_offset_deg: float = 0.0  # for uncertainty sweeps


class NominalShipEphemeris(ShipEphemeris):
    """Orbit through the launch site at liftoff, phased by an assumed downrange angle at SECO.

    After SECO the transatmospheric trajectory has nearly orbital energy; the
    insertion burn changes the orbit only slightly, so a single near-circular
    orbit is used for the whole coast. The argument of latitude of the site at
    liftoff is ``asin(sin(lat)/sin(i))`` on an ascending pass and ``pi`` minus
    that on a descending pass (Starbase launches toward the ESE are descending).
    """

    def __init__(self, p: NominalParams, phase_sigma_deg: float = 5.0):
        self.p = p
        i = np.deg2rad(p.inc_deg)
        site_ecef = geodetic_to_ecef(p.site_lat_deg, p.site_lon_deg, 0.0)
        site = ecef_to_teme(site_ecef, p.liftoff_utc)
        lat_gc = np.arcsin(site[2] / np.linalg.norm(site))
        if abs(np.sin(lat_gc)) > np.sin(i):
            raise ValueError("inclination lower than launch-site latitude: site cannot lie in the plane")
        u0 = np.arcsin(np.sin(lat_gc) / np.sin(i))
        if p.launch_pass == "descending":
            u0 = np.pi - u0
        elif p.launch_pass != "ascending":
            raise ValueError("launch_pass must be 'ascending' or 'descending'")
        alpha = np.arctan2(site[1], site[0])
        dlam = np.arctan2(np.cos(i) * np.sin(u0), np.cos(u0))
        raan = alpha - dlam
        rp, ra = R_EARTH + p.perigee_km, R_EARTH + p.apogee_km
        a = 0.5 * (rp + ra)
        e = (ra - rp) / (ra + rp)
        u_seco = u0 + np.deg2rad(p.downrange_deg_at_seco + p.phase_offset_deg)
        # perigee placed at the SECO point (argp = u_seco, M = 0 there): only phase matters at e ~ 1e-3
        self.orbit = KeplerOrbit(p.liftoff_utc + p.seco_met_s, a, e, i, raan, u_seco, 0.0, use_j2=True)
        self.description = (
            f"nominal {p.perigee_km:.0f}x{p.apogee_km:.0f} km i={p.inc_deg:.1f} deg, "
            f"downrange {p.downrange_deg_at_seco:.0f}+{p.phase_offset_deg:.1f} deg at SECO"
        )
        self.phase_sigma_deg = phase_sigma_deg
        self.position_sigma_km = float(np.deg2rad(phase_sigma_deg) * a)

    def state(self, utc):
        return self.orbit.state(utc)

    def with_phase_offset(self, deg: float) -> "NominalShipEphemeris":
        q = NominalParams(**{**self.p.__dict__, "phase_offset_deg": deg})
        return NominalShipEphemeris(q, self.phase_sigma_deg)

    @classmethod
    def from_mission(cls, mission, liftoff_utc: float) -> "NominalShipEphemeris":
        on = mission.orbit_nominal
        dr = on.downrange_deg_at_seco
        p = NominalParams(
            liftoff_utc=liftoff_utc,
            site_lat_deg=mission.launch_site.lat_deg,
            site_lon_deg=mission.launch_site.lon_deg,
            inc_deg=on.inc_deg,
            perigee_km=on.perigee_km,
            apogee_km=on.apogee_km,
            launch_pass=on.launch_pass,
            downrange_deg_at_seco=14.0 if dr is None else dr,
        )
        return cls(p, on.phase_sigma_deg)


def altitude_speed(ship: ShipEphemeris, utc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geodetic altitude (km), inertial speed and Earth-relative speed (km/h) for HUD cross-checks."""
    from .frames import ecef_to_geodetic, teme_to_ecef, teme_velocity_relative_to_ground

    r, v = ship.state(utc)
    _, _, alt = ecef_to_geodetic(teme_to_ecef(r, utc))
    v_in = np.linalg.norm(v, axis=-1) * 3600.0
    v_gr = np.linalg.norm(teme_velocity_relative_to_ground(r, v), axis=-1) * 3600.0
    return alt, v_in, v_gr


def circular_speed_kmh(alt_km: float) -> float:
    return float(np.sqrt(MU / (R_EARTH + alt_km)) * 3600.0)
