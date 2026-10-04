"""Slow, one-off preparation: time map, camera, ship ephemeris, catalogue predictions.

Everything here runs before the first frame (or when the clip's time window
grows), never per frame. Each component records where it came from so the
overlay can state which information the identities rest on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..geometry.camera import PinholeCamera, camera_from_config
from ..identify.predictions import PredictionCache
from ..io.timemap import TimeMap, parse_utc
from ..orbit.catalog import best_record_per_object, select_records
from ..orbit.omm import OMMRecord, load_omm_json
from ..orbit.release import ReleaseEstimate, estimate_release_times
from ..orbit.ship import GroupCentroidEphemeris, NominalShipEphemeris, SGP4ShipEphemeris, ShipEphemeris

log = logging.getLogger(__name__)


@dataclass
class MissionContext:
    timemap: TimeMap
    camera_native: PinholeCamera
    ship: ShipEphemeris | None = None
    records: list[OMMRecord] = field(default_factory=list)
    group: list[OMMRecord] = field(default_factory=list)
    cache: PredictionCache | None = None
    release_estimates: list[ReleaseEstimate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    mode: str = "replay"  # replay | live | synthetic


def build_context(
    cfg: Config,
    width: int,
    height: int,
    clip_t0: float,
    clip_t1: float,
    catalog_path: str | None = None,
    mode: str = "replay",
    catalog_mode: str = "retrospective",
) -> MissionContext:
    m = cfg.mission
    tm = TimeMap.from_config(cfg.timemap, m.liftoff_utc)
    cam = camera_from_config(cfg.camera, width, height)
    ctx = MissionContext(tm, cam, mode=mode)
    if not tm.is_calibrated:
        ctx.notes.append("no time anchors: MET/UTC unknown, catalogue association disabled")
    if not cam.calibrated:
        ctx.notes.append("camera attitude not calibrated: catalogue association disabled")

    path = catalog_path or m.catalog_file
    if path:
        ctx.records = load_omm_json(path)
        ctx.notes.append(f"catalogue: {len(ctx.records)} element sets from {path}")
    liftoff = parse_utc(m.liftoff_utc) if m.liftoff_utc else None
    u0, _ = tm.utc(clip_t0)
    u1, _ = tm.utc(clip_t1)
    if u0 is not None and ctx.records:
        ctx.records = best_record_per_object(ctx.records, u0, catalog_mode)
        if catalog_mode == "retrospective":
            ctx.notes.append("RETROSPECTIVE: element sets chosen with hindsight (may postdate the video)")
        pg = m.payload_group
        if pg.norad_range or pg.name_contains or pg.intdes:
            ctx.group = select_records(ctx.records, pg.norad_range, pg.name_contains, pg.intdes)
            if pg.norad_range:
                ctx.notes.append(f"payload group {pg.name}: {len(ctx.group)} objects (NORAD range is a candidate set)")

    if liftoff is not None:
        ctx.ship = _ship_ephemeris(cfg, liftoff, ctx)
    if ctx.ship is not None and u0 is not None and u1 is not None and ctx.records:
        pad = 5.0
        near = _within_range(ctx.records, ctx.ship, u0 - pad, u1 + pad, cfg.identify.max_range_km)
        ctx.notes.append(f"{len(near)} of {len(ctx.records)} objects come within {cfg.identify.max_range_km:.0f} km during the clip")
        ctx.cache = PredictionCache.build(near, ctx.ship, u0 - pad, u1 + pad, dt=0.25,
                                          note="retrospective" if catalog_mode == "retrospective" else "as-of")
        ctx.notes.append(f"ship ephemeris: {ctx.ship.description} (~{ctx.ship.position_sigma_km:.1f} km 1-sigma)")
    if ctx.group and liftoff is not None and m.deployment.first_met_s is not None:
        if isinstance(ctx.ship, SGP4ShipEphemeris):
            t0 = liftoff + m.deployment.first_met_s - 60.0
            t1 = liftoff + (m.deployment.last_met_s or m.deployment.first_met_s) + 60.0
            ctx.release_estimates = estimate_release_times(ctx.group, t0, t1, ship=ctx.ship, dt=0.5,
                                                           along_track_sigma_km=cfg.identify.ephem_sigma_km)
            spread = np.median([r.t_sigma_s for r in ctx.release_estimates]) if ctx.release_estimates else float("nan")
            ctx.notes.append(f"release-time estimates for {len(ctx.release_estimates)} objects, median sigma {spread:.0f} s")
        else:
            # the group's own centroid is a biased reference for release times (it moves with the
            # group); without an independent ship ephemeris, release slots stay unidentified
            ctx.notes.append("no independent ship ephemeris: DEPLOY-k slots are not mapped to catalogue numbers")
    for n in ctx.notes:
        log.info(n)
    return ctx


def _within_range(records: list[OMMRecord], ship: ShipEphemeris, t0: float, t1: float, max_km: float,
                  step_s: float = 20.0, chunk: int = 2000) -> list[OMMRecord]:
    """Keep objects whose distance to the ship drops below ``max_km`` (coarse grid + speed margin).

    Bounds memory for long clips and whole-catalogue files: predictions are cached only for these.
    """
    from ..orbit.propagate import Propagator

    t = np.arange(t0, t1 + step_s, step_s)
    rs, _ = ship.state(t)
    keep: list[OMMRecord] = []
    margin = 15.0 * step_s  # km: relative speed bound (km/s) times the grid step
    for k in range(0, len(records), chunk):
        part = records[k : k + chunk]
        r, _, ok = Propagator(part).states(t)
        d = np.where(ok, np.linalg.norm(r - rs[None], axis=-1), np.inf)
        keep += [rec for rec, dmin in zip(part, d.min(1)) if dmin <= max_km + margin]
    return keep


def _ship_ephemeris(cfg: Config, liftoff: float, ctx: MissionContext) -> ShipEphemeris | None:
    spec = cfg.mission.ship_ephemeris
    if spec == "nominal":
        return NominalShipEphemeris.from_mission(cfg.mission, liftoff)
    if spec == "group_centroid":
        if not ctx.group:
            raise ValueError("ship_ephemeris=group_centroid needs mission.payload_group to select records")
        return GroupCentroidEphemeris(ctx.group)
    if spec.startswith("omm:"):
        recs = load_omm_json(spec[4:])
        if not recs:
            raise ValueError(f"no records in {spec[4:]}")
        return SGP4ShipEphemeris(recs[0])
    raise ValueError(f"unknown mission.ship_ephemeris {spec!r}")
