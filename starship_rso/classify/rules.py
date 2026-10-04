"""Transparent, physics-motivated category rules (the reference classifier).

Each category gets an evidence score in [0, 1] built from named cues, and the
cues are reported as reasons so the overlay can show *why*. The answer is
``unknown`` unless one category clearly wins. These rules encode hypotheses
(e.g. "defocused + fast + flickering suggests a near-field particle"); they
are not proof, and the trained model in :mod:`classify.model` should replace
them wherever labelled data exist.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import ClassifyCfg, MissionCfg
from ..types import Category, CategoryDecision


def _ramp(v: float, lo: float, hi: float) -> float:
    """0 at ``lo``, 1 at ``hi`` (works for decreasing ramps when lo > hi)."""
    if hi == lo:
        return float(v >= hi)
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _noisy_or(ps: list[float]) -> float:
    q = 1.0
    for p in ps:
        q *= 1.0 - float(np.clip(p, 0.0, 1.0))
    return 1.0 - q


def classify_rules(f: dict[str, float], cfg: ClassifyCfg, mission: MissionCfg | None = None) -> CategoryDecision:
    reasons: list[str] = []
    s: dict[str, float] = {c.value: 0.0 for c in Category if c is not Category.UNKNOWN}
    if f["n_obs"] < cfg.min_obs:
        return CategoryDecision(Category.UNKNOWN, s, [f"only {int(f['n_obs'])} observations"])

    dur_ok = _ramp(f["duration_s"], 0.2, 1.0)
    speed, rel = f["speed_px_s"], f["rel_speed_px_s"]

    # Fixed relative to the camera: glints, tile edges, antennas.
    static = _ramp(speed, 2 * cfg.static_speed_px_s, cfg.static_speed_px_s)
    s[Category.VEHICLE_FEATURE.value] = static * dur_ok
    if static > 0.5:
        reasons.append(f"fixed in image ({speed:.1f} px/s)")

    # Moving with the Earth: cloud puffs, sun glint on water.
    if f["has_bg"] > 0 and f["bg_speed_px_s"] > 2 * cfg.bg_rel_speed_px_s:
        with_bg = _ramp(rel, 2 * cfg.bg_rel_speed_px_s, cfg.bg_rel_speed_px_s)
        s[Category.BACKGROUND_FEATURE.value] = with_bg * dur_ok
        if with_bg > 0.5:
            reasons.append(f"moves with background (rel {rel:.1f} px/s)")

    # Near-field particle cues (noisy-OR of independent-ish cues).
    cues = []
    defocus = _ramp(f["sigma_med"], cfg.defocus_sigma_px, 2 * cfg.defocus_sigma_px)
    if defocus > 0:
        cues.append(0.6 * defocus)
        reasons.append(f"defocused (sigma {f['sigma_med']:.1f} px)")
    fast = _ramp(rel, cfg.fast_rel_speed_px_s, 3 * cfg.fast_rel_speed_px_s)
    if fast > 0:
        cues.append(0.6 * fast)
        reasons.append(f"fast relative motion ({rel:.0f} px/s)")
    flick = _ramp(f["flux_cv"], cfg.flicker_cv, 2 * cfg.flicker_cv)
    periodic = _ramp(f["flicker_power"], 0.25, 0.6)
    if flick > 0 or periodic > 0:
        cues.append(0.5 * max(flick, periodic))
        reasons.append(f"flickering (cv {f['flux_cv']:.2f}, peak {f['flicker_power']:.2f} @ {f['flicker_hz']:.1f} Hz)")
    curved = _ramp(f["curv_rms_px"], cfg.payload_max_curv_px, 4 * cfg.payload_max_curv_px)
    if curved > 0:
        cues.append(0.3 * curved)
        reasons.append(f"curved/accelerating path ({f['curv_rms_px']:.1f} px)")
    growth = _ramp(abs(f["log_sigma_rate"]), 0.3, 1.0)
    if growth > 0:
        cues.append(0.3 * growth)
        reasons.append("rapid size change (near the camera)")
    s[Category.NEAR_FIELD_PARTICLE.value] = _noisy_or(cues) * dur_ok * (1.0 - static)

    # Deployed payload: steady, in focus, slow, appears near the door during the deployment window.
    steady = _ramp(f["curv_rms_px"], 2 * cfg.payload_max_curv_px, cfg.payload_max_curv_px)
    slowish = _ramp(rel, 1.5 * cfg.payload_max_speed_px_s, cfg.payload_max_speed_px_s)
    calm = _ramp(f["flux_cv"], cfg.flicker_cv, 0.5 * cfg.flicker_cv)
    long = _ramp(f["duration_s"], 1.0, 3.0)
    p = steady * slowish * calm * long * (1.0 - static)
    # a departing payload is close: resolved, or visibly changing size; a far point stays point-like.
    # Brightness change alone is not evidence of nearness (a distant pass brightens and fades too).
    resolved = _ramp(f["sigma_med"], 1.8, 3.0)
    # a size trend is measurable only on an object that is at least marginally resolved
    size_change = _ramp(abs(f["log_sigma_rate"]), 0.05, 0.3) * _ramp(f["sigma_med"], 1.2, 1.8)
    p *= 0.25 + 0.75 * max(resolved, size_change)
    door = f["door_dist_norm"]
    if door >= 0:
        # a payload leaving the door is metres away, hence resolved when first seen; a point that
        # first appears there is more likely a distant object emerging from behind the vehicle
        near_door = _ramp(door, 2 * cfg.payload_door_radius, cfg.payload_door_radius)
        near_door *= _ramp(f.get("first_sigma", 2.0), 1.2, 2.0)
        p *= 0.3 + 0.7 * near_door
        if near_door > 0.5:
            reasons.append("first seen near the payload door")
    if mission is not None and mission.deployment.first_met_s is not None and math.isfinite(f["met_first"]):
        lo = mission.deployment.first_met_s - 30.0
        hi = (mission.deployment.last_met_s or lo) + 600.0
        in_window = 1.0 if lo <= f["met_first"] <= hi else 0.2
        p *= in_window
        if in_window < 1:
            reasons.append("outside the deployment window")
    s[Category.PAYLOAD.value] = p
    if p > 0.5:
        reasons.append("steady, slow, non-flickering motion")

    ranked = sorted(s.items(), key=lambda kv: -kv[1])
    (best, sb), (_, s2) = ranked[0], ranked[1]
    if sb < 0.5 or sb - s2 < cfg.margin:
        return CategoryDecision(Category.UNKNOWN, s, reasons + ["no category clearly supported"])
    return CategoryDecision(Category.parse(best), s, reasons)
