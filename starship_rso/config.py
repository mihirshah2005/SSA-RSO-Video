"""Typed configuration loaded from one or more YAML files plus dotted overrides.

Later files override earlier ones (deep merge). Unknown keys are an error so
that a typo never silently falls back to a default.

    cfg = load_config(["configs/default.yaml", "configs/flight14.yaml"],
                      overrides=["detector.type=heatmap"])
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------- sections
@dataclass
class ProcessingCfg:
    scale: float = 1.0  # resize factor applied before processing (1.0 = native)


@dataclass
class RealtimeCfg:
    enabled: bool = True  # pace frames by wall clock (live replay); False = as fast as possible
    speed: float = 1.0  # replay speed multiplier
    drop_policy: str = "latest"  # latest: skip stale frames when behind | none: never drop
    queue_size: int = 4


@dataclass
class StaticMaskCfg:
    enabled: bool = True
    tau_s: float = 8.0  # memory of the temporal statistics (s); long, so slow movers are not masked
    std_thresh: float = 3.0  # grey levels; below this a pixel is "static"
    bright_thresh: float = 40.0  # static AND brighter than this -> vehicle
    grad_thresh: float = 10.0  # static AND textured (mean-image gradient) -> vehicle
    min_area_px: int = 1500  # small static blobs stay unmasked (hovering objects)
    dilate_px: int = 5
    warmup_s: float = 1.0
    update_every: int = 5  # recompute the mask every N frames


@dataclass
class MaskCfg:
    hud_rects: list[list[float]] = field(default_factory=list)  # normalised [x0, y0, x1, y1]
    border_px: int = 4
    static: StaticMaskCfg = field(default_factory=StaticMaskCfg)


@dataclass
class RegistrationCfg:
    work_width: int = 960
    max_corners: int = 600
    quality: float = 0.01
    min_distance: int = 12
    lk_win: int = 21
    lk_levels: int = 3
    fb_thresh_px: float = 1.0  # forward-backward consistency (work pixels)
    ransac_thresh_px: float = 1.5  # work pixels
    min_inliers: int = 25
    model: str = "homography"  # homography | affine | similarity (the most general model allowed)
    # Sparse or clustered support cannot pin down perspective terms, and an over-fitted model
    # extrapolates badly (several px) across the rest of the image. Below these limits the fit
    # steps down: homography -> affine (keeps a linear flow gradient) -> similarity.
    full_model_min_inliers: int = 150
    full_model_min_spread: float = 0.3  # inlier convex hull / usable image area
    affine_min_inliers: int = 60
    affine_min_spread: float = 0.15
    static_px: float = 0.4  # corners moving less than this (work px/frame) are treated as camera-fixed
    min_moving_spread: float = 0.05  # moving inliers must cover this much of the image to be the background


@dataclass
class SceneCutCfg:
    work_width: int = 320
    hist_corr_thresh: float = 0.55  # histogram correlation below this -> cut
    mean_abs_thresh: float = 40.0  # mean abs difference above this -> cut


@dataclass
class ScaleCfg:
    radius: int = 3  # top-hat structuring element radius (in downsampled pixels)
    downsample: int = 1


@dataclass
class ClassicalCfg:
    polarity: str = "bright"  # bright | dark | both
    scales: list[ScaleCfg] = field(default_factory=lambda: [ScaleCfg(3, 1), ScaleCfg(4, 4), ScaleCfg(4, 8)])
    snr_clean: float = 6.0  # appearance-only threshold in low-clutter regions (black sky)
    snr_app: float = 5.0  # appearance threshold when temporal support is required
    snr_res: float = 6.0  # temporal residual threshold
    low_factor: float = 0.6  # detections down to low_factor*threshold are kept as "low" (2nd stage)
    noise_every: int = 3  # recompute the robust noise maps every N frames
    clutter_window: int = 97  # median window (px), much larger than any object so objects cannot raise their own clutter
    clutter_thresh: float = 1.0  # local median |high-pass| (grey levels) above which motion is required
    history: list[int] = field(default_factory=lambda: [2, 5])  # past-frame offsets (full-res scale)
    coarse_history: list[int] = field(default_factory=lambda: [8, 16])  # offsets for downsampled scales
    residual_dilate: int = 3  # max-filter size on past frames (tolerates sub-pixel misregistration)
    nms_radius: int = 4
    max_detections: int = 400
    centroid_halfwin: int = 4
    noise_block: int = 64  # block size for the robust noise map
    residual_block: int = 96  # block size for the local motion-residual noise map
    min_residual_frac: float = 0.5  # residual must be at least this fraction of the object's contrast


@dataclass
class HeatmapCfg:
    weights: str | None = None
    frames: int = 3
    threshold: float = 0.35
    low_threshold: float = 0.15
    tile: int = 0  # 0 = whole frame; else tile size with 64 px overlap
    device: str = "auto"  # auto | cpu | cuda | mps
    half: bool = True


@dataclass
class DetectorCfg:
    type: str = "classical"  # classical | heatmap | fused
    classical: ClassicalCfg = field(default_factory=ClassicalCfg)
    heatmap: HeatmapCfg = field(default_factory=HeatmapCfg)
    fuse_radius: float = 3.0


@dataclass
class TrackerCfg:
    q_accel: float = 2000.0  # white-acceleration spectral density (px^2/s^3)
    init_vel_sigma: float = 400.0  # px/s, prior velocity spread of a new track
    meas_sigma_floor: float = 0.7  # px
    gate_chi2: float = 16.0  # Mahalanobis gate (2 dof)
    max_gate_px: float = 120.0
    confirm_hits: int = 4
    confirm_window: int = 6
    max_tentative_misses: int = 2
    max_coast_s: float = 0.5
    high_conf: float = 0.5  # detections at/above -> stage 1; below -> stage 2 (existing tracks only)
    history_len: int = 900  # points kept per track
    min_dt: float = 1e-3


@dataclass
class ClassifyCfg:
    min_obs: int = 6
    static_speed_px_s: float = 3.0  # |v| below this relative to the camera -> vehicle-fixed
    vehicle_near_norm: float = 0.03  # a fixed point counts as a vehicle feature within this distance of the ship (x width)
    bg_rel_speed_px_s: float = 4.0  # |v - v_bg| below this -> moves with the Earth
    defocus_sigma_px: float = 3.5  # PSF sigma above this suggests a near-field (defocused) object
    fast_rel_speed_px_s: float = 150.0  # relative speed above this suggests near-field
    flicker_cv: float = 0.45  # flux coefficient of variation above this suggests tumbling particle
    payload_max_speed_px_s: float = 80.0
    payload_max_curv_px: float = 2.0  # RMS residual (px) of a quadratic fit allowed for a payload
    payload_door_radius: float = 0.15  # normalised distance from door point at first sighting
    margin: float = 0.15  # winning score must exceed the runner-up by this to leave "unknown"
    model_path: str | None = None  # optional trained gradient-boosting model (joblib)


@dataclass
class DeploymentCfg:
    door_xy: list[float] | None = None  # normalised image position of the payload door (per shot)
    min_track_s: float = 1.0
    min_payload_passes: int = 3  # consecutive "payload" classifications before a release event is logged


@dataclass
class IdentifyCfg:
    slow_loop_hz: float = 2.0
    p_unknown_prior: float = 0.5
    accept_posterior: float = 0.95
    accept_chi2: float = 13.3  # 4 dof, 99%
    min_track_obs: int = 15
    attitude_sigma_deg: float = 1.0
    ephem_sigma_km: float = 2.0  # relative position uncertainty (ship vs object), 1-sigma
    rate_sigma_deg_s: float = 0.05
    time_sigma_s: float = 0.5
    max_rate_deg_s: float = 20.0  # extent of the "unknown" rate density
    release_time_sigma_s: float = 10.0  # delay between release and first sighting (1-sigma)
    object_size_m: float = 10.0  # assumed size of catalogue objects for the size-consistency gate
    max_range_km: float = 2000.0  # only objects that come this close are cached for association
    max_bias_sigma_deg: float = 2.0  # no acceptance if the predicted direction is more uncertain than this
    max_rate_sigma_deg_s: float = 0.3  # ... or the predicted angular rate is more uncertain than this
    max_chance_matches: float = 0.01  # expected unrelated tracks fitting the accepted object by chance
    psf_sigma_px: float = 1.2  # optical PSF sigma of an unresolved point (native px)


@dataclass
class CameraCfg:
    hfov_deg: float = 90.0  # assumption until calibrated (see docs/FEASIBILITY.md)
    k1: float = 0.0
    k2: float = 0.0
    cx: float | None = None  # native pixels; None -> image centre
    cy: float | None = None
    # attitude of the camera in the ship RIC frame as yaw/pitch/roll (deg); None = uncalibrated
    attitude_ypr_deg: list[float] | None = None
    calibration_file: str | None = None  # JSON from `rso calibrate`


@dataclass
class OverlayCfg:
    trail_len: int = 40
    show_tentative: bool = False
    min_track_s: float = 0.3  # hide confirmed tracks younger than this (declutters noise-linked tracks)
    label_min_s: float = 1.0  # unknown tracks get a text label only after this age
    show_categories: list[str] = field(
        default_factory=lambda: [
            "unknown",
            "near_field_particle",
            "payload_candidate",
            "background_feature",
            "vehicle_feature",
            "artifact",
        ]
    )
    panel_width: int = 420
    font_scale: float = 0.5
    max_panel_rows: int = 18
    display_max_width: int = 1600  # the live window is shrunk to this width (the saved video is not)


@dataclass
class LaunchSiteCfg:
    lat_deg: float = 25.997
    lon_deg: float = -97.157
    alt_km: float = 0.0


@dataclass
class OrbitNominalCfg:
    perigee_km: float = 262.0
    apogee_km: float = 277.0
    inc_deg: float = 30.5
    insertion_met_s: float = 1520.0
    launch_pass: str = "descending"  # descending: launch azimuth > 90 deg (ESE)
    # inertial argument-of-latitude advance from the launch site to the ship at SECO (T+8:11);
    # None -> 14 deg assumption, with the phase uncertainty below
    downrange_deg_at_seco: float | None = None
    phase_sigma_deg: float = 5.0


@dataclass
class DeploymentScheduleCfg:
    first_met_s: float | None = None
    last_met_s: float | None = None
    count: int = 0


@dataclass
class PayloadGroupCfg:
    name: str = ""
    intdes: str | None = None
    norad_range: list[int] | None = None  # inclusive [first, last]; candidate only until verified
    name_contains: str | None = None


@dataclass
class MissionCfg:
    name: str = "unspecified"
    liftoff_utc: str | None = None  # ISO 8601, e.g. 2026-09-28T12:48:59Z
    launch_site: LaunchSiteCfg = field(default_factory=LaunchSiteCfg)
    orbit_nominal: OrbitNominalCfg = field(default_factory=OrbitNominalCfg)
    deployment: DeploymentScheduleCfg = field(default_factory=DeploymentScheduleCfg)
    payload_group: PayloadGroupCfg = field(default_factory=PayloadGroupCfg)
    ship_ephemeris: str = "nominal"  # nominal | omm:<path> | group_centroid
    catalog_file: str | None = None  # OMM JSON used for association / screening


@dataclass
class TimeAnchorCfg:
    video_s: float = 0.0
    met_s: float = 0.0
    sigma_s: float = 0.5


@dataclass
class TimemapCfg:
    anchors: list[TimeAnchorCfg] = field(default_factory=list)
    clock_readings_csv: str | None = None  # from `rso ocr`
    broadcast_delay_s: float = 0.0  # overlay clock minus exposure time, if known


@dataclass
class HudCfg:
    clock: list[float] | None = None  # normalised [x0, y0, x1, y1]
    speed: list[float] | None = None
    altitude: list[float] | None = None


@dataclass
class LoggingCfg:
    out_dir: str = "runs"
    write_video: bool = True
    jsonl: bool = True


@dataclass
class Config:
    processing: ProcessingCfg = field(default_factory=ProcessingCfg)
    realtime: RealtimeCfg = field(default_factory=RealtimeCfg)
    masks: MaskCfg = field(default_factory=MaskCfg)
    registration: RegistrationCfg = field(default_factory=RegistrationCfg)
    scenecut: SceneCutCfg = field(default_factory=SceneCutCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    classify: ClassifyCfg = field(default_factory=ClassifyCfg)
    deployment: DeploymentCfg = field(default_factory=DeploymentCfg)
    identify: IdentifyCfg = field(default_factory=IdentifyCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    overlay: OverlayCfg = field(default_factory=OverlayCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)
    timemap: TimemapCfg = field(default_factory=TimemapCfg)
    hud: HudCfg = field(default_factory=HudCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        """Stable short hash of the full configuration (for run provenance)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# ------------------------------------------------------------------ machinery
class ConfigError(ValueError):
    pass


def _strip_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _build(tp: Any, value: Any, path: str) -> Any:
    tp = _strip_optional(tp)
    if value is None:
        return None
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        hints = typing.get_type_hints(tp)
        names = {f.name for f in dataclasses.fields(tp)}
        unknown = set(value) - names
        if unknown:
            raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
        kwargs = {k: _build(hints[k], v, f"{path}.{k}") for k, v in value.items()}
        return tp(**kwargs)
    origin = typing.get_origin(tp)
    if origin is list:
        (inner,) = typing.get_args(tp) or (Any,)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        return [_build(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if tp is int and isinstance(value, bool):
        raise ConfigError(f"{path}: expected int, got bool")
    if tp is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if tp in (int, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}, got {value!r}")
    return value


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_override(d: dict, item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"override {item!r} must look like a.b.c=value")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    parts = key.strip().split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"override {item!r}: {p} is not a mapping")
    cur[parts[-1]] = value


def config_from_dict(d: dict) -> Config:
    return _build(Config, d, "config")


def load_config(paths: list[str | Path] | None = None, overrides: list[str] | None = None) -> Config:
    merged: dict = {}
    for p in paths or []:
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        merged = _deep_merge(merged, data)
    for item in overrides or []:
        _apply_override(merged, item)
    return config_from_dict(merged)


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
