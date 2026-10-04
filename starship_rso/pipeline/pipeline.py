"""Causal per-frame pipeline: fast loop (every frame) + slow loop (association, ~1-2 Hz).

Execution order for each decoded frame, using only this and earlier frames:

1. time: video t -> MET/UTC (with uncertainty)
2. cut detection (resets image-plane state)
3. masks (HUD + camera-fixed vehicle structure)
4. background registration against the previous processed frame
5. detection (classical, learned heatmap, or both fused)
6. tracking
7. category evidence for confirmed tracks (every ``classify_every`` frames)
8. slow loop: catalogue association and release-event identities
9. audit record for the frame

An identity update never rewrites history: decisions carry the time they were
made, and the log shows when a name first became supported.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..classify.deployment import DeploymentMonitor
from ..classify.features import track_features
from ..classify.model import TrackCategoryModel
from ..classify.rules import classify_rules
from ..config import Config
from ..identify.association import CatalogAssociator
from ..identify.deployment_id import EventTime, match_release_events
from ..tracking.tracker import Tracker
from ..types import Category, Detection, FrameInfo, IdentityDecision, IdentityStatus, Scaler
from ..vision.classical import ClassicalDetector
from ..vision.masks import MaskManager
from ..vision.ops import resize_frame, to_gray_f32
from ..vision.registration import BackgroundRegistrar, History, Registration
from ..vision.scenecut import SceneCutDetector
from .context import MissionContext

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    info: FrameInfo
    detections: list[Detection]
    tracks: list
    registration: Registration
    valid: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)
    slow_loop_ran: bool = False
    warming_up: bool = False  # vehicle mask not learned yet for this shot: detections withheld


class Pipeline:
    def __init__(self, cfg: Config, native_width: int, native_height: int, ctx: MissionContext,
                 classify_every: int = 5):
        self.cfg = cfg
        self.ctx = ctx
        self.scaler = Scaler(cfg.processing.scale)
        self.native = (native_width, native_height)
        self.w = max(1, int(round(native_width * cfg.processing.scale)))
        self.h = max(1, int(round(native_height * cfg.processing.scale)))
        shape = (self.h, self.w)
        self.masks = MaskManager(cfg.masks, shape)
        self.registrar = BackgroundRegistrar(cfg.registration, shape)
        self.history = History(max(cfg.detector.classical.history + [cfg.detector.heatmap.frames - 1, 1]))
        self.cuts = SceneCutDetector(cfg.scenecut)
        self.detector = self._make_detector(shape)
        self.tracker = Tracker(cfg.tracker)
        self.deploy = DeploymentMonitor(cfg.mission.deployment, cfg.deployment.min_track_s)
        self.model = TrackCategoryModel.load(cfg.classify.model_path) if cfg.classify.model_path else None
        self.associator = CatalogAssociator(cfg.identify, ctx.camera_native, ctx.cache, self.scaler)
        self.classify_every = max(1, classify_every)
        self._prev_u8: np.ndarray | None = None
        self._prev_t: float | None = None
        self._last_slow_t = -np.inf
        self._n = 0
        self._reg_fail_streak = 0
        self.soft_cut_frames = 5
        self.release_ids: dict = {}
        self._warming = True

    def _make_detector(self, shape):
        d = self.cfg.detector
        if d.type == "classical":
            return ClassicalDetector(d.classical, shape)
        from ..vision.heatmap import FusedDetector, HeatmapDetector

        hm = HeatmapDetector(d.heatmap, shape)
        if d.type == "heatmap":
            return hm
        if d.type == "fused":
            return FusedDetector(ClassicalDetector(d.classical, shape), hm, d.fuse_radius)
        raise ValueError(f"unknown detector.type {d.type}")

    # ------------------------------------------------------------------ main
    def process(self, index: int, t: float, frame_bgr: np.ndarray, available_wall: float | None = None) -> FrameResult:
        c = self.cfg
        tm: dict[str, float] = {}
        t_start = time.perf_counter()
        img = resize_frame(frame_bgr, c.processing.scale)
        gray = to_gray_f32(img)
        gray_u8 = gray.astype(np.uint8)
        met, msig = self.ctx.timemap.met(t)
        utc, _ = self.ctx.timemap.utc(t)
        is_cut = self.cuts.update(gray_u8) if self._n > 0 else False
        if not is_cut and self._reg_fail_streak >= self.soft_cut_frames:
            # several registration failures on a textured background: likely a cut between similar
            # views that the histogram test missed; image-plane state cannot be trusted any more
            is_cut = True
            self.cuts.shot_id += 1
            self._reg_fail_streak = 0
        info = FrameInfo(index, t, self.w, self.h, met, utc, msig, self.cuts.shot_id, is_cut, available_wall)
        if is_cut:
            self.masks.reset()
            self.history.clear()
            self.detector.reset()
            self.registrar.reset()
            self._prev_u8 = None
            self._last_slow_t = -np.inf
        tm["prep"] = _ms(t_start)

        t1 = time.perf_counter()
        if self._prev_u8 is not None and self._prev_t is not None:
            # register with the previous frame's mask (the vehicle mask needs this registration)
            reg = self.registrar.estimate(self._prev_u8, gray_u8, self.masks.valid, max(t - self._prev_t, 1e-3))
        else:
            reg = Registration(np.eye(3), False, reason="first frame of shot")
        self.history.advance(reg)
        if self._prev_u8 is not None and not reg.valid and not reg.static_background:
            self._reg_fail_streak += 1
        else:
            self._reg_fail_streak = 0
        tm["register"] = _ms(t1)

        t1 = time.perf_counter()
        valid = self.masks.update(gray, t, reg.H, reg.valid, reg.static_background)
        tm["mask"] = _ms(t1)

        t1 = time.perf_counter()
        # until the vehicle mask exists, the ship's own texture slides against the registered Earth
        # and looks like motion: the detector only fills its frame history and reports nothing
        warming = not self.masks.ready
        if not warming and self._warming:
            # first frame with a vehicle mask: frames stored during warm-up must not offer the
            # background behind the ship's edge to the motion residual
            self.history.restrict(valid)
            self.detector.restrict_history(valid)
        self._warming = warming
        dets = self.detector.detect(gray, self.history.items(), valid, frame_index=index, t=t,
                                    static_background=reg.static_background, report=not warming)
        tm["detect"] = _ms(t1)

        t1 = time.perf_counter()
        if reg.valid:
            flow = reg.flow_at
        elif reg.static_background:
            flow = lambda x, y: (0.0, 0.0)  # noqa: E731 - black sky: background fixed in the image
        else:
            flow = None
        tracks = self.tracker.update(dets, info, flow)
        tm["track"] = _ms(t1)

        t1 = time.perf_counter()
        if self._n % self.classify_every == 0:
            self._classify(tracks, info)
        tm["classify"] = _ms(t1)

        slow = False
        t1 = time.perf_counter()
        period = 1.0 / max(c.identify.slow_loop_hz, 1e-3)
        if t - self._last_slow_t >= period or t < self._last_slow_t:  # also after a backwards time jump
            self._slow_loop(tracks, t)
            self._last_slow_t = t
            slow = True
        tm["identify"] = _ms(t1)

        self.history.push(gray, valid)
        self._prev_u8, self._prev_t = gray_u8, t
        self._n += 1
        tm["total"] = _ms(t_start)
        return FrameResult(info, dets, list(tracks), reg, valid, tm, slow, warming)

    # ------------------------------------------------------------- classify
    def _vehicle_distance(self) -> np.ndarray | None:
        """Distance (processing px, at 1/4 resolution) to the masked ship structure, or None."""
        vm = self.masks.static.mask
        if not self.masks.ready or not vm.any():
            return None
        small = cv2.resize(vm.astype(np.uint8), (max(1, self.w // 4), max(1, self.h // 4)), interpolation=cv2.INTER_NEAREST)
        return cv2.distanceTransform((small == 0).astype(np.uint8), cv2.DIST_L2, 3) * 4.0

    def _classify(self, tracks, info: FrameInfo) -> None:
        door = self.cfg.deployment.door_xy
        vdist = self._vehicle_distance()
        for tr in tracks:
            if not tr.is_confirmed:
                continue
            f = track_features(tr.history, self.w, self.h, door, first=tr.first, n_obs=tr.n_obs,
                               vehicle_dist=vdist)
            tr.features = f
            if self.model is not None:
                dec = self.model.predict(f)
            else:
                dec = classify_rules(f, self.cfg.classify, self.cfg.mission)
            tr.category = dec
            tr.payload_streak = tr.payload_streak + 1 if dec.category == Category.PAYLOAD else 0
            # a release event is logged once and never retracted, so it needs a stable decision
            if (tr.payload_streak >= self.cfg.deployment.min_payload_passes
                    and f["duration_s"] >= self.cfg.deployment.min_track_s):
                door_ok = door is None or (0 <= f["door_dist_norm"] <= self.cfg.classify.payload_door_radius)
                if door_ok and tr.first is not None:
                    first = tr.first
                    tsig = first.time_sigma if first.time_sigma is not None else 1.0
                    self.deploy.observe(tr, first.met, first.utc, tsig)

    # ------------------------------------------------------------ slow loop
    def _slow_loop(self, tracks, t: float) -> None:
        confirmed = [tr for tr in tracks if tr.is_confirmed]
        self.associator.evaluate(confirmed, t)
        # release events: local DEPLOY-k labels; catalogue identity only when supported
        if self.deploy.events:
            rs = self.cfg.identify.release_time_sigma_s
            evs = [EventTime(e.label, e.utc_first, float(np.hypot(e.time_sigma, rs)))
                   for e in self.deploy.events if e.utc_first is not None]
            if evs and self.ctx.release_estimates:
                d = self.cfg.mission.deployment
                window = (d.last_met_s - d.first_met_s + 120.0) if d.first_met_s is not None and d.last_met_s is not None else 600.0
                self.release_ids = match_release_events(evs, self.ctx.release_estimates, window)
        for tr in confirmed:
            ev = self.deploy.event_for(tr.uid)
            if ev is not None:
                dec = self.release_ids.get(ev.label)
                if dec is None:
                    # decided when the release was first seen; stays fixed while nothing better exists
                    dec = IdentityDecision(IdentityStatus.UNAVAILABLE, ev.label, decided_t=ev.t_first,
                                           reasons=[f"local release label (schedule slot {ev.k_schedule})"])
                tr.identity_release = dec
            tr.identity = combine_identities(tr.identity_geo, tr.identity_release, t)


_RANK = {IdentityStatus.VERIFIED: 5, IdentityStatus.ACCEPTED: 4, IdentityStatus.CANDIDATES: 2,
         IdentityStatus.UNAVAILABLE: 1, IdentityStatus.NONE: 0}


def combine_identities(geo: IdentityDecision, rel: IdentityDecision | None, t: float) -> IdentityDecision:
    """Display the strongest supported identity; geometry wins ties (it is per-track evidence).

    A release-route candidate list never hides an accepted geometric identity,
    and a local DEPLOY-k label is kept visible when nothing stronger exists.
    """
    if rel is None:
        return geo
    if _RANK[rel.status] > _RANK[geo.status]:
        out = rel
    elif _RANK[geo.status] >= _RANK[IdentityStatus.CANDIDATES]:
        out = geo
    else:  # neither route names anything: show the local release label
        out = IdentityDecision(rel.status if rel.status != IdentityStatus.NONE else geo.status, rel.label,
                               rel.candidates, rel.p_unknown, rel.decided_t, rel.reasons + geo.reasons)
    if out.decided_t is None:
        out.decided_t = t
    return out


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
