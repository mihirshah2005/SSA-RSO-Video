"""Multi-target point tracker: Kalman CV + Hungarian assignment + ByteTrack-style second stage.

* Stage 1: high-confidence detections vs every live track (confirmed, lost,
  tentative), cost ``d^2 + ln|S|`` with a Mahalanobis gate and a pixel cap.
* Stage 2: remaining low-confidence detections vs remaining confirmed/lost
  tracks only, so faint responses can extend an established track but never
  start one.
* Unmatched high-confidence detections start tentative tracks; a tentative
  track is confirmed after ``confirm_hits`` hits within ``confirm_window``
  processed frames. Confirmed tracks coast (state LOST) for at most
  ``max_coast_s``; a coasted position is a prediction, never an observation.
* A camera cut kills every track: image-plane state does not survive a cut.

Display ids are assigned only at confirmation, so the visible numbering has
no gaps from short-lived tentatives. A track id is never a satellite id.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import TrackerCfg
from ..types import (
    CategoryDecision,
    Detection,
    FrameInfo,
    IdentityDecision,
    TrackPoint,
    TrackState,
)
from .kalman import KalmanCV

_INF = 1e9


@dataclass
class Track:
    uid: int
    kf: KalmanCV
    shot_id: int
    created_t: float
    created_frame: int
    display_id: int | None = None
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    misses: int = 0
    last_t: float = 0.0  # time of the last *observed* update
    last_pred_t: float = 0.0  # time the filter state refers to
    hit_window: deque = field(default_factory=lambda: deque(maxlen=16))
    history: list[TrackPoint] = field(default_factory=list)
    category: CategoryDecision = field(default_factory=CategoryDecision)
    identity: IdentityDecision = field(default_factory=IdentityDecision)  # combined, what is displayed
    identity_geo: IdentityDecision = field(default_factory=IdentityDecision)  # catalogue geometry
    identity_release: IdentityDecision | None = None  # release-event route (payloads only)
    features: dict = field(default_factory=dict)
    death_reason: str = ""
    first: TrackPoint | None = None  # first observation, kept even when history is trimmed
    n_obs: int = 0
    payload_streak: int = 0  # consecutive classification passes that returned "payload"

    @property
    def label(self) -> str:
        return f"T{self.display_id:03d}" if self.display_id is not None else f"t{self.uid}"

    @property
    def is_live(self) -> bool:
        return self.state != TrackState.DEAD

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.LOST)

    def observed_points(self) -> list[TrackPoint]:
        return [p for p in self.history if p.detection is not None]

    @property
    def first_t(self) -> float:
        return self.first.t if self.first is not None else self.created_t


class Tracker:
    """``on_finish(track)`` is called when a confirmed track dies; with ``keep_finished=False``
    dead tracks are not retained (long live runs stream them to disk instead)."""

    def __init__(self, cfg: TrackerCfg, on_finish=None, keep_finished: bool = True):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self.on_finish = on_finish
        self.keep_finished = keep_finished
        self.n_finished = 0
        self._next_uid = 1
        self._next_display = 1
        self._frames_seen = 0

    # ------------------------------------------------------------- lifecycle
    def reset(self, reason: str = "reset") -> None:
        for tr in self.tracks:
            self._kill(tr, reason)
        self.tracks = []

    def _kill(self, tr: Track, reason: str) -> None:
        tr.state = TrackState.DEAD
        tr.death_reason = reason
        if tr.display_id is not None:
            self.n_finished += 1
            if self.on_finish is not None:
                self.on_finish(tr)
            if self.keep_finished:
                self.finished.append(tr)

    def live_tracks(self, confirmed_only: bool = False) -> list[Track]:
        return [t for t in self.tracks if t.is_live and (t.is_confirmed or not confirmed_only)]

    def all_confirmed(self) -> list[Track]:
        """Finished + live confirmed tracks (for export)."""
        return self.finished + [t for t in self.tracks if t.display_id is not None]

    # ----------------------------------------------------------------- core
    def _r(self, d: Detection) -> float:
        return d.pos_sigma**2 + self.cfg.meas_sigma_floor**2

    def _cost_matrix(self, tracks: list[Track], dets: list[Detection]) -> np.ndarray:
        """Vectorised gated cost ``d^2 + ln|S|`` (Mahalanobis gate and pixel cap)."""
        c = self.cfg
        if not tracks or not dets:
            return np.full((len(tracks), len(dets)), _INF)
        Z = np.array([[d.x, d.y] for d in dets])
        R = np.array([self._r(d) for d in dets])[None, :]
        X = np.array([t.kf.x[:2] for t in tracks])
        P = np.array([t.kf.P[:2, :2] for t in tracks])
        nu = Z[None, :, :] - X[:, None, :]
        a = P[:, None, 0, 0] + R
        dd = P[:, None, 1, 1] + R
        b = P[:, None, 0, 1]
        det = a * dd - b * b
        safe = np.where(det > 0, det, 1.0)
        d2 = (dd * nu[..., 0] ** 2 - 2 * b * nu[..., 0] * nu[..., 1] + a * nu[..., 1] ** 2) / safe
        ok = (np.hypot(nu[..., 0], nu[..., 1]) <= c.max_gate_px) & (det > 0) & (d2 <= c.gate_chi2)
        return np.where(ok, d2 + np.log(safe), _INF)

    @staticmethod
    def _assign(C: np.ndarray) -> list[tuple[int, int]]:
        if C.size == 0:
            return []
        rows, cols = linear_sum_assignment(C)
        return [(int(r), int(k)) for r, k in zip(rows, cols) if C[r, k] < _INF / 2]

    def update(
        self, dets: list[Detection], frame: FrameInfo, bg_flow=None
    ) -> list[Track]:
        """Advance all tracks to ``frame.t`` and associate ``dets``.

        ``bg_flow(x, y) -> (vx, vy) | None`` gives the background image velocity
        used later for classification. Returns the live tracks.
        """
        c = self.cfg
        self._frames_seen += 1
        if frame.is_cut:
            self.reset("camera cut")
        for tr in self.tracks:
            dt = frame.t - tr.last_pred_t
            if dt > c.min_dt:
                tr.kf.predict(dt)
                tr.last_pred_t = frame.t

        high = [d for d in dets if d.confidence >= c.high_conf]
        low = [d for d in dets if d.confidence < c.high_conf]
        live = [t for t in self.tracks if t.is_live]

        matches: list[tuple[Track, Detection]] = []
        # stage 1
        C1 = self._cost_matrix(live, high)
        m1 = self._assign(C1)
        used_t = {i for i, _ in m1}
        used_d = {j for _, j in m1}
        matches += [(live[i], high[j]) for i, j in m1]
        # stage 2: low-confidence detections extend established tracks only
        rest = [t for k, t in enumerate(live) if k not in used_t and t.is_confirmed]
        if rest and low:
            for i, j in self._assign(self._cost_matrix(rest, low)):
                matches.append((rest[i], low[j]))
        matched_tracks = {id(t) for t, _ in matches}

        for tr, d in matches:
            tr.kf.update(np.array([d.x, d.y]), self._r(d))
            tr.hits += 1
            tr.misses = 0
            tr.last_t = frame.t
            tr.hit_window.append(self._frames_seen)
            self._append_point(tr, frame, d, bg_flow)
            if tr.state == TrackState.LOST:
                tr.state = TrackState.CONFIRMED

        for tr in live:
            if id(tr) in matched_tracks:
                continue
            tr.misses += 1
            if tr.state == TrackState.TENTATIVE:
                if tr.misses > c.max_tentative_misses:
                    self._kill(tr, "tentative expired")
                continue
            tr.state = TrackState.LOST
            if frame.t - tr.last_t > c.max_coast_s:
                self._kill(tr, "coast limit")
            else:
                self._append_point(tr, frame, None, bg_flow)

        # confirmation
        for tr in live:
            if tr.state == TrackState.TENTATIVE and tr.is_live:
                # count processed frames, not source indices: the live reader may drop frames
                recent = [f for f in tr.hit_window if f > self._frames_seen - c.confirm_window]
                if len(recent) >= c.confirm_hits:
                    tr.state = TrackState.CONFIRMED
                    tr.display_id = self._next_display
                    self._next_display += 1

        # births
        for j, d in enumerate(high):
            if j in used_d:
                continue
            tr = Track(
                uid=self._next_uid,
                kf=KalmanCV(d.x, d.y, np.sqrt(self._r(d)), c.init_vel_sigma, c.q_accel),
                shot_id=frame.shot_id,
                created_t=frame.t,
                created_frame=frame.index,
                last_t=frame.t,
                last_pred_t=frame.t,
            )
            tr.hit_window.append(self._frames_seen)
            self._next_uid += 1
            self._append_point(tr, frame, d, bg_flow)
            self.tracks.append(tr)

        self.tracks = [t for t in self.tracks if t.is_live]
        return self.tracks

    def _append_point(self, tr: Track, frame: FrameInfo, d: Detection | None, bg_flow) -> None:
        x, y, vx, vy = (float(v) for v in tr.kf.x)
        flow = bg_flow(x, y) if bg_flow is not None else None
        pt = TrackPoint(frame.index, frame.t, x, y, vx, vy, d, flow, frame.met, frame.utc,
                        self._frames_seen, frame.time_sigma)
        tr.history.append(pt)
        if d is not None:
            tr.n_obs += 1
            if tr.first is None:
                tr.first = pt
        if len(tr.history) > self.cfg.history_len:
            del tr.history[: len(tr.history) - self.cfg.history_len]
