"""Release events: tracks classified as payloads become ``DEPLOY-k`` labels.

The label is a *local* identity. When the mission deployment schedule is
known (first/last release MET and count), the expected release index is
``k = round((MET_first - first_met) / spacing) + 1`` with ``spacing =
(last_met - first_met) / (count - 1)``. This is an index into the release
sequence, not a catalogue number: release order says nothing about NORAD
numbering. Catalogue identities are attached later, only when supported (see
:mod:`identify.deployment_id`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DeploymentScheduleCfg


@dataclass
class ReleaseEvent:
    track_uid: int
    t_first: float  # video time of first sighting
    met_first: float | None
    utc_first: float | None
    k_schedule: int | None
    k_sigma: float | None
    label: str
    time_sigma: float = 1.0  # 1-sigma of met_first/utc_first from the time map
    extra: dict = field(default_factory=dict)


class DeploymentMonitor:
    def __init__(self, schedule: DeploymentScheduleCfg, min_track_s: float = 1.0):
        self.schedule = schedule
        self.min_track_s = min_track_s
        self.events: list[ReleaseEvent] = []
        self._by_track: dict[int, ReleaseEvent] = {}
        self._seq = 0

    @property
    def spacing(self) -> float | None:
        s = self.schedule
        if s.first_met_s is None or s.last_met_s is None or s.count < 2:
            return None
        return (s.last_met_s - s.first_met_s) / (s.count - 1)

    def schedule_index(self, met: float | None, met_sigma: float = 1.0) -> tuple[int | None, float | None]:
        sp = self.spacing
        if sp is None or met is None or self.schedule.first_met_s is None:
            return None, None
        kf = (met - self.schedule.first_met_s) / sp + 1.0
        k = int(np.clip(round(kf), 1, self.schedule.count))
        # a payload is first *seen* after release; allow ~0.3 spacing slack plus timing error
        return k, float(np.hypot(0.3, met_sigma / sp))

    def observe(self, track, met_first: float | None, utc_first: float | None, met_sigma: float = 1.0) -> ReleaseEvent:
        if track.uid in self._by_track:
            return self._by_track[track.uid]
        self._seq += 1
        k, ks = self.schedule_index(met_first, met_sigma)
        label = f"DEPLOY-{k:02d}" if k is not None else f"DEPLOY-L{self._seq:02d}"
        dup = [e for e in self.events if e.k_schedule == k and k is not None]
        if dup:
            n = len(dup)  # two or more tracks claim the same release slot
            label += chr(ord("a") + n) if n < 26 else f"+{n}"
        ev = ReleaseEvent(track.uid, track.first_t, met_first, utc_first, k, ks, label, met_sigma)
        self.events.append(ev)
        self._by_track[track.uid] = ev
        return ev

    def event_for(self, uid: int) -> ReleaseEvent | None:
        return self._by_track.get(uid)
