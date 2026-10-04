"""Map video presentation time -> mission elapsed time (MET) -> UTC.

Anchors are ``(video_s, met_s, sigma_s)`` pairs. They come from a manual
reading of the on-screen T+ clock or from :func:`anchors_from_clock_readings`,
which locates the instants where the OCR'd integer clock ticks over (sub-second
accuracy, about one frame).

Between anchors of the same continuous segment the map is linear; outside it
extrapolates with slope 1 (the clock runs in real time) from the nearest
anchor. A segment boundary is inferred where neighbouring anchors disagree
with slope 1 by more than ``break_tol_s`` (replays, edits, freezes). The
overlay clock is not the exposure time: ``broadcast_delay_s`` (overlay minus
exposure) is subtracted when known and the uncertainty is always reported.
"""

from __future__ import annotations

import bisect
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_CLOCK_RE = re.compile(r"T\s*([+-])\s*(\d{1,2}):(\d{2}):(\d{2})")


def parse_clock(text: str) -> float | None:
    """Parse ``T+00:49:34`` / ``T-00:00:10`` into signed seconds (None if no match)."""
    m = _CLOCK_RE.search(text.replace(" ", ""))
    if not m:
        return None
    sign = 1.0 if m.group(1) == "+" else -1.0
    h, mi, s = (int(m.group(i)) for i in (2, 3, 4))
    if mi >= 60 or s >= 60:
        return None
    return sign * (h * 3600 + mi * 60 + s)


def format_met(met: float | None) -> str:
    if met is None or not np.isfinite(met):
        return "T? --:--:--"
    sign = "+" if met >= 0 else "-"
    m = abs(met)
    h, rem = divmod(int(m), 3600)
    mi, s = divmod(rem, 60)
    return f"T{sign}{h:02d}:{mi:02d}:{s:02d}"


def parse_utc(text: str) -> float:
    """ISO-8601 UTC string -> POSIX seconds."""
    s = text.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def format_utc(utc: float | None) -> str:
    if utc is None:
        return "UTC unknown"
    return datetime.fromtimestamp(utc, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True)
class Anchor:
    video_s: float
    met_s: float
    sigma_s: float = 0.5


class TimeMap:
    def __init__(
        self,
        anchors: list[Anchor] | None = None,
        liftoff_utc: float | None = None,
        broadcast_delay_s: float = 0.0,
        break_tol_s: float = 1.5,
        drift_per_s: float = 2e-4,
    ):
        self.anchors = sorted(anchors or [], key=lambda a: a.video_s)
        self.liftoff_utc = liftoff_utc
        self.broadcast_delay_s = float(broadcast_delay_s)
        self.drift_per_s = drift_per_s
        # segment id per anchor: a new segment starts where slope-1 continuity breaks
        self._seg: list[int] = []
        seg = 0
        for i, a in enumerate(self.anchors):
            if i > 0:
                p = self.anchors[i - 1]
                if abs((a.met_s - p.met_s) - (a.video_s - p.video_s)) > break_tol_s:
                    seg += 1
            self._seg.append(seg)
        self._vs = [a.video_s for a in self.anchors]

    @property
    def is_calibrated(self) -> bool:
        return bool(self.anchors)

    def met(self, t: float) -> tuple[float | None, float | None]:
        """Return (met_s, sigma_s) at video time ``t``; (None, None) without anchors."""
        if not self.anchors:
            return None, None
        i = bisect.bisect_right(self._vs, t)
        lo = self.anchors[i - 1] if i > 0 else None
        hi = self.anchors[i] if i < len(self.anchors) else None
        if lo is not None and hi is not None and self._seg[i - 1] == self._seg[i] and hi.video_s > lo.video_s:
            w = (t - lo.video_s) / (hi.video_s - lo.video_s)
            met = lo.met_s + w * (hi.met_s - lo.met_s)
            sig = max(lo.sigma_s, hi.sigma_s)
        else:
            ref = lo if (lo is not None and (hi is None or t - lo.video_s <= hi.video_s - t)) else hi
            assert ref is not None
            met = ref.met_s + (t - ref.video_s)
            sig = ref.sigma_s + self.drift_per_s * abs(t - ref.video_s)
        return met - self.broadcast_delay_s, sig

    def utc(self, t: float) -> tuple[float | None, float | None]:
        met, sig = self.met(t)
        if met is None or self.liftoff_utc is None:
            return None, None
        return self.liftoff_utc + met, sig

    def segment_of(self, t: float) -> int:
        if not self.anchors:
            return 0
        i = bisect.bisect_right(self._vs, t)
        return self._seg[max(0, i - 1)]

    @classmethod
    def from_config(cls, tm_cfg, liftoff_utc_text: str | None) -> "TimeMap":
        anchors = [Anchor(a.video_s, a.met_s, a.sigma_s) for a in tm_cfg.anchors]
        if tm_cfg.clock_readings_csv:
            anchors += anchors_from_clock_readings(load_clock_readings(tm_cfg.clock_readings_csv))
        liftoff = parse_utc(liftoff_utc_text) if liftoff_utc_text else None
        return cls(anchors, liftoff, tm_cfg.broadcast_delay_s)


def load_clock_readings(path: str | Path) -> list[tuple[float, float]]:
    """Read ``video_s,met_s`` rows (``met_s`` may be empty when OCR failed)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("met_s") not in (None, ""):
                rows.append((float(r["video_s"]), float(r["met_s"])))
    return rows


def anchors_from_clock_readings(
    readings: list[tuple[float, float]], max_gap_s: float = 1.0, min_support: int = 3, tol_s: float = 0.6
) -> list[Anchor]:
    """Turn sampled integer clock readings into anchors at tick instants.

    A tick is a pair of consecutive readings ``(t0, m)`` and ``(t1, m + 1)`` with
    ``t1 - t0 <= max_gap_s``; the tick happened in ``(t0, t1]`` so the anchor is
    ``(midpoint, m + 1)`` with sigma ``(t1 - t0) / sqrt(12)`` (uniform). Ticks
    are grouped into segments of consistent offset; isolated ticks inconsistent
    with their neighbours (OCR errors) are discarded. Each segment contributes
    its first and last tick, using the segment-median offset.
    """
    r = sorted(readings)
    ticks: list[tuple[float, float, float]] = []  # (t_mid, met, sigma)
    for (t0, m0), (t1, m1) in zip(r, r[1:]):
        if 0 < t1 - t0 <= max_gap_s and abs((m1 - m0) - 1.0) < 1e-6:
            ticks.append((0.5 * (t0 + t1), m1, (t1 - t0) / np.sqrt(12.0)))
    if not ticks:
        return []
    offsets = np.array([m - t for t, m, _ in ticks])
    segments: list[list[int]] = [[0]]
    for i in range(1, len(ticks)):
        if abs(offsets[i] - offsets[segments[-1][-1]]) <= tol_s:
            segments[-1].append(i)
        else:
            segments.append([i])
    anchors: list[Anchor] = []
    for seg in segments:
        if len(seg) < min_support:
            continue
        off = float(np.median(offsets[seg]))
        # sampling at a fixed rate is phase-locked to the clock ticks, so tick errors are correlated:
        # do not shrink by sqrt(N); one sampling interval / sqrt(12) is the honest floor
        sig = float(max(np.median([ticks[i][2] for i in seg]), 0.02))
        for i in (seg[0], seg[-1]):
            t = ticks[i][0]
            anchors.append(Anchor(t, t + off, sig))
    return sorted(set(anchors), key=lambda a: a.video_s)
