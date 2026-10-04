"""Data contracts shared by every stage of the pipeline.

Coordinate convention: pixel ``(x, y)`` with ``x`` the column and ``y`` the
row, pixel centres at integer coordinates (OpenCV convention). All pipeline
coordinates are in *processing* pixels; :class:`Scaler` converts to native
video pixels for logs and exports.

Times: ``t`` is video presentation time in seconds. ``met`` is mission
elapsed time in seconds (T+). ``utc`` is POSIX seconds (UTC). The three are
never interchangeable; :mod:`starship_rso.io.timemap` maps between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Category(str, Enum):
    """Physical category of a track. ``UNKNOWN`` is always allowed."""

    UNKNOWN = "unknown"
    NEAR_FIELD_PARTICLE = "near_field_particle"  # ice/frost/vent/debris within metres
    PAYLOAD = "payload_candidate"  # resolved or steady object leaving the payload door
    VEHICLE_FEATURE = "vehicle_feature"  # fixed relative to the camera (glint, tile edge)
    BACKGROUND_FEATURE = "background_feature"  # moves with the Earth (cloud puff, sun glint)
    ARTIFACT = "artifact"  # compression or overlay artefact

    @classmethod
    def parse(cls, value: str) -> "Category":
        for c in cls:
            if c.value == value or c.name.lower() == str(value).lower():
                return c
        raise ValueError(f"unknown category {value!r}")


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"  # confirmed but currently unobserved (coasting)
    DEAD = "dead"


class IdentityStatus(str, Enum):
    """Levels of identity evidence. Display only what the evidence supports."""

    NONE = "none"  # not evaluated (e.g. near-field particle)
    UNAVAILABLE = "identity_unavailable"  # geometry/catalogue cannot support a name
    CANDIDATES = "candidates"  # several plausible catalogue objects
    ACCEPTED = "accepted"  # unique, stable, passes rejection tests
    VERIFIED = "verified"  # independently verified (never set automatically)


@dataclass
class FrameInfo:
    index: int
    t: float  # presentation time (s)
    width: int
    height: int
    met: float | None = None  # mission elapsed time (s)
    utc: float | None = None  # POSIX seconds
    time_sigma: float | None = None  # 1-sigma uncertainty of met/utc (s)
    shot_id: int = 0
    is_cut: bool = False
    available_wall: float | None = None  # wall-clock time the frame became available (live mode)


@dataclass
class Detection:
    x: float
    y: float
    score: float  # SNR-like strength used by the tracker (higher = stronger)
    confidence: float  # score mapped to [0, 1]
    flux: float = 0.0  # background-subtracted integrated intensity
    peak: float = 0.0
    sigma: float = 1.0  # PSF width (px) from second moments
    ellipticity: float = 0.0
    area: int = 1
    residual: float = 0.0  # temporal residual SNR (motion evidence); 0 if not computed
    pos_sigma: float = 0.5  # 1-sigma centroid uncertainty (px)
    polarity: int = 1  # +1 bright-on-background, -1 dark-on-background
    source: str = "classical"
    frame_index: int = -1
    t: float = 0.0

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 4),
            "flux": round(self.flux, 2),
            "peak": round(self.peak, 2),
            "sigma": round(self.sigma, 3),
            "ellipticity": round(self.ellipticity, 3),
            "area": int(self.area),
            "residual": round(self.residual, 3),
            "pos_sigma": round(self.pos_sigma, 3),
            "polarity": self.polarity,
            "source": self.source,
        }


@dataclass
class TrackPoint:
    """One entry of a track history. ``detection`` is None when coasting."""

    frame_index: int
    t: float
    x: float  # filtered position
    y: float
    vx: float
    vy: float
    detection: Detection | None = None
    bg_flow: tuple[float, float] | None = None  # background image velocity (px/s) at (x, y)
    met: float | None = None
    utc: float | None = None
    seq: int = -1  # processed-frame counter (robust to dropped source frames)
    time_sigma: float | None = None  # 1-sigma uncertainty of met/utc (s)


@dataclass
class CategoryDecision:
    category: Category = Category.UNKNOWN
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class IdentityCandidate:
    object_id: str  # NORAD id as string, or synthetic id
    name: str
    log_likelihood: float
    posterior: float
    mahalanobis2: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentityDecision:
    status: IdentityStatus = IdentityStatus.NONE
    label: str = ""  # e.g. DEPLOY-13 or a catalogue name
    candidates: list[IdentityCandidate] = field(default_factory=list)
    p_unknown: float = 1.0
    decided_t: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def best(self) -> IdentityCandidate | None:
        return self.candidates[0] if self.candidates else None


class Scaler:
    """Maps processing pixels <-> native pixels for a resize factor ``scale``.

    Uses the pixel-centre convention of ``cv2.resize``:
    ``x_proc = (x_native + 0.5) * scale - 0.5``.
    """

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)

    def to_native(self, x: float, y: float) -> tuple[float, float]:
        s = self.scale
        return (x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5

    def to_proc(self, x: float, y: float) -> tuple[float, float]:
        s = self.scale
        return (x + 0.5) * s - 0.5, (y + 0.5) * s - 0.5
