"""OMM (CCSDS Orbit Mean-elements Message) records and SGP4 initialisation.

CelesTrak and Space-Track both serve OMM keywords as JSON/CSV. Catalogue
numbers >= 100000 are not representable in classic TLEs, so this project
never depends on TLE text. ``sgp4init`` accepts satnum < 340000 (Alpha-5);
larger ids are kept in our own record and passed to SGP4 as 0.

Provenance fields matter for honest evaluation: ``epoch`` is when the
elements are valid; ``creation_utc`` (Space-Track ``CREATION_DATE``) is when
they were produced; ``retrieved_utc`` is when we downloaded them. A record is
usable "as of" time t in live mode only if it existed at t.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sgp4.api import WGS72, Satrec

from .timeutil import parse_epoch, sgp4_epoch_days

_DEG = math.pi / 180.0
_XPDOTP = 1440.0 / (2.0 * math.pi)  # rev/day -> rad/min factor


@dataclass
class OMMRecord:
    norad_id: int
    name: str
    object_id: str  # international designator, e.g. 2026-123A
    epoch_utc: float
    fields: dict[str, Any]
    source: str = "unknown"
    creation_utc: float | None = None
    retrieved_utc: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return str(self.norad_id)

    def to_json(self) -> dict[str, Any]:
        d = dict(self.fields)
        d["_source"] = self.source
        if self.retrieved_utc is not None:
            d["_retrieved_utc"] = self.retrieved_utc
        return d


def _f(d: dict, key: str, default: float | None = None) -> float:
    v = d.get(key, default)
    if v is None or v == "":
        if default is None:
            raise KeyError(key)
        return float(default)
    return float(v)


def record_from_omm(d: dict[str, Any], source: str = "unknown", retrieved_utc: float | None = None) -> OMMRecord:
    d = {str(k).upper() if not str(k).startswith("_") else k: v for k, v in d.items()}
    creation = None
    if d.get("CREATION_DATE"):
        try:
            creation = parse_epoch(d["CREATION_DATE"])
        except ValueError:
            creation = None
    return OMMRecord(
        norad_id=int(float(d["NORAD_CAT_ID"])),
        name=str(d.get("OBJECT_NAME", "")).strip(),
        object_id=str(d.get("OBJECT_ID", "") or ""),
        epoch_utc=parse_epoch(d["EPOCH"]),
        fields=d,
        source=str(d.get("_source", source)),
        creation_utc=creation,
        retrieved_utc=d.get("_retrieved_utc", retrieved_utc),
    )


def to_satrec(rec: OMMRecord) -> Satrec:
    d = rec.fields
    sat = Satrec()
    satnum = rec.norad_id if rec.norad_id < 340000 else 0
    sat.sgp4init(
        WGS72,
        "i",
        satnum,
        sgp4_epoch_days(rec.epoch_utc),
        _f(d, "BSTAR", 0.0),
        _f(d, "MEAN_MOTION_DOT", 0.0) / (_XPDOTP * 1440.0),
        _f(d, "MEAN_MOTION_DDOT", 0.0) / (_XPDOTP * 1440.0 * 1440.0),
        _f(d, "ECCENTRICITY"),
        _f(d, "ARG_OF_PERICENTER") * _DEG,
        _f(d, "INCLINATION") * _DEG,
        _f(d, "MEAN_ANOMALY") * _DEG,
        _f(d, "MEAN_MOTION") / _XPDOTP,
        _f(d, "RA_OF_ASC_NODE") * _DEG,
    )
    return sat


def load_omm_json(path: str | Path, source: str | None = None) -> list[OMMRecord]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "records" in data:
        meta = data.get("meta", {})
        items = data["records"]
        src = source or meta.get("source", "file")
        ret = meta.get("retrieved_utc")
    else:
        items = data
        src = source or "file"
        ret = None
    return [record_from_omm(d, src, ret) for d in items]


def save_omm_json(records: list[OMMRecord], path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta or {}, "records": [r.to_json() for r in records]}, fh, indent=1)


def make_omm_fields(
    norad_id: int,
    name: str,
    epoch_iso: str,
    mean_motion_rev_day: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    mean_anom_deg: float,
    bstar: float = 0.0,
    object_id: str = "SIM",
) -> dict[str, Any]:
    """Build an OMM dict (used by the simulator and tests)."""
    return {
        "OBJECT_NAME": name,
        "OBJECT_ID": object_id,
        "EPOCH": epoch_iso,
        "MEAN_MOTION": mean_motion_rev_day,
        "ECCENTRICITY": ecc,
        "INCLINATION": inc_deg,
        "RA_OF_ASC_NODE": raan_deg,
        "ARG_OF_PERICENTER": argp_deg,
        "MEAN_ANOMALY": mean_anom_deg,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": norad_id,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 0,
        "BSTAR": bstar,
        "MEAN_MOTION_DOT": 0.0,
        "MEAN_MOTION_DDOT": 0.0,
    }
