"""Time scales used by the orbit code.

UTC is carried as POSIX seconds. SGP4 takes Julian dates split into whole and
fractional parts for precision. UT1 is approximated by UTC (|UT1-UTC| < 0.9 s,
about 0.004 deg of Earth rotation) which is far below the other error terms
in this project.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

JD_UNIX_EPOCH = 2440587.5
JD_J2000 = 2451545.0
JD_SGP4_EPOCH0 = 2433281.5  # 1949-12-31 00:00 UT, epoch origin of sgp4init


def utc_to_jd(utc) -> tuple[np.ndarray, np.ndarray]:
    """POSIX seconds -> (jd_whole, jd_fraction) with jd_whole ending in .5."""
    u = np.asarray(utc, dtype=np.float64)
    days = u / 86400.0
    whole = np.floor(days)
    jd = whole + JD_UNIX_EPOCH
    fr = days - whole
    return jd, fr


def jd_to_utc(jd, fr=0.0):
    return (np.asarray(jd, dtype=np.float64) - JD_UNIX_EPOCH + np.asarray(fr, dtype=np.float64)) * 86400.0


def gmst_rad(utc) -> np.ndarray:
    """Greenwich mean sidereal time (IAU 1982), radians, for UT1 ~ UTC."""
    jd, fr = utc_to_jd(utc)
    tut1 = ((jd - JD_J2000) + fr) / 36525.0
    sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 0.093104 * tut1**2
        - 6.2e-6 * tut1**3
    )
    return np.mod(np.deg2rad(sec / 240.0), 2 * np.pi)


def parse_epoch(text: str) -> float:
    """OMM epoch string (with or without fractional seconds / 'Z') -> POSIX seconds UTC."""
    s = str(text).strip().replace("Z", "")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"cannot parse epoch {text!r}")


def iso(utc: float) -> str:
    return datetime.fromtimestamp(float(utc), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def sgp4_epoch_days(utc: float) -> float:
    jd, fr = utc_to_jd(utc)
    return float(jd - JD_SGP4_EPOCH0 + fr)
