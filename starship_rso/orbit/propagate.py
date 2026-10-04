"""Vectorised SGP4 propagation of many OMM records to many times (TEME, km, km/s)."""

from __future__ import annotations

import numpy as np
from sgp4.api import SatrecArray

from .omm import OMMRecord, to_satrec
from .timeutil import utc_to_jd


class Propagator:
    def __init__(self, records: list[OMMRecord]):
        self.records = list(records)
        self.ids = [r.key for r in self.records]
        self.names = [r.name for r in self.records]
        self._sats = [to_satrec(r) for r in self.records]
        self._arr = SatrecArray(self._sats) if self._sats else None

    def __len__(self) -> int:
        return len(self.records)

    def states(self, utc, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (r, v, ok) with shapes (n_obj, n_t, 3), (n_obj, n_t, 3), (n_obj, n_t).

        ``ok`` is False where SGP4 reported an error (e.g. decayed); r/v are NaN there.
        """
        t = np.atleast_1d(np.asarray(utc, dtype=np.float64))
        n = len(self._sats)
        if n == 0:
            return np.zeros((0, t.size, 3)), np.zeros((0, t.size, 3)), np.zeros((0, t.size), bool)
        jd, fr = utc_to_jd(t)
        rs, vs, es = [], [], []
        for k in range(0, n, chunk):
            arr = SatrecArray(self._sats[k : k + chunk]) if n > chunk else self._arr
            e, r, v = arr.sgp4(jd, fr)
            rs.append(r)
            vs.append(v)
            es.append(e)
        r = np.concatenate(rs, 0)
        v = np.concatenate(vs, 0)
        ok = np.concatenate(es, 0) == 0
        r[~ok] = np.nan
        v[~ok] = np.nan
        return r, v, ok

    def state_one(self, idx: int, utc) -> tuple[np.ndarray, np.ndarray]:
        t = np.atleast_1d(np.asarray(utc, dtype=np.float64))
        jd, fr = utc_to_jd(t)
        e, r, v = self._sats[idx].sgp4_array(jd, fr)
        r = np.asarray(r, float)
        v = np.asarray(v, float)
        r[np.asarray(e) != 0] = np.nan
        v[np.asarray(e) != 0] = np.nan
        return r, v
