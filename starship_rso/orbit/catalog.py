"""Catalogue access (CelesTrak, Space-Track) with a local, versioned cache.

Rules that keep this honest and polite:

* Downloads happen in a separate preparation step (``rso fetch-catalog``),
  never per video frame. The live pipeline only reads the cache.
* CelesTrak updates every 2 hours and blocks repeat downloads; a cached
  response younger than ``min_age_s`` is reused, and any HTTP error stops the
  run (no retry loops).
* Space-Track needs a free account (``SPACETRACK_USER`` /
  ``SPACETRACK_PASSWORD`` environment variables). Its documented limits are
  < 30 requests/minute and < 300/hour; this client sleeps to stay well under.
* Every cached file records the query, source and retrieval time, so a result
  can say which data were available when.

Element sets for the past: use Space-Track ``gp_history`` for the epochs
around the video, or CelesTrak ``gp-first.php`` for the first elements of a
launch. A *current* element set back-propagated days or weeks is not a
historical record and is flagged as such.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .omm import OMMRecord, record_from_omm, save_omm_json
from .timeutil import parse_epoch

log = logging.getLogger(__name__)

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_GP_FIRST = "https://celestrak.org/NORAD/elements/gp-first.php"
CELESTRAK_SUPGP = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php"
CELESTRAK_SATCAT = "https://celestrak.org/satcat/records.php"
SPACETRACK_BASE = "https://www.space-track.org"


class CatalogError(RuntimeError):
    pass


def _cache_path(cache_dir: Path, url: str, params: dict) -> Path:
    key = json.dumps([url, sorted(params.items())], sort_keys=True)
    return cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".json")


def _read_cache(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _write_cache(path: Path, url: str, params: dict, payload, source: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "meta": {"url": url, "params": params, "source": source, "retrieved_utc": time.time()},
        "records": payload,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    return blob


@dataclass
class CelesTrakClient:
    cache_dir: Path = Path("data/catalog_cache")
    min_age_s: float = 2 * 3600.0
    timeout_s: float = 60.0
    offline: bool = False

    def _get(self, url: str, params: dict, source: str) -> list[OMMRecord]:
        params = {**params, "FORMAT": "JSON"}
        path = _cache_path(Path(self.cache_dir), url, params)
        cached = _read_cache(path)
        if cached is not None and (self.offline or time.time() - cached["meta"]["retrieved_utc"] < self.min_age_s):
            log.info("using cached %s (%s)", path.name, cached["meta"]["url"])
            blob = cached
        else:
            if self.offline:
                raise CatalogError(f"offline and no cache for {url} {params}")
            import requests

            resp = requests.get(url, params=params, timeout=self.timeout_s, headers={"User-Agent": "starship-rso/0.1 (FYP research)"})
            if resp.status_code != 200:
                raise CatalogError(f"CelesTrak HTTP {resp.status_code} for {resp.url}: stopping (no retries)")
            text = resp.text.strip()
            if not text.startswith("["):
                raise CatalogError(f"CelesTrak returned no JSON for {resp.url}: {text[:200]!r}")
            blob = _write_cache(path, url, params, json.loads(text), source)
        ret = blob["meta"]["retrieved_utc"]
        return [record_from_omm(d, source, ret) for d in blob["records"]]

    def gp(self, **query) -> list[OMMRecord]:
        """``gp(CATNR=...)``, ``gp(INTDES='2026-xxx')``, ``gp(GROUP='starlink')``, ``gp(NAME=...)``."""
        return self._get(CELESTRAK_GP, _one_query(query), "celestrak-gp")

    def gp_first(self, **query) -> list[OMMRecord]:
        return self._get(CELESTRAK_GP_FIRST, _one_query(query), "celestrak-gp-first")

    def supgp(self, **query) -> list[OMMRecord]:
        """``supgp(FILE='starlink')``, ``supgp(INTDES=...)``, ``supgp(SOURCE='SpaceX-E')``."""
        return self._get(CELESTRAK_SUPGP, _one_query(query), "celestrak-supgp")

    def satcat(self, **query) -> list[dict]:
        params = {**_one_query(query), "FORMAT": "JSON"}
        path = _cache_path(Path(self.cache_dir), CELESTRAK_SATCAT, params)
        cached = _read_cache(path)
        if cached is not None:
            return cached["records"]
        if self.offline:
            raise CatalogError("offline and no cached SATCAT response")
        import requests

        resp = requests.get(CELESTRAK_SATCAT, params=params, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise CatalogError(f"CelesTrak SATCAT HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise CatalogError(f"CelesTrak SATCAT returned no JSON: {resp.text[:200]!r}") from exc
        return _write_cache(path, CELESTRAK_SATCAT, params, payload, "celestrak-satcat")["records"]


def _one_query(query: dict) -> dict:
    if len(query) != 1:
        raise ValueError("exactly one query parameter (CATNR, INTDES, GROUP, NAME, FILE, SOURCE, SPECIAL)")
    k, v = next(iter(query.items()))
    return {k.upper(): str(v)}


class SpaceTrackClient:
    """Minimal Space-Track client (login session + rate limiting + cache)."""

    def __init__(self, user: str | None = None, password: str | None = None, cache_dir: Path = Path("data/catalog_cache"),
                 min_interval_s: float = 13.0):
        # 13 s between requests (logins included) keeps under 300/hour and 30/minute
        self.user = user or os.environ.get("SPACETRACK_USER")
        self.password = password or os.environ.get("SPACETRACK_PASSWORD")
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self._session = None
        self._last = 0.0

    def _login(self):
        if (not self.user or not self.password) and sys.stdin is not None and sys.stdin.isatty():
            # prompting keeps the password out of the shell history (and away from shell expansion:
            # zsh rewrites an unquoted "!" in a typed password)
            import getpass

            self.user = self.user or input("Space-Track user (e-mail): ").strip()
            self.password = self.password or getpass.getpass("Space-Track password: ")
        if not self.user or not self.password:
            raise CatalogError("set SPACETRACK_USER and SPACETRACK_PASSWORD (free account at space-track.org)")
        import requests

        self._throttle()
        s = requests.Session()
        r = s.post(f"{SPACETRACK_BASE}/ajaxauth/login", data={"identity": self.user, "password": self.password}, timeout=60)
        self._last = time.time()
        if r.status_code != 200 or "Failed" in r.text:
            raise CatalogError(
                f"Space-Track login failed (HTTP {r.status_code}): wrong user/password, or the account is not "
                "activated yet (log in once at space-track.org). Credentials come from SPACETRACK_USER/"
                "SPACETRACK_PASSWORD if set; 'unset SPACETRACK_USER SPACETRACK_PASSWORD' to be prompted instead.")
        self._session = s

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)

    def query(self, path: str, source: str = "spacetrack", complete_before_utc: float | None = None) -> list[dict]:
        """Run a query (cached). A response is cached permanently only if it cannot change any more,
        i.e. its epoch window ended at least a day before retrieval (``complete_before_utc``)."""
        url = f"{SPACETRACK_BASE}/basicspacedata/query/{path.strip('/')}"
        cache = _cache_path(self.cache_dir, url, {})
        cached = _read_cache(cache)
        if cached is not None and (cached["meta"].get("complete", True)
                                   or time.time() - cached["meta"]["retrieved_utc"] < 6 * 3600):
            return cached["records"]
        if self._session is None:
            self._login()
        self._throttle()
        r = self._session.get(url, timeout=300)
        self._last = time.time()
        if r.status_code != 200:
            raise CatalogError(f"Space-Track HTTP {r.status_code} for {url}: stopping (no retries)")
        blob = _write_cache(cache, url, {}, r.json(), source)
        complete = complete_before_utc is None or complete_before_utc < time.time() - 86400.0
        if not complete:
            blob["meta"]["complete"] = False
            cache.write_text(json.dumps(blob))
        return blob["records"]

    def gp_history(self, norad_first: int, norad_last: int, epoch_from: str, epoch_to: str) -> list[OMMRecord]:
        """All element sets for a NORAD id range with epochs in [epoch_from, epoch_to] (YYYY-MM-DD)."""
        path = (
            f"class/gp_history/NORAD_CAT_ID/{norad_first}--{norad_last}/EPOCH/{epoch_from}--{epoch_to}"
            "/orderby/EPOCH asc/format/json"
        )
        end = parse_epoch(epoch_to)
        return [record_from_omm(d, "spacetrack-gp_history")
                for d in self.query(path, "spacetrack-gp_history", complete_before_utc=end)]

    def gp_history_window(self, epoch_from: str, epoch_to: str) -> list[OMMRecord]:
        """Every catalogued object with an epoch in the window, as ONE query (run once; cached).

        Space-Track asks users not to pull large gp_history sets repeatedly; for anything bigger
        than a few days prefer their yearly bulk files, or the current ``gp`` class when the
        video is recent (and say so in the results).
        """
        path = f"class/gp_history/EPOCH/{epoch_from}--{epoch_to}/orderby/NORAD_CAT_ID asc/format/json"
        end = parse_epoch(epoch_to)
        return [record_from_omm(d, "spacetrack-gp_history")
                for d in self.query(path, "spacetrack-gp_history", complete_before_utc=end)]

    def satcat(self, intdes_prefix: str) -> list[dict]:
        return self.query(f"class/satcat/INTLDES/~~{intdes_prefix}/format/json", "spacetrack-satcat")


# ----------------------------------------------------------------- selection
def select_records(
    records: Iterable[OMMRecord],
    norad_range: list[int] | None = None,
    name_contains: str | None = None,
    intdes: str | None = None,
) -> list[OMMRecord]:
    out = []
    for r in records:
        if norad_range is not None and not (norad_range[0] <= r.norad_id <= norad_range[1]):
            continue
        if name_contains and name_contains.upper() not in r.name.upper():
            continue
        if intdes and not r.object_id.startswith(intdes):
            continue
        out.append(r)
    return out


def best_record_per_object(
    records: Iterable[OMMRecord], t_utc: float, mode: str = "retrospective", publication_lag_s: float = 6 * 3600.0
) -> list[OMMRecord]:
    """Pick one element set per object for time ``t_utc``.

    * ``retrospective``: the epoch closest to t (before or after), labelled as
      possibly using information published after the video.
    * ``as_of``: only sets that existed at t. Space-Track records carry
      ``CREATION_DATE``; CelesTrak records do not, so for them an element set is
      assumed available ``publication_lag_s`` after its epoch (elements are
      published hours after their epoch, never before).
    """
    by_obj: dict[int, list[OMMRecord]] = {}
    for r in records:
        by_obj.setdefault(r.norad_id, []).append(r)
    out = []
    for recs in by_obj.values():
        if mode == "as_of":
            avail = [r for r in recs
                     if (r.creation_utc if r.creation_utc is not None else r.epoch_utc + publication_lag_s) <= t_utc]
            if avail:
                out.append(max(avail, key=lambda r: r.epoch_utc))
        elif mode == "retrospective":
            out.append(min(recs, key=lambda r: abs(r.epoch_utc - t_utc)))
        else:
            raise ValueError(mode)
    return sorted(out, key=lambda r: r.norad_id)


def write_catalog(records: list[OMMRecord], path: str | Path, note: str) -> None:
    meta = {
        "note": note,
        "written_utc": time.time(),
        "sources": sorted({r.source for r in records}),
        "n": len(records),
    }
    save_omm_json(records, path, meta)
