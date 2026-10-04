"""Append-only JSON-lines logs and run directories with provenance."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "value"):
        return o.value
    return str(o)


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_default, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def git_revision(repo: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo or Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo or Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return (rev or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_run_dir(base: str | Path, tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(base) / f"{stamp}_{tag}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def provenance(cfg_digest: str, extra: dict | None = None) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(),
        "config_digest": cfg_digest,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wall_start": time.time(),
        **(extra or {}),
    }
