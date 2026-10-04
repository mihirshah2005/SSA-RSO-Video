"""Optional OCR of the webcast HUD: T+ clock, speed (km/h) and altitude (km).

Requires ``pytesseract`` and the ``tesseract`` binary. Without them the
pipeline still runs; supply manual time anchors in the mission config instead.

The HUD layout is broadcaster-specific. Regions are normalised
``[x0, y0, x1, y1]`` boxes in the mission config (``hud.clock`` etc.); check
them with ``rso check-layout`` before trusting any reading. Readings are
measurements with errors, never ground truth: the overlay is asynchronous with
the camera and OCR can misread digits, so :func:`anchors_from_clock_readings`
rejects inconsistent ticks.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from ..io.timemap import parse_clock

log = logging.getLogger(__name__)


def have_tesseract() -> bool:
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def crop_norm(frame: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    xa, xb = int(round(x0 * w)), int(round(x1 * w))
    ya, yb = int(round(y0 * h)), int(round(y1 * h))
    return frame[max(0, ya) : min(h, yb), max(0, xa) : min(w, xb)]


def _prep(crop: np.ndarray, upscale: int = 3) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() < 127:  # white text on dark background -> dark text on white
        b = 255 - b
    return cv2.copyMakeBorder(b, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def _ocr(img: np.ndarray, whitelist: str) -> str:
    import pytesseract  # type: ignore

    cfg = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg).strip()


def read_number(crop: np.ndarray) -> float | None:
    txt = _ocr(_prep(crop), "0123456789.,")
    digits = re.sub(r"[^0-9.]", "", txt.replace(",", ""))
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def read_clock(crop: np.ndarray) -> float | None:
    txt = _ocr(_prep(crop), "T+-0123456789:")
    if not txt.startswith("T"):
        txt = "T" + txt  # the T glyph is often dropped
    return parse_clock(txt)


def read_hud(frame: np.ndarray, hud_cfg) -> dict[str, float | None]:
    out: dict[str, float | None] = {"met_s": None, "speed_kmh": None, "alt_km": None}
    if hud_cfg.clock:
        out["met_s"] = read_clock(crop_norm(frame, hud_cfg.clock))
    if hud_cfg.speed:
        out["speed_kmh"] = read_number(crop_norm(frame, hud_cfg.speed))
    if hud_cfg.altitude:
        out["alt_km"] = read_number(crop_norm(frame, hud_cfg.altitude))
    return out


def ocr_video(source, hud_cfg, every_s: float = 0.2, out_csv: str | Path | None = None) -> list[dict]:
    """OCR the HUD every ``every_s`` seconds of video. Returns rows and optionally writes CSV."""
    if not have_tesseract():
        raise RuntimeError("pytesseract/tesseract not available: pip install pytesseract and install tesseract")
    rows: list[dict] = []
    next_t = -np.inf
    for _, t, frame in source:
        if t + 1e-9 < next_t:
            continue
        next_t = t + every_s
        r = read_hud(frame, hud_cfg)
        rows.append({"video_s": round(t, 4), **r})
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["video_s", "met_s", "speed_kmh", "alt_km"])
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        log.info("wrote %d HUD readings to %s", len(rows), out_csv)
    return rows
