"""Shot list: camera cuts over a long stretch of broadcast, with a contact sheet.

The pipeline resets image-plane state at every cut, and each camera view needs
its own payload-door position and calibration, so the first job on a new video
is to know which views appear when. Frames are sampled (``every``-th frame),
cuts come from the same detector the pipeline uses, and very short shots
(flashes, graphics) are flagged rather than hidden.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..vision.scenecut import SceneCutDetector
from .timemap import TimeMap, format_met
from .video import VideoSource


@dataclass
class Shot:
    shot: int
    video_start: float
    video_end: float
    met_start: float | None
    met_end: float | None
    mean_level: float  # mean grey level (black sky vs Earth vs bright vehicle views)
    short: bool = False

    @property
    def duration(self) -> float:
        return self.video_end - self.video_start


def find_shots(cfg: Config, video: str, start_s: float | None = None, end_s: float | None = None,
               every: int = 3, thumb_width: int = 320, min_shot_s: float = 1.0):
    """Return (shots, thumbnails). One thumbnail per shot, taken one second into the shot."""
    tm = TimeMap.from_config(cfg.timemap, cfg.mission.liftoff_utc)
    det = SceneCutDetector(cfg.scenecut)
    shots: list[Shot] = []
    thumbs: dict[int, np.ndarray] = {}
    cur: dict | None = None
    levels: list[float] = []
    t = None
    for _, t, frame in VideoSource(video, start_s, end_s, stride=every):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cut = det.update(gray) if cur is not None else False
        if cur is None or cut:
            if cur is not None:
                shots.append(_close(cur, t, levels, tm, min_shot_s))
            cur = {"shot": len(shots), "t0": t}
            levels = []
        levels.append(float(gray.mean()))
        if cur["shot"] not in thumbs and t - cur["t0"] >= min(1.0, 0.5 * min_shot_s):
            h, w = frame.shape[:2]
            thumbs[cur["shot"]] = cv2.resize(frame, (thumb_width, int(round(h * thumb_width / w))),
                                             interpolation=cv2.INTER_AREA)
    if cur is not None and t is not None:
        shots.append(_close(cur, t, levels, tm, min_shot_s))
    return shots, thumbs


def _close(cur: dict, t_end: float, levels: list[float], tm: TimeMap, min_shot_s: float) -> Shot:
    m0, _ = tm.met(cur["t0"])
    m1, _ = tm.met(t_end)
    return Shot(cur["shot"], round(cur["t0"], 3), round(t_end, 3), None if m0 is None else round(m0, 2),
                None if m1 is None else round(m1, 2), round(float(np.mean(levels)) if levels else 0.0, 1),
                (t_end - cur["t0"]) < min_shot_s)


def write_shots(shots: list[Shot], thumbs: dict[int, np.ndarray], out_dir: str | Path, cols: int = 5) -> Path:
    """``shots.csv`` plus ``contact_sheet.jpg`` (one labelled thumbnail per shot of 1 s or longer)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shots.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(asdict(shots[0]).keys()) + ["duration"] if shots else ["shot"])
        wr.writeheader()
        for s in shots:
            wr.writerow({**asdict(s), "duration": round(s.duration, 2)})
    tiles = []
    for s in shots:
        im = thumbs.get(s.shot)
        if im is None or s.short:
            continue
        im = im.copy()
        label = f"S{s.shot:02d} {s.video_start:.0f}-{s.video_end:.0f}s"
        met = format_met(s.met_start) if s.met_start is not None else "MET ?"
        for k, text in enumerate((label, met)):
            y = 16 + 16 * k
            cv2.putText(im, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    if tiles:
        h, w = tiles[0].shape[:2]
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, im in enumerate(tiles):
            r, c = divmod(k, cols)
            sheet[r * h : r * h + im.shape[0], c * w : c * w + im.shape[1]] = im[:h, :w]
        cv2.imwrite(str(out / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
