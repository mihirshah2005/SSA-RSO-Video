"""Timestamp-preserving video input, plus a paced reader that imitates a live feed.

``VideoSource`` yields ``(index, t, frame_bgr)`` where ``t`` is the
presentation time in seconds. PyAV gives exact presentation timestamps and is
used when installed; otherwise OpenCV is used, with ``CAP_PROP_POS_MSEC`` when
it is monotonic and ``index / fps`` as the fallback.

``PacedReader`` makes a recorded file behave like a live stream: frame ``i``
becomes *available* at ``wall0 + (t_i - t_0) / speed``. With
``drop_policy="latest"`` a slow consumer skips stale frames instead of falling
further and further behind (the queue never grows into a smooth-looking but
minutes-late display). The true time gap is preserved in ``t`` so the
tracker integrates over the real interval.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    width: int
    height: int
    backend: str


def resolve_stream_url(url: str) -> str:
    """Resolve a web page URL (e.g. a YouTube live page) to a direct media URL with yt-dlp.

    Local paths and direct media URLs are returned unchanged. Requires the
    ``yt-dlp`` executable; downloading or restreaming content may be subject to
    the platform's terms, so prefer an original source you are allowed to use.
    """
    if Path(url).exists() or not url.startswith(("http://", "https://")):
        return url
    if any(url.lower().split("?")[0].endswith(ext) for ext in (".mp4", ".m3u8", ".mkv", ".webm", ".ts")):
        return url
    exe = shutil.which("yt-dlp")
    if exe is None:
        raise RuntimeError("yt-dlp is not installed; pass a local file or a direct media URL")
    out = subprocess.run([exe, "-g", "-f", "best[height<=1080]", url], capture_output=True, text=True, check=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp returned no stream URL for {url}")
    return lines[0]


class VideoSource:
    """Iterate frames of a file or stream with presentation timestamps."""

    def __init__(
        self,
        path: str | Path,
        start_s: float | None = None,
        end_s: float | None = None,
        stride: int = 1,
        backend: str = "auto",
    ):
        self.path = str(path)
        self.start_s = start_s
        self.end_s = end_s
        self.stride = max(1, int(stride))
        if backend == "auto":
            # RSO_VIDEO_BACKEND=opencv avoids importing PyAV (on macOS the opencv-python and av
            # wheels each bundle libavdevice, which prints duplicate-class warnings)
            forced = os.environ.get("RSO_VIDEO_BACKEND", "").strip().lower()
            if forced in ("pyav", "opencv"):
                backend = forced
            else:
                backend = "pyav" if _have_pyav() and Path(self.path).exists() else "opencv"
        if backend not in ("pyav", "opencv"):
            raise ValueError(f"unknown backend {backend}")
        self.backend = backend
        self.meta = self._probe()

    # ------------------------------------------------------------------ probe
    def _probe(self) -> VideoMeta:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if not (fps > 0 and np.isfinite(fps)):
            fps = 30.0
            log.warning("FPS unknown for %s; assuming 30", self.path)
        return VideoMeta(fps=float(fps), frame_count=n, width=w, height=h, backend=self.backend)

    # --------------------------------------------------------------- iterate
    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        gen = self._iter_pyav() if self.backend == "pyav" else self._iter_opencv()
        k = 0
        for idx, t, get in gen:
            if self.start_s is not None and t < self.start_s - 1e-6:
                continue
            if self.end_s is not None and t > self.end_s + 1e-6:
                break
            if k % self.stride == 0:
                yield idx, t, get()  # pixels are converted only for frames that are used
            k += 1

    def _iter_opencv(self) -> Iterator[tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        fps = self.meta.fps
        idx = 0
        if self.start_s and self.start_s > 0:
            # Seek slightly early; frames before start_s are discarded by __iter__.
            target = max(0.0, self.start_s - 2.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            idx = int(round(pos)) if pos and pos > 0 else int(round(target * fps))
        last_t = -np.inf
        msec_ok = True
        try:
            while True:
                if not cap.grab():  # decodes; colour conversion happens in retrieve()
                    break
                # once container timestamps fail, continue from the last good time (never jump back)
                t = (last_t + 1.0 / fps) if np.isfinite(last_t) else idx / fps
                if msec_ok:
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if ms is not None and ms > 0 and ms / 1000.0 > last_t:
                        t = ms / 1000.0
                    elif idx > 0 and np.isfinite(last_t):
                        msec_ok = False  # timestamps unusable for this container/backend
                last_t = t
                yield idx, float(t), lambda: cap.retrieve()[1]
                idx += 1
        finally:
            cap.release()

    def _iter_pyav(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import av  # type: ignore

        container = av.open(self.path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # presentation times relative to the stream start (HLS/TS captures often start at > 0)
            start = stream.start_time or 0
            if self.start_s and self.start_s > 2.0:
                container.seek(int((self.start_s - 2.0) / stream.time_base) + start, stream=stream, backward=True)
            idx = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float((frame.pts - start) * stream.time_base)
                if idx is None:
                    idx = int(round(t * self.meta.fps))
                yield idx, t, (lambda f=frame: f.to_ndarray(format="bgr24"))
                idx += 1
        finally:
            container.close()


class PacedReader:
    """Wrap an iterable of ``(index, t, frame)`` and release frames on a wall clock.

    Iteration yields ``(index, t, frame, available_wall, dropped_before)``.
    """

    _SENTINEL = object()

    def __init__(self, frames, speed: float = 1.0, drop_policy: str = "latest", buffer: int = 8):
        if speed <= 0:
            raise ValueError("speed must be positive")
        if drop_policy not in ("latest", "none"):
            raise ValueError("drop_policy must be 'latest' or 'none'")
        self._frames = frames
        self.speed = float(speed)
        self.drop_policy = drop_policy
        self._buf: deque = deque()
        self._cond = threading.Condition()
        self._max = max(2, int(buffer))
        self._stop = False
        self._done = False
        self._error: BaseException | None = None
        self.dropped = 0
        self._wall0: float | None = None
        self._t0: float | None = None
        self._thread = threading.Thread(target=self._run, name="decoder", daemon=True)

    def _avail(self, t: float) -> float:
        assert self._wall0 is not None and self._t0 is not None
        return self._wall0 + (t - self._t0) / self.speed

    def _run(self) -> None:
        try:
            for item in self._frames:
                with self._cond:
                    if self._wall0 is None:
                        self._wall0, self._t0 = time.perf_counter(), item[1]
                    # Drop the oldest frame if the buffer is full and the frame is already stale.
                    while len(self._buf) >= self._max and not self._stop:
                        if self.drop_policy == "latest" and time.perf_counter() > self._avail(self._buf[0][1]):
                            self._buf.popleft()
                            self.dropped += 1
                            break
                        self._cond.wait(timeout=0.005)
                    if self._stop:
                        return
                    self._buf.append(item)
                    self._cond.notify_all()
        except BaseException as exc:  # surface decode errors to the consumer
            self._error = exc
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def __iter__(self):
        self._thread.start()
        try:
            while True:
                with self._cond:
                    while not self._buf and not self._done:
                        self._cond.wait(timeout=0.05)
                    if self._error is not None:
                        raise self._error
                    if not self._buf and self._done:
                        return
                    skipped = 0
                    if self.drop_policy == "latest":
                        now = time.perf_counter()
                        while len(self._buf) > 1 and self._avail(self._buf[1][1]) <= now:
                            self._buf.popleft()
                            skipped += 1
                    item = self._buf.popleft()
                    self.dropped += skipped
                    self._cond.notify_all()
                avail = self._avail(item[1])
                delay = avail - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                yield (*item, avail, skipped)
        finally:
            with self._cond:
                self._stop = True
                self._cond.notify_all()


def read_frame_at(path: str | Path, t: float) -> np.ndarray:
    """Return the first frame at or after presentation time ``t`` (seconds)."""
    for _, ft, frame in VideoSource(path, start_s=t, end_s=t + 5.0):
        if ft >= t - 1e-6:
            return frame
    raise ValueError(f"no frame at t={t} in {path}")


def _have_pyav() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:
        return False
