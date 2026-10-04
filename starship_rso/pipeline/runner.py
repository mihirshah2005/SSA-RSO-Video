"""Run the pipeline on a file or stream: optional wall-clock pacing, display, recording, logs.

Latency is measured from the moment a frame became *available* (its paced
release time in replay mode) to the moment its result was rendered. In
replay mode with ``realtime.enabled`` the reader drops stale frames instead
of queueing, so the display never drifts behind the source.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..config import Config, save_config
from ..io.runlog import JsonlWriter, make_run_dir, provenance, read_jsonl
from ..io.video import PacedReader, VideoSource, resolve_stream_url
from .context import build_context
from .overlay import OverlayRenderer
from .pipeline import Pipeline

log = logging.getLogger(__name__)


def track_record(tr, scaler) -> dict:
    pts = []
    for p in tr.history:
        x, y = scaler.to_native(p.x, p.y)
        d = p.detection
        e = {"f": p.frame_index, "t": round(p.t, 4), "x": round(x, 2), "y": round(y, 2), "obs": d is not None}
        if d is not None:
            dx, dy = scaler.to_native(d.x, d.y)
            e.update({"dx": round(dx, 2), "dy": round(dy, 2), "conf": round(d.confidence, 3), "sigma": round(d.sigma / scaler.scale, 2),
                      "flux": round(d.flux, 1)})
        if p.met is not None:
            e["met"] = round(p.met, 3)
        pts.append(e)
    idd = tr.identity
    return {
        "uid": tr.uid,
        "id": tr.display_id,
        "label": tr.label,
        "shot": tr.shot_id,
        "category": tr.category.category.value,
        "category_scores": {k: round(v, 3) for k, v in tr.category.scores.items()},
        "category_reasons": tr.category.reasons,
        "identity_status": idd.status.value,
        "identity_label": idd.label,
        "identity_candidates": [{"id": c.object_id, "name": c.name, "posterior": round(c.posterior, 4),
                                 "chi2": round(c.mahalanobis2, 2), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.detail.items()}}
                                for c in idd.candidates],
        "identity_reasons": idd.reasons,
        "identity_decided_t": idd.decided_t,
        "death": tr.death_reason,
        "features": {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else None) for k, v in tr.features.items()},
        "points": pts,
    }


def run(
    cfg: Config,
    video: str,
    out_dir: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    display: bool = True,
    mode: str = "replay",
    catalog_mode: str = "retrospective",
    max_frames: int | None = None,
    tag: str = "run",
    write_frames_jsonl: bool = True,
) -> Path:
    src_path = resolve_stream_url(video)
    source = VideoSource(src_path, start_s, end_s)
    meta = source.meta
    t0 = start_s or 0.0
    t1 = end_s if end_s is not None else (meta.frame_count / meta.fps if meta.frame_count else t0 + 3600.0)
    ctx = build_context(cfg, meta.width, meta.height, t0, t1, mode=mode, catalog_mode=catalog_mode)
    run_dir = Path(out_dir) if out_dir else make_run_dir(cfg.logging.out_dir, tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("frames.jsonl", "tracks.jsonl", "tracks.json", "identity_log.jsonl"):
        (run_dir / stale).unlink(missing_ok=True)  # a re-run into the same directory starts clean
    # dead tracks are streamed to disk, so memory stays bounded on long live runs
    tracks_log = JsonlWriter(run_dir / "tracks.jsonl")
    counts = {"n": 0, "categories": {}, "identity_status": {}}
    pipe_ref: dict = {}

    def on_finish(tr) -> None:
        tracks_log.write(track_record(tr, pipe_ref["pipe"].scaler))
        counts["n"] += 1
        for key, val in (("categories", tr.category.category.value), ("identity_status", tr.identity.status.value)):
            counts[key][val] = counts[key].get(val, 0) + 1

    pipe = Pipeline(cfg, meta.width, meta.height, ctx)
    pipe.tracker.on_finish = on_finish
    pipe.tracker.keep_finished = False
    pipe_ref["pipe"] = pipe
    overlay = OverlayRenderer(cfg.overlay, pipe.scaler, cfg.mission.name, mode, ctx.notes)
    save_config(cfg, run_dir / "config.yaml")
    prov = provenance(cfg.digest(), {"video": str(video), "start_s": start_s, "end_s": end_s, "mode": mode,
                                     "catalog_mode": catalog_mode, "fps": meta.fps, "size": [meta.width, meta.height],
                                     "notes": ctx.notes})
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    writer_path = run_dir / "overlay.mp4"
    frames_log = JsonlWriter(run_dir / "frames.jsonl") if (cfg.logging.jsonl and write_frames_jsonl) else None
    # video encoding and log writing run on a background thread, so they never delay the display
    sink = _OutputSink(writer_path if cfg.logging.write_video else None, meta.fps, frames_log)
    ident_log = JsonlWriter(run_dir / "identity_log.jsonl")

    paced = PacedReader(source, cfg.realtime.speed, cfg.realtime.drop_policy, cfg.realtime.queue_size) \
        if cfg.realtime.enabled else None
    it = paced if paced is not None else ((i, t, f, None, 0) for i, t, f in source)
    lat_recent: deque = deque(maxlen=900)  # rolling window for the on-screen p95
    lat_hist = np.zeros(2001, np.int64)  # 1 ms bins up to 2 s for the run summary (bounded memory)
    proc_hist = np.zeros(2001, np.int64)
    loop_hist = np.zeros(2001, np.int64)  # whole iteration incl. overlay and display
    n, dropped, lat_p95 = 0, 0, 0.0
    wall0 = time.perf_counter()
    win = "Starship RSO"
    paused = False
    try:
        for idx, t, frame, avail, skipped in it:
            loop0 = time.perf_counter()
            dropped = paced.dropped if paced is not None else 0
            res = pipe.process(idx, t, frame, avail)
            proc_hist[min(int(res.timings_ms["total"]), 2000)] += 1
            elapsed = time.perf_counter() - wall0
            if n % 15 == 0 and lat_recent:
                lat_p95 = float(np.percentile(lat_recent, 95))
            stats = {"fps": (n + 1) / max(elapsed, 1e-6), "dropped": dropped, "lat_p95_ms": lat_p95}
            vis = overlay.render(frame, res, ctx, stats)
            done = time.perf_counter()
            if avail is not None:
                ms = (done - avail) * 1000.0
                lat_recent.append(ms)
                lat_hist[min(int(ms), 2000)] += 1
            sink.put(vis, _frame_record(res, pipe.scaler) if frames_log is not None else None)
            for rec in pipe.associator.history:
                ident_log.write(rec)
            pipe.associator.history.clear()
            if display:
                dw = cfg.overlay.display_max_width
                shown = vis if not dw or vis.shape[1] <= dw else cv2.resize(
                    vis, (dw, int(round(vis.shape[0] * dw / vis.shape[1]))), interpolation=cv2.INTER_AREA)
                cv2.imshow(win, shown)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("m"):
                    overlay.show_masks = not overlay.show_masks
                elif key == ord("t"):
                    overlay.show_trails = not overlay.show_trails
                elif key == ord("p"):
                    overlay.show_predictions = not overlay.show_predictions
                elif key == ord("s"):
                    cv2.imwrite(str(run_dir / f"snapshot_{idx:06d}.png"), vis)
            n += 1
            loop_hist[min(int((time.perf_counter() - loop0) * 1000.0), 2000)] += 1
            if max_frames is not None and n >= max_frames:
                break
    finally:
        if paced is not None:
            dropped = paced.dropped
        sink.close()
        if frames_log is not None:
            frames_log.close()
        if display:
            cv2.destroyAllWindows()
        for tr in pipe.tracker.tracks:  # still alive at the end of the clip
            if tr.display_id is not None:
                on_finish(tr)
        tracks_log.close()
        ident_log.close()
    events = [e.__dict__ for e in pipe.deploy.events]
    wall = time.perf_counter() - wall0
    summary = {
        "frames_processed": n,
        "frames_dropped": dropped,
        "wall_s": wall,
        "effective_fps": n / max(wall, 1e-6),
        "proc_ms_p50": _hist_pct(proc_hist, 50),
        "proc_ms_p95": _hist_pct(proc_hist, 95),
        "loop_ms_p50": _hist_pct(loop_hist, 50),  # pipeline + overlay + display, per processed frame
        "loop_ms_p95": _hist_pct(loop_hist, 95),
        "latency_ms_p50": _hist_pct(lat_hist, 50),
        "latency_ms_p95": _hist_pct(lat_hist, 95),
        "confirmed_tracks": counts["n"],
        "categories": counts["categories"],
        "identity_status": counts["identity_status"],
        "release_events": events,
        "release_estimates": [r.to_dict() for r in ctx.release_estimates],
        "notes": ctx.notes,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("run written to %s", run_dir)
    return run_dir


class _OutputSink:
    """Encodes the overlay video and writes the frame log on one background thread.

    The queue is bounded and ``put`` blocks when it is full, so no logged frame is ever
    dropped; with a fast enough disk the display loop never waits for the encoder.
    """

    def __init__(self, video_path: Path | None, fps: float, frames_log, maxsize: int = 32):
        self.video_path, self.fps, self.frames_log = video_path, fps, frames_log
        self._writer = None
        self._error: BaseException | None = None
        self._q: queue.Queue = queue.Queue(maxsize)
        self._thread = threading.Thread(target=self._run, name="rso-output", daemon=True)
        self._thread.start()

    def put(self, vis: np.ndarray, record: dict | None) -> None:
        if self._error is not None:
            raise RuntimeError("output writer failed") from self._error
        self._q.put((vis, record))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            if self._error is not None:
                continue  # keep draining so put() never blocks forever
            vis, record = item
            try:
                if self.video_path is not None:
                    if self._writer is None:
                        self._writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                       self.fps, (vis.shape[1], vis.shape[0]))
                    self._writer.write(vis)
                if record is not None and self.frames_log is not None:
                    self.frames_log.write(record)
            except BaseException as exc:  # surfaced on the next put() / close()
                self._error = exc

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._writer is not None:
            self._writer.release()
        if self._error is not None:
            raise RuntimeError("output writer failed") from self._error


def _hist_pct(hist: np.ndarray, q: float) -> float | None:
    total = int(hist.sum())
    if total == 0:
        return None
    k = int(np.searchsorted(np.cumsum(hist), q / 100.0 * total))
    return float(k) + 0.5


def load_tracks(run_dir: str | Path) -> list[dict]:
    """Track records of a run (``tracks.jsonl``; older runs used ``tracks.json``)."""
    run = Path(run_dir)
    if (run / "tracks.jsonl").exists():
        return read_jsonl(run / "tracks.jsonl")
    return json.loads((run / "tracks.json").read_text())


def _count(it) -> dict:
    out: dict = {}
    for k in it:
        out[k] = out.get(k, 0) + 1
    return out


def _frame_record(res, scaler) -> dict:
    i = res.info
    return {
        "f": i.index,
        "t": round(i.t, 4),
        "met": None if i.met is None else round(i.met, 3),
        "utc": i.utc,
        "shot": i.shot_id,
        "cut": i.is_cut,
        "warm": res.warming_up,
        "reg": {"valid": res.registration.valid, "inliers": res.registration.n_inliers,
                "rms": None if not np.isfinite(res.registration.rms) else round(res.registration.rms, 3),
                "reason": res.registration.reason},
        "dets": [
            {**d.to_dict(), "x": round(scaler.to_native(d.x, d.y)[0], 2), "y": round(scaler.to_native(d.x, d.y)[1], 2)}
            for d in res.detections
        ],
        "tracks": [
            {"uid": tr.uid, "id": tr.display_id, "state": tr.state.value, "cat": tr.category.category.value,
             "x": round(scaler.to_native(tr.kf.x[0], tr.kf.x[1])[0], 2),
             "y": round(scaler.to_native(tr.kf.x[0], tr.kf.x[1])[1], 2),
             "obs": bool(tr.history and tr.history[-1].frame_index == i.index and tr.history[-1].detection is not None),
             "ident": tr.identity.status.value, "label": tr.identity.label}
            for tr in res.tracks
        ],
        "ms": {k: round(v, 2) for k, v in res.timings_ms.items()},
    }
