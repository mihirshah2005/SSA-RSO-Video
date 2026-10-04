"""Demo overlay: markers, trails, catalogue predictions and an evidence panel.

Visual rules: colour *and* text carry the category; coasting (predicted-only)
tracks are drawn hollow and dashed; catalogue predictions use a distinct
diamond marker so they are never confused with detections; the header always
states the data mode (causal replay / live / synthetic) and whether the
catalogue was used retrospectively.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import OverlayCfg
from ..io.timemap import format_met, format_utc
from ..types import Category, IdentityStatus, TrackState

COLORS = {  # BGR
    Category.UNKNOWN: (200, 200, 200),
    Category.NEAR_FIELD_PARTICLE: (255, 170, 60),
    Category.PAYLOAD: (60, 220, 60),
    Category.VEHICLE_FEATURE: (120, 120, 120),
    Category.BACKGROUND_FEATURE: (150, 110, 200),
    Category.ARTIFACT: (90, 90, 160),
}
ID_COLOR = (40, 220, 255)
PRED_COLOR = (255, 80, 255)
SHORT = {
    Category.UNKNOWN: "unknown",
    Category.NEAR_FIELD_PARTICLE: "particle?",
    Category.PAYLOAD: "payload?",
    Category.VEHICLE_FEATURE: "vehicle",
    Category.BACKGROUND_FEATURE: "background",
    Category.ARTIFACT: "artefact",
}


def _dashed_circle(img, c, r, color, n=12):
    for k in range(n):
        if k % 2 == 0:
            a0, a1 = 360 * k / n, 360 * (k + 1) / n
            cv2.ellipse(img, c, (r, r), 0, a0, a1, color, 1, cv2.LINE_AA)


def _put(img, text, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class OverlayRenderer:
    def __init__(self, cfg: OverlayCfg, scaler, mission_name: str = "", mode: str = "replay", notes=None):
        self.cfg = cfg
        self.scaler = scaler
        self.mission_name = mission_name
        self.mode = mode
        self.notes = notes or []
        self.show_masks = False
        self.show_trails = True
        self.show_predictions = True

    def _nat(self, x, y):
        return self.scaler.to_native(x, y)

    def render(self, frame_bgr, result, ctx, stats: dict | None = None, release_events=None) -> np.ndarray:
        c = self.cfg
        img = frame_bgr.copy()
        h, w = img.shape[:2]
        fs = c.font_scale * max(0.6, w / 1600.0)
        if self.show_masks:
            inv = ~result.valid
            if inv.shape != img.shape[:2]:
                inv = cv2.resize(inv.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            img[inv] = (0.6 * img[inv] + 0.4 * np.array([60, 0, 60])).astype(np.uint8)
        allowed = set(c.show_categories)
        rows = []
        for tr in result.tracks:
            if tr.state == TrackState.TENTATIVE and not c.show_tentative:
                continue
            if tr.is_confirmed and (result.info.t - tr.first_t) < c.min_track_s:
                continue
            cat = tr.category.category
            if cat.value not in allowed:
                continue
            col = COLORS.get(cat, (200, 200, 200))
            x, y = self._nat(tr.kf.x[0], tr.kf.x[1])
            p = (int(round(x)), int(round(y)))
            obs = tr.observed_points()
            idl = self._identity_text(tr)
            age = result.info.t - tr.first_t
            # declutter: particles and young unknowns get a small marker only; names and labels are
            # drawn for identified, candidate, payload or long-lived tracks
            prominent = bool(idl) or cat == Category.PAYLOAD or (age >= c.label_min_s and cat != Category.NEAR_FIELD_PARTICLE)
            if self.show_trails and len(obs) > 1 and prominent:
                pts = np.array([self._nat(q.detection.x, q.detection.y) for q in obs[-c.trail_len :]], np.int32)
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
            r = int(max(6, 2.5 * obs[-1].detection.sigma / self.scaler.scale)) if obs else 6
            if not prominent:
                r = max(4, r // 2)
            if tr.state == TrackState.LOST:
                _dashed_circle(img, p, r, col)
            else:
                cv2.circle(img, p, r, col, 2 if prominent else 1, cv2.LINE_AA)
            label = f"{tr.label} {SHORT.get(cat, cat.value)}"
            if prominent:
                _put(img, label, (p[0] + r + 3, p[1] - 3), col, fs)
                if idl:
                    _put(img, idl, (p[0] + r + 3, p[1] + int(16 * fs / 0.5)), ID_COLOR, fs)
            if tr.is_confirmed:
                rows.append((tr, label, idl or self._panel_identity(tr)))
        if self.show_predictions and ctx.cache is not None and ctx.camera_native.calibrated and result.info.utc is not None:
            self._draw_predictions(img, ctx, result.info.utc, fs)
        return self._panel(img, result, ctx, rows, stats or {}, fs)

    @staticmethod
    def _identity_text(tr) -> str:
        """Identity text drawn on the image: names, candidate lists and DEPLOY-k labels only."""
        d = tr.identity
        if d.status == IdentityStatus.ACCEPTED:
            return f"ID: {d.label}"
        if d.status == IdentityStatus.CANDIDATES:
            best = d.best
            extra = f" best {best.name} p={best.posterior:.2f}" if best is not None and "DEPLOY" not in d.label else ""
            return f"{d.label}{extra}" if d.label else "candidates"
        if d.label.startswith("DEPLOY"):
            return d.label
        return ""

    @staticmethod
    def _panel_identity(tr) -> str:
        d = tr.identity
        if d.status == IdentityStatus.UNAVAILABLE:
            return d.label or (d.reasons[0] if d.reasons else "unavailable")
        return ""

    def _draw_predictions(self, img, ctx, utc, fs):
        cam = ctx.camera_native
        rel = ctx.cache.rel_at(utc)[:, 0]
        rng = np.linalg.norm(rel, axis=1)
        uv, front = cam.project_ric(rel / np.maximum(rng[:, None], 1e-9))
        ok = cam.in_image(uv, front)
        lit = ctx.cache.sunlit_at(utc)[:, 0]
        for k in np.nonzero(ok & lit & (rng < 2000))[0]:
            u, v = int(uv[k, 0]), int(uv[k, 1])
            d = 6
            cv2.polylines(img, [np.array([[u, v - d], [u + d, v], [u, v + d], [u - d, v]], np.int32)], True, PRED_COLOR, 1, cv2.LINE_AA)
            _put(img, f"{ctx.cache.names[k]} {rng[k]:.0f} km (pred)", (u + 8, v + 4), PRED_COLOR, 0.8 * fs)

    def _panel(self, img, result, ctx, rows, stats, fs):
        c = self.cfg
        h, w = img.shape[:2]
        pw = int(c.panel_width * max(0.6, w / 1600.0))
        panel = np.full((h, pw, 3), 22, np.uint8)
        y = int(22 * fs / 0.5)
        dy = int(18 * fs / 0.5)
        info = result.info

        def line(text, color=(230, 230, 230), scale=1.0):
            nonlocal y
            if y < h - 4:
                _put(panel, text, (8, y), color, fs * scale)
            y += dy

        mode = {"replay": "CAUSAL REPLAY", "live": "LIVE", "synthetic": "SYNTHETIC DATA"}.get(self.mode, self.mode.upper())
        line(self.mission_name or "Starship RSO", (255, 255, 255), 1.1)
        line(mode, (0, 200, 255) if self.mode == "synthetic" else (120, 255, 120))
        line(f"{format_met(info.met)}  (+/-{info.time_sigma or 0:.1f}s)" if info.met is not None else "MET unknown")
        line(format_utc(info.utc))
        if getattr(result, "warming_up", False):
            line("learning vehicle mask (new shot): detections held", (0, 200, 255))
        if stats:
            line(f"proc {stats.get('fps', 0):.1f} fps  lat p95 {stats.get('lat_p95_ms', 0):.0f} ms  drop {stats.get('dropped', 0)}",
                 (180, 180, 180))
        counts: dict[str, int] = {}
        for tr, _, _ in rows:
            counts[tr.category.category.value] = counts.get(tr.category.category.value, 0) + 1
        line("confirmed tracks: " + str(len(rows)))
        for cat in Category:
            if counts.get(cat.value):
                line(f"  {SHORT[cat]}: {counts[cat.value]}", COLORS[cat])
        n_acc = sum(1 for tr, _, _ in rows if tr.identity.status == IdentityStatus.ACCEPTED)
        line(f"catalogue identities accepted: {n_acc}", ID_COLOR)
        y += dy // 2
        line("track      category     identity", (160, 160, 160))
        shown = sorted(rows, key=lambda r: (r[0].identity.status != IdentityStatus.ACCEPTED,
                                            r[0].category.category == Category.NEAR_FIELD_PARTICLE, r[0].display_id))
        for tr, label, idl in shown[: c.max_panel_rows]:
            col = ID_COLOR if tr.identity.status == IdentityStatus.ACCEPTED else COLORS.get(tr.category.category)
            line(f"{tr.label:<6} {SHORT.get(tr.category.category, ''):<11} {idl[:28]}", col, 0.9)
        y = max(y, h - dy * (len(ctx.notes[:4]) + 2))
        for n in ctx.notes[:4]:
            line(n[:60], (140, 140, 140), 0.8)
        return np.hstack([img, panel])
