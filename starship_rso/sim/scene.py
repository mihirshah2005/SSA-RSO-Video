"""Synthetic Starship-like onboard video with complete ground truth.

What it renders (all clearly synthetic, watermarked "SYNTHETIC"):

* a scrolling cloud/ocean background (or black sky, or an Earth limb),
* a static ship body occupying part of the frame, and a HUD band with a T+ clock,
* near-field particles: defocused, flickering discs on curved image paths,
* deployed payloads: released from a door on the ship and drifting away under
  Clohessy-Wiltshire relative motion, projected through the camera,
* catalogued objects passing within tens of km: propagated with SGP4 from
  their own element sets and rendered as point sources with a brightness model,
* noise and JPEG compression.

Independence, stated precisely: the pinhole projection and the RIC
conversion of the truth are written out inline (not imported from
:mod:`identify` or :mod:`geometry`), so a projection bug in the association
engine cannot make a test pass. The orbit propagation (SGP4), the nominal
ship ephemeris and the attitude convention ARE shared with the pipeline, so
this simulator does not validate those; :mod:`tests.test_orbit_geometry`
checks them against astropy and the sgp4 reference instead. The *published*
catalogue has an injected along-track error, its stated quality is written to
``mission.yaml`` (``identify.ephem_sigma_km``), and payload catalogue numbers
are deliberately not in release order.

The output folder contains ``video.mp4``, ``truth.json``, ``catalog.json``,
``camera.json`` and ``mission.yaml`` (a config overlay to run the pipeline on it).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..geometry.camera import PinholeCamera, attitude_from_ypr
from ..io.timemap import format_met, parse_utc
from ..orbit.frames import R_EARTH, ric_matrix
from ..orbit.kepler import MU, state_to_elements
from ..orbit.omm import make_omm_fields, record_from_omm, save_omm_json
from ..orbit.propagate import Propagator
from ..orbit.ship import NominalParams, NominalShipEphemeris
from ..orbit.sun import is_sunlit, phase_angle, sun_position_teme
from ..orbit.timeutil import iso
from .render import add_box, add_disc, add_gaussian_spot, compress, earth_texture, mag_to_amp, vehicle_layer


@dataclass
class SimSpec:
    width: int = 960
    height: int = 540
    fps: float = 30.0
    duration_s: float = 12.0
    seed: int = 0
    hfov_deg: float = 90.0
    attitude_ypr_deg: list[float] = field(default_factory=lambda: [0.0, 40.0, 0.0])
    liftoff_utc: str = "2026-09-28T12:48:59Z"
    met_start_s: float = 2960.0
    inc_deg: float = 30.5
    perigee_km: float = 262.0
    apogee_km: float = 277.0
    background: str = "earth"  # earth | black | limb
    earth_flow_px_s: list[float] = field(default_factory=lambda: [0.0, 70.0])
    vehicle_polygon: list[list[float]] = field(
        default_factory=lambda: [[0.70, 0.0], [1.0, 0.0], [1.0, 0.84], [0.86, 0.84], [0.64, 0.40]]
    )
    hud: bool = True
    hud_band: float = 0.16  # bottom fraction of the frame
    n_particles: int = 14
    particle_speed_px_s: list[float] = field(default_factory=lambda: [80.0, 420.0])
    particle_radius_px: list[float] = field(default_factory=lambda: [1.0, 9.0])
    n_releases: int = 2
    release_interval_s: float = 5.0
    first_release_s: float = 0.5
    door_px_norm: list[float] = field(default_factory=lambda: [0.64, 0.42])  # where the door appears
    door_range_m: float = 40.0  # camera-to-door distance
    release_speed_m_s: float = 2.0
    release_dir_cam: list[float] = field(default_factory=lambda: [-1.0, -0.15, 0.5])  # camera axes
    payload_size_m: float = 3.0
    payload_std_mag: float = 4.0
    n_passes: int = 2
    pass_range_km: list[float] = field(default_factory=lambda: [6.0, 15.0])
    pass_rate_deg_s: list[float] = field(default_factory=lambda: [1.5, 4.0])
    pass_std_mag: float = 4.5
    n_distractors: int = 40
    noise_sigma: float = 2.0
    jpeg_quality: int = 85
    catalog_error_km: float = 0.05  # along-track error injected into the published element sets
    ship_phase_error_deg: float = 0.0  # error of the ship ephemeris handed to the pipeline


# --------------------------------------------------------------------------- helpers
def _cw(x0: np.ndarray, v0: np.ndarray, n: float, t: np.ndarray) -> np.ndarray:
    """Clohessy-Wiltshire relative position (RIC: x radial, y in-track, z cross-track)."""
    t = np.asarray(t, float)[:, None] if np.ndim(t) else np.array([[float(t)]])
    s, c = np.sin(n * t), np.cos(n * t)
    x = (4 - 3 * c) * x0[0] + s / n * v0[0] + 2 / n * (1 - c) * v0[1]
    y = 6 * (s - n * t) * x0[0] + x0[1] - 2 / n * (1 - c) * v0[0] + (4 * s - 3 * n * t) / n * v0[1]
    z = x0[2] * c + v0[2] / n * s
    return np.concatenate([x, y, z], axis=1)


def _project(R_cam_ric: np.ndarray, f: float, cx: float, cy: float, v_ric: np.ndarray):
    """Inline pinhole projection (independent of geometry.camera)."""
    v = v_ric @ R_cam_ric.T
    z = v[..., 2]
    with np.errstate(all="ignore"):
        u = f * v[..., 0] / z + cx
        w = f * v[..., 1] / z + cy
    return u, w, z > 1e-6


def _omm_from_state(norad: int, name: str, r: np.ndarray, v: np.ndarray, epoch_utc: float, objid: str) -> dict:
    el = state_to_elements(r, v)
    n_rev_day = np.sqrt(MU / el["a"] ** 3) * 86400.0 / (2 * np.pi)
    return make_omm_fields(
        norad, name, iso(epoch_utc), float(n_rev_day), max(float(el["e"]), 1e-7), float(np.rad2deg(el["i"])),
        float(np.rad2deg(el["raan"])), float(np.rad2deg(el["argp"])), float(np.rad2deg(el["M"])), 0.0, objid,
    )


def _fit_omm_to_state(norad, name, r_target, v_target, epoch_utc, objid, iters: int = 6) -> dict:
    """Element set whose SGP4 state at its epoch matches (r, v) (fixed-point correction)."""
    r_in, v_in = r_target.copy(), v_target.copy()
    fields = _omm_from_state(norad, name, r_in, v_in, epoch_utc, objid)
    for _ in range(iters):
        p = Propagator([record_from_omm(fields)])
        r_out, v_out, _ = p.states([epoch_utc])
        dr, dv = r_target - r_out[0, 0], v_target - v_out[0, 0]
        if np.linalg.norm(dr) < 1e-4 and np.linalg.norm(dv) < 1e-7:
            break
        r_in, v_in = r_in + dr, v_in + dv
        fields = _omm_from_state(norad, name, r_in, v_in, epoch_utc, objid)
    return fields


def _perturb_along_track(fields: dict, km: float) -> dict:
    out = dict(fields)
    n = fields["MEAN_MOTION"] * 2 * np.pi / 86400.0
    a = (MU / n**2) ** (1 / 3)
    out["MEAN_ANOMALY"] = float((fields["MEAN_ANOMALY"] + np.rad2deg(km / a)) % 360.0)
    return out


# --------------------------------------------------------------------------- simulator
class Simulator:
    def __init__(self, spec: SimSpec):
        self.spec = s = spec
        self.rng = np.random.default_rng(s.seed)
        self.liftoff = parse_utc(s.liftoff_utc)
        self.utc0 = self.liftoff + s.met_start_s
        self.n_frames = int(round(s.duration_s * s.fps))
        self.t = np.arange(self.n_frames) / s.fps
        self.utc = self.utc0 + self.t
        self.ship = NominalShipEphemeris(
            NominalParams(self.liftoff, 25.997, -97.157, s.inc_deg, s.perigee_km, s.apogee_km)
        )
        self.rs, self.vs = self.ship.state(self.utc)
        self.M = ric_matrix(self.rs, self.vs)  # (T, 3, 3) TEME -> RIC
        self.n_orb = float(np.sqrt(MU / np.linalg.norm(self.rs[0]) ** 3))
        self.R = attitude_from_ypr(*s.attitude_ypr_deg)
        self.f = 0.5 * s.width / np.tan(0.5 * np.deg2rad(s.hfov_deg))
        self.cx, self.cy = (s.width - 1) / 2, (s.height - 1) / 2
        self.ifov = 1.0 / self.f
        self.objects: dict[str, dict] = {}
        self.tracks: dict[str, dict] = {}  # id -> arrays x, y, visible, radius, amp, kind
        self.published: list[dict] = []
        h, w = s.height, s.width
        self._veh_img, self._veh_mask = vehicle_layer(h, w, s.vehicle_polygon, self.rng)
        self._hud_mask = np.zeros((h, w), bool)
        if s.hud:
            self._hud_mask[int(h * (1 - s.hud_band)) :, :] = True
        self._occluded = self._veh_mask | self._hud_mask
        self._tex = earth_texture(2 * h, 2 * w, self.rng) if s.background in ("earth", "limb") else None

    # ------------------------------------------------------------ objects
    def _free_pixel(self, margin: float = 0.12) -> tuple[float, float]:
        s = self.spec
        for _ in range(200):
            u = self.rng.uniform(margin, 1 - margin) * s.width
            v = self.rng.uniform(margin, 1 - s.hud_band - margin) * s.height
            if not self._occluded[int(v), int(u)]:
                return u, v
        return s.width * 0.3, s.height * 0.4

    def build_passes(self) -> None:
        s = self.spec
        for j in range(s.n_passes):
            norad = 990001 + j * 7
            name = f"SIM-SAT-{j + 1:02d}"
            tc = s.duration_s * (j + 1) / (s.n_passes + 1)
            k = int(round(tc * s.fps))
            u, v = self._free_pixel(0.2)
            d_cam = np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
            d_cam /= np.linalg.norm(d_cam)
            d_ric = self.R.T @ d_cam
            rho = self.rng.uniform(*s.pass_range_km)
            rate = np.deg2rad(self.rng.uniform(*s.pass_rate_deg_s))
            perp = np.cross(d_ric, self.rng.standard_normal(3))
            perp /= np.linalg.norm(perp)
            vrel_ric = perp * rate * rho
            r_obj = self.rs[k] + self.M[k].T @ (rho * d_ric)
            # relative velocity in the rotating RIC frame -> inertial (add omega x r_rel)
            w_ric = np.array([0.0, 0.0, self.n_orb])
            v_obj = self.vs[k] + self.M[k].T @ (vrel_ric + np.cross(w_ric, rho * d_ric))
            fields = _fit_omm_to_state(norad, name, r_obj, v_obj, float(self.utc[k]), "SIM-PASS")
            self.objects[str(norad)] = {"kind": "catalog", "norad_id": str(norad), "name": name, "t_closest": tc,
                                        "range_km": rho}
            self._truth_from_elements(str(norad), fields, s.pass_std_mag, size_m=8.0)
            self.published.append(_perturb_along_track(fields, self.rng.normal(0, s.catalog_error_km)))

    def build_distractors(self) -> None:
        s = self.spec
        for j in range(s.n_distractors):
            norad = 980001 + j
            a = R_EARTH + self.rng.uniform(300, 1200)
            fields = make_omm_fields(
                norad, f"SIM-FAR-{j:03d}", iso(self.utc0), float(np.sqrt(MU / a**3) * 86400 / (2 * np.pi)), 0.001,
                float(self.rng.uniform(0, 98)), float(self.rng.uniform(0, 360)), 0.0, float(self.rng.uniform(0, 360)),
            )
            self.published.append(fields)

    def _truth_from_elements(self, oid: str, fields: dict, std_mag: float, size_m: float) -> None:
        r, _, ok = Propagator([record_from_omm(fields)]).states(self.utc)
        rel_teme = r[0] - self.rs
        rel_ric = np.einsum("tij,tj->ti", self.M, rel_teme)
        rng_km = np.linalg.norm(rel_ric, axis=1)
        u, v, front = _project(self.R, self.f, self.cx, self.cy, rel_ric)
        sun = sun_position_teme(self.utc)
        lit = is_sunlit(r[0], sun)
        ph = phase_angle(self.rs, r[0], sun)
        f_ph = (np.sin(ph) + (np.pi - ph) * np.cos(ph)) / np.pi
        mag = std_mag - 2.5 * np.log10(np.maximum(f_ph, 1e-4) / (1 / np.pi)) + 5 * np.log10(rng_km / 1000.0)
        amp = np.array([mag_to_amp(m) for m in mag]) * lit
        self.tracks[oid] = {"x": u, "y": v, "front": front & ok[0], "amp": amp, "sigma": np.full(len(u), 0.9),
                            "kind": "catalog", "size_px": (size_m / 1000.0) / rng_km / self.ifov}

    def _door_ric_km(self) -> np.ndarray:
        s = self.spec
        u, v = s.door_px_norm[0] * s.width, s.door_px_norm[1] * s.height
        d = np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        return self.R.T @ (d / np.linalg.norm(d)) * s.door_range_m / 1000.0

    def build_payloads(self) -> None:
        s = self.spec
        if s.n_releases <= 0:
            return
        ids = list(range(990501, 990501 + s.n_releases))
        perm = self.rng.permutation(len(ids))  # catalogue numbers NOT in release order
        door = self._door_ric_km()
        cam = np.zeros(3)  # camera at the ship reference point; the door is placed relative to it
        dcam = np.array(s.release_dir_cam, float)
        dv = self.R.T @ (dcam / np.linalg.norm(dcam)) * s.release_speed_m_s / 1000.0
        for k in range(s.n_releases):
            t_rel = s.first_release_s + k * s.release_interval_s
            norad = ids[perm[k]]
            name = f"SIM-STARLINK-{norad - 990500:04d}"
            jitter = self.rng.normal(0, 0.05, 3) / 1000.0
            v0 = dv + jitter
            tau = self.t - t_rel
            rel = _cw(door, v0, self.n_orb, np.maximum(tau, 0.0))  # km, ship RIC
            rel_cam = rel - cam
            u, v, front = _project(self.R, self.f, self.cx, self.cy, rel_cam)
            dist_m = np.linalg.norm(rel_cam, axis=1) * 1000.0
            size_px = s.payload_size_m / np.maximum(dist_m, 1e-3) / self.ifov
            amp = np.where(tau >= 0, 200.0, 0.0)
            oid = str(norad)
            self.objects[oid] = {"kind": "payload", "norad_id": oid, "name": name, "release_k": k + 1,
                                 "t_release": t_rel, "met_release": s.met_start_s + t_rel}
            self.tracks[oid] = {"x": u, "y": v, "front": front & (tau >= 0), "amp": amp, "sigma": np.full(len(u), 1.0),
                                "kind": "payload", "size_px": size_px}
            # published element set fitted to the post-release state (epoch = 10 min after release)
            kk = int(np.clip(round(t_rel * s.fps), 0, self.n_frames - 1))
            r_p = self.rs[kk] + self.M[kk].T @ door
            w_ric = np.array([0.0, 0.0, self.n_orb])
            v_p = self.vs[kk] + self.M[kk].T @ (v0 + np.cross(w_ric, door))
            fields = _fit_omm_to_state(norad, name, r_p, v_p, float(self.utc[kk]), "SIM-DEPLOY")
            self.published.append(_perturb_along_track(fields, self.rng.normal(0, s.catalog_error_km)))

    def build_particles(self) -> None:
        s = self.spec
        h, w = s.height, s.width
        poly = (np.array(s.vehicle_polygon) * [w, h])
        for j in range(s.n_particles):
            t0 = self.rng.uniform(0, s.duration_s * 0.85)
            # spawn on the vehicle edge (left boundary of the polygon)
            a, b = poly[self.rng.integers(len(poly))], poly[self.rng.integers(len(poly))]
            p0 = a + self.rng.uniform() * (b - a)
            p0 = np.clip(p0 + self.rng.normal(0, 10, 2), [5, 5], [w - 5, h * (1 - s.hud_band) - 5])
            ang = self.rng.uniform(np.pi * 0.5, np.pi * 1.5)  # mostly leftwards, away from the ship
            spd = self.rng.uniform(*s.particle_speed_px_s)
            vel = spd * np.array([np.cos(ang), np.sin(ang)])
            acc = self.rng.normal(0, 40, 2)
            rad = self.rng.uniform(*s.particle_radius_px)
            amp0 = self.rng.uniform(70, 200)
            fl_f, fl_d, fl_p = self.rng.uniform(0.7, 6.0), self.rng.uniform(0.3, 0.85), self.rng.uniform(0, 2 * np.pi)
            life = self.rng.uniform(1.5, 5.0)
            tau = self.t - t0
            alive = (tau >= 0) & (tau <= life)
            x = p0[0] + vel[0] * tau + 0.5 * acc[0] * tau**2
            y = p0[1] + vel[1] * tau + 0.5 * acc[1] * tau**2
            amp = amp0 * (1 - fl_d * 0.5 * (1 + np.sin(2 * np.pi * fl_f * tau + fl_p))) * alive
            oid = f"P{j:03d}"
            self.objects[oid] = {"kind": "particle", "radius_px": rad, "flicker_hz": fl_f}
            self.tracks[oid] = {"x": x, "y": y, "front": alive, "amp": amp, "sigma": np.full(len(x), rad),
                                "kind": "particle", "size_px": np.full(len(x), 2 * rad)}

    # ------------------------------------------------------------ rendering
    def _background(self, k: int) -> np.ndarray:
        """Background scrolled by ``earth_flow_px_s * t`` (content moves with the flow), wrapping."""
        s = self.spec
        h, w = s.height, s.width
        if self._tex is None:
            return np.zeros((h, w, 3), np.float32) + 3.0
        dx, dy = s.earth_flow_px_s[0] * self.t[k], s.earth_flow_px_s[1] * self.t[k]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])  # dst(x, y) = tex(x - dx, y - dy)
        bg = cv2.warpAffine(self._tex, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_WRAP).astype(np.float32)
        if s.background == "limb":
            yy, xx = np.mgrid[0:h, 0:w]
            limb = (xx - w * 0.2) ** 2 + (yy - h * 2.2) ** 2 < (h * 1.85) ** 2
            bg[~limb] = 3.0
        return bg

    def _visible(self, x, y, front) -> bool:
        s = self.spec
        if not front or not (np.isfinite(x) and np.isfinite(y)):
            return False
        if not (0 <= x < s.width and 0 <= y < s.height):
            return False
        return not self._occluded[int(y), int(x)]

    def render(self, out_dir: str | Path, progress: bool = False) -> Path:
        s = self.spec
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.build_passes()
        self.build_payloads()
        self.build_particles()
        self.build_distractors()
        vw = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), s.fps, (s.width, s.height))
        if not vw.isOpened():
            raise IOError("cannot open video writer (mp4v)")
        frames_truth = []
        for k in range(self.n_frames):
            img = self._background(k)
            truth_k = []
            # objects behind the ship (catalog, payload) first
            for oid, tr in self.tracks.items():
                if tr["kind"] == "particle":
                    continue
                x, y = float(tr["x"][k]), float(tr["y"][k])
                if not self._visible(x, y, bool(tr["front"][k])) or tr["amp"][k] <= 0:
                    continue
                size = float(tr["size_px"][k])
                if tr["kind"] == "payload" and size > 2.5:
                    add_box(img, x, y, 0.5 * size, 0.25 * size, float(tr["amp"][k]), angle_deg=20.0)
                else:
                    add_gaussian_spot(img, x, y, float(tr["sigma"][k]), float(tr["amp"][k]))
                truth_k.append({"id": oid, "kind": tr["kind"], "x": round(x, 3), "y": round(y, 3),
                                "r": round(max(size / 2, 1.0), 2)})
            img[self._veh_mask] = self._veh_img[self._veh_mask]
            for oid, tr in self.tracks.items():
                if tr["kind"] != "particle" or not tr["front"][k] or tr["amp"][k] <= 1:
                    continue
                x, y = float(tr["x"][k]), float(tr["y"][k])
                if not (0 <= x < s.width and 0 <= y < s.height * (1 - s.hud_band)):
                    continue
                r = float(tr["sigma"][k])
                if r < 2.0:
                    add_gaussian_spot(img, x, y, max(r * 0.7, 0.8), float(tr["amp"][k]))
                else:
                    add_disc(img, x, y, r, float(tr["amp"][k]) * 0.6)
                truth_k.append({"id": oid, "kind": "particle", "x": round(x, 3), "y": round(y, 3), "r": round(r, 2)})
            img += self.rng.normal(0, s.noise_sigma, img.shape).astype(np.float32)
            u8 = np.clip(img, 0, 255).astype(np.uint8)
            if s.hud:
                self._draw_hud(u8, k)
            u8 = compress(u8, s.jpeg_quality)
            vw.write(u8)
            frames_truth.append(truth_k)
            if progress and k % 60 == 0:
                print(f"  rendered {k}/{self.n_frames}")
        vw.release()
        self._write_outputs(out, frames_truth)
        return out

    def _draw_hud(self, img: np.ndarray, k: int) -> None:
        s = self.spec
        h, w = img.shape[:2]
        y0 = int(h * (1 - s.hud_band))
        img[y0:, :] = (img[y0:, :].astype(np.float32) * 0.35).astype(np.uint8)
        met = s.met_start_s + self.t[k]
        fs = h / 900.0
        cv2.putText(img, format_met(met), (int(w * 0.43), int(h * 0.94)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * fs,
                    (255, 255, 255), max(1, int(2 * fs)), cv2.LINE_AA)
        cv2.putText(img, "SYNTHETIC", (int(w * 0.02), int(h * 0.94)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * fs,
                    (0, 200, 255), max(1, int(2 * fs)), cv2.LINE_AA)

    def _write_outputs(self, out: Path, frames_truth: list) -> None:
        s = self.spec
        with open(out / "truth.json", "w", encoding="utf-8") as fh:
            json.dump({"spec": asdict(s), "fps": s.fps, "width": s.width, "height": s.height,
                       "utc0": self.utc0, "objects": self.objects, "frames": frames_truth}, fh)
        recs = [record_from_omm(d, "synthetic") for d in self.published]
        save_omm_json(recs, out / "catalog.json", {"source": "synthetic", "note": "SYNTHETIC catalogue with injected errors"})
        cam = PinholeCamera.from_hfov(s.width, s.height, s.hfov_deg)
        cam.R_cam_ric = self.R
        cam.meta = {"source": "simulation truth"}
        cam.save(out / "camera.json")
        payloads = sorted((o for o in self.objects.values() if o["kind"] == "payload"), key=lambda o: o["release_k"])
        door_px = [float(s.door_px_norm[0]), float(s.door_px_norm[1])] if s.n_releases > 0 else None
        met_rel = [p["met_release"] for p in payloads]
        mission = {
            "mission": {
                "name": f"SYNTHETIC (seed {s.seed})",
                "liftoff_utc": s.liftoff_utc,
                # the simulated ship follows this orbit exactly; phase_sigma states what the pipeline may assume
                "orbit_nominal": {"perigee_km": s.perigee_km, "apogee_km": s.apogee_km, "inc_deg": s.inc_deg,
                                  "downrange_deg_at_seco": 14.0 + s.ship_phase_error_deg,
                                  "phase_sigma_deg": max(abs(s.ship_phase_error_deg), 1e-4)},
                "deployment": {"first_met_s": met_rel[0] if met_rel else None,
                               "last_met_s": met_rel[-1] if met_rel else None, "count": len(met_rel)},
                "payload_group": {"name": "SIM deployment", "norad_range": [990501, 990599]},
                "catalog_file": str((out / "catalog.json").resolve()),
                "ship_ephemeris": "nominal",
            },
            "timemap": {"anchors": [{"video_s": 0.0, "met_s": s.met_start_s, "sigma_s": 0.05}]},
            # the synthetic catalogue's stated quality (2 x the injected error): identification is
            # demonstrated under good data; raise it to see the system withhold names
            "identify": {"ephem_sigma_km": max(0.1, 2.0 * s.catalog_error_km)},
            "camera": {"calibration_file": str((out / "camera.json").resolve())},
            "masks": {"hud_rects": [[0.0, 1 - s.hud_band, 1.0, 1.0]] if s.hud else []},
            "deployment": {"door_xy": door_px},
            "hud": {"clock": [0.40, 1 - s.hud_band + 0.02, 0.62, 0.99]},
        }
        with open(out / "mission.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(mission, fh, sort_keys=False)


def simulate(out_dir: str | Path, spec: SimSpec | None = None, progress: bool = False) -> Path:
    return Simulator(spec or SimSpec()).render(out_dir, progress=progress)
