"""Regression tests for defects found in the multi-agent code review."""

import json

import cv2
import numpy as np
import pytest

from conftest import cloud_texture
from starship_rso.config import ClassicalCfg, RegistrationCfg, TrackerCfg, load_config
from starship_rso.geometry.camera import PinholeCamera
from starship_rso.io.timemap import parse_utc
from starship_rso.orbit.conjunction import phase_sweep, screen
from starship_rso.orbit.omm import record_from_omm
from starship_rso.orbit.propagate import Propagator
from starship_rso.orbit.ship import NominalParams, NominalShipEphemeris, SGP4ShipEphemeris
from starship_rso.sim.scene import _fit_omm_to_state
from starship_rso.tracking.tracker import Tracker
from starship_rso.types import Detection, FrameInfo
from starship_rso.vision.classical import ClassicalDetector
from starship_rso.vision.registration import BackgroundRegistrar, History, Registration

L = parse_utc("2026-09-28T12:48:59Z")


def _det(x, y):
    return Detection(x=x, y=y, score=2.0, confidence=0.9, pos_sigma=0.3)


def test_tracks_confirm_when_the_live_reader_drops_frames():
    tr = Tracker(TrackerCfg())
    for k in range(12):  # only every third source frame reaches the tracker
        idx = 3 * k
        tr.update([_det(100 + 2.0 * idx, 50.0)], FrameInfo(idx, idx / 30, 640, 480))
    assert tr.live_tracks(confirmed_only=True), "track never confirmed with dropped frames"


def test_smooth_moving_ocean_is_not_declared_static(rng):
    h, w = 270, 480
    base = cloud_texture(h, w, rng, scale=12.0) * 0.08 + 90  # low contrast
    a = base.astype(np.uint8)
    b = cv2.warpAffine(base, np.float32([[1, 0, 6], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.uint8)
    r = BackgroundRegistrar(RegistrationCfg(work_width=w), (h, w)).estimate(a, b, None, 1 / 30)
    assert not r.static_background, r.reason


def test_thin_moving_earth_band_in_black_space_is_registered(rng):
    h, w = 270, 480
    img = np.full((h, w), 3.0, np.float32)
    band = cloud_texture(60, w + 40, rng, scale=2.0)
    a, b = img.copy(), img.copy()
    a[200:260] = band[:, 0:w]
    b[200:260] = band[:, 5 : w + 5]  # Earth band moves 5 px left, space stays black
    r = BackgroundRegistrar(RegistrationCfg(work_width=w), (h, w)).estimate(a.astype(np.uint8), b.astype(np.uint8), None, 1 / 30)
    assert r.valid and not r.static_background
    assert abs(r.H[0, 2] + 5.0) < 0.5


def _spot(img, x, y, amp, s):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img += (amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))).astype(np.float32)


def test_coarse_scale_positions_and_sizes_with_non_divisible_frame():
    h, w = 541, 965  # not divisible by the coarse factors
    rng = np.random.default_rng(0)
    frames = []
    for k in range(3):
        f = np.full((h, w), 5.0, np.float32) + rng.normal(0, 1.0, (h, w)).astype(np.float32)
        _spot(f, 700.3, 500.6, 120, 1.2)  # point source near the bottom-right corner
        _spot(f, 300.0, 200.0, 90, 5.0)  # defocused blob
        frames.append(f)
    det = ClassicalDetector(ClassicalCfg(), (h, w))
    hist = History(16)
    out = None
    for k, f in enumerate(frames):
        hist.advance(Registration(np.eye(3), False, static_background=True))
        out = det.detect(f, hist.items(), np.ones((h, w), bool), k, k / 30, static_background=True)
        hist.push(f, np.ones((h, w), bool))
    pt = min(out, key=lambda d: np.hypot(d.x - 700.3, d.y - 500.6))
    blob = min(out, key=lambda d: np.hypot(d.x - 300.0, d.y - 200.0))
    assert np.hypot(pt.x - 700.3, pt.y - 500.6) < 0.3
    assert pt.sigma < 2.0, f"point source measured as sigma {pt.sigma:.1f} px"
    assert blob.sigma > pt.sigma and np.hypot(blob.x - 300, blob.y - 200) < 1.0


def test_noise_maps_recomputed_every_frame_when_requested():
    cfg = ClassicalCfg(noise_every=1)
    det = ClassicalDetector(cfg, (100, 100))
    calls = []
    import starship_rso.vision.classical as mod

    orig = mod.peak_stats_map
    mod.peak_stats_map = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        rng = np.random.default_rng(1)
        for k in range(3):
            det.detect(rng.normal(50, 3, (100, 100)).astype(np.float32), [], np.ones((100, 100), bool), k, k / 30)
    finally:
        mod.peak_stats_map = orig
    assert len(calls) >= 3 * len(cfg.scales)


def test_phase_sweep_has_no_gaps_and_screen_finds_close_pass():
    ship = NominalShipEphemeris(NominalParams(L, 25.997, -97.157, 30.5, 262.0, 277.0), phase_sigma_deg=5.0)
    ships, spacing = phase_sweep(ship, range_km=100.0)
    assert spacing <= 100.0 and len(ships) > 7
    # an object passing 1 km from the TRUE ship, which is 2.5 deg (0.5 sigma) off the central hypothesis
    true_ship = ship.with_phase_offset(2.5)
    tc = L + 3000.0
    r, v = true_ship.state(tc)
    fields = _fit_omm_to_state(990777, "CLOSE", r + np.array([0.0, 0.0, 1.0]), v + np.array([0.0, 0.0, 0.05]), tc, "T")
    hits = screen(Propagator([record_from_omm(fields)]), ships, tc - 60, tc + 60, range_km=100.0,
                  extra_radius_km=0.5 * spacing)
    assert hits, "close pass fell between phase hypotheses"


def test_closest_approach_refined_between_samples():
    ship = NominalShipEphemeris(NominalParams(L, 25.997, -97.157, 30.5, 262.0, 277.0))
    tc = L + 3000.0
    r, v = ship.state(tc)
    vhat = v / np.linalg.norm(v)
    # 1 km cross-track miss, 2 km/s relative speed; grid samples fall 0.25 s either side of closest approach
    off = np.cross(r, v)
    off = off / np.linalg.norm(off)
    fields = _fit_omm_to_state(990778, "X", r + off, v + 2.0 * vhat, tc, "T")
    hits = screen(Propagator([record_from_omm(fields)]), [ship], tc - 30.25, tc + 30, range_km=50.0)
    assert hits and hits[0].range_min_km < 1.2


def test_sgp4_ship_scalar_time_shape():
    ship = NominalShipEphemeris(NominalParams(L, 25.997, -97.157, 30.5, 262.0, 277.0))
    r, v = ship.state(L + 3000.0)
    rec = record_from_omm(_fit_omm_to_state(990779, "SHIP", r, v, L + 3000.0, "T"))
    rs, vs = SGP4ShipEphemeris(rec).state(L + 3000.0)
    assert rs.shape == (3,) and vs.shape == (3,)


def test_distortion_never_folds_back_into_the_image():
    cam = PinholeCamera.from_hfov(1920, 1080, 90.0, k1=-0.25)
    ang = np.deg2rad(np.array([50.0, 60.0, 70.0]))
    rays = np.stack([np.sin(ang), np.zeros(3), np.cos(ang)], 1)
    uv, ok = cam.project_cam(rays)
    assert not (ok & cam.in_image(uv, ok)).any()
    cam2 = PinholeCamera.from_hfov(1920, 1080, 90.0, k1=-0.1)
    corner = np.array([[0.0, 0.0], [1919.0, 1079.0]])
    uv2, ok2 = cam2.project_cam(cam2.unproject(corner))
    assert ok2.all() and np.allclose(uv2, corner, atol=1e-3)


def test_export_cvat_respects_task_range_and_yolo_normalisation(tmp_path):
    from starship_rso.io.cvat import LabeledPoint, LabeledTrack, export_tracks, import_tracks
    from starship_rso.train.export_yolo import export_yolo

    tr = [LabeledTrack(1, "unknown", [LabeledPoint(f, 10.0, 20.0) for f in range(95, 105)])]
    export_tracks(tr, tmp_path / "c.xml", 640, 360, n_frames=10, frame_offset=95)
    back, _ = import_tracks(tmp_path / "c.xml")
    frames = [p.frame for p in back[0].points]
    assert min(frames) >= 0 and max(frames) <= 9
    # YOLO labels on a frame smaller than the tile
    store = tmp_path / "store"
    (store / "frames").mkdir(parents=True)
    img = np.zeros((270, 480), np.uint8)
    cv2.imwrite(str(store / "frames" / "000000.jpg"), img)
    (store / "meta.json").write_text(json.dumps({"width": 480, "height": 270, "frames": [{"i": 0}]}))
    (store / "labels.json").write_text(json.dumps({"points": {"0": [[240.0, 135.0, 1.0, 1, "x"]]}, "ignore": {}}))
    (store / "splits.json").write_text(json.dumps({"train": [0], "val": [], "test": []}))
    out = export_yolo(store, tmp_path / "yolo", tile=640, every=1)
    lab = (out / "labels" / "train" / "000000_0_0.txt").read_text().split()
    assert abs(float(lab[1]) - 0.5) < 1e-6 and abs(float(lab[2]) - 0.5) < 1e-6


@pytest.mark.slow
def test_training_stack_resume_done_and_fused_pipeline(tmp_path):
    """sim -> frame store -> 2-epoch train -> resume (no-op after DONE) -> fused detector run."""
    pytest.importorskip("torch")
    from starship_rso.pipeline.runner import run
    from starship_rso.sim.scene import SimSpec, simulate
    from starship_rso.train.data import build_frame_store, labels_from_sim, make_splits, write_labels
    from starship_rso.train.train_heatmap import TrainCfg, train

    sim = simulate(tmp_path / "sim", SimSpec(width=320, height=180, duration_s=3.0, n_passes=1, n_releases=0,
                                             n_particles=3, seed=2))
    cfg = load_config(["configs/default.yaml", str(sim / "mission.yaml")], ["realtime.enabled=false"])
    store = build_frame_store(cfg, str(sim / "video.mp4"), tmp_path / "store")
    write_labels(store, labels_from_sim(sim / "truth.json"))
    make_splits(store, frames=3, block_s=0.5)
    tc = TrainCfg(stores=[str(store)], out_dir=str(tmp_path / "hm"), crop=96, batch_size=2, base=8, depth=2,
                  samples_per_epoch=4, val_samples=2, workers=0, epochs=2, patience=10)
    out = train(tc)
    assert (out / "best.pt").exists() and (out / "DONE").exists()
    assert len((out / "log.csv").read_text().strip().splitlines()) == 3
    train(tc)  # finished: must return immediately without training more
    assert len((out / "log.csv").read_text().strip().splitlines()) == 3
    cfg.detector.type = "fused"
    cfg.detector.heatmap.weights = str(out / "best.pt")
    cfg.detector.heatmap.device = "cpu"
    d = run(cfg, str(sim / "video.mp4"), str(tmp_path / "run"), display=False, mode="synthetic", max_frames=30)
    assert json.loads((d / "summary.json").read_text())["frames_processed"] == 30
