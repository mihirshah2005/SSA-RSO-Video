"""Regression tests for defects found while validating on synthetic clips."""

import numpy as np

from starship_rso.classify.features import _robust_slope
from starship_rso.classify.rules import classify_rules
from starship_rso.config import ClassicalCfg, ClassifyCfg, StaticMaskCfg
from starship_rso.types import Category
from starship_rso.vision.classical import ClassicalDetector
from starship_rso.vision.masks import StaticStructureMask
from starship_rso.vision.registration import History, Registration
from test_tracking_classify import _features_for


def _spot(img, x, y, amp, s):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img += (amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))).astype(np.float32)


def test_resolved_object_gives_one_centred_detection():
    """A payload tens of pixels across (a flat rotated box, as the simulator renders it) drifting
    slowly against the moving Earth: motion evidence exists only at its leading edge, and each
    edge or corner used to become its own detection 10-12 px from the centre, hence several
    jittering tracks for one object."""
    import cv2

    from starship_rso.sim.render import add_box

    h, w = 400, 640
    rng = np.random.default_rng(4)
    det = ClassicalDetector(ClassicalCfg(), (h, w))
    hist = History(16)
    tex = cv2.GaussianBlur(rng.normal(0, 6, (h, w + 100)).astype(np.float32), (0, 0), 2.0) * 4 + 60
    H = np.array([[1, 0, -2.0], [0, 1, 0], [0, 0, 1]])  # background moves 2 px/frame left
    for k in range(24):
        f = tex[:, 2 * k : 2 * k + w].copy() + rng.normal(0, 1.0, (h, w)).astype(np.float32)
        add_box(f, 300.0 - k, 200.0, 17.5, 8.7, 150.0, angle_deg=20.0)  # 35 x 17 px, 1 px/frame
        _spot(f, 300.0 - k, 290.0, 60, 1.0)  # an unrelated point 90 px below, moving with it
        hist.advance(Registration(H, True))
        out = det.detect(f, hist.items(), np.ones((h, w), bool), k, k / 30)
        hist.push(f, np.ones((h, w), bool))
        if k >= 4:
            on_body = [d for d in out if np.hypot(d.x - 300 + k, d.y - 200) < 30]
            assert len(on_body) == 1, (k, [(round(d.x), round(d.y), d.source) for d in on_body])
            assert np.hypot(on_body[0].x - 300 + k, on_body[0].y - 200) < 1.5 and on_body[0].sigma > 5.0
            assert any(np.hypot(d.x - 300 + k, d.y - 290) < 1.5 for d in out), "separate point lost"


def test_vehicle_mask_reports_ready_only_after_warmup():
    m = StaticStructureMask(StaticMaskCfg(warmup_s=1.0, update_every=5), (120, 160))
    g = np.random.default_rng(0).uniform(0, 255, (120, 160)).astype(np.float32)
    for k in range(20):  # 0.63 s at 30 fps
        m.update(g, k / 30)
    assert not m.ready
    for k in range(20, 40):
        m.update(g, k / 30)
    assert m.ready
    assert StaticStructureMask(StaticMaskCfg(enabled=False), (10, 10)).ready


def test_size_trend_ignores_a_few_blended_frames():
    t = np.arange(60) / 30.0
    y = np.zeros(60)
    y[:5] = 1.5  # first frames blended with the vehicle edge
    assert abs(_robust_slope(t, y)) < 0.02


def test_point_first_seen_at_the_door_is_not_a_payload():
    """A distant object emerging from behind the vehicle near the door is point-like when first
    seen; a released payload, metres away, is resolved."""
    cfg = ClassifyCfg()
    point = _features_for(lambda t: (576 - 20 * t, 216 - 5 * t), lambda t: 400 * np.exp(-0.4 * t), 0.8, n=120)
    assert classify_rules(point, cfg).category != Category.PAYLOAD


def test_sparse_clustered_corners_do_not_overfit_perspective(rng):
    """Smooth ocean with a few clouds: a homography fitted to clustered, noisy corners used to
    extrapolate several pixels of false motion into the rest of the image."""
    import cv2

    from starship_rso.config import RegistrationCfg
    from starship_rso.vision.registration import BackgroundRegistrar, apply_h

    h, w = 360, 640
    tex = np.full((h + 40, w + 40), 90.0, np.float32)
    for _ in range(150):  # cloud puffs, all in the upper-left corner of the scene
        x, y = rng.uniform([20, 20], [w * 0.4, h * 0.35])
        _spot(tex, x, y, rng.uniform(30, 90), rng.uniform(1.5, 4.0))
    tex = cv2.GaussianBlur(tex, (0, 0), 0.8)
    errs = []
    reg = BackgroundRegistrar(RegistrationCfg(work_width=w), (h, w))
    grid = np.array([[x, y] for x in (10, w / 2, w - 10) for y in (10, h / 2, h - 10)], float)
    for k in range(12):
        a = tex[k * 2 : k * 2 + h, 0:w] + rng.normal(0, 2.0, (h, w)).astype(np.float32)
        b = tex[k * 2 + 2 : k * 2 + 2 + h, 0:w] + rng.normal(0, 2.0, (h, w)).astype(np.float32)  # content moves up 2 px
        r = reg.estimate(np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8), None, 1 / 30)
        assert r.valid, r.reason
        d = apply_h(r.H, grid) - grid
        errs.append(np.max(np.hypot(d[:, 0], d[:, 1] + 2.0)))
    assert np.max(errs) < 0.5, np.round(errs, 2)


# ---------------------------------------------------------------- second review pass
def test_faint_defocused_disc_is_measured_as_a_disc():
    """A near-field particle can be a large disc only 1-2 noise sigma above the background per
    pixel; re-measurement used to lock onto one noise spike (sigma ~0.3 px, positions off by 14 px)."""
    h, w = 270, 480
    yy, xx = np.mgrid[0:h, 0:w]
    for seed in range(3):
        rng = np.random.default_rng(seed)
        g = (20 + rng.normal(0, 3.0, (h, w)) + 4.0 * (np.hypot(xx - 240.3, yy - 130.6) <= 20)).astype(np.float32)
        det = ClassicalDetector(ClassicalCfg(clutter_thresh=5.0), (h, w))
        d = det.detect(g, [], np.ones((h, w), bool), static_background=True)
        near = [x for x in d if np.hypot(x.x - 240.3, x.y - 130.6) < 25]
        assert near, "disc not detected"
        b = max(near, key=lambda x: x.score)
        assert b.sigma > 6.0 and np.hypot(b.x - 240.3, b.y - 130.6) < 2.0, (b.sigma, b.x, b.y)


def test_measurement_near_the_image_edge_is_not_biased_by_repeated_pixels():
    from starship_rso.vision.classical import remeasure
    from starship_rso.types import Detection

    h, w = 120, 160
    g = np.full((h, w), 10.0, np.float32)
    _spot(g, 4.0, 60.0, 100.0, 3.0)
    d = remeasure(g, [Detection(x=4.0, y=60.0, score=5, confidence=0.9, sigma=3.0, pos_sigma=0.3)], noise=1.0)
    assert d and abs(d[0].x - 4.0) < 0.25, d[0].x


def test_noise_level_is_not_carried_into_a_new_shot():
    h, w = 200, 300
    rng = np.random.default_rng(0)
    det = ClassicalDetector(ClassicalCfg(), (h, w))
    det.detect((30 + rng.normal(0, 8.0, (h, w))).astype(np.float32), [], np.ones((h, w), bool), static_background=True)
    det.reset()
    quiet = (30 + rng.normal(0, 0.7, (h, w))).astype(np.float32)
    det.detect(quiet, [], np.ones((h, w), bool))  # first frame of the new shot finds nothing
    assert det._noise_hp is not None and det._noise_hp < 1.5


def test_frames_stored_during_mask_warmup_get_the_learned_vehicle_mask():
    """After warm-up, Earth points that were behind the ship in stored frames looked available to
    the motion residual: a band of false motion above the ship's edge for the first frames."""
    import cv2

    h, w = 270, 480
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h + 400, w)).astype(np.float32), (0, 0), 1.5)
    tex = np.clip(110 + 60 * tex / tex.std(), 0, 255).astype(np.float32)
    veh = np.zeros((h, w), bool)
    veh[200:, :] = True  # camera-fixed vehicle at the bottom; the Earth emerges from behind it
    vy = 6.0
    T = np.array([[1, 0, 0], [0, 1, -vy], [0, 0, 1.0]])
    cfg = ClassicalCfg()
    det = ClassicalDetector(cfg, (h, w))
    hist = History(max(cfg.history))
    band = []
    for k in range(20):
        g = tex[int(k * vy) : int(k * vy) + h].copy()
        g[veh] = 15.0 + 3 * rng.normal(0, 1, veh.sum())
        g = (g + rng.normal(0, 1.0, g.shape)).astype(np.float32)
        hist.advance(Registration(T, k > 0))
        warming = k < 10
        valid = np.ones((h, w), bool) if warming else ~veh
        if k == 10:  # what the pipeline does on its first frame with a vehicle mask
            hist.restrict(valid)
            det.restrict_history(valid)
        d = det.detect(g, hist.items(), valid, frame_index=k, report=not warming)
        if not warming:
            band.append(sum(1 for x in d if 150 <= x.y < 200 and x.score >= 0.6))
        hist.push(g, valid)
    assert sum(band) == 0, band
