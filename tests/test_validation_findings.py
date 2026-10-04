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
