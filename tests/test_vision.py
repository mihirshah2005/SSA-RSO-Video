import cv2
import numpy as np

from conftest import cloud_texture
from starship_rso.config import ClassicalCfg, MaskCfg, RegistrationCfg
from starship_rso.vision.classical import ClassicalDetector
from starship_rso.vision.masks import MaskManager, hud_mask
from starship_rso.vision.registration import BackgroundRegistrar, History, Registration


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _spot(img, x, y, amp=120.0, s=1.0):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32)


def test_hud_mask_covers_rects_and_border():
    m = hud_mask((100, 200), [[0.0, 0.8, 1.0, 1.0]], border_px=2)
    assert m[90, 100] and m[0, 100] and not m[50, 100]


def test_registration_recovers_translation_and_ignores_static_corners(rng):
    h, w = 360, 640
    base = cloud_texture(h, w, rng)
    a = base.copy()
    b = _shift(base, 4.0, -2.5)
    # camera-fixed structure: identical textured block in both frames
    block = cloud_texture(120, 160, np.random.default_rng(7)) * 0.5
    a[0:120, 480:640] = block
    b[0:120, 480:640] = block
    reg = BackgroundRegistrar(RegistrationCfg(work_width=640), (h, w)).estimate(
        a.astype(np.uint8), b.astype(np.uint8), None, 1 / 30
    )
    assert reg.valid
    assert reg.H[0, 2] == np.float64(reg.H[0, 2])
    assert abs(reg.H[0, 2] - 4.0) < 0.15 and abs(reg.H[1, 2] + 2.5) < 0.15
    vx, vy = reg.flow_at(100, 100)
    assert abs(vx - 120) < 6 and abs(vy + 75) < 6


def test_registration_flags_black_sky_as_static_background():
    z = np.full((240, 320), 3, np.uint8)
    reg = BackgroundRegistrar(RegistrationCfg(), (240, 320)).estimate(z, z, None, 1 / 30)
    assert not reg.valid and reg.static_background


def _run_detector(frames, valid=None, cfg=None):
    h, w = frames[0].shape
    det = ClassicalDetector(cfg or ClassicalCfg(), (h, w))
    reg = BackgroundRegistrar(RegistrationCfg(work_width=w), (h, w))
    hist = History(16)
    valid = np.ones((h, w), bool) if valid is None else valid
    out, prev = [], None
    for i, f in enumerate(frames):
        r = reg.estimate(prev.astype(np.uint8), f.astype(np.uint8), valid, 1 / 30) if prev is not None else Registration(np.eye(3), False)
        hist.advance(r)
        out.append(det.detect(f, hist.items(), valid, i, i / 30, static_background=r.static_background))
        hist.push(f, valid)
        prev = f
    return out


def test_moving_dot_over_moving_clouds_is_detected_and_clouds_are_not(rng):
    h, w = 270, 480
    base = cloud_texture(h, w, rng, scale=2.5)
    frames = []
    for k in range(12):
        f = _shift(base, 2.0 * k, 1.0 * k).copy()
        f += rng.normal(0, 1.5, f.shape).astype(np.float32)
        _spot(f, 100 + 6 * k, 200 - 3 * k, amp=110)
        frames.append(np.clip(f, 0, 255))
    dets = _run_detector(frames)
    last = [d for d in dets[-1] if d.confidence >= 0.5]
    near = [d for d in last if np.hypot(d.x - (100 + 66), d.y - (200 - 33)) < 2.0]
    assert near, "moving dot not detected"
    assert len(last) <= 6, f"too many false detections on moving clouds: {len(last)}"


def test_point_on_black_sky_detected_without_motion():
    h, w = 200, 300
    frames = []
    rng = np.random.default_rng(3)
    for k in range(4):
        f = np.full((h, w), 4.0, np.float32) + rng.normal(0, 1.0, (h, w)).astype(np.float32)
        _spot(f, 150.3, 80.6, amp=60)  # hovering: fixed in the image
        frames.append(np.clip(f, 0, 255))
    dets = _run_detector(frames)
    best = max(dets[-1], key=lambda d: d.confidence)
    assert abs(best.x - 150.3) < 0.3 and abs(best.y - 80.6) < 0.3
    assert best.confidence > 0.9


def test_vehicle_mask_learns_camera_fixed_structure(rng):
    h, w = 240, 320
    base = cloud_texture(h, w, rng, scale=2.0)
    ship = cloud_texture(h, w, np.random.default_rng(9), scale=1.0) * 0.6
    mm = MaskManager(MaskCfg(), (h, w))
    reg = BackgroundRegistrar(RegistrationCfg(work_width=w), (h, w))
    prev = None
    for k in range(60):
        f = _shift(base, 3.0 * k, 0.0).copy()
        f[:, 200:] = ship[:, 200:]
        r = reg.estimate(prev.astype(np.uint8), f.astype(np.uint8), mm.valid, 1 / 30) if prev is not None else Registration(np.eye(3), False)
        mm.update(f, k / 30, r.H, r.valid)
        prev = f
    inv = mm.invalid
    assert inv[:, 230:].mean() > 0.8, "vehicle not masked"
    assert inv[10:-10, 10:150].mean() < 0.02, "moving Earth wrongly masked"
