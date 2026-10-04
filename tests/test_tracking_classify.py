import numpy as np

from starship_rso.classify.deployment import DeploymentMonitor
from starship_rso.classify.features import track_features
from starship_rso.classify.rules import classify_rules
from starship_rso.config import ClassifyCfg, DeploymentScheduleCfg, TrackerCfg
from starship_rso.tracking.kalman import KalmanCV
from starship_rso.tracking.tracker import Tracker
from starship_rso.types import Category, Detection, FrameInfo, TrackState


def _det(x, y, conf=0.9, sigma=1.0, flux=100.0):
    return Detection(x=x, y=y, score=2.0, confidence=conf, sigma=sigma, flux=flux, pos_sigma=0.3)


def test_kalman_converges_on_constant_velocity():
    kf = KalmanCV(0, 0, 0.5, 300, 100.0)
    for k in range(1, 30):
        kf.predict(1 / 30)
        kf.update(np.array([3.0 * k, -1.5 * k]), 0.1)
    assert abs(kf.vel[0] - 90) < 3 and abs(kf.vel[1] + 45) < 3


def test_two_crossing_targets_keep_identities(rng):
    tr = Tracker(TrackerCfg())
    for k in range(60):
        t = k / 30
        a = (100 + 120 * t, 100 + 60 * t)
        b = (100 + 120 * t, 160 - 60 * t)  # crosses a at t = 0.5 s
        dets = [_det(a[0] + rng.normal(0, 0.3), a[1] + rng.normal(0, 0.3)),
                _det(b[0] + rng.normal(0, 0.3), b[1] + rng.normal(0, 0.3))]
        tr.update(dets, FrameInfo(k, t, 640, 480))
    live = [t for t in tr.tracks if t.is_confirmed]
    assert len(live) == 2
    for t in live:
        vy = t.kf.vel[1]
        ys = [p.y for p in t.history[-5:]]
        # a track that ends moving down must have started at the top and vice versa (no swap)
        start_y = t.history[0].y
        assert (vy > 0 and start_y < 130) or (vy < 0 and start_y > 130), (vy, start_y, ys)


def test_random_clutter_rarely_confirms(rng):
    tr = Tracker(TrackerCfg())
    for k in range(90):
        dets = [_det(*rng.uniform(0, [960, 540])) for _ in range(10)]
        tr.update(dets, FrameInfo(k, k / 30, 960, 540))
    assert len(tr.all_confirmed()) <= 3


def test_camera_cut_kills_tracks():
    tr = Tracker(TrackerCfg())
    for k in range(10):
        tr.update([_det(50 + k, 50)], FrameInfo(k, k / 30, 640, 480))
    assert tr.live_tracks(confirmed_only=True)
    tr.update([], FrameInfo(10, 10 / 30, 640, 480, is_cut=True))
    assert not tr.tracks
    assert all(t.state == TrackState.DEAD for t in tr.finished)


def _features_for(path_fn, flux_fn, sigma, n=60, bg=(0.0, 60.0)):
    tr = Tracker(TrackerCfg())
    for k in range(n):
        t = k / 30
        x, y = path_fn(t)
        tr.update([_det(x, y, sigma=sigma, flux=flux_fn(t))], FrameInfo(k, t, 960, 540, met=2100 + t),
                  bg_flow=lambda *_: bg)
    track = tr.live_tracks(True)[0]
    return track_features(track.history, 960, 540, door_xy=[0.6, 0.4])


def test_rules_separate_particle_vehicle_and_payload():
    cfg = ClassifyCfg()
    particle = _features_for(lambda t: (500 - 250 * t, 200 + 80 * t * t), lambda t: 100 * (1 + 0.8 * np.sin(20 * t)), 6.0)
    assert classify_rules(particle, cfg).category == Category.NEAR_FIELD_PARTICLE
    glint = _features_for(lambda t: (700.0, 300.0), lambda t: 80.0, 1.2)
    assert classify_rules(glint, cfg).category == Category.VEHICLE_FEATURE
    payload = _features_for(lambda t: (576 - 20 * t, 216 - 5 * t), lambda t: 400 * np.exp(-0.4 * t), 4.0, n=120)
    assert classify_rules(payload, cfg).category == Category.PAYLOAD
    cloud = _features_for(lambda t: (300.0, 100 + 60 * t), lambda t: 50.0, 1.5)
    assert classify_rules(cloud, cfg).category == Category.BACKGROUND_FEATURE


def test_deployment_schedule_index():
    mon = DeploymentMonitor(DeploymentScheduleCfg(first_met_s=2047, last_met_s=3879, count=26))
    k, ks = mon.schedule_index(2974.0)
    assert k == 13 or k == 14
    assert mon.spacing > 70
