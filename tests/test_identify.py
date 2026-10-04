import numpy as np

from starship_rso.config import IdentifyCfg
from starship_rso.geometry.camera import PinholeCamera, attitude_from_ypr
from starship_rso.identify.association import CatalogAssociator, TrackObs, score_track
from starship_rso.identify.deployment_id import EventTime, match_release_events
from starship_rso.identify.predictions import PredictionCache
from starship_rso.orbit.release import ReleaseEstimate
from starship_rso.types import CategoryDecision, Detection, IdentityStatus, TrackPoint


GOOD = IdentifyCfg(ephem_sigma_km=0.1)  # good relative ephemeris: geometry is informative at 12-15 km


def _cache(t, rels, names, ship_sigma_km=0.0):
    T = len(t)
    return PredictionCache(t=t, ids=[str(i) for i in range(len(rels))], names=names, rel_ric=np.stack(rels),
                           sunlit=np.ones((len(rels), T), bool), ship_r=np.tile([6650.0, 0, 0], (T, 1)),
                           ship_v=np.tile([0, 7.7, 0], (T, 1)), ship_description="test", ship_sigma_km=ship_sigma_km)


def _scene():
    cam = PinholeCamera.from_hfov(1280, 720, 90.0)
    cam.R_cam_ric = attitude_from_ypr(0, 30, 0)
    t = np.arange(0.0, 10.0, 0.25) + 1.79e9
    tt = t - t[0]
    d0 = cam.R_cam_ric.T @ np.array([0.1, -0.05, 1.0])
    d0 /= np.linalg.norm(d0)
    perp = np.cross(d0, [0, 0, 1.0])
    perp /= np.linalg.norm(perp)
    a = 12.0 * d0[None] + 0.2 * tt[:, None] * perp[None]  # 12 km, crossing at 0.2 km/s
    b = 15.0 * d0[None] - 0.2 * tt[:, None] * perp[None]  # same start direction, opposite motion
    c = -400.0 * d0[None] + 0 * tt[:, None]  # behind the camera
    return cam, t, _cache(t, [a, b, c], ["SAT-A", "SAT-B", "SAT-C"])


def _obs_from(cache, cam, k, t_obs, noise=0.4, seed=0):
    rel = cache.rel_at(t_obs)[k]
    uv, _ = cam.project_ric(rel / np.linalg.norm(rel, axis=1, keepdims=True))
    uv += np.random.default_rng(seed).normal(0, noise, uv.shape)
    return TrackObs(t_obs, uv, np.full(len(t_obs), 0.5), 1.0)


def test_track_from_known_object_is_identified():
    cam, t, cache = _scene()
    t_obs = t[0] + np.arange(0, 3, 1 / 30)
    cands, pu, diag = score_track(_obs_from(cache, cam, 0, t_obs), cam, cache, GOOD)
    assert cands[0].name == "SAT-A" and cands[0].posterior > 0.95 and pu < 0.05
    assert diag["n_candidates_in_view"] == 2  # SAT-C is behind the camera


def test_uncatalogued_track_prefers_unknown():
    cam, t, cache = _scene()
    t_obs = t[0] + np.arange(0, 3, 1 / 30)
    uv = np.c_[np.linspace(100, 300, len(t_obs)), np.linspace(600, 500, len(t_obs))]
    cands, pu, _ = score_track(TrackObs(t_obs, uv, np.full(len(t_obs), 0.5), 1.0), cam, cache, GOOD)
    assert pu > 0.9


def test_resolved_blob_cannot_match_distant_point():
    cam, t, cache = _scene()
    t_obs = t[0] + np.arange(0, 3, 1 / 30)
    obs = _obs_from(cache, cam, 0, t_obs)
    obs.psf_sigma_px = 12.0  # a large resolved object
    cands, pu, diag = score_track(obs, cam, cache, GOOD)
    assert not cands and diag["n_rejected_by_size"] >= 1


class _FakeTrack:
    def __init__(self, uid, obs, cat="unknown"):
        from starship_rso.types import Category

        self.uid = uid
        self.category = CategoryDecision(Category.parse(cat))
        self._pts = [TrackPoint(i, float(u), float(x), float(y), 0, 0,
                                Detection(x=float(x), y=float(y), score=2, confidence=0.9, sigma=1.0, pos_sigma=0.5),
                                None, None, float(u)) for i, (u, (x, y)) in enumerate(zip(obs.utc, obs.uv_native))]
        from starship_rso.types import IdentityDecision

        self.identity_geo = IdentityDecision()

    def observed_points(self):
        return self._pts


def test_associator_accepts_after_two_stable_evaluations_and_resolves_conflicts():
    cam, t, cache = _scene()
    t_obs = t[0] + np.arange(0, 3, 1 / 30)
    good = _FakeTrack(1, _obs_from(cache, cam, 0, t_obs))
    dup = _FakeTrack(2, _obs_from(cache, cam, 0, t_obs, noise=3.0, seed=5))
    particle = _FakeTrack(3, _obs_from(cache, cam, 1, t_obs), cat="near_field_particle")
    assoc = CatalogAssociator(GOOD, cam, cache)
    assoc.evaluate([good, particle], 1.0)
    assert good.identity_geo.status == IdentityStatus.CANDIDATES  # first evaluation: not yet stable
    assert particle.identity_geo.status == IdentityStatus.NONE
    assoc.evaluate([good, particle], 1.5)
    assert good.identity_geo.status == IdentityStatus.ACCEPTED and "SAT-A" in good.identity_geo.label
    for k in range(3):
        assoc.evaluate([good, dup], 2.0 + k)
    statuses = {good.identity_geo.status, dup.identity_geo.status}
    assert IdentityStatus.ACCEPTED not in statuses or good.identity_geo.status == IdentityStatus.ACCEPTED


def test_uncalibrated_camera_disables_identity():
    cam, t, cache = _scene()
    cam.R_cam_ric = None
    tr = _FakeTrack(1, TrackObs(t[:20], np.zeros((20, 2)) + 100, np.ones(20), 1.0))
    CatalogAssociator(IdentifyCfg(), cam, cache).evaluate([tr], 0.0)
    assert tr.identity_geo.status == IdentityStatus.UNAVAILABLE
    assert "not calibrated" in tr.identity_geo.reasons[0]


def test_release_event_matching_accepts_only_separable_estimates():
    base = 1.79e9
    est = [ReleaseEstimate(str(100 + k), f"S{k}", base + 70 * k, 3.0, 0.1, "ship") for k in range(5)]
    events = [EventTime(f"DEPLOY-{k + 1:02d}", base + 70 * k + 1.0, 2.0) for k in range(5)]
    out = match_release_events(events, est, window_s=500)
    assert all(d.status == IdentityStatus.ACCEPTED for d in out.values())
    assert "S2" in out["DEPLOY-03"].label
    flat = [ReleaseEstimate(e.norad_id, e.name, base, float("inf"), 0.1, "ship") for e in est]
    out = match_release_events(events, flat, window_s=500)
    assert all(d.status != IdentityStatus.ACCEPTED for d in out.values())


def test_uninformative_geometry_is_never_accepted():
    """Review finding: with km-level ephemeris error, a nearby object's predicted direction carries
    no information, so even a dot that matches its angular rate must not be named."""
    cam, t, _ = _scene()
    tt = t - t[0]
    d0 = cam.R_cam_ric.T @ np.array([0.0, 0.0, 1.0])
    near = 0.6 * d0[None] + 0 * tt[:, None]  # a payload 0.6 km away, predicted at the image centre
    cache = _cache(t, [near], ["STARLINK-NEAR"], ship_sigma_km=2.0)
    t_obs = t[0] + np.arange(0, 3, 1 / 30)
    uv = np.tile([[200.0, 150.0]], (len(t_obs), 1))  # stationary dot far from the prediction
    tr = _FakeTrack(1, TrackObs(t_obs, uv, np.full(len(t_obs), 0.5), 1.0))
    assoc = CatalogAssociator(IdentifyCfg(), cam, cache)
    for k in range(4):
        assoc.evaluate([tr], 1.0 + k)
    assert tr.identity_geo.status != IdentityStatus.ACCEPTED
    assert any("not informative" in r for r in tr.identity_geo.reasons)
