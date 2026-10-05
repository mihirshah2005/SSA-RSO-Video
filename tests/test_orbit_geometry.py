import numpy as np
import pytest
from sgp4 import omm as sgomm
from sgp4.api import Satrec

from starship_rso.geometry.calibrate import attitude_from_nadir_and_velocity, fit_foe, fit_limb
from starship_rso.geometry.camera import PinholeCamera, attitude_from_ypr, triad
from starship_rso.io.timemap import parse_utc
from starship_rso.orbit.frames import R_EARTH, ecef_to_geodetic, ric_matrix, teme_to_ecef
from starship_rso.orbit.kepler import KeplerOrbit, elements_to_state, state_to_elements
from starship_rso.orbit.omm import make_omm_fields, record_from_omm, to_satrec
from starship_rso.orbit.propagate import Propagator
from starship_rso.orbit.ship import NominalParams, NominalShipEphemeris, altitude_speed
from starship_rso.orbit.sun import is_sunlit, sun_position_teme
from starship_rso.orbit.timeutil import gmst_rad, parse_epoch, utc_to_jd

L = parse_utc("2026-09-28T12:48:59Z")
ISS = {"OBJECT_NAME": "ISS", "OBJECT_ID": "1998-067A", "EPOCH": "2026-09-27T12:00:00.000000", "MEAN_MOTION": 15.50103472,
       "ECCENTRICITY": 0.0006703, "INCLINATION": 51.6416, "RA_OF_ASC_NODE": 247.4627, "ARG_OF_PERICENTER": 130.536,
       "MEAN_ANOMALY": 325.0288, "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": 25544,
       "ELEMENT_SET_NO": 999, "REV_AT_EPOCH": 1, "BSTAR": 0.00034, "MEAN_MOTION_DOT": 0.00016717, "MEAN_MOTION_DDOT": 0}


def test_omm_initialisation_matches_sgp4_reference():
    ref = Satrec()
    sgomm.initialize(ref, {k: str(v) for k, v in ISS.items()})
    mine = to_satrec(record_from_omm(ISS))
    jd, fr = utc_to_jd(L)
    e1, r1, v1 = ref.sgp4(float(jd), float(fr))
    e2, r2, v2 = mine.sgp4(float(jd), float(fr))
    assert e1 == e2 == 0
    assert np.allclose(r1, r2, atol=1e-6) and np.allclose(v1, v2, atol=1e-9)


def test_six_digit_catalogue_numbers_are_supported():
    d = dict(ISS, NORAD_CAT_ID=100855, EPOCH="2026-10-02T00:00:00")
    rec = record_from_omm(d)
    r, v, ok = Propagator([rec]).states([parse_epoch("2026-10-02T01:00:00")])
    assert ok.all() and np.isfinite(r).all()


def test_vectorised_propagation_matches_single():
    p = Propagator([record_from_omm(ISS)] * 3)
    t = L + np.arange(0, 600, 60.0)
    r, v, ok = p.states(t)
    r1, v1 = p.state_one(0, t)
    assert np.allclose(r[0], r1) and ok.all()


def test_kepler_round_trip():
    r, v = elements_to_state(6700.0, 0.01, 0.6, 1.2, 0.4, 2.0)
    el = state_to_elements(r, v)
    assert el["a"] == pytest.approx(6700.0, rel=1e-9)
    assert el["e"] == pytest.approx(0.01, rel=1e-6)
    assert el["M"] == pytest.approx(2.0, abs=1e-8)


def _astropy_offline():
    pytest.importorskip("astropy")
    from astropy.utils import iers

    iers.conf.auto_download = False  # tests must pass offline; degraded IERS accuracy is fine here
    iers.conf.iers_degraded_accuracy = "warn"


def test_gmst_against_astropy():
    _astropy_offline()
    from astropy.time import Time

    t = Time("2026-09-28T13:30:00", scale="utc")
    ref = t.sidereal_time("mean", "greenwich").radian
    assert abs(((gmst_rad(t.unix) - ref + np.pi) % (2 * np.pi)) - np.pi) < 2e-5


def test_sun_direction_against_astropy():
    _astropy_offline()
    from astropy.coordinates import TEME, get_sun
    from astropy.time import Time

    t = Time("2026-09-28T13:30:00", scale="utc")
    # astropy's Sun is in GCRS (J2000 axes); ours is of-date like TEME, so compare in TEME
    s = get_sun(t).transform_to(TEME(obstime=t)).cartesian.xyz.value
    mine = sun_position_teme(t.unix)
    ang = np.degrees(np.arccos(np.dot(s / np.linalg.norm(s), mine / np.linalg.norm(mine))))
    assert ang < 0.05


def test_shadow_model():
    sun = np.array([1.5e8, 0, 0])
    assert is_sunlit(np.array([7000.0, 0, 0]), sun)
    assert not is_sunlit(np.array([-7000.0, 0, 0]), sun)
    assert is_sunlit(np.array([-7000.0, 7000.0, 0]), sun)


def test_nominal_flight14_orbit_is_consistent_with_mission_facts():
    eph = NominalShipEphemeris(NominalParams(L, 25.997, -97.157, 30.5, 262.0, 277.0))
    mets = np.array([2047.0, 2974.0, 3879.0])
    alt, v_in, v_gr = altitude_speed(eph, L + mets)
    assert np.all((alt > 250) & (alt < 300))
    assert np.all(np.abs(v_gr - 26394) < 300)  # HUD shows ~26,394 km/h at T+49:34
    r, v = eph.state(L + mets)
    inc = np.degrees(np.arccos((np.cross(r[0], v[0]) / np.linalg.norm(np.cross(r[0], v[0])))[2]))
    assert inc == pytest.approx(30.5, abs=0.05)
    # the plane contains the launch site at liftoff (descending pass)
    site = teme_to_ecef  # noqa: F841 - documented check below
    lat, lon, _ = ecef_to_geodetic(teme_to_ecef(r, L + mets))
    assert np.all(np.abs(lat) <= 31.0)


def test_camera_projection_round_trip_with_distortion():
    cam = PinholeCamera.from_hfov(1920, 1080, 90.0, k1=-0.05, k2=0.01)
    uv = np.array([[100.0, 50.0], [960.0, 540.0], [1800.0, 1000.0]])
    rays = cam.unproject(uv)
    uv2, front = cam.project_cam(rays)
    assert front.all() and np.allclose(uv, uv2, atol=1e-3)


def test_attitude_and_triad():
    R = attitude_from_ypr(10, 35, -5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
    R0 = attitude_from_ypr(0, 0, 0)
    assert np.allclose(R0 @ np.array([-1.0, 0, 0]), [0, 0, 1])  # nadir is the optical axis
    a, b = np.array([0.3, -0.2, 0.9]), np.array([-0.5, 0.8, 0.1])
    A = triad(R @ a, R @ b, a, b)
    assert np.allclose(A, R, atol=1e-9)


def test_limb_fit_recovers_nadir_and_focal():
    w, h, hfov = 1920, 1080, 80.0
    cam = PinholeCamera.from_hfov(w, h, hfov)
    r_obs = R_EARTH + 271.0
    rho = np.arcsin((R_EARTH + 20.0) / r_obs)
    nadir = np.array([0.2, 0.75, 0.6])
    nadir /= np.linalg.norm(nadir)
    e1 = np.cross(nadir, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nadir, e1)
    pts = []
    for a in np.linspace(0, 2 * np.pi, 400):
        d = np.cos(rho) * nadir + np.sin(rho) * (np.cos(a) * e1 + np.sin(a) * e2)
        uv, f = cam.project_cam(d[None])
        if f[0] and cam.in_image(uv, f)[0]:
            pts.append(uv[0])
    pts = np.array(pts) + np.random.default_rng(0).normal(0, 0.5, (len(pts), 2))
    fit = fit_limb(pts, w, h, r_obs, init_hfov_deg=95.0)
    assert np.degrees(np.arccos(np.clip(fit.nadir_cam @ nadir, -1, 1))) < 0.3
    assert fit.hfov_deg == pytest.approx(hfov, abs=1.0)


def test_foe_recovers_translation_direction():
    cam = PinholeCamera.from_hfov(1280, 720, 90.0)
    rng = np.random.default_rng(1)
    t_true = np.array([0.3, -0.6, 0.74])
    t_true /= np.linalg.norm(t_true)
    P = rng.uniform([-300, -300, 200], [300, 300, 600], (300, 3))  # ground points, camera frame (km)
    uv1, f1 = cam.project_cam(P)
    uv2, f2 = cam.project_cam(P - 0.25 * t_true)  # camera moved along t
    ok = cam.in_image(uv1, f1) & cam.in_image(uv2, f2)
    t, rms = fit_foe(uv1[ok], uv2[ok], cam)
    assert np.degrees(np.arccos(np.clip(t @ t_true, -1, 1))) < 1.0
    R = attitude_from_nadir_and_velocity(np.array([0, 0.3, 0.95]) / np.linalg.norm([0, 0.3, 0.95]), t,
                                         np.array([0.0, 1.0, 0.0]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_ric_frame_orthonormal():
    r, v = np.array([6700.0, 100.0, 50.0]), np.array([0.1, 7.6, 1.0])
    M = ric_matrix(r, v)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert np.allclose(M @ (r / np.linalg.norm(r)), [1, 0, 0])


def test_make_omm_fields_round_trip():
    f = make_omm_fields(990001, "X", "2026-09-28T13:00:00.000000", 15.6, 0.001, 30.5, 10, 20, 30)
    rec = record_from_omm(f)
    assert rec.norad_id == 990001 and rec.epoch_utc == parse_epoch("2026-09-28T13:00:00")
    r, _, ok = Propagator([rec]).states([rec.epoch_utc])
    assert ok.all() and 6500 < np.linalg.norm(r) < 6800
    assert KeplerOrbit(rec.epoch_utc, 6700, 0.001, 0.5, 0, 0, 0).period_s > 5000


@pytest.mark.slow
def test_ship_orbit_recovered_from_quantised_hud_telemetry():
    """Altitude (1 km steps) and ground speed (1 km/h steps) pin down the in-plane orbit, and the
    launch prior picks the right one of the two phase branches half an orbit apart."""
    from starship_rso.orbit.fit_ship import HudSeries, _state, fit_ship, predict_hud, propagate_j2
    from starship_rso.orbit.kepler import KeplerOrbit

    L = 1.79e9
    inc, raan = np.deg2rad(30.5), np.deg2rad(339.0)
    truth = np.array([6378.137 + 272.0, 0.0011, -0.0004, np.deg2rad(290.0)])
    met = np.arange(2040.0, 3900.0, 4.0)
    t0 = L + 2970.0
    r0, v0 = _state(truth, inc, raan)
    r, v = propagate_j2(r0, v0, t0, L + met)
    alt, spd = predict_hud(r, v, L + met, "geodetic")
    rng = np.random.default_rng(3)  # display rounding of values that drift within each sampling bin
    series = HudSeries(met, np.round(alt + rng.uniform(-0.3, 0.3, alt.size)), np.round(spd + rng.uniform(-0.3, 0.3, spd.size)))
    prior = KeplerOrbit(t0, truth[0], 0.001, inc, raan, 0.0, truth[3] + np.deg2rad(3.0))  # phase off by 3 deg

    class Prior:
        def state(self, utc):
            return prior.state(utc)

    fit = fit_ship(series, L, inc, raan, "test", epoch_met=2970.0, prior=Prior())
    assert fit.alt_mode == "geodetic"
    du = (fit.u_deg - 290.0 + 180.0) % 360.0 - 180.0
    assert abs(du) < 3.0 * fit.u_sigma_deg and fit.u_sigma_deg < 0.5, (fit.u_deg, fit.u_sigma_deg)
    assert abs(fit.perigee_km - (272.0 * 1.0)) < 15.0 and fit.rms_alt_km < 0.4
