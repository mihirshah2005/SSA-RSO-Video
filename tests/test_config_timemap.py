import numpy as np
import pytest

from starship_rso.config import ConfigError, config_from_dict, load_config
from starship_rso.io.timemap import Anchor, TimeMap, anchors_from_clock_readings, format_met, parse_clock, parse_utc


def test_default_and_mission_configs_load():
    cfg = load_config(["configs/default.yaml", "configs/flight14.yaml"], ["detector.classical.snr_res=7"])
    assert cfg.detector.classical.snr_res == 7.0
    assert cfg.mission.deployment.count == 26
    assert cfg.mission.liftoff_utc.startswith("2026-09-28")
    assert len(cfg.digest()) == 12


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError):
        config_from_dict({"detector": {"clasical": {}}})
    with pytest.raises(ConfigError):
        config_from_dict({"tracker": {"confirm_hits": True}})


def test_parse_clock_and_format():
    assert parse_clock("T+00:49:34") == 2974
    assert parse_clock("T - 00:00:10") == -10
    assert parse_clock("garbage") is None
    assert format_met(2974.4) == "T+00:49:34"


def test_timemap_interpolates_and_extrapolates():
    tm = TimeMap([Anchor(10.0, 100.0, 0.1), Anchor(20.0, 110.0, 0.1)], parse_utc("2026-09-28T12:48:59Z"))
    m, s = tm.met(15.0)
    assert m == pytest.approx(105.0)
    m, s = tm.met(30.0)  # extrapolate with slope 1
    assert m == pytest.approx(120.0)
    assert s > 0.1
    u, _ = tm.utc(10.0)
    assert u == pytest.approx(parse_utc("2026-09-28T12:48:59Z") + 100.0)


def test_timemap_segments_break_at_replay():
    # second pair jumps back 60 s (a replay): no interpolation across the break
    tm = TimeMap([Anchor(0, 100), Anchor(10, 110), Anchor(20, 60), Anchor(30, 70)])
    assert tm.segment_of(5) != tm.segment_of(25)
    m, _ = tm.met(14.0)  # closer to the 10 s anchor -> slope 1 from it
    assert m == pytest.approx(114.0)


def test_anchors_from_clock_readings_recover_offset():
    rng = np.random.default_rng(0)
    offset = 2974.0 - 2986.0 + 0.37  # met = video + offset
    t = np.arange(2980.0, 3000.0, 0.2) + rng.uniform(0, 0.02, 100)
    readings = [(float(v), float(np.floor(v + offset))) for v in t]
    readings[30] = (readings[30][0], readings[30][1] + 7)  # an OCR error
    anchors = anchors_from_clock_readings(readings)
    assert anchors
    tm = TimeMap(anchors)
    m, _ = tm.met(2990.0)
    assert m == pytest.approx(2990.0 + offset, abs=0.12)
