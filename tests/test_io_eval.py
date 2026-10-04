import numpy as np

from starship_rso.eval.metrics import detection_metrics, identity_metrics, match_points, tracking_metrics
from starship_rso.io.cvat import LabeledPoint, LabeledTrack, export_tracks, import_tracks, interpolate_track
from starship_rso.orbit.photometry import apparent_magnitude, feasibility_table


def test_cvat_round_trip(tmp_path):
    tr = [LabeledTrack(3, "payload_candidate", [LabeledPoint(10, 1.5, 2.5), LabeledPoint(11, 2.5, 3.5)],
                       {"identity": "100855", "identity_source": "test"})]
    p = tmp_path / "a.xml"
    export_tracks(tr, p, 1920, 1080, 100)
    back, meta = import_tracks(p)
    assert meta == {"width": 1920, "height": 1080}
    assert back[0].label == "payload_candidate" and len(back[0].visible()) == 2
    assert back[0].attributes["identity"] == "100855"
    assert back[0].visible()[1].x == 2.5


def test_interpolate_track_fills_gaps():
    tr = LabeledTrack(1, "unknown", [LabeledPoint(0, 0, 0), LabeledPoint(4, 8, 4)])
    out = interpolate_track(tr)
    assert [p.frame for p in out.points] == [0, 1, 2, 3, 4]
    assert out.points[2].x == 4.0


def test_match_points_one_to_one():
    pairs, d = match_points(np.array([[0, 0], [10, 0]]), np.array([[0.5, 0], [0.6, 0], [30, 0]]), 3.0)
    assert len(pairs) == 1


def test_detection_and_tracking_metrics_perfect_case():
    T = [[{"id": "a", "x": 1.0 * k, "y": 0.0, "kind": "catalog"}] for k in range(10)]
    P = [[{"x": 1.0 * k + 0.2, "y": 0.0, "confidence": 0.9}] for k in range(10)]
    m = detection_metrics(T, P, tols=(1.0,))
    assert m["tol1"]["precision"] == 1.0 and m["tol1"]["recall"] == 1.0
    TR = [[{"id": 7, "x": 1.0 * k, "y": 0.1}] for k in range(10)]
    tm = tracking_metrics(T, TR, tol=1.0)
    assert tm["idf1"] == 1.0 and tm["id_switches"] == 0 and tm["mota"] == 1.0


def test_identity_metrics_counts_wrong_names():
    truth = {1: "A", 2: "B", 3: None}
    dec = {1: ("accepted", "A"), 2: ("accepted", "C"), 3: ("candidates", "A")}
    m = identity_metrics(truth, dec)
    assert m["accepted"] == 2 and m["accepted_wrong"] == 1 and m["coverage"] == 0.5


def test_photometry_scales_with_range():
    m1 = apparent_magnitude(1000.0, np.pi / 2, 4.5)
    m2 = apparent_magnitude(100.0, np.pi / 2, 4.5)
    assert m1 == np.float64(4.5) or abs(m1 - 4.5) < 1e-9
    assert abs((m1 - m2) - 5.0) < 1e-9
    rows = feasibility_table()
    assert rows[1]["px_30.0m"] > 3.0  # 30 m object at 10 km is a few pixels


def test_shot_list_finds_cuts_and_flags_flashes(tmp_path):
    import cv2

    from starship_rso.config import load_config
    from starship_rso.io.shots import find_shots, write_shots

    w, h = 320, 180
    rng = np.random.default_rng(0)
    earth = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3)).astype(np.uint8), (0, 0), 4)
    sky = np.full((h, w, 3), 5, np.uint8)
    vw = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for k in range(150):  # earth 2 s, sky 2 s, a 3-frame flash, then earth again
        vw.write(earth if k < 60 else sky if k < 120 else (np.full((h, w, 3), 250, np.uint8) if k < 123 else earth))
    vw.release()
    cfg = load_config(["configs/default.yaml", "configs/flight14.yaml"])
    shots, thumbs = find_shots(cfg, str(tmp_path / "v.mp4"), every=1)
    assert [round(s.video_start) for s in shots[:3]] == [0, 2, 4]
    assert any(s.short for s in shots) and shots[1].mean_level < 10
    assert shots[0].met_start is not None  # the mission time map applies
    out = write_shots(shots, thumbs, tmp_path / "shots")
    assert (out / "shots.csv").exists() and (out / "contact_sheet.jpg").exists()
