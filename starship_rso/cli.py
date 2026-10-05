"""Command-line interface: ``rso <command> ...`` (or ``python -m starship_rso.cli``).

Commands, in the order a project usually uses them:

  feasibility      visibility / error-budget tables behind docs/FEASIBILITY.md
  extract-clip     cut a segment from a downloaded video with ffmpeg
  check-layout     draw HUD mask and OCR boxes on a frame to verify the config
  ocr              read the T+ clock / speed / altitude from the HUD -> CSV (+ fitted anchors)
  fetch-catalog    download element sets (CelesTrak / Space-Track) into a cached OMM file
  ship             print the ship ephemeris (altitude/speed) for HUD cross-checks
  screen           which catalogued objects came within range of the ship during a window
  release-times    estimate release times of a deployed group from its element sets
  calibrate        camera pointing from the Earth limb (+ optional focus of expansion)
  simulate         render a synthetic video with full ground truth
  run              live / replay pipeline with overlay, logs and summary
  evaluate         metrics of a run against synthetic truth or CVAT labels
  export-cvat      export a run's tracks as CVAT XML for correction
  make-dataset     frame store + labels + splits for training the heatmap detector
  export-yolo      tiled YOLO dataset from a frame store (baseline)
  train-classifier gradient-boosting category model from labelled tracks
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("rso")


def load_tracks(run_dir):
    from .pipeline.runner import load_tracks as _lt

    return _lt(run_dir)


def _cfg(args):
    from .config import load_config

    paths = list(args.config or [])
    if not paths and Path("configs/default.yaml").exists():
        paths = ["configs/default.yaml"]
    return load_config(paths, args.set)


def _add_cfg(p):
    p.add_argument("-c", "--config", action="append", help="YAML config (repeatable; later overrides earlier)")
    p.add_argument("--set", action="append", default=[], help="override, e.g. --set detector.type=fused")


# ------------------------------------------------------------------ commands
def cmd_feasibility(args):
    from .orbit.photometry import feasibility_table

    rows = feasibility_table(hfov_deg=args.hfov, width_px=args.width)
    print(f"Apparent size (px) and rough magnitude, HFOV {args.hfov} deg over {args.width} px "
          f"(IFOV {np.rad2deg(np.deg2rad(args.hfov) / args.width) * 3600:.0f} arcsec)")
    print(f"{'range km':>9} {'3U 0.34 m':>10} {'3 m bus':>9} {'30 m':>8} {'mag (large sat)':>16}")
    for r in rows:
        print(f"{r['range_km']:>9} {r['px_0.34m']:>10.3f} {r['px_3.0m']:>9.3f} {r['px_30.0m']:>8.2f} {r['mag_large_sat']:>16.1f}")
    ifov = np.deg2rad(args.hfov) / args.width
    print("\nError budget (illustrative):")
    print(f"  0.1 deg pointing error   = {np.deg2rad(0.1) / ifov:.1f} px")
    for rng in (10, 100, 1000):
        print(f"  1 km position error @ {rng:>4} km = {np.degrees(1.0 / rng):.3f} deg = {(1.0 / rng) / ifov:.1f} px")


def cmd_extract_clip(args):
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found")
    cmd = ["ffmpeg", "-y", "-ss", str(args.start), "-i", args.video, "-t", str(args.duration)]
    cmd += ["-c", "copy"] if args.copy else ["-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-an"]
    cmd.append(args.out)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"note: clip time 0 corresponds to source time {args.start} s; shift time anchors accordingly")


def cmd_check_layout(args):
    import cv2

    from .io.video import read_frame_at
    from .vision.masks import hud_mask

    cfg = _cfg(args)
    frame = read_frame_at(args.video, args.t)
    h, w = frame.shape[:2]
    m = hud_mask((h, w), cfg.masks.hud_rects, cfg.masks.border_px)
    vis = frame.copy()
    vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for name in ("clock", "speed", "altitude"):
        box = getattr(cfg.hud, name)
        if box:
            x0, y0, x1, y1 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, name, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if cfg.deployment.door_xy:
        cv2.circle(vis, (int(cfg.deployment.door_xy[0] * w), int(cfg.deployment.door_xy[1] * h)), 12, (0, 255, 255), 2)
    cv2.imwrite(args.out, vis)
    print(f"wrote {args.out} ({w}x{h}); red = masked HUD, green = OCR boxes, yellow = payload door")


def cmd_ocr(args):
    from .io.ocr import ocr_video
    from .io.timemap import anchors_from_clock_readings
    from .io.video import VideoSource

    cfg = _cfg(args)
    rows = ocr_video(VideoSource(args.video, args.start, args.end), cfg.hud, args.every, args.out)
    readings = [(r["video_s"], r["met_s"]) for r in rows if r["met_s"] is not None]
    anchors = anchors_from_clock_readings(readings)
    print(f"{len(readings)}/{len(rows)} clock readings, {len(anchors)} anchors")
    print("timemap:\n  anchors:")
    for a in anchors:
        print(f"    - {{video_s: {a.video_s:.3f}, met_s: {a.met_s:.3f}, sigma_s: {a.sigma_s:.3f}}}")


def cmd_shots(args):
    from .io.shots import find_shots, write_shots

    cfg = _cfg(args)
    shots, thumbs = find_shots(cfg, args.video, args.start, args.end, every=args.every)
    out = write_shots(shots, thumbs, args.out)
    long_ = [s for s in shots if not s.short]
    print(f"{len(shots)} shots ({len(shots) - len(long_)} shorter than 1 s, likely flashes or graphics)")
    for s in long_:
        met = f"MET {s.met_start:8.1f} .. {s.met_end:8.1f}" if s.met_start is not None else "MET unknown"
        print(f"  S{s.shot:02d}  video {s.video_start:8.1f} .. {s.video_end:8.1f}  ({s.duration:6.1f} s)  {met}  level {s.mean_level:5.1f}")
    print(f"wrote {out / 'shots.csv'} and {out / 'contact_sheet.jpg'}")


def cmd_fetch_catalog(args):
    from .orbit.catalog import CelesTrakClient, SpaceTrackClient, write_catalog

    if args.source == "celestrak":
        cl = CelesTrakClient(Path(args.cache))
        q = {args.query: args.value}
        fn = {"gp": cl.gp, "gp-first": cl.gp_first, "supgp": cl.supgp}[args.endpoint]
        recs = fn(**q)
    else:
        st = SpaceTrackClient(cache_dir=Path(args.cache))
        if args.norad_range:
            a, b = (int(v) for v in args.norad_range.split("-"))
            recs = st.gp_history(a, b, args.epoch_from, args.epoch_to)
        else:
            recs = st.gp_history_window(args.epoch_from, args.epoch_to)
    write_catalog(recs, args.out, note=f"{args.source} {vars(args)}")
    names = sorted({r.name for r in recs})
    print(f"{len(recs)} element sets, {len({r.norad_id for r in recs})} objects -> {args.out}")
    print("first names:", names[:10])


def _liftoff(cfg):
    from .io.timemap import parse_utc

    if not cfg.mission.liftoff_utc:
        sys.exit("mission.liftoff_utc is not set")
    return parse_utc(cfg.mission.liftoff_utc)


def _ship(cfg, liftoff):
    from .orbit.omm import load_catalog, load_omm_json
    from .orbit.ship import GroupCentroidEphemeris, NominalShipEphemeris, SGP4ShipEphemeris

    spec = cfg.mission.ship_ephemeris
    if spec == "nominal":
        return NominalShipEphemeris.from_mission(cfg.mission, liftoff)
    if spec.startswith("omm:"):
        return SGP4ShipEphemeris(load_omm_json(spec[4:])[0])
    if spec.startswith("fitted:"):
        from .orbit.fit_ship import FittedShipEphemeris

        return FittedShipEphemeris.load(spec[7:])
    if spec == "group_centroid":
        from .orbit.catalog import select_records

        pg = cfg.mission.payload_group
        recs = select_records(load_catalog(cfg.mission.catalog_file), pg.norad_range, pg.name_contains, pg.intdes)
        return GroupCentroidEphemeris(recs)
    sys.exit(f"unknown ship_ephemeris {spec}")


def cmd_ship(args):
    from .orbit.frames import ecef_to_geodetic, teme_to_ecef
    from .orbit.ship import altitude_speed

    cfg = _cfg(args)
    L = _liftoff(cfg)
    ship = _ship(cfg, L)
    mets = np.arange(args.met_from, args.met_to + 1e-9, args.step)
    alt, vin, vgr = altitude_speed(ship, L + mets)
    r, _ = ship.state(L + mets)
    lat, lon, _ = ecef_to_geodetic(teme_to_ecef(r, L + mets))
    print(ship.description)
    print(f"{'MET':>8} {'alt km':>8} {'v_inertial km/h':>16} {'v_ground km/h':>14} {'lat':>7} {'lon':>8}")
    for m, a, b, c, la, lo in zip(mets, alt, vin, vgr, lat, lon):
        print(f"{m:>8.0f} {a:>8.1f} {b:>16.0f} {c:>14.0f} {la:>7.2f} {lo:>8.2f}")
    print("Compare with the HUD (the SpaceX overlay speed matches the ground-relative value).")


def cmd_fit_ship(args):
    from .io.timemap import TimeMap
    from .orbit.fit_ship import FittedShipEphemeris, fit_ship, load_hud_series, plane_from_group, plane_from_site
    from .orbit.frames import ecef_to_geodetic, teme_to_ecef
    from .orbit.ship import NominalShipEphemeris

    cfg = _cfg(args)
    L = _liftoff(cfg)
    tm = TimeMap.from_config(cfg.timemap, cfg.mission.liftoff_utc)
    if not tm.is_calibrated:
        sys.exit("the time map has no anchors: run `rso ocr` first")
    series = load_hud_series(args.hud, tm, every_s=args.every)
    epoch_met = float(np.median(series.met))
    nominal = NominalShipEphemeris.from_mission(cfg.mission, L)
    if args.group:
        from .orbit.catalog import select_records
        from .orbit.omm import load_omm_json

        pg = cfg.mission.payload_group
        recs = select_records(load_omm_json(args.group), pg.norad_range, pg.name_contains, pg.intdes)
        first: dict = {}
        for r in recs:  # earliest element set per object: least orbit raising since release
            if r.norad_id not in first or r.epoch_utc < first[r.norad_id].epoch_utc:
                first[r.norad_id] = r
        inc, raan, n = plane_from_group(list(first.values()), L + epoch_met)
        src = f"payload group ({n} objects, earliest element sets)"
    else:
        from .orbit.kepler import j2_rates

        m = cfg.mission
        inc, raan0 = plane_from_site(L, m.launch_site.lat_deg, m.launch_site.lon_deg, m.orbit_nominal.inc_deg,
                                     m.orbit_nominal.launch_pass)
        rd, _, _ = j2_rates(6378.137 + 0.5 * (m.orbit_nominal.perigee_km + m.orbit_nominal.apogee_km), 0.001, inc)
        raan = raan0 + rd * epoch_met
        src = "launch site + inclination"
    print(f"HUD series: {len(series.met)} samples over MET {series.met.min():.0f}-{series.met.max():.0f} s "
          f"(altitude {np.isfinite(series.alt_km).sum()}, speed {np.isfinite(series.speed_kmh).sum()})")
    print(f"orbital plane from {src}: i = {np.rad2deg(inc):.3f} deg, RAAN = {np.rad2deg(raan) % 360:.3f} deg (TEME)")
    fit = fit_ship(series, L, inc, raan, src, epoch_met=epoch_met, prior=nominal)
    fit.save(args.out)
    print("altitude convention   rms altitude   rms speed   (display resolution alone gives 0.29 / 0.29)")
    for mode, a in fit.alternatives.items():
        mark = "  <- best" if mode == fit.alt_mode else ""
        print(f"  {mode:<12}        {a['rms_alt_km']:7.3f} km   {a['rms_speed_kmh']:7.3f} km/h{mark}")
    print(f"orbit {fit.perigee_km:.1f} x {fit.apogee_km:.1f} km, i {fit.inc_deg:.3f} deg; along-track 1-sigma "
          f"{fit.u_sigma_deg:.3f} deg = {fit.position_sigma_km:.1f} km")
    eph = FittedShipEphemeris(fit)
    print("   MET    lat      lon     alt   offset from nominal (km)")
    for met in (args.met_marks or [500, 2063, 2976, 3034, 3878]):
        r, v = eph.state(L + met)
        rn, _ = nominal.state(L + met)
        lat, lon, alt = ecef_to_geodetic(teme_to_ecef(r, L + met))
        print(f"  {met:6.0f} {float(lat):7.2f} {float(lon):8.2f} {float(alt):7.1f}   {np.linalg.norm(rn - r):8.1f}")
    print(f"wrote {args.out}; use it with  mission.ship_ephemeris: fitted:{args.out}")


def cmd_screen(args):
    from .orbit.conjunction import phase_sweep, screen
    from .orbit.omm import load_catalog
    from .orbit.propagate import Propagator

    cfg = _cfg(args)
    L = _liftoff(cfg)
    ship = _ship(cfg, L)
    recs = load_catalog(args.catalog or cfg.mission.catalog_file)
    if args.best_epoch:
        from .orbit.catalog import best_record_per_object

        recs = best_record_per_object(recs, L + args.met_from, "retrospective")
    if args.no_phase_sweep:
        ships, spacing = [ship], 0.0
    else:
        ships, spacing = phase_sweep(ship, args.range_km)
    extra = 0.5 * spacing
    if spacing == 0 and np.isfinite(getattr(ship, "position_sigma_km", np.nan)):
        extra = 3.0 * ship.position_sigma_km  # a single ship estimate: widen by its 3-sigma uncertainty
    hits = screen(Propagator(recs), ships, L + args.met_from, L + args.met_to, args.range_km,
                  ifov_rad=np.deg2rad(cfg.camera.hfov_deg) / args.width, extra_radius_km=extra)
    print(f"ship: {ship.description}")
    if spacing > 0:
        print(f"phase sweep: {len(ships)} ship hypotheses {spacing:.0f} km apart; screen radius widened by "
              f"{extra:.0f} km so nothing falls between them")
    elif extra > 0:
        print(f"screen radius widened by {extra:.0f} km (3 x the ship's position uncertainty)")
    out = [h.to_dict() for h in hits]
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(recs)} objects screened over MET {args.met_from}-{args.met_to} s with {len(ships)} ship hypotheses")
    print(f"{len(hits)} within {args.range_km} km; visible by the photometric model: {sum(h.visible_est for h in hits)}")
    epoch = {str(r.key): r.epoch_utc for r in recs}
    for h in hits[:25]:
        age = (epoch.get(str(h.norad_id), np.nan) - h.t_closest_utc) / 86400.0
        flag = "  STALE: >1 day of propagation" if abs(age) > 1.0 else ""
        print(f"  {h.norad_id:>7} {h.name[:24]:<24} {h.range_min_km:8.1f} km  MET {h.t_closest_utc - L:7.1f}  "
              f"lit={h.sunlit} mag~{h.mag_est:5.1f} size~{h.size_px_est:6.2f}px visible={h.visible_est}  "
              f"elements {age:+.1f} d{flag}")


def cmd_release_times(args):
    from .orbit.catalog import best_record_per_object, select_records
    from .orbit.omm import load_catalog
    from .orbit.release import estimate_release_times

    cfg = _cfg(args)
    L = _liftoff(cfg)
    d = cfg.mission.deployment
    pg = cfg.mission.payload_group
    path = args.catalog or cfg.mission.catalog_file
    if not path:
        sys.exit("no catalogue: pass --catalog or set mission.catalog_file (see `rso fetch-catalog`)")
    if d.first_met_s is None or d.last_met_s is None:
        sys.exit("mission.deployment.first_met_s / last_met_s are not set")
    recs = select_records(load_catalog(path), pg.norad_range, pg.name_contains, pg.intdes)
    recs = best_record_per_object(recs, L + (d.first_met_s or 0), "retrospective")
    if not recs:
        sys.exit("no payload-group records selected (check mission.payload_group)")
    ship = None
    if cfg.mission.ship_ephemeris.startswith(("omm:", "fitted:")):
        ship = _ship(cfg, L)
        print(f"ship: {ship.description}")
    else:
        print("WARNING: no independent ship ephemeris (mission.ship_ephemeris=omm:<file> or fitted:<file>); the group "
              "centroid is a biased reference, so these release times cannot identify release slots")
    est = estimate_release_times(recs, L + d.first_met_s - 120, L + d.last_met_s + 120, ship=ship,
                                 along_track_sigma_km=args.along_track_sigma_km)
    sp = (d.last_met_s - d.first_met_s) / max(d.count - 1, 1)
    print(f"{len(est)} objects; release spacing {sp:.1f} s; method {est[0].method}")
    for e in sorted(est, key=lambda e: (not np.isfinite(e.t_sigma_s), e.t_release_utc)):
        note = "" if np.isfinite(e.t_sigma_s) else "  (never near the ship: elements not valid back to release)"
        print(f"  {e.norad_id:>7} {e.name[:24]:<24} MET {e.t_release_utc - L:8.1f} +/- {e.t_sigma_s:6.1f} s  "
              f"min sep {e.min_sep_km:7.2f} km{note}")
    ages = [(r.epoch_utc - L) / 86400.0 for r in recs]
    print(f"element sets used: epochs {min(ages):.2f}-{max(ages):.2f} days after liftoff")
    med = float(np.median([e.t_sigma_s for e in est]))
    verdict = "separable" if (ship is not None and med < 0.3 * sp) else "NOT separable (identity stays a candidate list)"
    print(f"median sigma {med:.1f} s vs spacing {sp:.1f} s -> {verdict}")
    Path(args.out).write_text(json.dumps([e.to_dict() for e in est], indent=1))


def cmd_calibrate(args):
    import cv2

    from .geometry.calibrate import attitude_from_nadir_and_velocity, extract_limb_points, fit_foe, fit_limb
    from .geometry.camera import PinholeCamera
    from .io.video import read_frame_at
    from .orbit.frames import R_EARTH, ric_matrix, teme_velocity_relative_to_ground
    from .vision.masks import hud_mask
    from .vision.registration import BackgroundRegistrar

    cfg = _cfg(args)
    frame = read_frame_at(args.video, args.t)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args.limb_points:
        pts = np.array(json.loads(Path(args.limb_points).read_text()), float)
    else:
        pts = extract_limb_points(gray, hud_mask((h, w), cfg.masks.hud_rects, cfg.masks.border_px))
    r_obs = R_EARTH + args.altitude_km
    fit = fit_limb(pts, w, h, r_obs, args.hfov, fix_focal=args.fix_focal)
    print(f"limb: {fit.n_points} points, rms {fit.rms_deg:.3f} deg, HFOV {fit.hfov_deg:.2f} deg, nadir_cam {np.round(fit.nadir_cam, 4)}")
    cam = PinholeCamera(w, h, fit.fx, fit.fx, (w - 1) / 2, (h - 1) / 2)
    cam.meta = {"limb_rms_deg": fit.rms_deg, "limb_points": fit.n_points, "t": args.t, "video": args.video}
    if args.foe_dt > 0:
        L = _liftoff(cfg)
        from .io.timemap import TimeMap

        tm = TimeMap.from_config(cfg.timemap, cfg.mission.liftoff_utc)
        utc, _ = tm.utc(args.t)
        if utc is None:
            sys.exit("time anchors needed for the velocity direction (timemap.anchors)")
        f2 = read_frame_at(args.video, args.t + args.foe_dt)
        reg = BackgroundRegistrar(cfg.registration, (h, w))
        mask = ~hud_mask((h, w), cfg.masks.hud_rects, cfg.masks.border_px)
        r = reg.estimate(gray, cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), mask, args.foe_dt)
        if not r.valid:
            sys.exit(f"registration failed ({r.reason}); cannot estimate the focus of expansion")
        t_cam, rms = fit_foe(r.pts_prev, r.pts_cur, cam)
        ship = _ship(cfg, L)
        rs, vs = ship.state(utc)
        vg_ric = ric_matrix(rs, vs) @ teme_velocity_relative_to_ground(rs, vs)
        cam.R_cam_ric = attitude_from_nadir_and_velocity(fit.nadir_cam, t_cam, vg_ric)
        cam.meta.update({"foe_cam": t_cam.tolist(), "foe_rms": rms, "foe_dt": args.foe_dt})
        print(f"FOE direction (camera) {np.round(t_cam, 4)}, residual {rms:.2e}; attitude set from nadir + velocity")
    else:
        print("no --foe-dt: intrinsics + nadir only (attitude about nadir undetermined, association stays disabled)")
    cam.save(args.out)
    print(f"wrote {args.out}")


def cmd_simulate(args):
    from .sim.scene import SimSpec, simulate

    spec = SimSpec(width=args.width, height=args.height, duration_s=args.duration, seed=args.seed,
                   background=args.background, n_passes=args.passes, n_releases=args.releases,
                   n_particles=args.particles)
    out = simulate(args.out, spec, progress=True)
    print(f"synthetic clip in {out}. Run it with:\n  rso run --video {out}/video.mp4 -c configs/default.yaml "
          f"-c {out}/mission.yaml --mode synthetic")


def cmd_run(args):
    from .pipeline.runner import run

    cfg = _cfg(args)
    if args.no_realtime:
        cfg.realtime.enabled = False
    d = run(cfg, args.video, args.out, args.start, args.end, display=not args.no_display, mode=args.mode,
            catalog_mode=args.catalog_mode, max_frames=args.max_frames)
    s = json.loads((d / "summary.json").read_text())
    print(json.dumps({k: s[k] for k in ("frames_processed", "frames_dropped", "proc_ms_p50", "proc_ms_p95",
                                        "latency_ms_p95", "confirmed_tracks", "categories", "identity_status")}, indent=1))
    print(f"outputs in {d}")


def cmd_evaluate(args):
    from .eval.metrics import assign_tracks_to_truth, detection_metrics, identity_metrics, tracking_metrics

    run = Path(args.run)
    frames = [json.loads(line) for line in open(run / "frames.jsonl", encoding="utf-8")]
    if args.truth:
        truth = json.loads(Path(args.truth).read_text())
        tf = truth["frames"]
        kinds = {o: v["kind"] for o, v in truth["objects"].items()}
        ids = {o: v.get("norad_id") for o, v in truth["objects"].items()}
    else:
        from .io.cvat import import_tracks

        tracks, _ = import_tracks(args.cvat, args.frame_offset)
        n = max(f["f"] for f in frames) + 1
        tf = [[] for _ in range(n)]
        kinds, ids = {}, {}
        for tr in tracks:
            for p in tr.visible():
                if 0 <= p.frame < n:
                    # ambiguous or occluded points are neither required nor counted as false positives
                    ign = p.occluded or p.attributes.get("visibility") == "ambiguous"
                    tf[p.frame].append({"id": str(tr.track_id), "x": p.x, "y": p.y,
                                        "kind": "ignore" if ign else tr.label, "r": 4.0})
            kinds[str(tr.track_id)] = tr.label
            ids[str(tr.track_id)] = tr.attributes.get("identity") or None
    by_f = {f["f"]: f for f in frames}
    idx = sorted(i for i in by_f if i < len(tf))
    if args.skip:
        idx = idx[args.skip :]
    T = [tf[i] for i in idx]
    T_scored = [[o for o in fr if o.get("kind") != "ignore"] for fr in T]
    P = [[{"x": d["x"], "y": d["y"], "confidence": d["confidence"]} for d in by_f[i]["dets"]] for i in idx]
    TR = [[{"id": t["uid"], "x": t["x"], "y": t["y"]} for t in by_f[i]["tracks"] if t["state"] == "confirmed" and t["obs"]]
          for i in idx]
    real_kinds = set(kinds.values()) - {"ignore"}
    res = {"detection": detection_metrics(T, P, kinds=real_kinds, ignore_kinds={"ignore"}, min_conf=args.min_conf),
           "detection_by_kind": {k: detection_metrics(T, P, (3.0,), kinds={k}, ignore_kinds=(real_kinds - {k}) | {"ignore"},
                                                      min_conf=args.min_conf)["tol3"] for k in sorted(real_kinds)},
           "tracking": tracking_metrics(T_scored, TR, tol=args.track_tol)}
    T = T_scored
    amap = assign_tracks_to_truth(T, TR, tol=args.track_tol)
    tracks = load_tracks(run)
    truth_id = {t["uid"]: (ids.get(amap.get(t["uid"])) if amap.get(t["uid"]) else None) for t in tracks}
    dec = {t["uid"]: (t["identity_status"], (t["identity_candidates"][0]["id"] if t["identity_candidates"] else None))
           for t in tracks}
    res["identity"] = identity_metrics(truth_id, dec)
    res["category_vs_truth_kind"] = {}
    for t in tracks:
        g = amap.get(t["uid"])
        key = f"{kinds.get(g, 'no-truth-match')} -> {t['category']}"
        res["category_vs_truth_kind"][key] = res["category_vs_truth_kind"].get(key, 0) + 1
    out = Path(args.out or run / "metrics.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_export_cvat(args):
    from .io.cvat import LabeledPoint, LabeledTrack, export_tracks

    run = Path(args.run)
    tracks = load_tracks(run)
    prov = json.loads((run / "provenance.json").read_text())
    w, h = prov["size"]
    out = []
    n_seen = 0
    lo = args.frame_offset
    hi = lo + args.n_frames if args.n_frames else None
    for t in tracks:
        obs = [p for p in t["points"] if p["obs"] and p["f"] >= lo and (hi is None or p["f"] < hi)]
        if len(obs) < args.min_obs:
            continue
        out.append(LabeledTrack(t["uid"], t["category"], [LabeledPoint(p["f"], p["dx"], p["dy"]) for p in obs]))
        n_seen = max(n_seen, obs[-1]["f"] + 1 - lo)
    n_frames = args.n_frames or n_seen
    export_tracks(out, args.out, w, h, n_frames, frame_offset=lo)
    print(f"{len(out)} tracks -> {args.out} (CVAT frame 0 = source frame {lo}, {n_frames} frames)")


def cmd_make_dataset(args):
    from .train.data import build_frame_store, labels_from_cvat, labels_from_run, labels_from_sim, make_splits, write_labels

    cfg = _cfg(args)
    store = build_frame_store(cfg, args.video, args.out, args.start, args.end, args.stride, args.max_frames)
    s = cfg.processing.scale
    if args.sim_truth:
        lab = labels_from_sim(args.sim_truth, s)
    elif args.cvat:
        lab = labels_from_cvat(args.cvat, s, args.frame_offset)
    elif args.run:
        lab = labels_from_run(args.run, s, frame_offset=args.frame_offset)
    else:
        lab = {"points": {}, "ignore": {}, "source": "none"}
    write_labels(store, lab)
    stored = {f["i"] for f in json.loads((Path(store) / "meta.json").read_text())["frames"]}
    overlap = len(stored & {int(k) for k, v in lab["points"].items() if v})
    if lab["points"] and overlap == 0:
        print("WARNING: no labelled frame matches a stored frame; check --frame-offset and --start")
    print(f"labelled frames in the store: {overlap}")
    sp = make_splits(store, args.frames, args.val_frac, args.test_frac, args.block_s, args.seed)
    print(f"store {store}: labels from {lab['source']}; split sizes " + str({k: len(v) for k, v in sp.items() if k != 'meta'}))


def cmd_export_yolo(args):
    from .train.export_yolo import export_yolo

    out = export_yolo(args.store, args.out, args.tile, every=args.every)
    print(f"YOLO dataset in {out}: yolo detect train data={out}/data.yaml model=yolo11s.pt imgsz={args.tile}")


def cmd_train_classifier(args):
    from .classify.model import TrackCategoryModel

    feats, labels = [], []
    offsets = args.frame_offsets or [0] * len(args.runs)
    if not (len(args.runs) == len(args.labels) == len(offsets)):
        sys.exit("--runs, --labels and --frame-offsets need the same number of entries")
    for run_dir, cvat, off in zip(args.runs, args.labels, offsets):
        tracks = load_tracks(run_dir)
        from .eval.metrics import assign_tracks_to_truth
        from .io.cvat import import_tracks

        lt, _ = import_tracks(cvat, off)
        frames = [json.loads(line) for line in open(Path(run_dir, "frames.jsonl"), encoding="utf-8")]
        n = max(f["f"] for f in frames) + 1
        tf = [[] for _ in range(n)]
        lab_of = {}
        for t in lt:
            lab_of[str(t.track_id)] = t.label
            for p in t.visible():
                if p.frame < n:
                    tf[p.frame].append({"id": str(t.track_id), "x": p.x, "y": p.y})
        TR = [[{"id": t["uid"], "x": t["x"], "y": t["y"]} for t in f["tracks"] if t["state"] == "confirmed" and t["obs"]]
              for f in frames]
        amap = assign_tracks_to_truth([tf[f["f"]] for f in frames], TR)
        for t in tracks:
            g = amap.get(t["uid"])
            if g is None or not t["features"]:
                continue
            feats.append({k: (v if v is not None else float("nan")) for k, v in t["features"].items()})
            labels.append(lab_of[g])
    if len(set(labels)) < 2:
        sys.exit("need at least two labelled categories")
    model = TrackCategoryModel.train(feats, labels)
    model.save(args.out)
    print(f"trained on {len(labels)} tracks {dict(zip(*np.unique(labels, return_counts=True)))} -> {args.out}")


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rso", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("feasibility", help="visibility and error-budget tables")
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--width", type=int, default=1920)
    p.set_defaults(fn=cmd_feasibility)

    p = sub.add_parser("extract-clip", help="cut a segment with ffmpeg")
    p.add_argument("video")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--copy", action="store_true", help="stream copy (fast, keyframe-aligned)")
    p.add_argument("--crf", type=int, default=12)
    p.set_defaults(fn=cmd_extract_clip)

    p = sub.add_parser("check-layout", help="draw HUD mask/OCR boxes on a frame")
    _add_cfg(p)
    p.add_argument("--video", required=True)
    p.add_argument("--t", type=float, default=0.0)
    p.add_argument("--out", default="layout_check.png")
    p.set_defaults(fn=cmd_check_layout)

    p = sub.add_parser("ocr", help="OCR the HUD clock/speed/altitude")
    _add_cfg(p)
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--every", type=float, default=0.2)
    p.add_argument("--out", default="hud_readings.csv")
    p.set_defaults(fn=cmd_ocr)

    p = sub.add_parser("shots", help="camera cuts over a stretch of video, with a contact sheet")
    _add_cfg(p)
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--every", type=int, default=3, help="analyse every n-th frame")
    p.add_argument("--out", default="data/shots")
    p.set_defaults(fn=cmd_shots)

    p = sub.add_parser("fetch-catalog", help="download element sets into an OMM JSON file")
    p.add_argument("--source", choices=["celestrak", "spacetrack"], default="celestrak")
    p.add_argument("--endpoint", choices=["gp", "gp-first", "supgp"], default="gp")
    p.add_argument("--query", default="GROUP", help="CATNR | INTDES | GROUP | NAME | FILE | SOURCE")
    p.add_argument("--value", default="starlink")
    p.add_argument("--norad-range", help="Space-Track: e.g. 100855-100880")
    p.add_argument("--epoch-from", help="Space-Track: YYYY-MM-DD")
    p.add_argument("--epoch-to", help="Space-Track: YYYY-MM-DD")
    p.add_argument("--cache", default="data/catalog_cache")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fetch_catalog)

    p = sub.add_parser("ship", help="ship ephemeris table for HUD cross-checks")
    _add_cfg(p)
    p.add_argument("--met-from", type=float, default=1500)
    p.add_argument("--met-to", type=float, default=4000)
    p.add_argument("--step", type=float, default=250)
    p.set_defaults(fn=cmd_ship)

    p = sub.add_parser("fit-ship", help="fit the ship orbit to the HUD altitude/speed readings")
    _add_cfg(p)
    p.add_argument("--hud", default="hud_readings.csv", help="CSV written by `rso ocr`")
    p.add_argument("--group", help="OMM JSON of the deployed payload group (gives the orbital plane)")
    p.add_argument("--every", type=float, default=2.0, help="seconds per fitted sample")
    p.add_argument("--met-marks", type=float, nargs="*", help="METs at which to print the ship position")
    p.add_argument("--out", default="data/ship/ship_fit.json")
    p.set_defaults(fn=cmd_fit_ship)

    p = sub.add_parser("screen", help="catalogued objects near the ship")
    _add_cfg(p)
    p.add_argument("--catalog")
    p.add_argument("--met-from", type=float, required=True)
    p.add_argument("--met-to", type=float, required=True)
    p.add_argument("--range-km", type=float, default=100.0)
    p.add_argument("--no-phase-sweep", action="store_true",
                   help="use only the central ship ephemeris (default: sweep the along-track uncertainty)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--best-epoch", action="store_true", help="one element set per object, nearest the window")
    p.add_argument("--out", default="screen.json")
    p.set_defaults(fn=cmd_screen)

    p = sub.add_parser("release-times", help="release-time estimates for the payload group")
    _add_cfg(p)
    p.add_argument("--catalog")
    p.add_argument("--along-track-sigma-km", type=float, default=1.0)
    p.add_argument("--out", default="release_times.json")
    p.set_defaults(fn=cmd_release_times)

    p = sub.add_parser("calibrate", help="camera intrinsics/pointing from the Earth limb (+ FOE)")
    _add_cfg(p)
    p.add_argument("--video", required=True)
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--altitude-km", type=float, required=True, help="from the HUD or the ship ephemeris")
    p.add_argument("--limb-points", help="JSON [[x, y], ...] clicked limb points (default: automatic)")
    p.add_argument("--hfov", type=float, default=90.0, help="initial guess")
    p.add_argument("--fix-focal", action="store_true")
    p.add_argument("--foe-dt", type=float, default=0.0, help="seconds between frames for the focus of expansion")
    p.add_argument("--out", default="camera.json")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("simulate", help="render a synthetic clip with truth")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--background", choices=["earth", "black", "limb"], default="earth")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--releases", type=int, default=2)
    p.add_argument("--particles", type=int, default=14)
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("run", help="run the pipeline (live replay by default)")
    _add_cfg(p)
    p.add_argument("--video", required=True, help="file, direct stream URL, or page URL (needs yt-dlp)")
    p.add_argument("--out")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-realtime", action="store_true", help="process every frame as fast as possible")
    p.add_argument("--mode", choices=["replay", "live", "synthetic"], default="replay")
    p.add_argument("--catalog-mode", choices=["retrospective", "as_of"], default="retrospective")
    p.add_argument("--max-frames", type=int)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("evaluate", help="metrics vs synthetic truth or CVAT labels")
    p.add_argument("--run", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--truth", help="synthetic truth.json")
    g.add_argument("--cvat", help="CVAT XML")
    p.add_argument("--frame-offset", type=int, default=0)
    p.add_argument("--skip", type=int, default=0, help="ignore the first N frames (warm-up)")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--track-tol", type=float, default=6.0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("export-cvat", help="export a run's tracks to CVAT XML")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-obs", type=int, default=8)
    p.add_argument("--frame-offset", type=int, default=0, help="source frame index of the CVAT clip's frame 0")
    p.add_argument("--n-frames", type=int, help="number of frames in the CVAT task (points outside are dropped)")
    p.set_defaults(fn=cmd_export_cvat)

    p = sub.add_parser("make-dataset", help="frame store + labels + splits for the heatmap detector")
    _add_cfg(p)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sim-truth")
    src.add_argument("--cvat")
    src.add_argument("--run", help="pipeline run directory (pseudo-labels)")
    p.add_argument("--frame-offset", type=int, default=0)
    p.add_argument("--frames", type=int, default=3, help="model window, used as the split gap")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--block-s", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_make_dataset)

    p = sub.add_parser("export-yolo", help="tiled YOLO dataset from a frame store")
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--every", type=int, default=5)
    p.set_defaults(fn=cmd_export_yolo)

    p = sub.add_parser("train-classifier", help="GBM category model from labelled tracks")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="CVAT XML per run (same order)")
    p.add_argument("--frame-offsets", nargs="+", type=int, help="source frame of each CVAT clip's frame 0")
    p.add_argument("--out", default="category_model.pkl")
    p.set_defaults(fn=cmd_train_classifier)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
