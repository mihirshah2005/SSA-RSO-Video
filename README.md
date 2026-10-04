# Video_CV: small-object detection, tracking and identification in Starship onboard video

Real-time pipeline for the white dots seen around Starship in onboard footage (Flight 14, Ship 41, Starlink V3 deployment). It detects and tracks every moving point, classifies what each track most likely is, and attaches a catalogue identity **only when the evidence is unique**. It says `unknown`, lists candidates, or shows `identity unavailable` (with the reason) otherwise.

## Read this first: what is feasible

- **Most dots are near-field particles** (ice, frost, vented material) within metres of the camera. No catalogue can name them.
- **The only catalogued objects close enough to see are Flight 14's own 26 Starlink V3s** leaving the payload door (T+34:07 to T+1:04:39).
- **Distant satellites are sub-pixel and far too faint** for a camera exposed for the sunlit Earth.
- **SpaceX publishes no Starship ephemeris or attitude**, so geometric matching needs the camera calibration and ship-orbit reconstruction built in here, each with stated uncertainty.

Full numbers and sources: [docs/FEASIBILITY.md](docs/FEASIBILITY.md). The live demo of exact names runs end to end on synthetic clips with known truth, clearly labelled SYNTHETIC; on real footage the system shows what the data support.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
python -m pytest -q                                   # 69 tests incl. end-to-end synthetic runs (-m "not slow" to skip those)

rso simulate --out data/sim/seed0 --duration 20       # synthetic clip + truth + catalogue + camera
rso run -c configs/default.yaml -c data/sim/seed0/mission.yaml --video data/sim/seed0/video.mp4 --mode synthetic
rso evaluate --run runs/<newest> --truth data/sim/seed0/truth.json --skip 45
```

Real footage: follow [docs/MANUAL_STEPS.md](docs/MANUAL_STEPS.md) (download, HUD/time anchors, catalogue, calibration, labels, GPU training).

## What you see in the demo

| On screen | Meaning |
|---|---|
| `T012 particle?` (blue) | tracked; behaves like a near-field particle (defocused / fast / flickering) |
| `T031 payload?` + `DEPLOY-13` (green) | departing object first seen at the door; release slot 13 of 26 from the schedule |
| `T040 unknown` | tracked, category not supported |
| `3 candidates, best X p=0.62` | geometry informative but not unique |
| `ID: STARLINK-xxxx (100861)` (yellow) | accepted: unique, stable, passes the chi-square and conflict tests |
| magenta diamond `NAME 12 km (pred)` | where a catalogued object is predicted (needs a calibrated camera) |
| panel | MET/UTC with uncertainty, data mode, fps and p95 latency, counts, reasons |

## Architecture

```
video ─► time map ─► cut detection ─► registration (moving Earth) ─► masks (HUD + ship)
      ─► detector (classical | temporal heatmap | fused) ─► tracker (Kalman + Hungarian, 2 stages)
      ─► category evidence ─► [2 Hz] catalogue association + release-slot identity ─► overlay + logs
```

Design details and decision rules: [docs/DESIGN.md](docs/DESIGN.md).

## Repository layout

```
configs/            default.yaml (pipeline), flight14.yaml (mission facts), train_heatmap.yaml
starship_rso/
  io/               video + paced live reader, time map, HUD OCR, CVAT, run logs
  vision/           registration, masks, classical detector, learned-detector wrapper
  tracking/         Kalman filter, two-stage tracker
  classify/         track features, rule classifier, GBM model, release events
  orbit/            OMM/SGP4, catalogue clients, ship ephemerides, Sun, photometry, screen, release times
  geometry/         camera model, limb + focus-of-expansion calibration
  identify/         prediction cache, track-catalogue association, release-event identities
  sim/              synthetic clips with independent truth
  train/            frame stores, temporal U-Net, training, YOLO export
  eval/             detection / tracking / identity metrics
  pipeline/         context, per-frame pipeline, overlay, runner
  cli.py            the `rso` command
cluster/            Slurm script for the NUS SoC cluster
notebooks/          Colab training notebook
docs/               FEASIBILITY, DESIGN, MANUAL_STEPS, ANNOTATION_GUIDE
tests/              unit tests + end-to-end synthetic test
```

## Commands

| Command | Purpose |
|---|---|
| `rso feasibility` | visibility and error-budget tables |
| `rso extract-clip` | cut a segment with ffmpeg |
| `rso check-layout` | verify HUD mask, OCR boxes, door position on a frame |
| `rso ocr` | HUD clock/speed/altitude → CSV and time anchors |
| `rso shots` | camera cuts over a stretch of video → shot list and contact sheet |
| `rso fetch-catalog` | CelesTrak / Space-Track element sets → cached OMM JSON |
| `rso ship` | ship ephemeris table to cross-check the HUD |
| `rso screen` | catalogued objects that came within range of the ship |
| `rso release-times` | release-time estimates for the deployed group, with a separability verdict |
| `rso calibrate` | camera FOV and pointing from the Earth limb and background flow |
| `rso simulate` | synthetic clip with truth |
| `rso run` | live/replay pipeline with overlay and logs |
| `rso evaluate` | metrics vs synthetic truth or CVAT labels |
| `rso export-cvat`, `rso make-dataset`, `rso export-yolo`, `rso train-classifier` | labelling and training workflow |

## Current measured status (synthetic, 960x540, classical detector, CPU container)

Three 20 s clips (600 frames each), first 1.5 s skipped while the vehicle mask is learned. Background: moving Earth with sparse clouds, an Earth limb, and black sky.

| Measure | Earth | Limb | Black sky |
|---|---|---|---|
| Detection recall @3 px, catalogued passes | 0.96 | 0.92 | 0.95 |
| Detection recall @3 px, released payloads | 0.99 | 0.49 | 1.00 |
| Detection recall @3 px, near-field particles | 0.50 | 0.43 | 0.70 |
| Tracking MOTA / IDF1 | 0.83 / 0.88 | 0.58 / 0.60 | 0.74 / 0.79 |
| Release events (truth: 2) | 2 | 2 | 2 |
| Accepted catalogue identities (correct / wrong) | 3 / 0 | 3 / 0 | 1 / 0 |
| Processing time per frame, p50 / p95 | 50 / 77 ms | 49 / 69 ms | 44 / 64 ms |

- **Identity**: no wrong name was accepted in any clip; the rest stay `candidates` or `unknown` with a stated reason.
- **Weak spots**: payloads drifting along the bright limb edge (limb clip), flickering particles, and many low-confidence detections on black sky (they rarely become tracks).
- **Speed**: the live reader drops frames rather than lag; `--set processing.scale=0.75` buys headroom.

These are software-validation numbers on synthetic data, not results on real footage. Real-footage numbers come after the manual steps.

## Honest-reporting rules built into the code

- Track ids are never satellite ids; `DEPLOY-k` is a release slot, not a catalogue number.
- Retrospective catalogue use is labelled on screen and in `provenance.json`; `--catalog-mode as_of` restricts to element sets that existed at the frame time.
- Identity decisions carry their decision time and are logged; nothing rewrites history.
- Synthetic results are always labelled SYNTHETIC and reported separately.
