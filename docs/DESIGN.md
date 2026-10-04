# Design

## Principle: four outputs, never merged

| Output | Meaning | Evidence | Where |
|---|---|---|---|
| Detection | an image feature exists here now | appearance + motion relative to the Earth | `vision/` |
| Track | the same feature across frames | Kalman prediction + assignment | `tracking/` |
| Category | a supported physical class, or `unknown` | track features, rules or a trained model | `classify/` |
| Identity | a catalogue object, only if unique | ship ephemeris + camera + element sets, or a release slot plus independent data | `identify/`, `orbit/`, `geometry/` |

Confidence numbers are kept separate too: detection confidence, category scores and identity posteriors are never multiplied together.

## Data flow (per frame, causal)

```
video ──► time map (video t → MET → UTC, with sigma)
      ──► cut detection ──(cut)──► reset masks, history, tracks
      ──► registration (prev → cur motion of the Earth; simplest adequate model; camera-fixed corners ignored)
      ──► masks (HUD + vehicle structure learned from registration)
      ──► detector (classical | heatmap | fused; held, not reported, until the vehicle mask exists)
      ──► tracker (two-stage assignment, M-of-N confirmation, coasting)
      ──► features + category (every 5 frames)
      ──► slow loop at 2 Hz: catalogue association + release-event identity
      ──► overlay + JSONL audit log
```

Slow, one-off work happens before the first frame (`pipeline/context.py`): load the catalogue cache, build the ship ephemeris, propagate every candidate on a 0.25 s grid in the ship's RIC frame, estimate release times. Nothing touches the network during a run.

## Modules

### `io/`
- `video.py`: PyAV timestamps when available, OpenCV otherwise. `PacedReader` releases frames on the wall clock and drops stale ones, so the display never drifts behind the source; the tracker still integrates over the true interval.
- `timemap.py`: piecewise-linear video→MET map from anchors; segments break at replays; OCR tick detection gives sub-second anchors and rejects misreads.
- `ocr.py`: optional Tesseract reading of the HUD clock, speed and altitude.
- `cvat.py`: CVAT 1.1 point-track import/export for annotation round trips.

### `vision/`
- `registration.py`: LK corners + forward/backward check + RANSAC. Corners that do not move are dropped when enough corners move, so the ship cannot pull the fit to the identity. When corners are scarce (smooth ocean), they are searched again with a bar ten times the noise-level corner response, so weak cloud texture counts but pixel noise does not. The motion model steps down from homography to affine to similarity: a more general model is used only when its inliers are numerous and spread out **and** it fits clearly better than the simpler one; an over-fitted homography on clustered corners extrapolated several pixels of false motion into the rest of the image. The background counts as moving only if the moving corners are numerous, cover a real part of the image and are explained far better by the motion model than by no motion; otherwise (particles crossing a black sky in front of a dark, noisy hull) the background is fixed in the image. Black sky (no texture **and** dark) is a static background; a bright featureless field is "unmeasurable motion", not static.
- `masks.py`: vehicle mask from the long-run frequency of "unchanged in camera coordinates but made worse by registration" on textured pixels. Slow movers (a payload drifting off) do not accumulate that evidence. On a fixed background (black sky) registration carries no such evidence, so the mask is everything steady and clearly above the sky level. Until the mask has been estimated for a shot (`masks.static.warmup_s`), the detector only fills its history and reports nothing (the overlay says so); on the first frame with a mask, the stored frames get it too.
- `classical.py`: multi-scale top-hat; appearance SNR calibrated on the distribution of *peaks* (the top-hat of noise is biased); motion residual sampled only at peaks by mapping them into registered past frames (max-filtered, min over several baselines; coarse scales use longer baselines). Residuals are unavailable where the past background was hidden by the vehicle, which removes the band of false motion along the ship's edge. Appearance-only detections are allowed only on a static background (black sky) in low-clutter areas (median local high-pass over a window much larger than any object). Every kept detection is re-measured on the full-resolution frame: the object is the connected region above 20% of its own peak, twice the noise and 1.5 times the texture spread on the window border (so a dot on clouds does not merge with them); large windows are box-smoothed first so a faint defocused disc is found as a disc. The window grows with the measured size, so a detection triggered by one corner of a resolved payload converges on the whole payload; fragments inside a larger object's extent are then absorbed into it, and detections with nothing measurable are dropped.
- `heatmap.py`: inference wrapper for the trained temporal U-Net (its detections are re-measured exactly like classical ones, so track features do not depend on the detector), plus `FusedDetector`.

### `tracking/`
- Constant-velocity Kalman filter with continuous white-acceleration noise and real time steps.
- Stage 1: high-confidence detections vs all live tracks, cost `d² + ln|S|`, Mahalanobis gate plus a pixel cap, Hungarian assignment.
- Stage 2: low-confidence detections may extend confirmed tracks only.
- Confirmation after 4 hits in 6 frames; coasting up to 0.5 s; a cut kills all tracks. Display ids are assigned at confirmation.

### `classify/`
- `features.py`: speed, speed relative to the local background flow, curvature, PSF size, detrended brightness scatter and periodicity (slow trends removed so a receding payload is not called "flickering"), size/brightness change rates, first-sighting position relative to the payload door, MET of first sighting.
- `rules.py`: reference classifier with named cues and an explicit `unknown` margin. Hypotheses, not proof. A payload must look near: resolved, or a measurable size trend on an at least marginally resolved object (brightness change alone is not evidence of nearness), and resolved when first seen at the door (a point first seen there is more likely a distant object emerging from behind the vehicle).
- `model.py`: gradient boosting trained on labelled tracks; returns `unknown` below a probability floor.
- `deployment.py`: payload-category tracks first seen near the door become `DEPLOY-k`, where k is the schedule slot from MET. An event is logged once and never retracted, so it needs `deployment.min_payload_passes` consecutive payload decisions.

### `orbit/`
- `omm.py`: OMM records (JSON/CSV fields, 6-digit catalogue numbers), SGP4 initialisation cross-checked against the `sgp4` reference parser.
- `catalog.py`: CelesTrak (gp, gp-first, SupGP, SATCAT) and Space-Track (gp_history) clients with a cache, polite rate limits and no retry loops; `best_record_per_object` in `retrospective` or `as_of` mode (creation date when known).
- `ship.py`: nominal orbit from launch site/time/inclination/apsides with an explicit phase uncertainty; SGP4 proxy; deployed-group centroid.
- `conjunction.py`: coarse-to-fine screen of the whole catalogue, with sunlight, Earth occultation and a photometric visibility estimate; phase sweep for the nominal ephemeris.
- `release.py`: release-time estimates by back-propagation with a Monte-Carlo spread; a minimum on the window edge is reported as not identifiable.

### `geometry/`
- `camera.py`: pinhole + Brown radial distortion; attitude stored relative to the ship's RIC frame.
- `calibrate.py`: limb-cone fit (nadir direction + focal length, FOV bounded to avoid the degenerate infinite-focal solution); focus of expansion from background flow (ground-relative velocity direction); TRIAD attitude. Calibration never uses the objects being identified.

### `identify/`
- `association.py`: observed rays vs predicted rays in the tangent plane of the mean observed direction; residual mean (shared biases) and residual rate, each with an error budget from attitude, ephemeris (catalogue and ship), timing and pixel noise; a size gate (a resolved blob cannot be a distant point); an explicit unknown hypothesis. ACCEPTED requires all of: posterior ≥ 0.95; χ² ≤ 13.3 (4 dof); **informative geometry** (direction sigma ≤ 2°, rate sigma ≤ 0.3°/s: with km-level ephemeris error, a nearby object's predicted direction carries no information, so a rate match alone must not name it); an expected number of chance matches ≤ 0.01 given how many tracks are evaluated; two consecutive evaluations; and winning any conflict with ≥ 0.95 assignment probability. The reason for any refusal is shown.
- `deployment_id.py`: release events vs release-time estimates with global assignment and an "unmatched" option; or an independent release-order list. Release-time estimates are made only with an independent ship ephemeris (an SGP4 element set for the ship); the deployed group's own centroid is biased for this, so without one `DEPLOY-k` stays a local label.
- Display precedence: accepted geometry > accepted release mapping > candidates > local `DEPLOY-k` label.

### `sim/`
Synthetic clips with independent truth: scrolling cloud background, static ship, HUD, flickering defocused particles, payloads under Clohessy-Wiltshire motion, catalogued passes propagated by SGP4 from their own element sets, distractor objects, noise and JPEG. The published catalogue carries an injected along-track error and payload numbers are permuted relative to release order.

### `train/`
Frame stores built with the same registration and masks as the live pipeline; labels from CVAT, pseudo-labels (other detections ignored, not taught as background) or synthetic truth; time-block splits with gaps; temporal U-Net with CenterNet focal loss and sub-pixel offsets at full resolution; synthetic point injection in camera coordinates before warping; resumable AMP training.

## Data modes shown on screen

- **CAUSAL REPLAY**: recorded video processed at normal speed using only past frames.
- **RETROSPECTIVE** catalogue: element sets chosen with hindsight; always labelled.
- **LIVE**: a stream; with `--catalog-mode as_of` only element sets that existed at the frame time are used.
- **SYNTHETIC DATA**: generated clips; identities here validate the software, not real objects.

## Run outputs

Every run directory holds `config.yaml`, `provenance.json` (git revision, config digest, notes), `frames.jsonl` (per-frame detections, tracks, timings, warm-up flag), `tracks.jsonl` (one line per finished track: history, features, category reasons, identity candidates and decision times; written as tracks end, so memory stays bounded on long live runs), `identity_log.jsonl` (every association decision), `summary.json` (latency percentiles from fixed-size histograms, counts) and `overlay.mp4`. A re-run into the same directory starts these files afresh.
