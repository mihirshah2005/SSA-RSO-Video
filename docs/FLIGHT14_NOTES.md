# Flight 14 footage notes (YouTube rebroadcast 3CVoSPIMx6I)

Measured on the downloaded 1080p30 H.264 file. Times are seconds into that file.

## Time map

- `rso ocr` over video 2040-3900 s read the HUD clock in 8,996 of 9,301 samples.
- 1,723 clock ticks fit one straight segment: **MET = video - 15.9 s** (1-sigma 0.06 s). No replays inside the deployment window.
- Deployment window (T+34:07 to T+1:04:39) = **video 2063-3895 s**.
- Altitude OCR drops the leading digit in about 5% of samples ("71" for "271"); only the clock is used for timing.

## Camera shots in the deployment window (`rso shots`, 29 shots)

| View | Shots (video s) | What it shows | Use |
|---|---|---|---|
| Side camera, Earth below | S00, S02, S04, S06, S08 (2040-2413) | ship side, bright Earth, black corner | particles over Earth |
| Payload bay interior ("PAYLOAD DEPLOY") | S01, S03 (2061-2093) | Starlink stack on the dispenser rail, light through the door | no dots; not for detection |
| Crowd split screen | S05, S07 (2202-2212, 2334-2338) | picture-in-picture of the crowd | **skip** |
| Under the flap, Earth beyond | S09, S21 (2413-2492, 3155-3197) | flap underside, black gap, Earth | mixed scene |
| Aft camera, hex tiles + Earth | S10, S12, S14, S16, S18, S20 (2492-3155) | the user's reference frame is S18 (2976-3034) | **main clip for particles over Earth** |
| Flap top, black sky + Earth limb | S11, S13, S15, S17, S19 (2558-3074) | dense fast particles, stars, one bright steady point | **main clip for black sky** |
| Night / split with globe graphic | S22-S28 (3197-3900) | ship in darkness, dim purple sky | little to detect |

Recommended clips: **S18 (video 2976-3034)** for particles over a moving Earth and **S19 (video 3034-3074)** for black sky with stars.

## What the first real runs showed (classical detector, full resolution)

- Released Starlinks are **not** seen leaving the door in the external views. The interior view shows the stack, not departing satellites. So `deployment.door_xy` stays unset, and `DEPLOY-k` events are not expected from this footage.
- Over the Earth (S18), heavy compression makes small cumulus puffs flicker. Before the fix, about 200 detections per frame landed on clouds. The detector now requires the motion residual to be at least half of the object's own contrast (`detector.classical.min_residual_frac`), with residual noise measured per block. That leaves about 50 per frame, mostly real moving specks with straight trails.
- On black sky (S19), many fast particles cross the frame. One bright point with a lens glow appears in all of S15-S19 (about 3 minutes). It drifts slowly from about (0.49, 0.33) to (0.44, 0.20) of the frame.
  - It could be a planet, a bright star, or a satellite moving with the ship.
  - The vehicle mask now leaves such isolated blobs alone. Steady points away from the ship are no longer called "vehicle features", so they stay eligible for catalogue association.
- Stars are visible in S19. A plate solve of those frames (for example with astrometry.net) would give the camera's inertial pointing and field of view far more precisely than the limb fit. This is the best route to a calibrated camera, and a calibrated camera is needed before any satellite can be named.
- Processing time at 1920x1080 is about 130-150 ms per frame on 4 CPU cores. Use `--set processing.scale=0.5` for the live demo, and full resolution with `--no-realtime` for analysis.
