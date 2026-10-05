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

## Second run (your Mac, 5 Oct)

- **S18 at full resolution:** 1,741 frames at about 9 frames per second, 4,630 tracks.
  - Most of the excess came from video 3005-3034 s. There the Earth drifts only about 0.5 px per frame, while compression scatters each tracked corner by about 0.3 px. The registration called the background "fixed in the image", and the clouds then flooded the detector with about 250 detections per frame.
  - The registration now asks whether the motion fitted to hundreds of corners is statistically significant, not whether each corner moved. Those frames are now registered.
  - What remains in late S18 is a real burst of particles streaming past the nose.
- **S19 live at half resolution:** 11 frames per second (loop 70 ms, pipeline 43 ms).
  - Track classification ran on every track every 5 frames, which dominated the cost with hundreds of particles in view. Each track is now reclassified only after 10 new observations.
  - Re-measurement now converges faster (measurement windows only grow).
- **Payload labels:** 52 release labels were created in S18 because no door position is set. A release event now requires `deployment.door_xy`; without it, "payload" stays a category only.

## Catalogue and orbit results (5 Oct)

**Catalogue.** Flight 14 is **2026-225**. The 26 Starlink V3s are NORAD 100855-100880, STARLINK-40075 to 40103. Space-Track `gp_history` gave 286 element sets between 30 Sep and 5 Oct, and the whole catalogue around the flight has 29,363 objects.

**Ship orbit fitted to the HUD** (`rso fit-ship`):
- The plane comes from the 26 V3s: i = 30.48 deg, RAAN 339.17 deg (TEME at MET 2970 s). This agrees with the plane through Starbase at liftoff to within 1.7 deg of RAAN.
- The fit uses altitude above the WGS84 ellipsoid and ground-relative speed. It reproduces 816 altitude and 930 speed samples (MET 2025-3883 s) to **0.29 km and 0.29 km/h rms**, which is the display's own resolution. Spherical or equatorial altitude conventions fit 6-7 times worse.
- The orbit is **264.6 x 281.1 km** (published nominal: 262 x 277 km). Along-track uncertainty is **0.12 deg = 13.6 km (1 sigma)**; a Monte Carlo on synthetic telemetry confirms the error estimate is calibrated.
- The fitted ship is 150-195 km behind the nominal model, which is well inside the nominal's 600 km uncertainty.
- Ship positions:
  - first release (MET 2063 s): lat -25.4, lon 4.2 (South Atlantic)
  - shots S18/S19 (video 2976-3074 s): lat about -26, lon 69-73 (southern Indian Ocean)

**Release slots cannot be mapped to catalogue numbers from public data.**
- The first public element sets are 2.13 days after launch, and orbit raising had already begun: the drag term is negative (thrust), and the orbit is 270 x 272 km, rising to 296 km circular by 5 Oct.
- Propagated back to the deployment window, the 26 V3s pass 13-1,650 km from the fitted ship instead of within metres. `rso release-times` therefore reports NOT separable.
- The display keeps `DEPLOY-k` local labels with a candidate list, which is the honest output.

**No foreign catalogued object came near the ship.** Screening all 29,363 objects against the fitted ship (100 km, widened by 3 sigma = 41 km):

| Object | Closest | Video time (shot) | Where | Angular rate |
|---|---|---|---|---|
| STARLINK-11422 (61944) | 107 km | 3126 s (S20, aft camera, looks down) | 97 km above | 4.7 deg/s |
| STARLINK-11297 (61060) | 113 km | 2407 s (S08, side camera) | 93 km above | 2.7 deg/s |
| STARLINK-38234 (100552) | 131 km | 3260 s (S22, night, unlit) | 75 km above | 4.9 deg/s |
| STARLINK-2448 (48111) | 135 km | 2565 s (S11, flap-top camera, black sky) | almost straight overhead | 2.6 deg/s |

- All four passed 75-135 km *above* the ship.
- STARLINK-2448 is the best chance of a real catalogued object on screen: sunlit, magnitude about +1, in a shot with black sky. In S11 the Earth limb runs down the right side, so the zenith lies about 100 deg from the limb direction, probably outside the field of view.
- Confirming this needs the camera's pointing, best from the stars visible in the same camera's shot S19 (plate solving).
- Conclusion for the report: the moving white dots in the deployment footage are near-field particles, not catalogued satellites. The only nameable objects are Flight 14's own V3s, and public data cannot tell them apart.
