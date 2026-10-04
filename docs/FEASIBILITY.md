# Feasibility: what this footage can and cannot support

Research snapshot: 3-4 October 2026, five days after the flight. Items marked **VERIFY** were not confirmed by a primary source at that time.

## 1. Flight 14 facts used by the code

| Item | Value | Where it is used |
|---|---|---|
| Liftoff | 28 Sep 2026, 12:48:59 UTC, Starbase Pad 2 | `mission.liftoff_utc` (UTC of every frame) |
| Vehicles | Booster 21, Ship 41 (the "S41" in the frame), Block 3 / V3 | context |
| Trajectory | First orbital Starship flight; insertion burn at T+25:20 | `orbit_nominal.insertion_met_s` |
| Orbit | about 262 x 277 km; inclination 30.5 deg (Wikipedia, McDowell) or 32 deg (NSF) **VERIFY** | nominal ship ephemeris |
| Payload | Starlink Group 31-1: 26 real Starlink V3 satellites (3 with cameras); no CubeSats, no third-party payloads | deployment schedule, payload group |
| Deployment window | planned T+34:07 to T+1:04:39 (SpaceX via Space.com); Wikipedia lists T+34:18 to T+1:04:50 | `mission.deployment` |
| Release cadence | about one per minute ("pez dispenser" door) | `DEPLOY-k` slot index |
| Your frame | T+00:49:34, 271 km, 26,394 km/h: mid-deployment, slot about 13-14 of 26 | `timemap.anchors` |
| Catalogue | Space-Track catalogued 26 new objects from NORAD 100855 on 2 Oct 2026. Count matches, link to Flight 14 **not confirmed** | `payload_group.norad_range` (candidate only) |
| Outcome | all 26 deployed; deorbit burn T+2:12:18; splashdown T+3:08:30 | context |

Cross-check built into the code: the nominal ephemeris gives a ground-relative speed of about 26,350 km/h at T+49:34, which matches the HUD value (26,394 km/h). The HUD speed is Earth-relative, not inertial (about 27,850 km/h).

## 2. What the white dots most likely are

- **Most specks are near-field particles** (ice, frost, vented condensate, small insulation or tile fragments) within metres of the lens.
  - Evidence by analogy: Space Shuttle water-dump studies resolved about 100 large ice particles per metre of path in onboard video, flickering as they tumbled (NASA NTRS 19910011413).
  - Out-of-focus discs mean the object is very close to the camera.
  - SpaceX has not stated the cause. Treat this as a hypothesis the classifier tests, not a fact.
- **Parallax misleads**: a particle a few metres away appears to turn or accelerate when the camera platform rotates.
- **Real satellites in view are almost certainly the departing Starlink V3s** during T+34 to T+65.

## 3. Physics: how close must a satellite be to appear?

Run `rso feasibility` to reproduce. Assumptions: 1920 px across a 90 deg field (IFOV 0.82 mrad, about 169 arcsec), a large satellite of magnitude 5 at 550 km.

| Range | 3U CubeSat (0.34 m) | 3 m bus | 30 m with arrays | Magnitude (large sat) |
|---|---|---|---|---|
| 1 km | 0.4 px | 3.7 px | 37 px | about -8.7 |
| 10 km | 0.04 px | 0.37 px | 3.7 px | about -3.7 |
| 100 km | 0.004 px | 0.04 px | 0.37 px | about +1.3 |
| 1000 km | below 0.001 px | 0.004 px | 0.04 px | about +6.3 |

Consequences:

- A camera exposed for sunlit Earth shows no stars. The faint limit is around magnitude 0 against cloud and maybe +2 against black sky.
- A visible catalogued satellite must be sunlit and within roughly tens of km.
- An unrelated satellite crosses a 90 deg field in seconds at km/s relative speeds. A chance pass within about 10 km during a short clip is very unlikely. `rso screen` checks this numerically instead of assuming it.

## 4. Error budget for naming an object

| Input error | Effect (1920 px over 90 deg) |
|---|---|
| 0.1 deg pointing | about 2 px |
| 1 km relative position at 10 km range | 5.7 deg, about 120 px |
| 1 km relative position at 100 km range | 0.57 deg, about 12 px |
| 1 km relative position at 1000 km range | 0.06 deg, about 1 px |

Why this matters here:

- **Distant objects** (100+ km) can be matched geometrically if the camera attitude is known, but they are too faint to see.
- **Nearby objects** (the departing Starlinks, 10 m to a few km) are visible, but km-level element-set errors make their predicted *directions* meaningless.
- **SpaceX publishes no Starship ephemeris or attitude.** The code reconstructs the ship orbit three ways (nominal mission facts, an SGP4 proxy, the deployed group's centroid) and states each one's uncertainty. The nominal orbit's along-track phase is uncertain by about 600 km.

## 5. What can honestly be claimed, by output level

| Output | Feasible on Flight 14 footage? | How |
|---|---|---|
| Detection | Yes | classical detector now, learned temporal detector after training |
| Tracking | Yes | Kalman + Hungarian + low-confidence second stage |
| Category | Partly | near-field particle / departing payload / vehicle-fixed / Earth-fixed / unknown, from motion, focus and flicker; validated only where labels exist |
| `DEPLOY-k` release slot | Yes, if the release is on camera | first sighting near the door, MET mapped to the published schedule |
| Exact catalogue name of a released Starlink | Only with independent data | an operator release-order list, or element sets accurate to tens of metres near release (not expected from public GP data) |
| Exact name of a background dot | No, for almost all dots | they are near-field particles, and foreign satellites are too faint or too far |

The system is built to say this itself: it shows `unknown`, a candidate list or `identity unavailable` with the reason, and only shows a name when the evidence is unique and stable.

## 6. Verification data available for the demo

- **Synthetic clips** (`rso simulate`): exact truth for detection, tracking, category and identity, including catalogue numbers deliberately not in release order. This is where named identification is demonstrated end to end. It is always labelled SYNTHETIC.
- **Flight 14 deployment footage**: satellite count (26) and approximate cadence are known, so `DEPLOY-k` counting can be checked.
- **Earlier flights as controls**: Flights 12 and 13 released objects that were never in the public catalogue (mass simulators on 12; Flight 13 objects reentered quickly). The correct output there is "no catalogue identity".
- **Catalogue negative result**: `rso screen` over the whole catalogue during the clip, with the ship's phase uncertainty swept, shows whether any foreign object came within visible range. "None did" is a defensible finding.

## Sources

1. Space.com live coverage, Flight 14: https://www.space.com/news/live/spacex-starship-flight-14-live-updates-sept-28-2026-starship-first-orbital-launch-attempt
2. SpaceX mission page: https://www.spacex.com/launches/starship-flight-14
3. NASASpaceflight, Flight 14 orbit: https://www.nasaspaceflight.com/2026/09/starship-flight-14-orbit/
4. Wikipedia, Starship flight 14: https://en.wikipedia.org/wiki/Starship_flight_14
5. NASA NTRS, Shuttle water-dump particles: https://ntrs.nasa.gov/api/citations/19910011413/downloads/19910011413.pdf
6. CelesTrak GP data formats (6-digit catalogue numbers have no TLE form): https://celestrak.org/NORAD/documentation/gp-data-formats.php
7. CelesTrak SupGP queries: https://celestrak.org/NORAD/documentation/sup-gp-queries.php
8. Space-Track API documentation and rate limits: https://www.space-track.org/documentation
9. NUS SoC GPU access via Slurm: https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus
