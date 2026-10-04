# Manual steps (what you run, in order)

Everything below runs on your Mac unless it says GPU. Each step says what to check before moving on.

## 0. Environment (once)

```bash
cd ~/Desktop/FYP_data/Video_CV
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"          # numpy, opencv, sgp4, torch (MPS on Apple silicon), sklearn, pytest
brew install ffmpeg yt-dlp tesseract   # video tools and optional HUD OCR
pip install pytesseract av             # optional: OCR + exact timestamps
python -m pytest -q                    # 69 tests, about 1 minute; all must pass
```

- On macOS, `opencv-python` and `av` each bundle FFmpeg's `libavdevice`, so every command prints two `objc[...] Class AVF... is implemented in both` lines. They concern camera/microphone capture, which this project never uses, so they are harmless here. To silence them, `export RSO_VIDEO_BACKEND=opencv` (PyAV is then never imported; timestamps come from OpenCV, which is exact for constant-frame-rate files like the rebroadcast).

## 1. Synthetic demo (no downloads needed)

```bash
rso simulate --out data/sim/seed0 --duration 20
rso run -c configs/default.yaml -c data/sim/seed0/mission.yaml --video data/sim/seed0/video.mp4 --mode synthetic
rso evaluate --run runs/<newest> --truth data/sim/seed0/truth.json --skip 45
```

- **Check**: passing objects show `ID: SIM-SAT-xx (990xxx)`, particles stay unnamed, `accepted_wrong` is 0.
- Keys in the window: space = pause, `m` = masks, `t` = trails, `p` = catalogue predictions, `s` = snapshot, `q` = quit.

## 2. Get the video (STOP: needs you)

YouTube's terms generally do not allow downloading. Prefer SpaceX's own webcast (spacex.com or X) if you can get a file; otherwise download for private research only and do not redistribute frames.

```bash
mkdir -p data/videos
yt-dlp -f "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4]" --merge-output-format mp4 \
  -o "data/videos/f14_rebroadcast.%(ext)s" "https://www.youtube.com/watch?v=3CVoSPIMx6I"
```

- The format filter asks for H.264 in an .mp4 (OpenCV cannot always decode AV1/VP9).
- Use `--start/--end` on the full file in later commands (time anchors then stay valid).
- For annotation, cut short clips: `rso extract-clip data/videos/f14_rebroadcast.mp4 --start 2880 --duration 120 --out data/videos/f14_clipA.mp4` (clip time 0 = source 2880 s).

## 3. Layout and time (check before trusting anything)

```bash
rso check-layout -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 --t 2986
rso ocr -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 --start 2040 --end 3900
```

- **Check** `layout_check.png`: red must cover the telemetry band and logo; green boxes must enclose the clock, speed and altitude digits. Edit `hud:` and `masks.hud_rects` in `configs/flight14.yaml` if not.
- Paste the printed anchors into `timemap.anchors` (replace the approximate screenshot anchor). **Done for Flight 14**: MET = video - 15.9 s across 2040-3900 s, one segment, no replays.
- Shot list for the deployment window (video 2063-3895 s = T+34:07 to T+1:04:39):

```bash
rso shots -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 --start 2040 --end 3900
```

  It prints each camera shot (video and MET range, mean brightness) and writes `data/shots/shots.csv` and `data/shots/contact_sheet.jpg`. Shots under 1 s are flagged as flashes or graphics.
- For the camera that looks at the payload door, set `deployment.door_xy: [x, y]` (normalised image position of the door). Without it the "first seen at the door" cue is off and payloads are recognised only by being resolved, slow and steady.

## 4. First real run (classical detector)

```bash
# analysis: every frame at full resolution, no display
rso run -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 \
  --start 2940 --end 3060 --no-realtime --no-display
# live demo: half resolution keeps up with 30 fps on a laptop
rso run -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 \
  --start 2940 --end 3060 --set processing.scale=0.5
```

- `summary.json` reports `proc_ms` (pipeline only), `loop_ms` (pipeline, overlay and display) and `effective_fps`. In live mode, frames the loop cannot keep up with are dropped, not queued (`frames_dropped`); the tracker uses the true time steps.
- **Check** with `m`: the ship should turn purple after about 1 s (the panel shows "learning vehicle mask" and no detections until then, and again after every camera cut); the Earth must stay unmasked.
- Tune on this clip only (keep a different clip for testing): `--set detector.classical.snr_res=7` if clouds produce too many detections; `--set processing.scale=0.75` if it runs below the video frame rate.
- Send me `runs/<dir>/summary.json`, a few snapshots and the `frames.jsonl` size if you want help tuning.

## 5. Catalogue data (STOP: needs a free Space-Track account)

```bash
export SPACETRACK_USER=...  SPACETRACK_PASSWORD=...
# confirm the candidate group really is Flight 14 (look for STARLINK names and launch date 2026-09-28)
curl "https://celestrak.org/satcat/records.php?CATNR=100855&FORMAT=JSON"
rso fetch-catalog --source spacetrack --norad-range 100855-100880 --epoch-from 2026-09-28 --epoch-to 2026-10-10 \
  --out data/catalog/f14_group.json
# whole catalogue near the flight, for the visibility screen (large download, run once)
rso fetch-catalog --source spacetrack --epoch-from 2026-09-27 --epoch-to 2026-09-29 --out data/catalog/all_0928.json
```

- Update `payload_group.norad_range` / `intdes` in `configs/flight14.yaml` with the confirmed values and set `mission.catalog_file`.
- If SupGP exists for the V3s, also fetch it (`--endpoint supgp --query INTDES --value 2026-xxx`): operator ephemerides are the only public data that might separate release slots.

## 6. Orbit analyses (minutes, CPU)

```bash
rso ship -c configs/default.yaml -c configs/flight14.yaml           # compare altitude/speed with the HUD
rso screen -c configs/default.yaml -c configs/flight14.yaml --catalog data/catalog/all_0928.json \
  --met-from 2040 --met-to 3900 --range-km 100 --best-epoch
rso release-times -c configs/default.yaml -c configs/flight14.yaml
```

- `screen` answers "could any foreign satellite have been visible?". Expect none within a few km; that is a result for the report.
- `release-times` prints whether release slots are separable from the element sets. If it says NOT separable, the demo shows `DEPLOY-k` with a candidate list, which is the honest answer.
- Better ship ephemeris: once group elements exist, try `mission.ship_ephemeris: group_centroid`.

## 7. Camera calibration (needed for any geometric identity)

Pick a shot where the Earth limb (horizon against black space) is visible and the camera is fixed to the ship.

```bash
rso calibrate -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 \
  --t <seconds> --altitude-km 271 --foe-dt 0.5 --out data/calib/cam_<shot>.json
```

- **Check** the printed limb RMS (aim for below 0.1 deg) and HFOV. If automatic limb points are poor, click 20-50 limb points in any image tool and pass `--limb-points points.json`.
- Set `camera.calibration_file` for runs of that shot only (calibration is per camera view).

## 8. Labels (STOP: needs you, CVAT)

```bash
rso export-cvat --run runs/<dir> --out data/labels/f14_clipA_pseudo.xml \
  --frame-offset <source frame of the clip's first frame> --n-frames <frames in the clip>
```

- Create a CVAT task from the same clip, upload the XML ("CVAT for video 1.1"), correct it following `docs/ANNOTATION_GUIDE.md`, and export it back.
- Target: 2-5 short clips covering clouds, black sky, glare, the deployment door, crowded particles. 2,000-5,000 frames and 100-300 tracks overall.

## 9. Datasets for training

```bash
rso make-dataset -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_clipA.mp4 \
  --cvat data/labels/f14_clipA.xml --out data/datasets/f14_clipA
rso simulate --out data/sim/seed0 --duration 60 && \
rso make-dataset -c configs/default.yaml -c data/sim/seed0/mission.yaml --video data/sim/seed0/video.mp4 \
  --sim-truth data/sim/seed0/truth.json --out data/datasets/sim_seed0
```

- Without CVAT labels you can start from pseudo-labels: `--run runs/<dir>` (weaker; other detections are ignored, not taught as background). If the run used the full video and the store a cut clip, add `--frame-offset <source frame of the clip's first frame>`; the command warns when labels and frames do not overlap.

## 10. Training (STOP: GPU, NUS SoC cluster or Colab)

```bash
# copy the repo + data/datasets to the cluster, then:
ssh <soc_unix_id>@xlogin.comp.nus.edu.sg
cd ~/Video_CV
sbatch cluster/train_slurm.sh configs/train_heatmap.yaml --pilot      # 200 steps: speed + memory
CHAIN=1 sbatch cluster/train_slurm.sh configs/train_heatmap.yaml       # full run, resubmits itself
```

- SoC GPU jobs are limited to 3 hours; training resumes from `last.pt` automatically and writes `DONE` when finished. With `CHAIN=1` each job queues its successor first, so a walltime kill still continues.
- Overrides stack: `--set frames=1 --set out_dir=runs/heatmap_k1`.
- Colab alternative: `notebooks/colab_train.ipynb`.
- Train three variants for the comparison table: `frames=1` (single-frame), `frames=3`, `frames=5` (different `out_dir`).
- Optional box baseline: `rso export-yolo --store data/datasets/f14_clipA --out data/yolo/f14` then train with Ultralytics.

## 11. Learned detector in the live demo

```bash
rso run -c configs/default.yaml -c configs/flight14.yaml --video data/videos/f14_rebroadcast.mp4 \
  --start 2940 --end 3240 --set detector.type=fused --set detector.heatmap.weights=runs/heatmap_k3/best.pt
rso evaluate --run runs/<dir> --cvat data/labels/f14_test.xml --frame-offset <offset>
```

## 12. Category model (after labels exist)

```bash
rso train-classifier --runs runs/<dirA> runs/<dirB> --labels data/labels/a.xml data/labels/b.xml \
  --frame-offsets <offsetA> <offsetB> --out models/category.pkl
rso run ... --set classify.model_path=models/category.pkl
```

## What to send back to me

- `summary.json` and `metrics.json` of runs, the `screen` and `release-times` outputs, the calibration JSON, and a few overlay snapshots. That is enough to tune thresholds and write the results section.
