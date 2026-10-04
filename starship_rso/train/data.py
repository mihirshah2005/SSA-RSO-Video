"""Training data for the temporal heatmap detector.

A *frame store* is a directory produced from one video clip:

    frames/000123.jpg   grey frames at processing resolution (JPEG q95)
    masks/000123.png    valid mask (255 = usable background, 0 = HUD/vehicle)
    meta.json           size, fps, scale, per-frame time, shot id, H (prev->cur), reg_ok
    labels.json         {"points": {idx: [[x, y, sigma, track, kind], ...]},
                         "ignore": {idx: [[x, y, r], ...]}, "source": ...}
    splits.json         {"train": [...], "val": [...], "test": [...]} frame indices

Labels can come from CVAT (manual), from a pipeline run (pseudo-labels:
confirmed tracks, with weaker detections as ignore regions so the network is
not taught that unlabelled faint objects are background), or from synthetic
truth. Splits are long disjoint time blocks with gaps larger than the model
window, so overlapping windows never straddle train and test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..io.video import VideoSource
from ..vision.masks import MaskManager
from ..vision.ops import resize_frame, to_gray_f32
from ..vision.registration import BackgroundRegistrar, Registration

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- building
def build_frame_store(cfg: Config, video: str, out_dir: str | Path, start_s=None, end_s=None, stride: int = 1,
                      max_frames: int | None = None) -> Path:
    """Decode, register and mask a clip exactly as the live pipeline does, and store it."""
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    src = VideoSource(video, start_s, end_s)
    s = cfg.processing.scale
    W, H = int(round(src.meta.width * s)), int(round(src.meta.height * s))
    masks = MaskManager(cfg.masks, (H, W))
    reg = BackgroundRegistrar(cfg.registration, (H, W))
    from ..vision.scenecut import SceneCutDetector

    cuts = SceneCutDetector(cfg.scenecut)
    prev_u8, prev_t = None, None
    H_acc = np.eye(3)
    ok_acc = True
    frames = []
    k = 0
    for idx, t, frame in src:
        g = to_gray_f32(resize_frame(frame, s))
        u8 = g.astype(np.uint8)
        cut = cuts.update(u8) if prev_u8 is not None else False
        if cut:
            masks.reset()
            prev_u8 = None
        if prev_u8 is not None:
            r = reg.estimate(prev_u8, u8, masks.valid, max(t - prev_t, 1e-3))
        else:
            r = Registration(np.eye(3), False, reason="start of shot")
        valid = masks.update(g, t, r.H, r.valid, r.static_background)
        H_acc = (r.H if r.valid else np.eye(3)) @ H_acc
        ok_acc = ok_acc and (r.valid or r.static_background)
        prev_u8, prev_t = u8, t
        if k % stride == 0:
            cv2.imwrite(str(out / "frames" / f"{idx:06d}.jpg"), u8, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / "masks" / f"{idx:06d}.png"), valid.astype(np.uint8) * 255)
            frames.append({"i": idx, "t": t, "shot": cuts.shot_id, "H": H_acc.ravel().tolist(),
                           "reg_ok": bool(ok_acc and len(frames) > 0 and frames[-1]["shot"] == cuts.shot_id),
                           # the live pipeline reports nothing before this, so run-derived labels are empty
                           "mask_ready": bool(masks.ready)})
            H_acc, ok_acc = np.eye(3), True
        k += 1
        if max_frames is not None and len(frames) >= max_frames:
            break
    meta = {"video": str(video), "width": W, "height": H, "fps": src.meta.fps / stride, "scale": s,
            "stride": stride, "frames": frames}
    (out / "meta.json").write_text(json.dumps(meta))
    log.info("frame store: %d frames -> %s", len(frames), out)
    return out


def labels_from_sim(truth_path: str | Path, scale: float = 1.0, kinds=("catalog", "payload", "particle")) -> dict:
    truth = json.loads(Path(truth_path).read_text())
    pts: dict[str, list] = {}
    for i, fr in enumerate(truth["frames"]):
        rows = []
        for o in fr:
            if o["kind"] not in kinds:
                continue
            x, y = (o["x"] + 0.5) * scale - 0.5, (o["y"] + 0.5) * scale - 0.5
            rows.append([x, y, float(np.clip(o.get("r", 1.0) * scale * 0.6, 0.8, 6.0)), o["id"], o["kind"]])
        pts[str(i)] = rows
    return {"points": pts, "ignore": {}, "source": f"synthetic truth {truth_path}"}


def labels_from_run(run_dir: str | Path, scale: float = 1.0, min_obs: int = 8,
                    exclude=("vehicle_feature", "background_feature", "artifact"), ignore: str = "all",
                    frame_offset: int = 0) -> dict:
    """Pseudo-labels from a pipeline run: observed points of confirmed tracks.

    Other detections become *ignore* regions (``ignore="all"``) or only the
    low-confidence ones do (``ignore="low"``). Ignoring is the safe default:
    an unlabelled detection may be a real short-lived particle, and teaching it
    as background would bias the detector toward the classical baseline's misses.
    """
    run = Path(run_dir)
    from ..pipeline.runner import load_tracks

    tracks = load_tracks(run)
    pts: dict[str, list] = {}
    for tr in tracks:
        obs = [p for p in tr["points"] if p["obs"]]
        if len(obs) < min_obs or tr["category"] in exclude:
            continue
        for p in obs:
            x, y = (p["dx"] + 0.5) * scale - 0.5, (p["dy"] + 0.5) * scale - 0.5
            pts.setdefault(str(p["f"] - frame_offset), []).append(
                [x, y, float(np.clip(p["sigma"] * scale, 0.8, 6.0)), tr["uid"], tr["category"]])
    ign: dict[str, list] = {}
    skip: list[str] = []
    fpath = run / "frames.jsonl"
    if fpath.exists():
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                fr = json.loads(line)
                if fr.get("warm"):
                    # the run reported nothing while learning the vehicle mask: no labels exist
                    # for this frame, so it must not be taught as empty background
                    skip.append(str(fr["f"] - frame_offset))
                    continue
                rows = [[(d["x"] + 0.5) * scale - 0.5, (d["y"] + 0.5) * scale - 0.5, max(3.0, 3 * d["sigma"] * scale)]
                        for d in fr["dets"] if ignore == "all" or d["confidence"] < 0.5]
                if rows:
                    ign[str(fr["f"] - frame_offset)] = rows
    return {"points": pts, "ignore": ign, "skip_frames": skip, "source": f"pseudo-labels from {run}"}


def labels_from_cvat(xml_path: str | Path, scale: float = 1.0, frame_offset: int = 0) -> dict:
    from ..io.cvat import import_tracks

    tracks, _ = import_tracks(xml_path, frame_offset)
    pts: dict[str, list] = {}
    ign: dict[str, list] = {}
    for tr in tracks:
        for p in tr.visible():
            x, y = (p.x + 0.5) * scale - 0.5, (p.y + 0.5) * scale - 0.5
            if p.occluded or p.attributes.get("visibility") == "ambiguous":
                ign.setdefault(str(p.frame), []).append([x, y, 6.0])
            else:
                pts.setdefault(str(p.frame), []).append([x, y, 1.2, tr.track_id, tr.label])
    return {"points": pts, "ignore": ign, "source": f"CVAT {xml_path}"}


def write_labels(store: str | Path, labels: dict) -> None:
    Path(store, "labels.json").write_text(json.dumps(labels))


def make_splits(store: str | Path, frames: int, val_frac: float = 0.15, test_frac: float = 0.15,
                block_s: float = 10.0, seed: int = 0) -> dict:
    """Assign contiguous time blocks to train/val/test with a gap of ``frames`` at each boundary."""
    meta = json.loads(Path(store, "meta.json").read_text())
    idx = [f["i"] for f in meta["frames"]]
    t = np.array([f["t"] for f in meta["frames"]])
    block = np.floor((t - t[0]) / block_s).astype(int)
    nb = int(block.max()) + 1 if len(block) else 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(nb)
    n_val = max(1, int(round(val_frac * nb))) if nb >= 3 else 0
    n_test = max(1, int(round(test_frac * nb))) if nb >= 3 else 0
    role = {}
    for k, b in enumerate(order):
        role[b] = "val" if k < n_val else ("test" if k < n_val + n_test else "train")
    out = {"train": [], "val": [], "test": []}
    for j, (i, b) in enumerate(zip(idx, block)):
        lo, hi = max(0, j - frames), min(len(idx) - 1, j + frames)
        if any(role[block[q]] != role[b] for q in (lo, hi)):
            continue  # gap at block boundaries
        out[role[b]].append(i)
    out["meta"] = {"block_s": block_s, "seed": seed, "gap_frames": frames}
    Path(store, "splits.json").write_text(json.dumps(out))
    return out


# --------------------------------------------------------------- rendering
def splat_targets(h: int, w: int, pts: np.ndarray, sig: np.ndarray):
    """Gaussian heatmap, offset map and positive mask for points (crop coordinates)."""
    heat = np.zeros((h, w), np.float32)
    off = np.zeros((2, h, w), np.float32)
    pos = np.zeros((h, w), np.float32)
    for (x, y), s in zip(pts, sig):
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        r = int(np.ceil(3 * s))
        x0, x1, y0, y1 = max(0, ix - r), min(w, ix + r + 1), max(0, iy - r), min(h, iy + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32)
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g)
        heat[iy, ix] = 1.0
        off[0, iy, ix], off[1, iy, ix] = x - ix, y - iy
        pos[iy, ix] = 1.0
    return heat, off, pos


def inject_point(img: np.ndarray, x: float, y: float, sigma: float, amp: float, disc: bool) -> None:
    from ..sim.render import add_disc, add_gaussian_spot

    if disc:
        add_disc(img, x, y, sigma * 2.0, amp)
    else:
        add_gaussian_spot(img, x, y, sigma, amp)


class HeatmapWindowDataset:
    """Random K-frame windows with targets. Imports torch lazily so the module loads without it."""

    def __init__(self, stores: list[str | Path], split: str, frames: int = 3, crop: int = 512,
                 positive_frac: float = 0.6, augment: bool = True, inject_rate: float = 0.5,
                 samples_per_epoch: int = 2000, seed: int = 0):
        self.frames = frames
        self.crop = crop
        self.positive_frac = positive_frac
        self.augment = augment
        self.inject_rate = inject_rate
        self.samples = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.items = []  # (store, meta, labels, position in meta.frames)
        self.stores = []
        for sd in stores:
            sd = Path(sd)
            meta = json.loads((sd / "meta.json").read_text())
            labels = json.loads((sd / "labels.json").read_text()) if (sd / "labels.json").exists() else {"points": {}, "ignore": {}}
            splits = json.loads((sd / "splits.json").read_text()) if (sd / "splits.json").exists() else None
            allowed = set(splits[split]) if splits else {f["i"] for f in meta["frames"]}
            skip = set(labels.get("skip_frames", []))
            si = len(self.stores)
            self.stores.append((sd, meta, labels))
            fr = meta["frames"]
            for j in range(frames - 1, len(fr)):
                win = fr[j - frames + 1 : j + 1]
                if fr[j]["i"] not in allowed:
                    continue
                if any(f["shot"] != fr[j]["shot"] for f in win) or not all(f["reg_ok"] for f in win[1:]):
                    continue
                if not fr[j].get("mask_ready", True) or str(fr[j]["i"]) in skip:
                    continue  # vehicle mask still being learned: labels from a run would be missing here
                self.items.append((si, j))
        self.pos_items = [it for it in self.items if self.stores[it[0]][2]["points"].get(str(self.stores[it[0]][1]["frames"][it[1]]["i"]))]
        if not self.items:
            raise ValueError(f"no usable windows for split {split!r}")

    def __len__(self) -> int:
        return self.samples

    def _load(self, sd: Path, i: int):
        g = cv2.imread(str(sd / "frames" / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(sd / "masks" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        return g.astype(np.float32), (m > 127) if m is not None else np.ones_like(g, bool)

    def get(self, k: int, deterministic: bool = False):
        rng = np.random.default_rng(k) if deterministic else self.rng
        use_pos = self.pos_items and rng.random() < self.positive_frac
        si, j = (self.pos_items if use_pos else self.items)[rng.integers(len(self.pos_items if use_pos else self.items))]
        sd, meta, labels = self.stores[si]
        fr = meta["frames"]
        win = fr[j - self.frames + 1 : j + 1]
        imgs, masks = zip(*[self._load(sd, f["i"]) for f in win])
        imgs = [im.copy() for im in imgs]
        h, w = imgs[-1].shape
        # homographies past -> current: each stored H maps the previous stored frame into this one
        Hc = [np.eye(3) for _ in range(self.frames)]
        for q in range(self.frames - 2, -1, -1):  # frame q -> current = H(q+1) composition
            Hc[q] = Hc[q + 1] @ np.array(win[q + 1]["H"]).reshape(3, 3)
        cur_i = win[-1]["i"]
        pts = [p for p in labels["points"].get(str(cur_i), [])]
        P = np.array([[p[0], p[1]] for p in pts]).reshape(-1, 2)
        S = np.array([p[2] for p in pts]).reshape(-1)
        ign = np.array(labels.get("ignore", {}).get(str(cur_i), [])).reshape(-1, 3)
        # synthetic injection (before warping; camera-frame motion)
        if self.augment and rng.random() < self.inject_rate:
            n_inj = int(rng.integers(1, 6))
            noise = float(np.std(imgs[-1] - cv2.GaussianBlur(imgs[-1], (0, 0), 1.0))) + 1.0
            ts = [f["t"] for f in win]
            new_p, new_s = [], []
            for _ in range(n_inj):
                x, y = rng.uniform(16, w - 16), rng.uniform(16, h - 16)
                if not masks[-1][int(y), int(x)]:
                    continue
                v = rng.uniform(-300, 300, 2)
                sg = float(rng.uniform(0.7, 3.0))
                amp = float(rng.uniform(3, 20)) * noise * (1 + sg)
                disc = sg > 2.2 and rng.random() < 0.5
                for q in range(self.frames):
                    dt = ts[-1] - ts[q]
                    fl = 1.0 - 0.6 * rng.random() if rng.random() < 0.3 else 1.0
                    inject_point(imgs[q], x - v[0] * dt, y - v[1] * dt, sg, amp * fl, disc)
                new_p.append([x, y])
                new_s.append(min(sg * (2.0 if disc else 1.0), 6.0))
            if new_p:
                P = np.vstack([P, np.array(new_p)])
                S = np.concatenate([S, np.array(new_s)])
        # crop
        cs = min(self.crop, h, w)
        if use_pos and len(P):
            c = P[rng.integers(len(P))] + rng.uniform(-cs * 0.35, cs * 0.35, 2)
        else:
            c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        x0 = int(np.clip(c[0] - cs / 2, 0, w - cs))
        y0 = int(np.clip(c[1] - cs / 2, 0, h - cs))
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        stack = []
        for q in range(self.frames):
            M = T @ Hc[q]
            stack.append(cv2.warpPerspective(imgs[q], M, (cs, cs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        valid = masks[-1][y0 : y0 + cs, x0 : x0 + cs].astype(np.float32)
        P = P - [x0, y0]
        inside = (P[:, 0] >= 0) & (P[:, 0] < cs) & (P[:, 1] >= 0) & (P[:, 1] < cs) if len(P) else np.zeros(0, bool)
        heat, off, pos = splat_targets(cs, cs, P[inside], S[inside] if len(S) else S)
        weight = valid.copy()
        for x, y, r in ign:
            x, y = x - x0, y - y0
            if -r <= x < cs + r and -r <= y < cs + r:
                cv2.circle(weight, (int(round(x)), int(round(y))), int(np.ceil(r)), 0.0, -1)
        weight = np.maximum(weight, (heat > 0.05).astype(np.float32))  # labelled points always count
        stack = np.stack(stack, 0)
        if self.augment:
            stack, heat, off, pos, weight = _augment(stack, heat, off, pos, weight, rng)
        return stack.astype(np.float32), heat[None], off, pos[None], weight[None]


def _augment(stack, heat, off, pos, weight, rng):
    # photometric (shared across the window)
    gain = rng.uniform(0.7, 1.3)
    bias = rng.uniform(-20, 20)
    gamma = rng.uniform(0.8, 1.25)
    stack = np.clip(255.0 * ((np.clip(stack * gain + bias, 0, 255) / 255.0) ** gamma), 0, 255)
    if rng.random() < 0.5:
        stack = stack + rng.normal(0, rng.uniform(0.5, 4.0), stack.shape)
    if rng.random() < 0.3:
        q = int(rng.integers(40, 95))
        stack = np.stack([cv2.imdecode(cv2.imencode(".jpg", np.clip(f, 0, 255).astype(np.uint8),
                                                    [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
                          for f in stack]).astype(np.float32)
    # geometric (flips / transposes keep the pixel grid exact)
    if rng.random() < 0.5:
        stack, heat, pos, weight = stack[..., ::-1], heat[:, ::-1], pos[:, ::-1], weight[:, ::-1]
        off = off[:, :, ::-1].copy()
        off[0] = -off[0]
    if rng.random() < 0.5:
        stack, heat, pos, weight = stack[..., ::-1, :], heat[::-1], pos[::-1], weight[::-1]
        off = off[:, ::-1].copy()
        off[1] = -off[1]
    if rng.random() < 0.5:
        stack, heat, pos, weight = stack.swapaxes(-1, -2), heat.T, pos.T, weight.T
        off = off.swapaxes(-1, -2)[::-1].copy()
    return (np.ascontiguousarray(stack), np.ascontiguousarray(heat), np.ascontiguousarray(off),
            np.ascontiguousarray(pos), np.ascontiguousarray(weight))
