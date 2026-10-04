"""Export a frame store to a tiled YOLO dataset (single-frame box-detector baseline).

Points become square boxes of side ``max(min_box, 4 * sigma)``. Tiles overlap
by 64 px so every point appears whole in at least one tile. Train with
Ultralytics (AGPL-3.0), for example::

    yolo detect train data=<out>/data.yaml model=yolo11s.pt imgsz=640 epochs=100

Evaluate it with point-distance metrics (``rso evaluate``) using the box
centres, so it is compared with the heatmap model on the same terms.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def export_yolo(store: str | Path, out_dir: str | Path, tile: int = 640, overlap: int = 64, every: int = 5,
                min_box: float = 8.0, keep_empty: float = 0.2, seed: int = 0) -> Path:
    store, out = Path(store), Path(out_dir)
    meta = json.loads((store / "meta.json").read_text())
    labels = json.loads((store / "labels.json").read_text())
    splits = json.loads((store / "splits.json").read_text())
    rng = np.random.default_rng(seed)
    W, H = meta["width"], meta["height"]
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(splits[split]):
            if n % every:
                continue
            img = cv2.imread(str(store / "frames" / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            pts = labels["points"].get(str(i), [])
            for y0 in ys:
                for x0 in xs:
                    crop = img[y0 : y0 + tile, x0 : x0 + tile]
                    ch, cw = crop.shape[:2]  # smaller than the tile when the frame is smaller
                    rows = []
                    for x, y, s, *_ in pts:
                        if x0 <= x < x0 + cw and y0 <= y < y0 + ch:
                            b = max(min_box, 4 * s)
                            rows.append(f"0 {(x - x0) / cw:.6f} {(y - y0) / ch:.6f} {b / cw:.6f} {b / ch:.6f}")
                    if not rows and rng.random() > keep_empty:
                        continue
                    name = f"{i:06d}_{x0}_{y0}"
                    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
                    (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows))
    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {0: "point"}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return out
