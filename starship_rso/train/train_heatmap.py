"""Train the temporal heatmap detector (GPU recommended; resumable; AMP).

    python -m starship_rso.train.train_heatmap --config configs/train_heatmap.yaml
    python -m starship_rso.train.train_heatmap --config configs/train_heatmap.yaml --pilot   # 200 steps, timing

Checkpoints: ``<out_dir>/last.pt`` (resume) and ``<out_dir>/best.pt`` (best
validation F1 at ``tol_px``). Metrics per epoch go to ``<out_dir>/log.csv``.
Validation uses fixed (deterministic) windows from the validation time
blocks, so numbers are comparable between epochs and runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("train_heatmap")


@dataclass
class TrainCfg:
    stores: list[str] = field(default_factory=list)
    out_dir: str = "runs/heatmap"
    frames: int = 3
    base: int = 32
    depth: int = 4
    crop: int = 512
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    samples_per_epoch: int = 4000
    val_samples: int = 400
    workers: int = 4
    amp: bool = True
    seed: int = 0
    offset_weight: float = 1.0
    patience: int = 8
    threshold: float = 0.35
    tol_px: float = 3.0
    inject_rate: float = 0.5
    positive_frac: float = 0.6
    device: str = "auto"


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _TorchDS:
    def __init__(self, base, deterministic: bool):
        self.base = base
        self.det = deterministic

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        import torch

        stack, heat, off, pos, weight = self.base.get(k, deterministic=self.det)
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (stack, heat, off, pos, weight))


def _worker_init(wid):
    import torch

    info = torch.utils.data.get_worker_info()
    info.dataset.base.rng = np.random.default_rng(int(torch.initial_seed()) % 2**32 + wid)


def evaluate(model, loader, device, threshold: float, tol: float) -> dict:
    import torch

    from ..eval.metrics import match_points
    from .model import build_input, decode

    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for stack, heat, off, pos, weight in loader:
            stack = stack.to(device)
            x = build_input(stack[:, -1:], [stack[:, q : q + 1] for q in range(stack.shape[1] - 2, -1, -1)])
            hl, of = model(x)
            dets = decode(hl.float(), of.float(), threshold)
            for b in range(stack.shape[0]):
                ys, xs = torch.nonzero(pos[b, 0] > 0.5, as_tuple=True)
                gt = np.stack([xs.numpy() + off[b, 0, ys, xs].numpy(), ys.numpy() + off[b, 1, ys, xs].numpy()], 1)
                d = dets[b].numpy()
                # predictions in zero-weight (ignored/masked) regions do not count as false positives
                w = weight[b, 0].numpy()
                if len(d):
                    ix = np.clip(np.rint(d[:, 0]).astype(int), 0, w.shape[1] - 1)
                    iy = np.clip(np.rint(d[:, 1]).astype(int), 0, w.shape[0] - 1)
                    d = d[w[iy, ix] > 0.5]
                pairs, _ = match_points(gt.reshape(-1, 2), d[:, :2].reshape(-1, 2), tol)
                tp += len(pairs)
                fn += len(gt) - len(pairs)
                fp += len(d) - len(pairs)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12), "tp": tp, "fp": fp, "fn": fn}


def train(cfg: TrainCfg, pilot: bool = False) -> Path:
    import torch

    from .data import HeatmapWindowDataset
    from .model import TemporalUNet, build_input, focal_loss, offset_loss

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    device = pick_device(cfg.device)
    log.info("device: %s", device)
    tr = HeatmapWindowDataset(cfg.stores, "train", cfg.frames, cfg.crop, cfg.positive_frac, True, cfg.inject_rate,
                              cfg.samples_per_epoch, cfg.seed)
    va = HeatmapWindowDataset(cfg.stores, "val", cfg.frames, cfg.crop, cfg.positive_frac, False, 0.0,
                              cfg.val_samples, cfg.seed + 1)
    dl = torch.utils.data.DataLoader(_TorchDS(tr, False), batch_size=cfg.batch_size, shuffle=True,
                                     num_workers=cfg.workers, worker_init_fn=_worker_init, drop_last=True,
                                     pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0)
    vl = torch.utils.data.DataLoader(_TorchDS(va, True), batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.workers)
    model = TemporalUNet(cfg.frames, cfg.base, cfg.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(total_steps, 1), pct_start=0.1)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_f1, bad = 0, -1.0, 0
    last = out / "last.pt"
    if (out / "DONE").exists() and not pilot:
        log.info("%s already finished (%s); delete DONE to train further", out, (out / "DONE").read_text().strip())
        return out
    if last.exists() and not pilot:
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck.get("total_steps", total_steps) != total_steps:
            raise SystemExit(
                f"{last} was trained with a different schedule (epochs x steps = {ck.get('total_steps')} vs "
                f"{total_steps}); use a new out_dir or restore epochs/batch_size/samples_per_epoch"
            )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, bad = ck["epoch"] + 1, ck["best_f1"], ck["bad"]
        # new random stream per resumed job, otherwise every chained job replays the same samples
        torch.manual_seed(cfg.seed + 1000 * start_epoch)
        tr.rng = np.random.default_rng(cfg.seed + 1000 * start_epoch)
        log.info("resumed from epoch %d (best F1 %.4f)", ck["epoch"], best_f1)
    meta = {"frames": cfg.frames, "base": cfg.base, "depth": cfg.depth, "threshold": cfg.threshold}
    logf = out / "log.csv"
    new_log = not logf.exists()
    with open(logf, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new_log:
            wr.writerow(["epoch", "train_loss", "val_precision", "val_recall", "val_f1", "lr", "sec"])
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            t0 = time.time()
            losses = []
            for step, (stack, heat, off, pos, weight) in enumerate(dl):
                stack, heat, off, pos, weight = (a.to(device, non_blocking=True) for a in (stack, heat, off, pos, weight))
                x = build_input(stack[:, -1:], [stack[:, q : q + 1] for q in range(stack.shape[1] - 2, -1, -1)])
                with (torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()):
                    hl, of = model(x)
                loss = focal_loss(hl.float(), heat, weight) + cfg.offset_weight * offset_loss(of.float(), off, pos)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(float(loss.detach()))
                if pilot and step + 1 >= 200:
                    break
            if pilot:
                sec = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
                res = {"steps": len(losses), "sec_per_step": sec / max(len(losses), 1), "max_mem_gib": mem,
                       "loss_first": losses[0], "loss_last": float(np.mean(losses[-20:]))}
                (out / "pilot.json").write_text(json.dumps(res, indent=2))
                log.info("pilot: %s", res)
                return out
            m = evaluate(model, vl, device, cfg.threshold, cfg.tol_px)
            sec = time.time() - t0
            wr.writerow([epoch, float(np.mean(losses)), m["precision"], m["recall"], m["f1"], sched.get_last_lr()[0], round(sec, 1)])
            fh.flush()
            log.info("epoch %d loss %.4f val P %.3f R %.3f F1 %.3f (%.0fs)", epoch, np.mean(losses), m["precision"],
                     m["recall"], m["f1"], sec)
            if m["f1"] > best_f1:
                best_f1, bad = m["f1"], 0
                _atomic_save(torch, {"model": model.state_dict(), "meta": {**meta, "val": m, "epoch": epoch}}, out / "best.pt")
            else:
                bad += 1
            _atomic_save(torch, {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                 "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1, "bad": bad,
                                 "meta": meta, "total_steps": total_steps}, last)
            if bad >= cfg.patience:
                log.info("early stop at epoch %d", epoch)
                (out / "DONE").write_text(f"early stop at epoch {epoch}, best F1 {best_f1:.4f}\n")
                return out
    (out / "DONE").write_text(f"completed {cfg.epochs} epochs, best F1 {best_f1:.4f}\n")
    return out


def _atomic_save(torch, obj, path: Path) -> None:
    """Write then rename, so a job killed mid-save never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_train_cfg(path: str | Path, overrides: list[str] | None = None) -> TrainCfg:
    d = yaml.safe_load(Path(path).read_text()) or {}
    for o in overrides or []:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    return TrainCfg(**d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="+", action="extend", default=[], help="overrides key=value (repeatable)")
    ap.add_argument("--print-out-dir", action="store_true", help="print the resolved out_dir and exit")
    ap.add_argument("--pilot", action="store_true", help="run 200 steps and report speed/memory")
    a = ap.parse_args(argv)
    cfg = load_train_cfg(a.config, a.set)
    if a.print_out_dir:
        print(cfg.out_dir)
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train(cfg, pilot=a.pilot)


if __name__ == "__main__":
    main()
