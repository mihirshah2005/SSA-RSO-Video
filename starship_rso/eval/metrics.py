"""Evaluation: point detection, image-plane tracking and identity metrics.

* Detection: one-to-one Hungarian matching of predicted to true centroids per
  frame within a pixel tolerance; precision/recall/F1 at several tolerances.
* Tracking: CLEAR-MOT style counts (matches, misses, false positives, identity
  switches, fragmentations) and IDF1 from a global track-to-track assignment.
* Identity: over tracks whose truth identity is known, the precision of
  *accepted* names, the coverage (fraction named), and the false-name rate;
  unknowns are reported, never dropped.

Truth and predictions are lists of per-frame dicts ``{"id": ..., "x": ..., "y": ...}``.
Bootstrap confidence intervals resample whole tracks/sequences, not frames.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_points(gt: np.ndarray, pred: np.ndarray, tol: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Hungarian one-to-one matching within ``tol`` px. Returns (pairs, distances)."""
    if len(gt) == 0 or len(pred) == 0:
        return [], np.zeros(0)
    D = np.hypot(gt[:, None, 0] - pred[None, :, 0], gt[:, None, 1] - pred[None, :, 1])
    C = np.where(D <= tol, D, 1e6)
    r, c = linear_sum_assignment(C)
    keep = C[r, c] < 1e6
    return list(zip(r[keep].tolist(), c[keep].tolist())), D[r[keep], c[keep]]


@dataclass
class DetCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    loc_err: list = None

    def __post_init__(self):
        if self.loc_err is None:
            self.loc_err = []

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-12)

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4),
                "loc_err_median_px": round(float(np.median(self.loc_err)), 3) if self.loc_err else None}


def detection_metrics(
    truth_frames: list[list[dict]],
    pred_frames: list[list[dict]],
    tols=(2.0, 3.0, 5.0),
    kinds: set[str] | None = None,
    ignore_kinds: set[str] | None = None,
    min_conf: float = 0.0,
) -> dict[str, dict]:
    """Per-tolerance detection counts. Predictions matched to ``ignore_kinds`` truth are neither TP nor FP."""
    out = {}
    for tol in tols:
        dc = DetCounts()
        for gt_f, pr_f in zip(truth_frames, pred_frames):
            gt = [o for o in gt_f if kinds is None or o.get("kind") in kinds]
            ign = [o for o in gt_f if ignore_kinds and o.get("kind") in ignore_kinds]
            pr = [p for p in pr_f if p.get("confidence", 1.0) >= min_conf]
            G = np.array([[o["x"], o["y"]] for o in gt]).reshape(-1, 2)
            P = np.array([[p["x"], p["y"]] for p in pr]).reshape(-1, 2)
            pairs, dist = match_points(G, P, tol)
            used = {j for _, j in pairs}
            dc.tp += len(pairs)
            dc.fn += len(G) - len(pairs)
            dc.loc_err += dist.tolist()
            rest = [j for j in range(len(P)) if j not in used]
            if ign and rest:
                I = np.array([[o["x"], o["y"]] for o in ign])
                R = P[rest]
                rad = np.array([max(tol, o.get("r", 0) + tol) for o in ign])
                D = np.hypot(I[:, None, 0] - R[None, :, 0], I[:, None, 1] - R[None, :, 1])
                rest = [j for k, j in enumerate(rest) if not (D[:, k] <= rad).any()]
            dc.fp += len(rest)
        out[f"tol{tol:g}"] = dc.to_dict()
    return out


def tracking_metrics(truth_frames: list[list[dict]], track_frames: list[list[dict]], tol: float = 5.0) -> dict:
    """CLEAR-MOT counts and IDF1. ``track_frames`` entries need ``id`` (track id), ``x``, ``y``."""
    matches = misses = fps = switches = 0
    last_match: dict = {}
    frag: dict = defaultdict(int)
    tracked_prev: dict = {}
    pair_counts: dict = defaultdict(int)
    gt_len: dict = defaultdict(int)
    pr_len: dict = defaultdict(int)
    for gt_f, tr_f in zip(truth_frames, track_frames):
        G = np.array([[o["x"], o["y"]] for o in gt_f]).reshape(-1, 2)
        P = np.array([[p["x"], p["y"]] for p in tr_f]).reshape(-1, 2)
        pairs, _ = match_points(G, P, tol)
        matched_g = set()
        for i, j in pairs:
            g, p = gt_f[i]["id"], tr_f[j]["id"]
            matched_g.add(g)
            pair_counts[(g, p)] += 1
            if g in last_match and last_match[g] != p:
                switches += 1
            if g in tracked_prev and not tracked_prev[g]:
                frag[g] += 1
            last_match[g] = p
        for o in gt_f:
            gt_len[o["id"]] += 1
            tracked_prev[o["id"]] = o["id"] in matched_g
        for p in tr_f:
            pr_len[p["id"]] += 1
        matches += len(pairs)
        misses += len(G) - len(pairs)
        fps += len(P) - len(pairs)
    n_gt = sum(gt_len.values())
    mota = 1 - (misses + fps + switches) / max(n_gt, 1)
    # IDF1: global one-to-one gt-track <-> pred-track assignment maximising co-matched frames
    gids, pids = sorted(gt_len), sorted(pr_len)
    idtp = 0
    if gids and pids:
        M = np.zeros((len(gids), len(pids)))
        gi = {g: k for k, g in enumerate(gids)}
        pi = {p: k for k, p in enumerate(pids)}
        for (g, p), n in pair_counts.items():
            M[gi[g], pi[p]] = n
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    idf1 = 2 * idtp / max(sum(gt_len.values()) + sum(pr_len.values()), 1)
    return {"mota": round(mota, 4), "idf1": round(idf1, 4), "matches": matches, "misses": misses,
            "false_positives": fps, "id_switches": switches, "fragmentations": int(sum(frag.values())),
            "gt_tracks": len(gids), "pred_tracks": len(pids)}


def identity_metrics(track_truth: dict[str, str | None], decisions: dict[str, tuple[str, str | None]]) -> dict:
    """``track_truth``: track -> true object id (None = no catalogue identity, e.g. particle).

    ``decisions``: track -> (status, accepted_object_id or None).
    """
    accepted = correct = wrong = named_truth = 0
    for tr, truth in track_truth.items():
        status, oid = decisions.get(tr, ("none", None))
        if truth is not None:
            named_truth += 1
        if status == "accepted":
            accepted += 1
            if truth is not None and oid == truth:
                correct += 1
            else:
                wrong += 1
    return {
        "accepted": accepted,
        "accepted_correct": correct,
        "accepted_wrong": wrong,
        "accepted_precision": round(correct / accepted, 4) if accepted else None,
        "coverage": round(correct / named_truth, 4) if named_truth else None,
        "tracks_with_true_identity": named_truth,
    }


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def assign_tracks_to_truth(truth_frames, track_frames, tol: float = 5.0, min_frac: float = 0.5) -> dict:
    """Majority truth id for each predicted track (None if matched in < ``min_frac`` of its frames)."""
    votes: dict = defaultdict(lambda: defaultdict(int))
    length: dict = defaultdict(int)
    for gt_f, tr_f in zip(truth_frames, track_frames):
        G = np.array([[o["x"], o["y"]] for o in gt_f]).reshape(-1, 2)
        P = np.array([[p["x"], p["y"]] for p in tr_f]).reshape(-1, 2)
        pairs, _ = match_points(G, P, tol)
        for i, j in pairs:
            votes[tr_f[j]["id"]][gt_f[i]["id"]] += 1
        for p in tr_f:
            length[p["id"]] += 1
    out = {}
    for tid, n in length.items():
        if votes[tid]:
            g, k = max(votes[tid].items(), key=lambda kv: kv[1])
            out[tid] = g if k >= min_frac * n else None
        else:
            out[tid] = None
    return out
