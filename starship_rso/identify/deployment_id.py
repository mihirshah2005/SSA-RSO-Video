"""Map observed release events (DEPLOY-k) to catalogue objects, only when supported.

Evidence sources, strongest first:

1. ``release_order`` -- an *independent* published mapping from release slot to
   NORAD id (e.g. operator data). Gives ACCEPTED only when its source is
   recorded; it is the only route to an exact name that does not depend on
   element-set accuracy.
2. Release-time estimates from back-propagated element sets
   (:mod:`orbit.release`). Each event's observed release MET is compared with
   each object's estimated release time; likelihoods are Gaussian in the time
   difference with both uncertainties, and an "unmatched" hypothesis is
   uniform over the deployment window. Exclusivity is enforced with a global
   assignment. When estimates are as uncertain as the release spacing the
   posteriors stay flat and the output remains a candidate list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..orbit.release import ReleaseEstimate
from ..types import IdentityCandidate, IdentityDecision, IdentityStatus


@dataclass
class EventTime:
    label: str
    utc: float
    sigma_s: float


def match_release_events(
    events: list[EventTime],
    estimates: list[ReleaseEstimate],
    window_s: float,
    p_unmatched: float = 0.2,
    accept_posterior: float = 0.95,
) -> dict[str, IdentityDecision]:
    out: dict[str, IdentityDecision] = {}
    if not events:
        return out
    if not estimates:
        return {e.label: IdentityDecision(IdentityStatus.UNAVAILABLE, e.label, reasons=["no release-time estimates"]) for e in events}
    t_e = np.array([e.utc for e in events])
    s_e = np.array([e.sigma_s for e in events])
    t_c = np.array([c.t_release_utc for c in estimates])
    # unidentifiable estimates (infinite sigma) become as wide as the window: they carry no timing information
    s_c = np.array([c.t_sigma_s if np.isfinite(c.t_sigma_s) else max(window_s, 1.0) for c in estimates])
    var = s_e[:, None] ** 2 + s_c[None, :] ** 2
    ll = -0.5 * (t_e[:, None] - t_c[None, :]) ** 2 / var - 0.5 * np.log(2 * np.pi * var)
    ll_u = -np.log(max(window_s, 1.0))
    prior_c = np.log((1 - p_unmatched) / len(estimates))
    lp = np.c_[np.full(len(events), np.log(p_unmatched) + ll_u), prior_c + ll]
    lp -= lp.max(1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(1, keepdims=True)
    # exclusivity: assignment on -log posterior with dummy "unmatched" columns
    cost = -np.log(np.maximum(post[:, 1:], 1e-300))
    dummy = np.tile(-np.log(np.maximum(post[:, :1], 1e-300)), (1, len(events)))
    rows, cols = linear_sum_assignment(np.c_[cost, dummy])
    for r, cidx in zip(rows, cols):
        e = events[r]
        order = np.argsort(-post[r, 1:])
        cands = [
            IdentityCandidate(estimates[k].norad_id, estimates[k].name, float(ll[r, k]), float(post[r, 1 + k]),
                              float((t_e[r] - t_c[k]) ** 2 / var[r, k]),
                              {"dt_s": float(t_e[r] - t_c[k]), "t_sigma_s": float(np.sqrt(var[r, k]))})
            for k in order[:3]
        ]
        pu = float(post[r, 0])
        if cidx < len(estimates) and post[r, 1 + cidx] >= accept_posterior:
            k = cidx
            out[e.label] = IdentityDecision(
                IdentityStatus.ACCEPTED, f"{e.label} = {estimates[k].name} ({estimates[k].norad_id})", cands, pu,
                reasons=[f"release-time match p={post[r, 1 + k]:.3f} ({estimates[k].method})"],
            )
        elif cands and cands[0].posterior >= 0.05:
            out[e.label] = IdentityDecision(IdentityStatus.CANDIDATES, f"{e.label}: {sum(c.posterior >= 0.05 for c in cands)}+ candidates",
                                            cands, pu, reasons=["release times not separable at this accuracy"])
        else:
            out[e.label] = IdentityDecision(IdentityStatus.UNAVAILABLE, e.label, cands, pu, reasons=["no compatible release time"])
    return out


def decisions_from_release_order(
    events: list[tuple[str, int | None]], release_order: list[str], names: dict[str, str] | None, source: str
) -> dict[str, IdentityDecision]:
    """Use an independent slot->NORAD mapping. ``events`` are (label, schedule index k)."""
    out = {}
    for label, k in events:
        if k is None or not (1 <= k <= len(release_order)):
            out[label] = IdentityDecision(IdentityStatus.UNAVAILABLE, label, reasons=["release slot unknown"])
            continue
        nid = release_order[k - 1]
        nm = (names or {}).get(nid, nid)
        out[label] = IdentityDecision(
            IdentityStatus.ACCEPTED, f"{label} = {nm} ({nid})",
            [IdentityCandidate(nid, nm, 0.0, 1.0, 0.0, {"source": source})], 0.0,
            reasons=[f"slot {k} from independent release order ({source})"],
        )
    return out
