"""Track-level catalogue association with an explicit "unknown/uncatalogued" hypothesis.

For each confirmed track, the observed image path is converted to ship-RIC
unit rays with the calibrated camera. For each candidate catalogue object
the cached prediction gives rays at the same instants. Residuals are taken in
the tangent plane of the mean observed direction and summarised by

* the mean offset (absorbs shared attitude/time/ephemeris biases), and
* the residual angular *rate* (sensitive to the object's apparent motion),

each with an uncertainty built from the stated error budget. Adjacent video
frames are not treated as independent evidence: the shared errors enter as a
single bias term. The unknown hypothesis has a uniform density over the
field of view and over plausible angular rates. Posteriors combine these
likelihoods with a prior that keeps ``p_unknown_prior`` on "none of the
catalogue". The best candidate is accepted only if its posterior is high,
its absolute fit passes a chi-square test, it is stable across consecutive
evaluations, and no other track claims the same object. Otherwise the output
is a candidate list or "identity unavailable".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import IdentifyCfg
from ..geometry.camera import PinholeCamera
from ..types import IdentityCandidate, IdentityDecision, IdentityStatus
from .predictions import PredictionCache


@dataclass
class TrackObs:
    utc: np.ndarray  # (n,)
    uv_native: np.ndarray  # (n, 2)
    pix_sigma: np.ndarray  # (n,)
    psf_sigma_px: float = 1.0  # median apparent size (second-moment sigma, native px)
    time_sigma_s: float = 0.0  # time-map uncertainty of these observations


def _tangent_basis(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 0.0, 1.0]) if abs(m[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(m, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(m, e1)


def _gnomonic(d: np.ndarray, m: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    dm = d @ m
    dm = np.where(dm > 1e-6, dm, np.nan)
    return np.stack([(d @ e1) / dm, (d @ e2) / dm], -1)


def score_track(
    obs: TrackObs, cam: PinholeCamera, cache: PredictionCache, cfg: IdentifyCfg
) -> tuple[list[IdentityCandidate], float, dict]:
    """Return (candidates sorted by posterior, p_unknown, diagnostics).

    Each candidate's ``detail`` carries the uncertainties its likelihood used
    (``sigma_bias_deg``, ``sigma_rate_deg_s``) and ``p_chance``: the probability
    that an unrelated track would fall inside its chi-square acceptance region
    under the unknown hypothesis. Acceptance (in :class:`CatalogAssociator`)
    requires these to be small, so a candidate whose predicted position carries
    no information can never be accepted on its angular rate alone.
    """
    n = len(obs.utc)
    rays = cam.rays_ric(obs.uv_native)  # (n, 3)
    m = rays.mean(0)
    m /= np.linalg.norm(m)
    e1, e2 = _tangent_basis(m)
    g_obs = _gnomonic(rays, m, e1, e2)  # (n, 2)
    tt = obs.utc - obs.utc.mean()
    T = max(float(np.ptp(obs.utc)), 1e-3)
    om_obs = np.zeros(2) if n < 3 else np.array([np.polyfit(tt, g_obs[:, k], 1)[0] for k in range(2)])

    rel = cache.rel_at(obs.utc)  # (N, n, 3)
    rng_km = np.linalg.norm(rel, axis=-1)
    pred = rel / np.maximum(rng_km[..., None], 1e-9)
    rho = np.maximum(rng_km.mean(1), 1e-6)  # (N,)

    pix = float(np.median(obs.pix_sigma)) * cam.ifov_rad
    att = np.deg2rad(cfg.attitude_sigma_deg)
    rate_floor = np.deg2rad(cfg.rate_sigma_deg_s)
    omega_mag = float(np.linalg.norm(om_obs))
    t_sig = max(cfg.time_sigma_s, obs.time_sigma_s)
    # relative-position uncertainty: catalogue error and ship-ephemeris error add in quadrature
    ship_sig = cache.ship_sigma_km if np.isfinite(cache.ship_sigma_km) else 0.0
    eph = float(np.hypot(cfg.ephem_sigma_km, ship_sig))
    v_rel = np.linalg.norm(np.diff(rel, axis=1), axis=-1).sum(1) / T if n > 1 else np.zeros(len(rho))
    sb2_all = att**2 + (eph / rho) ** 2 + (omega_mag * t_sig) ** 2 + pix**2 / n
    sr2_all = rate_floor**2 + (eph * v_rel / rho**2) ** 2 + 12 * pix**2 / (n * T**2)

    # gate: in front of the camera and inside the image expanded by each candidate's own uncertainty
    uv, front = cam.project_ric(pred)
    margin = np.minimum(3.0 * np.sqrt(sb2_all) * cam.fx, 4.0 * max(cam.width, cam.height)) + 20.0  # (N,)
    inside = (front & (uv[..., 0] >= -margin[:, None]) & (uv[..., 0] <= cam.width - 1 + margin[:, None])
              & (uv[..., 1] >= -margin[:, None]) & (uv[..., 1] <= cam.height - 1 + margin[:, None])).mean(1) > 0.5
    lit = cache.sunlit_at(obs.utc).mean(1) > 0.5
    # size consistency: a resolved blob cannot be a distant object that should be point-like
    size_px = (cfg.object_size_m / 1000.0) / rho * cam.fx
    expected_sigma = np.sqrt(cfg.psf_sigma_px**2 + (size_px / 2.355) ** 2)
    size_ok = obs.psf_sigma_px <= 3.0 * expected_sigma + 1.0
    keep = np.nonzero(inside & lit & size_ok)[0]

    fov_sr = cam.fov_solid_angle()
    unknown_density = 1.0 / (fov_sr * np.pi * np.deg2rad(cfg.max_rate_deg_s) ** 2)
    log_u = float(np.log(unknown_density))
    diag = {"n_obs": n, "span_s": T, "n_candidates_in_view": int(len(keep)), "omega_deg_s": float(np.rad2deg(omega_mag)),
            "n_rejected_by_size": int((inside & lit & ~size_ok).sum()), "eph_sigma_km": eph, "time_sigma_s": t_sig}
    if len(keep) == 0:
        return [], 1.0, diag

    cands: list[IdentityCandidate] = []
    logls = []
    c2 = cfg.accept_chi2
    for j in keep:
        g_p = _gnomonic(pred[j], m, e1, e2)
        e = g_p - g_obs
        if not np.all(np.isfinite(e)):
            continue
        bias = e.mean(0)
        rate = np.array([np.polyfit(tt, e[:, k], 1)[0] for k in range(2)]) if n >= 3 else np.zeros(2)
        sb2, sr2 = float(sb2_all[j]), float(sr2_all[j])
        d2 = float((bias @ bias) / sb2 + (rate @ rate) / sr2)
        ll = -0.5 * d2 - np.log(2 * np.pi * sb2) - np.log(2 * np.pi * sr2)
        # volume of the 4-D chi-square <= c2 region times the unknown density = chance of a false fit
        p_chance = float(min(1.0, 0.5 * np.pi**2 * c2**2 * sb2 * sr2 * unknown_density))
        logls.append(ll)
        cands.append(
            IdentityCandidate(
                cache.ids[j], cache.names[j], float(ll), 0.0, d2,
                {"range_km": float(rho[j]), "bias_deg": float(np.rad2deg(np.linalg.norm(bias))),
                 "rate_res_deg_s": float(np.rad2deg(np.linalg.norm(rate))),
                 "sigma_bias_deg": float(np.rad2deg(np.sqrt(sb2))), "sigma_rate_deg_s": float(np.rad2deg(np.sqrt(sr2))),
                 "p_chance": p_chance},
            )
        )
    if not cands:
        return [], 1.0, diag
    logls = np.array(logls)
    pu = cfg.p_unknown_prior
    lp = np.r_[np.log(pu) + log_u, np.log((1 - pu) / len(cands)) + logls]
    lp -= lp.max()
    post = np.exp(lp)
    post /= post.sum()
    for c, p in zip(cands, post[1:]):
        c.posterior = float(p)
    cands.sort(key=lambda c: -c.posterior)
    return cands, float(post[0]), diag


class CatalogAssociator:
    """Runs on the slow loop; keeps per-track hysteresis and global exclusivity."""

    def __init__(self, cfg: IdentifyCfg, cam: PinholeCamera, cache: PredictionCache | None, scaler=None):
        self.cfg = cfg
        self.cam = cam
        self.cache = cache
        self.scaler = scaler
        self._streak: dict[int, tuple[str, int]] = {}
        self.history: list[dict] = []

    def unavailable_reason(self) -> str | None:
        if self.cache is None or len(self.cache) == 0:
            return "no catalogue predictions loaded"
        if not self.cam.calibrated:
            return "camera attitude not calibrated"
        return None

    def _obs(self, track) -> TrackObs | None:
        pts = [p for p in track.observed_points() if p.utc is not None]
        if len(pts) < 2:
            return None
        uv = np.array([[p.detection.x, p.detection.y] for p in pts], float)
        if self.scaler is not None:
            uv = np.array([self.scaler.to_native(x, y) for x, y in uv])
            s = 1.0 / self.scaler.scale
        else:
            s = 1.0
        sig = np.array([p.detection.pos_sigma * s for p in pts])
        psf = float(np.median([p.detection.sigma for p in pts])) * s
        tsig = [p.time_sigma for p in pts if getattr(p, "time_sigma", None) is not None]
        return TrackObs(np.array([p.utc for p in pts]), uv, sig, psf, float(np.median(tsig)) if tsig else 0.0)

    def evaluate(self, tracks, now_t: float) -> None:
        """Update ``track.identity_geo`` for every confirmed track (slow loop)."""
        reason = self.unavailable_reason()
        accepted_by: dict[str, list] = {}
        excluded = ("near_field_particle", "vehicle_feature", "background_feature", "artifact")
        n_eval = max(1, sum(1 for tr in tracks if tr.category.category.value not in excluded))
        for tr in tracks:
            if tr.category.category.value in ("near_field_particle", "vehicle_feature", "background_feature", "artifact"):
                tr.identity_geo = IdentityDecision(IdentityStatus.NONE, "", reasons=[f"category {tr.category.category.value}"])
                self._streak.pop(tr.uid, None)
                continue
            if reason is not None:
                tr.identity_geo = IdentityDecision(IdentityStatus.UNAVAILABLE, "", reasons=[reason], decided_t=now_t)
                continue
            obs = self._obs(tr)
            if obs is None or len(obs.utc) < self.cfg.min_track_obs:
                tr.identity_geo = IdentityDecision(IdentityStatus.UNAVAILABLE, "", reasons=["track too short"], decided_t=now_t)
                continue
            if not (self.cache.covers(obs.utc[0]) and self.cache.covers(obs.utc[-1])):
                tr.identity_geo = IdentityDecision(IdentityStatus.UNAVAILABLE, "", reasons=["outside prediction window"], decided_t=now_t)
                continue
            cands, pu, diag = score_track(obs, self.cam, self.cache, self.cfg)
            dec = self._decide(tr, cands, pu, diag, now_t, n_eval)
            tr.identity_geo = dec
            if dec.status == IdentityStatus.ACCEPTED:
                accepted_by.setdefault(dec.best.object_id, []).append(tr)
            self.history.append({"t": now_t, "track": tr.uid, "status": dec.status.value, "label": dec.label,
                                 "p_unknown": pu, "diag": diag,
                                 "top": [(c.object_id, round(c.posterior, 4), round(c.mahalanobis2, 2)) for c in cands[:3]]})
        # exclusivity: one catalogue object explains at most one track. Among the tracks that claim
        # it, the object is assigned to the best only if that assignment alone is probable enough.
        for oid, trs in accepted_by.items():
            if len(trs) < 2:
                continue
            trs = sorted(trs, key=lambda t: -t.identity_geo.best.log_likelihood)
            ll = np.array([t.identity_geo.best.log_likelihood for t in trs])
            p_best = 1.0 / np.exp(ll - ll[0]).sum()
            losers = trs[1:] if p_best >= self.cfg.accept_posterior else trs
            for tr in losers:
                tr.identity_geo.status = IdentityStatus.CANDIDATES
                tr.identity_geo.label = f"{oid} claimed by {len(trs)} tracks"
                tr.identity_geo.reasons.append("conflict: same object claimed by several tracks")
                self._streak.pop(tr.uid, None)

    def _decide(self, tr, cands, pu, diag, now_t, n_eval: int = 1) -> IdentityDecision:
        c = self.cfg
        if not cands:
            self._streak.pop(tr.uid, None)
            return IdentityDecision(IdentityStatus.UNAVAILABLE, "no catalogue object in view", [], pu, now_t,
                                    [f"{diag['n_candidates_in_view']} candidates in view"])
        best = cands[0]
        d = best.detail
        informative = (d["sigma_bias_deg"] <= c.max_bias_sigma_deg and d["sigma_rate_deg_s"] <= c.max_rate_sigma_deg_s)
        # expected number of unrelated tracks that would fit this candidate as well, by chance
        chance = n_eval * d["p_chance"]
        good = (best.posterior >= c.accept_posterior and best.mahalanobis2 <= c.accept_chi2
                and informative and chance <= c.max_chance_matches)
        if good:
            prev = self._streak.get(tr.uid)
            n = prev[1] + 1 if prev and prev[0] == best.object_id else 1
            self._streak[tr.uid] = (best.object_id, n)
            if n >= 2:
                return IdentityDecision(IdentityStatus.ACCEPTED, f"{best.name} ({best.object_id})", cands[:3], pu, now_t,
                                        [f"posterior {best.posterior:.3f}, chi2 {best.mahalanobis2:.1f}, "
                                         f"chance matches {chance:.1e}, stable x{n}"])
        else:
            self._streak.pop(tr.uid, None)
        plausible = [k for k in cands if k.posterior >= 0.05]
        why = []
        if not informative:
            why.append(f"geometry not informative (sigma {d['sigma_bias_deg']:.1f} deg, {d['sigma_rate_deg_s']:.2f} deg/s)")
        elif chance > c.max_chance_matches:
            why.append(f"a chance match is too likely ({chance:.2f} expected)")
        if plausible:
            return IdentityDecision(IdentityStatus.CANDIDATES, f"{len(plausible)} candidates", plausible[:3], pu, now_t,
                                    [f"best {best.name} p={best.posterior:.2f}, unknown p={pu:.2f}"] + why)
        return IdentityDecision(IdentityStatus.UNAVAILABLE, "unknown / uncatalogued", cands[:3], pu, now_t,
                                [f"unknown hypothesis preferred (p={pu:.2f})"] + why)
