"""Pinhole camera with radial distortion and an attitude relative to the ship's RIC frame.

Camera axes follow OpenCV: x right, y down, z along the optical axis.
``R_cam_ric`` maps RIC vectors into camera coordinates: ``v_cam = R @ v_ric``.

Pixel coordinates are *native* video pixels (the HUD-free frame of the
broadcast). Distortion uses the Brown model ``x_d = x (1 + k1 r^2 + k2 r^4)``
on normalised coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Reference attitude: optical axis along -R (nadir), image "up" (-y) along +I (velocity).
# Rows are camera axes expressed in RIC: x_cam = +C? choose right-handed: x = z_cam x y_cam.
_NADIR_REF = np.array(
    [
        [0.0, 0.0, -1.0],  # x_cam = -C
        [0.0, -1.0, 0.0],  # y_cam = -I  (image down = backwards)
        [-1.0, 0.0, 0.0],  # z_cam = -R  (looking down)
    ]
)


def attitude_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R_cam_ric from yaw/pitch/roll applied to the nadir-looking reference (deg).

    yaw rotates about the optical axis' reference (radial), pitch about the
    cross-track axis, roll about the in-track axis, all in the RIC frame.
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    # rotate the camera body in RIC: R_ric_cam = Rz?(...) applied to reference
    R_ric_cam_ref = _NADIR_REF.T
    turn = rot_x(y) @ rot_z(p) @ rot_y(r)  # about R, C, I axes respectively
    R_ric_cam = turn @ R_ric_cam_ref
    return R_ric_cam.T


def triad(b1: np.ndarray, b2: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Rotation A with ``b ~ A @ r`` from two vector pairs (b1 is trusted most)."""

    def frame(a, b):
        t1 = a / np.linalg.norm(a)
        t2 = np.cross(a, b)
        t2 /= np.linalg.norm(t2)
        return np.column_stack([t1, t2, np.cross(t1, t2)])

    return frame(b1, b2) @ frame(r1, r2).T


@dataclass
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    R_cam_ric: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, k1: float = 0.0, k2: float = 0.0, cx=None, cy=None):
        f = 0.5 * width / np.tan(0.5 * np.deg2rad(hfov_deg))
        return cls(width, height, f, f, (width - 1) / 2 if cx is None else cx, (height - 1) / 2 if cy is None else cy, k1, k2)

    @property
    def hfov_deg(self) -> float:
        return float(np.rad2deg(2 * np.arctan(0.5 * self.width / self.fx)))

    @property
    def ifov_rad(self) -> float:
        return 1.0 / self.fx

    @property
    def calibrated(self) -> bool:
        return self.R_cam_ric is not None

    # ------------------------------------------------------------- distortion
    def _r_max(self) -> float:
        """Largest undistorted radius where the Brown model is still monotonic (inf if always)."""
        k1, k2 = self.k1, self.k2
        # d r_d / d r_u = 1 + 3 k1 r^2 + 5 k2 r^4 ; smallest positive root in x = r^2
        roots = np.roots([5 * k2, 3 * k1, 1.0]) if k2 != 0 else (np.array([-1.0 / (3 * k1)]) if k1 != 0 else np.array([]))
        pos = [float(np.real(x)) for x in np.atleast_1d(roots) if abs(np.imag(x)) < 1e-12 and np.real(x) > 0]
        return float(np.sqrt(min(pos))) if pos else float("inf")

    def _distort(self, xn: np.ndarray, yn: np.ndarray):
        r2 = xn * xn + yn * yn
        k = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return xn * k, yn * k

    def _undistort(self, xd: np.ndarray, yd: np.ndarray, iters: int = 20):
        """Invert the radial model with Newton's method on the radius (monotonic branch only)."""
        if self.k1 == 0 and self.k2 == 0:
            return xd.copy(), yd.copy()
        rd = np.sqrt(xd * xd + yd * yd)
        ru = rd.copy()
        rmax = self._r_max()
        for _ in range(iters):
            f = ru * (1 + self.k1 * ru**2 + self.k2 * ru**4) - rd
            fp = 1 + 3 * self.k1 * ru**2 + 5 * self.k2 * ru**4
            step = f / np.where(np.abs(fp) > 1e-9, fp, 1e-9)
            ru = np.clip(ru - step, 0.0, rmax if np.isfinite(rmax) else None)
            if np.all(np.abs(step) < 1e-12):
                break
        scale = np.where(rd > 1e-15, ru / np.maximum(rd, 1e-15), 1.0)
        return xd * scale, yd * scale

    # -------------------------------------------------------------- project
    def project_cam(self, v_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame vectors (..., 3) -> pixels (..., 2) and a validity mask.

        Points behind the camera, or beyond the radius where the distortion model
        folds back, are invalid (otherwise a ray 70 deg off-axis could land inside
        the image).
        """
        v = np.asarray(v_cam, dtype=np.float64)
        z = v[..., 2]
        front = z > 1e-9
        zs = np.where(front, z, np.nan)
        xn, yn = v[..., 0] / zs, v[..., 1] / zs
        rmax = self._r_max()
        if np.isfinite(rmax):
            front = front & (np.sqrt(xn * xn + yn * yn) < rmax)
        xd, yd = self._distort(xn, yn)
        uv = np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], -1)
        return uv, front

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (..., 2) -> unit rays in the camera frame (..., 3)."""
        uv = np.asarray(uv, dtype=np.float64)
        xd = (uv[..., 0] - self.cx) / self.fx
        yd = (uv[..., 1] - self.cy) / self.fy
        xn, yn = self._undistort(xd, yd)
        v = np.stack([xn, yn, np.ones_like(xn)], -1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def project_ric(self, v_ric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.R_cam_ric is None:
            raise RuntimeError("camera attitude not calibrated")
        v_cam = np.einsum("ij,...j->...i", self.R_cam_ric, v_ric)
        return self.project_cam(v_cam)

    def in_image(self, uv: np.ndarray, front: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (
            front
            & (uv[..., 0] >= -margin)
            & (uv[..., 0] <= self.width - 1 + margin)
            & (uv[..., 1] >= -margin)
            & (uv[..., 1] <= self.height - 1 + margin)
        )

    def rays_ric(self, uv: np.ndarray) -> np.ndarray:
        if self.R_cam_ric is None:
            raise RuntimeError("camera attitude not calibrated")
        return np.einsum("ji,...j->...i", self.R_cam_ric, self.unproject(uv))

    def fov_solid_angle(self) -> float:
        """Approximate solid angle of the image (sr)."""
        hw, hh = np.arctan(0.5 * self.width / self.fx), np.arctan(0.5 * self.height / self.fy)
        return float(4 * np.arcsin(np.sin(hw) * np.sin(hh)))

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "R_cam_ric": None if self.R_cam_ric is None else np.asarray(self.R_cam_ric).tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PinholeCamera":
        R = d.get("R_cam_ric")
        return cls(
            d["width"], d["height"], d["fx"], d["fy"], d["cx"], d["cy"], d.get("k1", 0.0), d.get("k2", 0.0),
            None if R is None else np.array(R, dtype=float), d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PinholeCamera":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def scaled(self, s: float) -> "PinholeCamera":
        """Same camera for an image resized by ``s`` (pixel-centre convention)."""
        c = 0.5 * s - 0.5
        return PinholeCamera(
            int(round(self.width * s)), int(round(self.height * s)), self.fx * s, self.fy * s,
            self.cx * s + c, self.cy * s + c, self.k1, self.k2, self.R_cam_ric, dict(self.meta),
        )


def camera_from_config(cam_cfg, width: int, height: int) -> PinholeCamera:
    if cam_cfg.calibration_file:
        cam = PinholeCamera.load(cam_cfg.calibration_file)
        if (cam.width, cam.height) != (width, height):
            cam = cam.scaled(width / cam.width)
        return cam
    cam = PinholeCamera.from_hfov(width, height, cam_cfg.hfov_deg, cam_cfg.k1, cam_cfg.k2, cam_cfg.cx, cam_cfg.cy)
    if cam_cfg.attitude_ypr_deg is not None:
        cam.R_cam_ric = attitude_from_ypr(*cam_cfg.attitude_ypr_deg)
    cam.meta["source"] = "config (assumed intrinsics)"
    return cam
