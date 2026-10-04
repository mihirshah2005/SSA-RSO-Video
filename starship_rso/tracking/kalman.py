"""Constant-velocity Kalman filter in the image plane with real time steps.

State ``[x, y, vx, vy]`` (px, px/s). Process noise is the continuous
white-acceleration model with spectral density ``q`` (px^2/s^3), so the
filter integrates over the actual interval between processed frames,
including frames dropped by the live reader.
"""

from __future__ import annotations

import numpy as np

_H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def transition(dt: float) -> np.ndarray:
    return np.array([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


def process_noise(dt: float, q: float) -> np.ndarray:
    a, b, c = dt**3 / 3.0, dt**2 / 2.0, dt
    return q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])


class KalmanCV:
    __slots__ = ("x", "P", "q")

    def __init__(self, x: float, y: float, pos_sigma: float, vel_sigma: float, q: float, vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([x, y, vx, vy], dtype=float)
        self.P = np.diag([pos_sigma**2, pos_sigma**2, vel_sigma**2, vel_sigma**2]).astype(float)
        self.q = float(q)

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + process_noise(dt, self.q)

    def predicted(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, P) predicted by ``dt`` without changing the filter."""
        if dt <= 0:
            return self.x.copy(), self.P.copy()
        F = transition(dt)
        return F @ self.x, F @ self.P @ F.T + process_noise(dt, self.q)

    def innovation(self, z: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
        S = _H @ self.P @ _H.T + np.eye(2) * r
        return z - self.x[:2], S

    def update(self, z: np.ndarray, r: float) -> None:
        nu, S = self.innovation(z, r)
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        I_KH = np.eye(4) - K @ _H
        # Joseph form keeps P symmetric positive definite
        self.P = I_KH @ self.P @ I_KH.T + K @ (np.eye(2) * r) @ K.T

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:]

    @property
    def pos_cov(self) -> np.ndarray:
        return self.P[:2, :2]
