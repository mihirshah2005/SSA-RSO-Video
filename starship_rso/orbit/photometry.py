"""Order-of-magnitude visibility model for an object seen from the ship camera.

Assumptions (all configurable, all stated in docs/FEASIBILITY.md):

* diffuse-sphere phase law;
* "standard magnitude" = apparent magnitude at 1000 km range and 90 deg
  phase. Typical values: large LEO broadband satellite ~4-5, 3U CubeSat ~8-9;
* a camera exposed for a sunlit Earth reaches only about magnitude 0 against
  cloud and maybe +2 against black sky (no stars are visible in the footage).

These numbers decide whether an object *could* appear; they never identify one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def diffuse_sphere_phase(phase_rad) -> np.ndarray:
    p = np.asarray(phase_rad, dtype=np.float64)
    return (np.sin(p) + (np.pi - p) * np.cos(p)) / np.pi


def apparent_magnitude(range_km, phase_rad, std_mag: float) -> np.ndarray:
    f = np.maximum(diffuse_sphere_phase(phase_rad), 1e-6)
    f90 = diffuse_sphere_phase(np.pi / 2)
    return std_mag - 2.5 * np.log10(f / f90) + 5.0 * np.log10(np.asarray(range_km, dtype=np.float64) / 1000.0)


def angular_size_px(size_m: float, range_km, ifov_rad: float) -> np.ndarray:
    return (size_m / 1000.0) / np.asarray(range_km, dtype=np.float64) / ifov_rad


@dataclass
class VisibilityModel:
    std_mag: float = 4.5
    size_m: float = 10.0
    mag_limit_earth: float = 0.0
    mag_limit_sky: float = 2.0
    min_resolved_px: float = 1.5

    def assess(self, range_km, phase_rad, sunlit, ifov_rad: float, earth_background) -> dict[str, np.ndarray]:
        mag = apparent_magnitude(range_km, phase_rad, self.std_mag)
        px = angular_size_px(self.size_m, range_km, ifov_rad)
        lim = np.where(earth_background, self.mag_limit_earth, self.mag_limit_sky)
        bright_enough = np.asarray(sunlit) & (mag <= lim)
        resolved = np.asarray(sunlit) & (px >= self.min_resolved_px)
        return {"mag": mag, "px": px, "visible": bright_enough | resolved}


def feasibility_table(
    ranges_km=(1, 10, 100, 1000),
    sizes_m=(0.34, 3.0, 30.0),
    hfov_deg: float = 90.0,
    width_px: int = 1920,
    std_mag_550km: float = 5.0,
) -> list[dict]:
    """Reproduce the order-of-magnitude visibility table used in the docs."""
    ifov = np.deg2rad(hfov_deg) / width_px
    rows = []
    for r in ranges_km:
        row = {"range_km": r}
        for s in sizes_m:
            row[f"px_{s}m"] = float(angular_size_px(s, r, ifov))
        row["mag_large_sat"] = float(std_mag_550km + 5 * np.log10(r / 550.0))
        rows.append(row)
    return rows
