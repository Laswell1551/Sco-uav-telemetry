"""Frozen metadata-driven cropping for the UZH-FPV replay protocol."""
from __future__ import annotations

import numpy as np


# Values reported by the official UZH-FPV dataset table.
OFFICIAL = {
    "indoor_forward_3": (54.63, 287.12, 9.50),
    "indoor_forward_5": (50.00, 156.47, 4.87),
    "indoor_forward_6": (32.93, 223.27, 12.52),
    "indoor_forward_7": (73.20, 333.59, 12.78),
    "indoor_forward_9": (34.04, 157.07, 11.42),
    "indoor_forward_10": (33.43, 149.36, 9.49),
    "indoor_45_2": (55.77, 218.90, 6.97),
    "indoor_45_4": (47.36, 168.06, 6.55),
    "indoor_45_9": (40.00, 215.58, 11.23),
    "indoor_45_12": (51.25, 124.56, 4.33),
    "indoor_45_13": (42.49, 166.62, 7.92),
    "indoor_45_14": (43.66, 220.40, 9.54),
    "outdoor_forward_1": (49.63, 258.23, 8.55),
    "outdoor_forward_3": (92.84, 735.51, 14.04),
    "outdoor_forward_5": (22.21, 189.63, 20.73),
    "outdoor_45_1": (24.49, 165.53, 15.62),
}


def crop_official_motion_window(name, t, xyz, rate_hz=20.0):
    """Select an official-duration window and resample it uniformly.

    The raw Leica logs include setup and tail periods. Among windows with the
    duration reported by the official dataset page, we select the one with
    maximum robust traveled distance. Segment contribution is capped at
    35 m/s times its duration so sub-millisecond timestamp jitter or isolated
    tracker glitches cannot determine the crop.
    """
    if name not in OFFICIAL:
        raise KeyError(f"No frozen official metadata for {name}")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    duration = OFFICIAL[name][0]
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    dt = np.diff(t)
    displacement = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    robust_distance = np.minimum(displacement, 35.0 * np.maximum(dt, 0.0))
    cumulative = np.concatenate(([0.0], np.cumsum(robust_distance)))

    best_score = -np.inf
    best_start = 0
    for start in range(t.size):
        end = int(np.searchsorted(t, t[start] + duration, side="right") - 1)
        if end <= start or t[end] - t[start] < 0.98 * duration:
            continue
        score = cumulative[end] - cumulative[start]
        if score > best_score:
            best_score = score
            best_start = start
    if not np.isfinite(best_score):
        raise ValueError(f"Trace {name} is shorter than its official duration")

    start_time = t[best_start]
    grid = start_time + np.arange(
        int(np.floor(duration * rate_hz)) + 1, dtype=float
    ) / rate_hz
    grid = grid[grid <= start_time + duration + 1e-9]
    out = np.column_stack(
        [np.interp(grid, t, xyz[:, axis]) for axis in range(3)]
    )
    return grid - grid[0], out


def robust_speed_summary(t, xyz):
    dt = np.diff(t)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
    return {
        "duration": float(t[-1] - t[0]),
        "path_length": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
        "speed_p50": float(np.quantile(speed, 0.50)),
        "speed_p95": float(np.quantile(speed, 0.95)),
        "speed_max": float(np.max(speed)),
    }

