"""Minimal parser and preprocessing for public UZH-FPV Leica traces."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_leica(path):
    """Parse valid Leica position replies into timestamps and XYZ positions.

    The public raw text contains metadata and command/response rows. Valid
    measurement replies have record type 3, ``%R1P`` at field 7, and the first
    XYZ triplet at fields 10:13. The response timestamp is field 6.
    """
    times = []
    positions = []
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 13 or row[0] != "3" or row[7] != "%R1P":
                continue
            try:
                stamp = datetime.strptime(row[6], "%Y-%m-%d %H:%M:%S.%f")
                xyz = tuple(float(v) for v in row[10:13])
            except (ValueError, IndexError):
                continue
            if np.all(np.isfinite(xyz)):
                times.append(stamp.timestamp())
                positions.append(xyz)
    if len(times) < 20:
        raise ValueError(f"Too few valid Leica measurements in {path}")

    t = np.asarray(times, dtype=float)
    xyz = np.asarray(positions, dtype=float)
    order = np.argsort(t, kind="stable")
    t, xyz = t[order], xyz[order]
    unique = np.concatenate(([True], np.diff(t) > 1e-6))
    t, xyz = t[unique], xyz[unique]
    return t - t[0], xyz


def active_interval(t, xyz, minimum_speed=0.20, pad_seconds=1.0):
    """Crop stationary setup/tail portions using a robust speed threshold."""
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    dt = np.diff(t)
    valid_dt = dt > 1e-6
    speed = np.zeros(t.size - 1)
    speed[valid_dt] = (
        np.linalg.norm(np.diff(xyz, axis=0)[valid_dt], axis=1) / dt[valid_dt]
    )
    # A short median suppresses isolated Leica spikes without smoothing flight
    # dynamics materially.
    width = 5
    padded = np.pad(speed, (width // 2, width // 2), mode="edge")
    smooth = np.array(
        [np.median(padded[i : i + width]) for i in range(speed.size)]
    )
    threshold = max(float(minimum_speed), 0.03 * float(np.quantile(smooth, 0.99)))
    moving = np.flatnonzero(smooth >= threshold)
    if moving.size < 10:
        raise ValueError("No sustained active-flight interval detected")
    start_time = max(float(t[moving[0]]) - pad_seconds, 0.0)
    end_time = min(float(t[moving[-1] + 1]) + pad_seconds, float(t[-1]))
    keep = (t >= start_time) & (t <= end_time)
    cropped_t = t[keep] - t[keep][0]
    cropped_xyz = xyz[keep]
    return cropped_t, cropped_xyz, threshold


def resample_normalized(t, xyz, length):
    """Interpolate a flight over normalized mission progress in [0,1]."""
    if length < 32:
        raise ValueError("length must be at least 32")
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    progress = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, int(length))
    out = np.column_stack(
        [np.interp(grid, progress, xyz[:, axis]) for axis in range(3)]
    )
    return grid, out


def deterministic_projection(name):
    """Fixed, data-independent unit vector for a sequence name."""
    # FNV-1a creates a stable seed without depending on Python's hash salt.
    seed = 2166136261
    for byte in name.encode("utf-8"):
        seed ^= byte
        seed = (seed * 16777619) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def projected_trace(name, t, xyz, length):
    """Return centered 1-D projection and metadata for replay."""
    grid, resampled = resample_normalized(t, xyz, length)
    direction = deterministic_projection(name)
    scalar = (resampled - resampled[0]) @ direction
    return grid, scalar, direction

