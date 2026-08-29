"""Stable final UZH-FPV trace preparation for scheduling replay.

This supersedes the first simultaneous despiking prototype in
``uzh_fpv_replay.py``. Replacing the single largest curvature residual at a
time prevents a one-sample glitch from falsely marking both neighbours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .uzh_fpv import deterministic_projection, parse_leica
from .uzh_fpv_protocol import OFFICIAL, crop_official_motion_window


def despike_positions(xyz, absolute_threshold=0.15, multiplier=10.0, max_steps=500):
    clean = np.asarray(xyz, dtype=float).copy()
    for _ in range(int(max_steps)):
        midpoint = 0.5 * (clean[:-2] + clean[2:])
        residual = np.linalg.norm(clean[1:-1] - midpoint, axis=1)
        positive = residual[residual > 0]
        scale = float(np.median(positive)) if positive.size else 0.0
        threshold = max(float(absolute_threshold), float(multiplier) * scale)
        local = int(np.argmax(residual))
        if residual[local] <= threshold:
            break
        index = local + 1
        clean[index] = midpoint[local]
    return clean


def load_replay_trace(root, name, rate_hz=20.0):
    root = Path(root)
    t, xyz = parse_leica(root / name / "leica.txt")
    t, xyz = crop_official_motion_window(name, t, xyz, rate_hz=rate_hz)
    clean = despike_positions(xyz)
    direction = deterministic_projection(name)
    scalar = (clean - clean[0]) @ direction
    return {
        "name": name,
        "t": t,
        "xyz": clean,
        "position": scalar,
        "projection": direction,
        "official": OFFICIAL[name],
    }


def load_all_replay_traces(root, rate_hz=20.0):
    return [
        load_replay_trace(root, name, rate_hz=rate_hz)
        for name in sorted(OFFICIAL)
    ]

