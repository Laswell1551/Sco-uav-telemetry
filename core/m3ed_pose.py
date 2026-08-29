"""Pose-only loader for public M3ED Falcon UAV ground truth."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_pose_file(path):
    """Load camera position in the initial-camera coordinate frame.

    M3ED stores ``Cn_T_C0``. Inverting it yields ``C0_T_Cn``; its translation
    is the current camera origin expressed in the initial-camera frame.
    Timestamps are integer microseconds in the released pose files.
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        transform = np.asarray(handle["Cn_T_C0"], dtype=float)
        timestamp = np.asarray(handle["ts"], dtype=np.int64)
    if transform.ndim != 3 or transform.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected transform shape in {path}")
    if timestamp.shape != (transform.shape[0],):
        raise ValueError(f"Timestamp/pose count mismatch in {path}")
    if not np.all(np.diff(timestamp) > 0):
        raise ValueError(f"Non-increasing timestamps in {path}")
    inverse = np.linalg.inv(transform)
    xyz = inverse[:, :3, 3]
    t = (timestamp - timestamp[0]).astype(float) / 1e6
    return t, xyz


def load_all_falcon_pose(root):
    root = Path(root)
    traces = []
    for path in sorted(root.glob("*_pose_gt.h5")):
        t, xyz = load_pose_file(path)
        name = path.name.removesuffix("_pose_gt.h5")
        traces.append({"name": name, "t": t, "xyz": xyz})
    if not traces:
        raise ValueError(f"No M3ED pose files found under {root}")
    return traces

