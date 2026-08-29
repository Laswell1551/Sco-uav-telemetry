"""Verify source hashes embedded in the frozen v16 result artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "results" / "frozen"
SNAPSHOT = ROOT / "provenance" / "frozen_source"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((FROZEN / name).read_text(encoding="utf-8"))


def check(name: str, runner: Path, sources: dict[str, Path]) -> int:
    record = load(name)["provenance"]
    if sha256(runner) != record["runner_sha256"]:
        raise AssertionError(f"runner provenance mismatch: {runner}")
    for logical, path in sources.items():
        expected = record["source_file_sha256"][logical]
        if sha256(path) != expected:
            raise AssertionError(f"source provenance mismatch: {logical}")
    return 1 + len(sources)


def main() -> None:
    baseline = check(
        "tmc_v16_baseline_expansion.json",
        SNAPSHOT / "run_tmc_v16_baseline_expansion.py",
        {
            "run_cv_piecewise_pilot.py": ROOT / "experiments" / "run_cv_piecewise_pilot.py",
            "core/change_detection_cv.py": ROOT / "core" / "change_detection_cv.py",
            "core/online_cv_moments_stable.py": ROOT / "core" / "online_cv_moments_stable.py",
            "core/cv_sequential_calibration.py": ROOT / "core" / "cv_sequential_calibration.py",
        },
    )
    traces = check(
        "tmc_v16_trace_baseline_expansion.json",
        SNAPSHOT / "run_tmc_v16_trace_baseline_expansion.py",
        {
            "run_tmc_v16_baseline_expansion.py": SNAPSHOT / "run_tmc_v16_baseline_expansion.py",
            "run_uzh_trace_replay.py": SNAPSHOT / "run_uzh_trace_replay.py",
            "run_m3ed_trace_replay.py": SNAPSHOT / "run_m3ed_trace_replay.py",
            "core/uzh_fpv_replay_v2.py": ROOT / "core" / "uzh_fpv_replay_v2.py",
            "core/m3ed_pose.py": ROOT / "core" / "m3ed_pose.py",
        },
    )
    print(f"provenance verification passed: {baseline + traces} source links")


if __name__ == "__main__":
    main()
