"""Prepare synchronized 1-s UAV/5G traces from 6GL-CLD26_v2.

This script does not invent latency. It aligns the dataset's measured UDP loss,
RAN reliability/congestion, and UAV pose fields for tests 12--21.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

UDP_TESTS = tuple(range(12, 22))


def _nearest(rows, times, target):
    i = bisect.bisect_left(times, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
    return rows[min(candidates, key=lambda j: abs(times[j] - target))]


def _to_us(iso_time):
    return int(datetime.fromisoformat(iso_time).replace(tzinfo=timezone.utc).timestamp() * 1e6)


def _f(row, key, default=0.0):
    value = row.get(key, "")
    return default if value in (None, "") else float(value)


def align_test(root: Path, test_id: int):
    test = root / f"test{test_id}"
    receiver = next(test.glob("*receiver*.csv"))
    ran_file = next(test.glob("*ran*.csv"))
    pose_file = next((test / "drone").glob("*pose.csv"))

    with receiver.open(encoding="utf-8-sig", newline="") as f:
        app = list(csv.DictReader(f))
    with ran_file.open(encoding="utf-8-sig", newline="") as f:
        ran = list(csv.DictReader(f))
    with pose_file.open(encoding="utf-8-sig", newline="") as f:
        pose = list(csv.DictReader(f))

    ran_times = [int(float(r["Timestamp"])) for r in ran]
    pose_times = [int(float(r["abs_time_us"])) for r in pose]
    out = []
    prev_received = prev_lost = 0
    for slot, row in enumerate(app):
        timestamp_us = _to_us(row["timestamp"])
        r = _nearest(ran, ran_times, timestamp_us)
        q = _nearest(pose, pose_times, timestamp_us)
        received = int(float(row["frames_received"]))
        lost = int(float(row["packets_lost"]))
        d_received = received - prev_received
        d_lost = lost - prev_lost
        if d_received < 0 or d_lost < 0:
            raise ValueError(f"nonmonotone receiver counters in test{test_id} slot {slot}")
        prev_received, prev_lost = received, lost
        denom = d_received + d_lost
        app_success = d_received / denom if denom else 1.0
        ul_ack, ul_nack = _f(r, "UlDataHARQAcks"), _f(r, "UlDataHARQNAcks")
        dl_ack, dl_nack = _f(r, "DlDataHARQAcks"), _f(r, "DlDataHARQNAcks")
        out.append({
            "test": f"test{test_id}", "slot": slot, "timestamp_us": timestamp_us,
            "received_delta": d_received, "lost_delta": d_lost,
            "app_success_rate": app_success,
            "ul_bler": ul_nack / (ul_ack + ul_nack) if ul_ack + ul_nack else 0.0,
            "dl_bler": dl_nack / (dl_ack + dl_nack) if dl_ack + dl_nack else 0.0,
            "ul_buffer_bytes": _f(r, "UlBufferStatusReport"),
            "dl_buffer_bytes": _f(r, "DlBufferStatus"),
            "ul_pusch_snr_db": _f(r, "AvgPuschSnr"),
            "ssb_sinr_db": _f(r, "SsbSinrResult"),
            "x_m": _f(q, "x"), "y_m": _f(q, "y"), "z_m": _f(q, "z"),
            "vx_mps": _f(q, "vx"), "vy_mps": _f(q, "vy"), "vz_mps": _f(q, "vz"),
        })
    return out


def build(root: Path, output_csv: Path, summary_json: Path):
    streams = {f"test{i}": align_test(root, i) for i in UDP_TESTS}
    fields = list(next(iter(streams.values()))[0])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rows in streams.values():
            writer.writerows(rows)
    summary = {
        "dataset": "6GL-CLD26_v2",
        "source_doi": "10.5281/zenodo.21240929",
        "tests": list(streams),
        "stream_lengths": {k: len(v) for k, v in streams.items()},
        "common_horizon": min(map(len, streams.values())),
        "loss_rate_percent": {
            k: 100.0 * sum(r["lost_delta"] for r in v) /
            max(sum(r["received_delta"] + r["lost_delta"] for r in v), 1)
            for k, v in streams.items()
        },
        "latency_status": "not measured in released per-second files; do not claim measured two-way delay",
        "aligned_csv_sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/6gl_cld26_v2/extracted/6gl-cld26-v2-main"))
    parser.add_argument("--csv", type=Path, default=Path("results/tmc_6gl_aligned_trace.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/tmc_6gl_aligned_trace_summary.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.csv, args.summary), indent=2))


if __name__ == "__main__":
    main()
