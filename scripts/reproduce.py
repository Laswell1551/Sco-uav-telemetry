"""Single entry point for SCO artifact verification and reproduction."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
FROZEN = ROOT / "results" / "frozen"

SMOKE_TESTS = (
    "tests/test_dare_monotonicity.py",
    "tests/test_topn_loss.py",
    "tests/test_tmc_channel_stress.py",
    "tests/test_tmc_inflight.py",
    "tests/test_tmc_random_delay.py",
    "tests/test_tmc_external_baselines.py",
    "tests/test_tmc_ts_baseline_expansion.py",
    "tests/test_tmc_v16_baseline_expansion.py",
    "tests/test_tmc_v16_trace_baseline_expansion.py",
)

FIGURE_SCRIPTS = (
    "figures/make_overview.py",
    "figures/make_runtime.py",
    "figures/make_timeline.py",
    "figures/make_controlled.py",
    "figures/make_external.py",
    "figures/make_channel_delay.py",
    "figures/make_boundary_certificate.py",
    "figures/make_capacity_delay.py",
)

EXPECTED_FIGURES = (
    "fig_overview_sco_pa_v21.pdf",
    "fig_runtime_scaling.pdf",
    "fig_pipeline_timeline_v21.pdf",
    "fig_controlled_grouped_v21.pdf",
    "fig_external_replay_compact.pdf",
    "fig_channel_pipeline_evidence_v21.pdf",
    "fig_random_delay_profiles_v21.pdf",
    "fig_random_delay_performance_mechanism_v21.pdf",
    "fig_model_boundary_compact.pdf",
    "fig_certificate_operating_compact.pdf",
    "fig_capacity_delay_trajectory_4x3_v36.pdf",
)


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT), str(EXPERIMENTS)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def run(argv: list[str | Path], *, cwd: Path = ROOT) -> None:
    command = [str(item) for item in argv]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=command_env(), check=True)


def python_file(relative: str, *args: str | Path, cwd: Path = ROOT) -> None:
    run([sys.executable, "-B", ROOT / relative, *args], cwd=cwd)


def new_run_dir(kind: str, output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    path = output_root.resolve() / f"{kind}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    print(f"run directory: {path}", flush=True)
    return path


def verify_frozen() -> None:
    python_file("scripts/verify_frozen.py")
    python_file("scripts/verify_provenance.py")
    python_file(
        "experiments/run_tmc_capacity_delay_trajectories.py",
        "--verify-frozen",
    )


def smoke() -> None:
    run([sys.executable, "-B", "-m", "pytest", "-q", *SMOKE_TESTS])


def figures() -> None:
    for script in FIGURE_SCRIPTS:
        python_file(script)
    output = ROOT / "figures" / "generated"
    missing = [name for name in EXPECTED_FIGURES if not (output / name).is_file()]
    if missing:
        raise SystemExit("missing expected figure PDFs: " + ", ".join(missing))
    bad = []
    for name in EXPECTED_FIGURES:
        path = output / name
        if path.stat().st_size < 1024 or path.read_bytes()[:4] != b"%PDF":
            bad.append(name)
    if bad:
        raise SystemExit("invalid figure PDFs: " + ", ".join(bad))
    print(f"validated {len(EXPECTED_FIGURES)} figure PDFs in {output}")


def build_tables() -> None:
    python_file("scripts/build_tables.py")


def synthetic(output_root: Path) -> None:
    run_dir = new_run_dir("synthetic", output_root)
    results = run_dir / "results"
    results.mkdir()
    controlled = results / "controlled-310001-310030.json"
    python_file(
        "experiments/run_cv_confirmatory_range.py",
        "--seed-start", "310001", "--seed-end", "310030",
        "--batches", "3", "--json-out", controlled,
    )
    python_file(
        "experiments/aggregate_cv_confirmatory_final.py",
        "--glob", results / "controlled-*.json",
        "--summary-out", results / "tmc_confirmatory_summary.csv",
    )
    v16 = results / "tmc_v16_baseline_expansion.json"
    ts = results / "tmc_ts_baseline_expansion.json"
    python_file(
        "experiments/run_tmc_v16_baseline_expansion.py",
        "--json-out", v16, "--bootstrap-replicates", "100000",
    )
    python_file("experiments/run_tmc_ts_baseline_expansion.py", "--output", ts)
    python_file(
        "experiments/analyze_tmc_ts_paired.py",
        "--ts-json", ts,
        "--output", results / "tmc_ts_paired_sco_addendum.json",
    )
    python_file(
        "experiments/run_tmc_external_baseline_addendum.py",
        "--seed-start", "310001", "--seed-end", "310030", "--batches", "3",
        "--json-out", results / "tmc_external_baseline_addendum_v1.json",
    )
    print(f"synthetic reproduction complete: {run_dir}")


def channel(output_root: Path) -> None:
    run_dir = new_run_dir("channel", output_root)
    (run_dir / "results").mkdir()
    python_file("experiments/run_tmc_channel_stress.py", "--out-dir", "results", cwd=run_dir)
    python_file("experiments/run_tmc_inflight_pilot.py", "--out-dir", "results", cwd=run_dir)
    python_file("experiments/run_tmc_inflight_pilot_extend.py", cwd=run_dir)
    python_file("experiments/run_tmc_inflight_formal.py", cwd=run_dir)
    python_file("experiments/audit_tmc_inflight_formal.py", cwd=run_dir)
    python_file("experiments/run_tmc_random_delay_eval.py", "--quick", cwd=run_dir)
    python_file("experiments/run_tmc_random_delay_eval.py", cwd=run_dir)
    python_file("experiments/run_tmc_he_baseline_formal_addendum.py", cwd=run_dir)
    python_file("experiments/audit_tmc_random_delay_formal.py", cwd=run_dir)
    print(f"channel reproduction complete: {run_dir}")


def boundary(output_root: Path) -> None:
    run_dir = new_run_dir("boundary", output_root)
    (run_dir / "results").mkdir()
    python_file("experiments/run_tmc_multiaxis_pilot.py", "--out-dir", "results", cwd=run_dir)
    for capacity in (4, 1):
        python_file(
            "experiments/run_tmc_ca_mismatch_pilot.py",
            "--capacity", str(capacity), "--out-dir", "results", cwd=run_dir,
        )
    python_file("experiments/run_tmc_certificate_sweep.py", "--out-dir", "results", cwd=run_dir)
    python_file("experiments/audit_tmc_round2_results.py", "--scope", "boundary", cwd=run_dir)
    print(f"boundary reproduction complete: {run_dir}")


def traces(output_root: Path, data_root: Path, controlled_json: Path) -> None:
    run_dir = new_run_dir("traces", output_root)
    results = run_dir / "results"
    results.mkdir()
    uzh_root = data_root.resolve() / "uzh_fpv_gt"
    m3ed_root = data_root.resolve() / "m3ed_falcon_pose"
    python_file(
        "experiments/run_uzh_trace_replay.py", "--data-root", uzh_root,
        "--output", results / "uzh_trace_replay_v1.json",
    )
    python_file(
        "experiments/run_m3ed_trace_replay.py", "--data-root", m3ed_root,
        "--output", results / "m3ed_trace_replay_v1.json",
    )
    python_file(
        "experiments/aggregate_uzh_trace_replay.py",
        "--input", results / "uzh_trace_replay_v1.json",
    )
    python_file(
        "experiments/aggregate_m3ed_trace_replay.py",
        "--input", results / "m3ed_trace_replay_v1.json",
    )
    python_file(
        "experiments/run_tmc_v16_trace_baseline_expansion.py",
        "--controlled-json", controlled_json.resolve(),
        "--json-out", results / "tmc_v16_trace_baseline_expansion.json",
        "--uzh-data-root", uzh_root, "--m3ed-data-root", m3ed_root,
        "--bootstrap-replicates", "20000",
    )
    print(f"trace reproduction complete: {run_dir}")


def sixgl(output_root: Path, data_root: Path) -> None:
    run_dir = new_run_dir("sixgl", output_root)
    results = run_dir / "results"
    results.mkdir()
    python_file(
        "experiments/prepare_tmc_6gl_trace.py", "--root", data_root.resolve(),
        "--csv", results / "tmc_6gl_aligned_trace.csv",
        "--summary", results / "tmc_6gl_aligned_trace_summary.json",
    )
    python_file("experiments/run_tmc_6gl_loss_replay.py", cwd=run_dir)
    python_file("experiments/audit_tmc_6gl_loss_replay.py", cwd=run_dir)
    print(f"6GL descriptive replay complete: {run_dir}")


def runtime(output_root: Path) -> None:
    run_dir = new_run_dir("runtime", output_root)
    results = run_dir / "results"
    results.mkdir()
    python_file("experiments/run_tmc_runtime_scaling.py", "--out-dir", results)
    print(f"runtime benchmark complete: {run_dir}")


def capacity_delay(output_root: Path) -> None:
    run_dir = new_run_dir("capacity-delay", output_root)
    results = run_dir / "results"
    results.mkdir()
    python_file(
        "experiments/run_tmc_capacity_delay_trajectories.py",
        "--out-dir", results,
    )
    print(f"capacity--delay reproduction complete: {run_dir}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=(
            "verify-frozen", "smoke", "figures", "tables", "verify",
            "synthetic", "channel", "boundary", "traces", "sixgl", "runtime",
            "capacity-delay",
        ),
    )
    value.add_argument("--output-root", type=Path, default=ROOT / "runs")
    value.add_argument("--data-root", type=Path, default=ROOT / "data")
    value.add_argument(
        "--controlled-json", type=Path,
        default=FROZEN / "tmc_v16_baseline_expansion.json",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    actions = {
        "verify-frozen": verify_frozen,
        "smoke": smoke,
        "figures": figures,
        "tables": build_tables,
    }
    if args.command in actions:
        actions[args.command]()
    elif args.command == "verify":
        verify_frozen()
        smoke()
        figures()
        build_tables()
    elif args.command == "synthetic":
        synthetic(args.output_root)
    elif args.command == "channel":
        channel(args.output_root)
    elif args.command == "boundary":
        boundary(args.output_root)
    elif args.command == "traces":
        traces(args.output_root, args.data_root, args.controlled_json)
    elif args.command == "sixgl":
        sixgl(args.output_root, args.data_root)
    elif args.command == "runtime":
        runtime(args.output_root)
    elif args.command == "capacity-delay":
        capacity_delay(args.output_root)


if __name__ == "__main__":
    main()
