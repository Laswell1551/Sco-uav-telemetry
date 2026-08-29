"""Fail-closed integrity, parseability, privacy, and size checks."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "results" / "frozen"
MANIFEST = FROZEN / "MANIFEST.sha256"
MAX_BYTES = 95 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".zip", ".pkl", ".pth", ".pt", ".pyc", ".ulg", ".xls", ".fls",
    ".fdb_latexmk",
}
REQUIRED = {
    "m3ed_trace_replay_v1.json",
    "uzh_trace_replay_v1.json",
    "tmc_confirmatory_summary.csv",
    "tmc_v16_baseline_expansion.json",
    "tmc_ts_baseline_expansion.json",
    "tmc_external_baseline_addendum_v1.json",
    "tmc_v16_trace_baseline_expansion.json",
    "tmc_channel_stress_summary.csv",
    "tmc_inflight_formal_summary.csv",
    "tmc_random_delay_formal_paired_summary.csv",
    "tmc_he_rm_formal_addendum_paired_summary.csv",
    "tmc_multiaxis_formal_summary.csv",
    "tmc_ca_mismatch_formal_v2_n1_summary.csv",
    "tmc_ca_mismatch_formal_v2_n4_summary.csv",
    "tmc_certificate_sweep_summary.csv",
    "tmc_runtime_scaling.csv",
    "tmc_capacity_delay_trajectory_raw_v36.csv",
    "tmc_capacity_delay_trajectory_summary_v36.csv",
    "tmc_capacity_delay_trajectory_meta_v36.json",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".cff", ".json", ".csv"
}
PRIVATE_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
    "workspace path": re.compile(
        "Baidu" + r"Syncdisk|AppData[\\/]Local[\\/]Temp", re.I
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{24,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{32,}"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest() -> dict[str, str]:
    if not MANIFEST.is_file():
        raise AssertionError("missing results/frozen/MANIFEST.sha256")
    entries: dict[str, str] = {}
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssertionError(f"invalid manifest line {number}")
        if relative in entries:
            raise AssertionError(f"duplicate manifest path: {relative}")
        entries[relative] = digest
    return entries


def verify_manifest() -> int:
    entries = parse_manifest()
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in FROZEN.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if actual != set(entries):
        unlisted = sorted(actual - set(entries))
        missing = sorted(set(entries) - actual)
        raise AssertionError(
            f"manifest set mismatch; unlisted={unlisted}, missing={missing}"
        )
    frozen_resolved = FROZEN.resolve()
    for relative, expected in entries.items():
        path = (ROOT / relative).resolve()
        if frozen_resolved not in path.parents:
            raise AssertionError(f"manifest path escapes frozen directory: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"hash mismatch: {relative}")
    return len(entries)


def verify_parseability() -> tuple[int, int]:
    json_count = 0
    csv_count = 0
    for path in sorted(FROZEN.iterdir()):
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8-sig") as handle:
                json.load(handle)
            json_count += 1
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header or len(header) != len(set(header)):
                    raise AssertionError(f"invalid or duplicate CSV header: {path.name}")
                if next(reader, None) is None:
                    raise AssertionError(f"CSV has no data rows: {path.name}")
            csv_count += 1
    return json_count, csv_count


def verify_release_tree() -> int:
    findings: list[str] = []
    count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        count += 1
        if path.stat().st_size > MAX_BYTES:
            findings.append(f"oversize file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden file type: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    if findings:
        raise AssertionError("release-tree audit failed:\n- " + "\n- ".join(findings))
    return count


def main() -> None:
    missing = sorted(name for name in REQUIRED if not (FROZEN / name).is_file())
    if missing:
        raise AssertionError("missing required frozen artifacts: " + ", ".join(missing))
    entries = verify_manifest()
    json_count, csv_count = verify_parseability()
    files = verify_release_tree()
    print(
        "frozen verification passed: "
        f"{entries} hashes, {json_count} JSON, {csv_count} CSV, {files} release files"
    )


if __name__ == "__main__":
    main()
