# SCO: Self-Exploring Whittle Control for UAV Telemetry

This repository is the release-candidate artifact for:

> SCO: Self-Exploring Whittle Control for UAV Telemetry with Model Drift and
> Two-Way Delay

It contains the SCO and PA-SCO research code, frozen paper-facing numerical
artifacts, figure renderers, targeted regression tests, and the experiment
protocols needed to reproduce the TMC results. The manuscript and IEEE
template are intentionally excluded from this software repository.

## What is reproducible now

| Layer | Status | Command |
| --- | --- | --- |
| Frozen-result integrity | Ready | `python scripts/reproduce.py verify-frozen` |
| Targeted SCO regression suite | Ready | `python scripts/reproduce.py smoke` |
| Paper figures from frozen results | Ready | `python scripts/reproduce.py figures` |
| Paper-facing tables | Ready | `python scripts/build_tables.py` |
| Synthetic experiments from scratch | Runner included; compute intensive | `python scripts/reproduce.py synthetic` |
| Fixed/random-delay experiments | Runner included; compute intensive | `python scripts/reproduce.py channel` |
| Crossed capacity--delay trajectories | Frozen 3x4 grid ready; full runner is compute intensive | `python scripts/reproduce.py capacity-delay` |
| Spatial/CA/certificate experiments | Runner included; compute intensive | `python scripts/reproduce.py boundary` |
| UZH-FPV and M3ED replays | Requires external data | See `data/README.md` |
| 6GL-CLD26_v2 replay | Code only; upstream redistribution license unresolved | See `data/README.md` |

The frozen-result path is deliberately read-only. Fresh runs are written below
`runs/`, and generated figures are written below `figures/generated/`.

## Quick start

Python 3.12 is the reference environment. Python 3.10 and 3.12 are exercised
by CI for the targeted smoke suite.

~~~bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt
python scripts/reproduce.py verify
~~~

Render the release-mapped paper figures and the paper-facing tables:

~~~bash
python scripts/reproduce.py figures
python scripts/build_tables.py
~~~

On Windows, activate the environment with
`.venv\\Scripts\\Activate.ps1`; on Linux/macOS use
`source .venv/bin/activate`.

## Repository layout

~~~text
core/                 SCO estimators, detectors, Whittle utilities, loaders
experiments/          controlled, trace, delay, boundary, and runtime runners
tables/               versioned method-design and claim-boundary manifests
tests/                targeted public regression suite
results/frozen/       paper-facing CSV/JSON artifacts plus SHA-256 manifest
figures/              publication renderers and generated PDF/SVG/PNG outputs
data/                 external-data policy and local-layout manifest
scripts/              verification, orchestration, manifest, and table tools
provenance/           exact source snapshots referenced by frozen JSON hashes
docs/                 artifact map, full DAG, limitations, release gate
LICENSES/             per-file terms for data-derived result sections
.github/workflows/    lightweight public CI
~~~

## Reproduction tiers

1. `verify` checks frozen hashes, parses result files, runs the targeted tests,
   and renders the current figures.
2. `synthetic`, `channel`, and `boundary` create isolated timestamped run
   directories. They never overwrite `results/frozen/`.
3. Trace replay requires the official UZH-FPV and M3ED inputs in the exact
   layout described by `data/README.md`.
4. Runtime numbers are hardware-specific. Reproduce their trend, not exact
   equality with the frozen machine.

See `docs/REPRODUCIBILITY.md` for the full dependency order and
`docs/ARTIFACT_MAP.md` for the figure/table-to-file map.

## Evidence boundary

- SCO, PA-SCO, confidence intervals, seeds, and frozen result values are
  preserved from the audited artifact lineage. The rc3 crossed
  capacity--delay trajectories add 40 observed checkpoints per curve and are
  checked against every overlapping formal endpoint.
- DTS-Whittle-CV, DE-CD-Whittle-CV, RM-ACK, and other source-inspired rows are
  matched adaptations under the paper's selected-only observation model, not
  official reproductions of the source algorithms.
- Third-party raw datasets are not distributed from this repository.
- No result should be interpreted as a field deployment or universal
  delay-optimality claim.

## License and citation

Original repository code is released under the MIT License. Third-party
datasets are governed by their own terms and are not covered by MIT; see
`NOTICE.md`, `THIRD_PARTY.md`, `LICENSES/DATA-RESULTS.md`, and
`data/README.md`.

Citation metadata lists Jiaqi Lin, Shi Yan, and Mugen Peng and records the
official source repository. The archival DOI will be added to `CITATION.cff`
after a versioned release has been deposited.

## Release status

This directory is the identifiable release candidate for
<https://github.com/Laswell1551/Sco-uav-telemetry>. Release tagging and DOI
registration remain explicit actions in `docs/RELEASE_CHECKLIST.md`.
