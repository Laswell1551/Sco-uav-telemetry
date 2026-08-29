# Reproducibility guide

## Reference environment

- Python 3.12.7
- NumPy 1.26.4
- SciPy 1.13.1
- Matplotlib 3.9.2
- SymPy 1.14.0
- h5py 3.11.0
- pytest 7.4.4

Install with 'requirements/dev.txt' or 'environment.yml'.

## Fast verification

~~~bash
python scripts/reproduce.py verify-frozen
python scripts/reproduce.py smoke
python scripts/reproduce.py figures
python scripts/build_tables.py
~~~

The public smoke suite covers the SCO theory utilities, channel coupling,
pipeline state, random delay, external matched baselines, and v16 frozen
baseline/trace adaptations. It deliberately excludes superseded legacy
estimators and tests that require raw third-party traces.

## Synthetic DAG

'python scripts/reproduce.py synthetic' runs:

1. 30 formal seeds with three paired instances per seed;
2. controlled aggregation and CSV export;
3. the frozen DTS/DE-CD addendum;
4. the TS addendum and paired SCO analysis;
5. the max-age external baseline.

Outputs are isolated below 'runs/synthetic-<timestamp>/'.

The original project did not archive the 30 per-seed controlled JSON shards.
The public runner regenerates them before aggregation; the frozen CSV remains
the reference for figure-only reproduction.

## Channel and pipeline DAG

'python scripts/reproduce.py channel' runs in this order:

1. fixed channel stress;
2. PA-SCO pilot;
3. pilot extension and frozen penalty selection;
4. formal in-flight evaluation;
5. in-flight audit;
6. seed-disjoint random-delay pilot (audit metadata only);
7. random-delay formal evaluation;
8. He-style matched adaptation;
9. random-delay audit.

These scripts are compute intensive. Do not interrupt a formal run after the
pilot selection and then reuse a partial 'results/' folder as evidence.

## Crossed capacity--delay trajectories

`python scripts/reproduce.py capacity-delay` runs the twelve-cell grid used by
the 3x4 recovery figure. It retains 40 observed post-change checkpoints for
each of four methods in every cell. The gate compares all previously scanned
endpoints with `tmc_inflight_formal_raw.csv` and checks the exact SCO/PA-SCO
trajectory identity at zero delay. Frozen rows can be audited without a full
simulation using:

~~~bash
python experiments/run_tmc_capacity_delay_trajectories.py --verify-frozen
~~~

## Boundary/calibration DAG

'python scripts/reproduce.py boundary' runs:

1. multi-axis experiment;
2. CA mismatch at N=4 and N=1;
3. certificate sweep;
4. boundary-scoped round-two audit.

## External traces

Run the UZH-FPV and M3ED base replays after preparing 'data/' as documented.
Then run 'run_tmc_v16_trace_baseline_expansion.py' using the newly generated
controlled addendum and the same official pose inputs.

The trace expansion accepts explicit '--uzh-data-root' and
'--m3ed-data-root' arguments; no original workstation path is required.

The trace runners now accept explicit '--data-root' and '--output' arguments;
they no longer require shell redirection.

## Runtime

~~~bash
python experiments/run_tmc_runtime_scaling.py --out-dir runs/runtime
~~~

Runtime is a machine-specific systems microbenchmark. Compare method ordering
and scaling, not exact millisecond equality with the frozen reference.

## Statistical integrity

- Formal seed sets are hard-coded and disjoint from pilot/tuning seed sets.
- Paired comparisons reuse the same latent problem, channel uniforms, and
  delay banks within each seed.
- Seed clusters, not within-seed rows, are the inferential unit.
- Figure QA compares plotted values and geometry; binary PDF/SVG hashes are not
  treated as cross-platform invariants.
