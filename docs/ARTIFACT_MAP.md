# Paper-to-artifact map

| Paper item | Main evidence | Renderer or generator |
| --- | --- | --- |
| Fig. 1 | Conceptual reliable/delayed information-state overview | 'figures/make_overview.py' |
| Fig. 2 | 'tmc_runtime_scaling.csv' | 'figures/make_runtime.py' |
| Fig. 3 | Conceptual packet/SCO/PA-SCO timeline | 'figures/make_timeline.py' |
| Fig. 4, Table II | confirmatory summary; v16/DTS/TS/max-age JSON | 'figures/make_controlled.py', 'scripts/build_tables.py' |
| Table III | frozen method-memory/exploration design map | 'scripts/build_tables.py' |
| Fig. 5, Table IV | UZH/M3ED replay JSON and v16 trace addendum | 'figures/make_external.py', 'scripts/build_tables.py' |
| 6GL paragraph | External-data replay code; no redistributed result | no main figure |
| Fig. 6 | channel summary and in-flight formal raw rows | 'figures/make_channel_delay.py' |
| Figs. 7-8, Table V | random-delay paired raw/summary plus He-style addendum | 'figures/make_channel_delay.py', 'scripts/build_tables.py' |
| Fig. 9 | multi-axis and CA N=1/N=4 summaries | 'figures/make_boundary_certificate.py' |
| Fig. 10 | certificate-sweep summary | 'figures/make_boundary_certificate.py' |
| Capacity--delay recovery figure | 'tmc_capacity_delay_trajectory_raw_v36.csv', summary, and audit metadata | 'experiments/run_tmc_capacity_delay_trajectories.py', 'figures/make_capacity_delay.py' |
| Tables I and VI | conceptual design and claim-boundary ledgers | manuscript-owned; no numeric source |

All numeric paths above resolve below 'results/frozen/'. The figure renderers
write PDF/SVG/PNG assets and machine-readable QA to 'figures/generated/'.

## Frozen controlled sources

- 'tmc_confirmatory_summary.csv'
- 'tmc_v16_baseline_expansion.json'
- 'tmc_ts_baseline_expansion.json'
- 'tmc_ts_paired_sco_addendum.json'
- 'tmc_external_baseline_addendum_v1.json'

## Frozen trace sources

- 'uzh_trace_replay_v1.json'
- 'm3ed_trace_replay_v1.json'
- 'tmc_v16_trace_baseline_expansion.json'

## Frozen delay sources

- 'tmc_channel_stress_summary.csv'
- 'tmc_inflight_formal_raw.csv'
- 'tmc_random_delay_formal_paired_raw.csv'
- 'tmc_random_delay_formal_paired_summary.csv'
- 'tmc_he_rm_formal_addendum_paired_summary.csv'
- 'tmc_he_rm_formal_addendum_summary.csv'
- 'tmc_capacity_delay_trajectory_raw_v36.csv'
- 'tmc_capacity_delay_trajectory_summary_v36.csv'
- 'tmc_capacity_delay_trajectory_meta_v36.json'

## Frozen boundary/calibration sources

- 'tmc_multiaxis_formal_summary.csv'
- 'tmc_ca_mismatch_formal_v2_n1_summary.csv'
- 'tmc_ca_mismatch_formal_v2_n4_summary.csv'
- 'tmc_certificate_sweep_summary.csv'
