# Data-derived result licensing and attribution

The root MIT license covers original SCO software and documentation only.
The following machine-readable statistics were computed from third-party
datasets and are conservatively distributed under the corresponding upstream
terms to the extent copyright or database rights attach to those statistics.

## UZH-FPV-derived section

- File: `results/frozen/uzh_trace_replay_v1.json`
- File section: `results/frozen/tmc_v16_trace_baseline_expansion.json` at
  `datasets.uzh_fpv`
- Source: UZH-FPV Drone Racing Dataset, https://fpv.ifi.uzh.ch/
- Citation: J. Delmerico et al., Are We Ready for Autonomous Drone Racing?
  The UZH-FPV Drone Racing Dataset, ICRA 2019.
- Upstream terms: CC BY-NC-SA 3.0,
  https://creativecommons.org/licenses/by-nc-sa/3.0/
- Modification notice: source trajectories were filtered, segmented, mapped
  to physical moment sequences, replayed through the included scheduling
  simulator, and aggregated into summary statistics and confidence intervals.

## M3ED-derived section

- File: `results/frozen/m3ed_trace_replay_v1.json`
- File section: `results/frozen/tmc_v16_trace_baseline_expansion.json` at
  `datasets.m3ed_falcon`
- Source: M3ED, https://m3ed.io/
- Citation: K. Chaney et al., M3ED: Multi-Robot, Multi-Sensor, Multi-
  Environment Event Dataset, CVPR Workshops 2023.
- Upstream terms: CC BY-SA 4.0,
  https://creativecommons.org/licenses/by-sa/4.0/
- Modification notice: Falcon pose streams were filtered, segmented, mapped
  to physical moment sequences, replayed through the included scheduling
  simulator, and aggregated into summary statistics and confidence intervals.

The combined v16 trace JSON is a collection with separately identified UZH
and M3ED sections; each section retains the terms above. No raw pose or image
data are included. The 6GL-derived inputs and outputs are not redistributed.
