# Table IV - External trace replay

| Method | UZH-FPV excess cost (%) | M3ED Falcon excess cost (%) | Evidence |
| --- | --- | --- | --- |
| Cumulative CE | 17.53 [16.38, 18.72] | 9.63 [9.24, 10.03] | primary replay |
| Cumulative UCB-CV | 17.31 [16.25, 18.40] | 8.45 [8.06, 8.84] | primary replay |
| SW-CE (32) | 17.61 [16.59, 18.67] | 9.16 [8.75, 9.60] | primary replay |
| SW-Whittle-CV (64) | 17.32 [16.24, 18.46] | 8.58 [8.22, 8.92] | primary replay |
| DTS-Whittle-CV | 16.84 [15.81, 17.90] | 8.18 [7.87, 8.50] | retrospective matched adaptation |
| DE-CD-Whittle-CV | 80.65 [73.89, 87.92] | 53.21 [48.20, 58.91] | retrospective matched adaptation |
| AoI / round robin | 40.62 [38.07, 43.29] | 33.10 [31.17, 35.02] | primary replay |
| Forced-reset-UCB | 18.40 [17.30, 19.54] | 8.89 [8.56, 9.22] | primary replay |
| SCO-reset-CE | 17.88 [16.57, 19.26] | 8.40 [8.03, 8.76] | primary replay |
| SCO-reset-UCB | **16.74 [15.50, 17.96]** | **7.14 [6.82, 7.45]** | primary replay |

Sources: results/frozen/uzh_trace_replay_v1.json, results/frozen/m3ed_trace_replay_v1.json, results/frozen/tmc_v16_trace_baseline_expansion.json.
