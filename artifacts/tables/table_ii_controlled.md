# Table II - Controlled drift

| Method | Full-horizon excess cost (%) | Post-change excess cost (%) | Evidence |
| --- | --- | --- | --- |
| Cumulative CE | 6.64 [6.16, 7.13] | 10.96 [10.09, 11.86] | primary |
| Cumulative UCB-CV | 7.03 [6.71, 7.34] | 8.87 [8.35, 9.39] | primary |
| SW-CE (32) | 4.70 [4.36, 5.04] | 5.96 [5.48, 6.46] | primary |
| SW-Whittle-CV (64) | 5.97 [5.62, 6.32] | 6.38 [5.96, 6.79] | primary |
| DTS-Whittle-CV | 5.20 [4.98, 5.40] | 5.91 [5.62, 6.21] | retrospective matched adaptation |
| TS-Whittle-CV | 6.80 [6.42, 7.18] | 10.78 [10.05, 11.55] | retrospective matched adaptation |
| DE-CD-Whittle-CV | 15.36 [12.99, 18.02] | 25.31 [20.49, 30.82] | retrospective matched adaptation |
| Max-Age | 29.50 [28.13, 30.89] | 24.32 [22.96, 25.67] | retrospective low-information comparator |
| Forced-reset-UCB | 6.62 [6.29, 6.93] | 5.62 [5.26, 5.99] | primary |
| SCO-reset-CE | **3.29 [2.95, 3.63]** | **3.69 [3.26, 4.15]** | primary |
| SCO-reset-UCB | 4.75 [4.43, 5.07] | 3.98 [3.64, 4.33] | primary |

Sources: results/frozen/tmc_confirmatory_summary.csv, results/frozen/tmc_v16_baseline_expansion.json, results/frozen/tmc_ts_baseline_expansion.json, results/frozen/tmc_external_baseline_addendum_v1.json.
