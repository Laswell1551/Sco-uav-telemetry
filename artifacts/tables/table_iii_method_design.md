# Table III - Matched method design

| Method | Matched design | Evidence |
| --- | --- | --- |
| Cumulative CE | All selected observations; point estimate; no forced exploration. | primary comparator |
| Cumulative UCB-CV | All selected observations; optimistic physical-moment estimate; no forced exploration. | primary comparator |
| SW-CE (32) | Last 32 selected observations; point estimate; no forced exploration. | primary comparator |
| SW-Whittle-CV (64) | Last 64 selected observations; optimistic estimate; no forced exploration. | primary comparator |
| DTS-Whittle-CV | Discounted selected-observation posterior; Thompson draw; gamma=0.99. | retrospective matched adaptation |
| TS-Whittle-CV | Episodic selected-observation posterior; Thompson draw; episode length=1. | retrospective matched adaptation |
| DE-CD-Whittle-CV | Detector reset plus explicit exploration blocks; alpha=0.5. | retrospective matched adaptation |
| Max-Age | Public age state only; no model estimation or explicit probes. | retrospective low-information comparator |
| Forced-reset-UCB | Reset-UCB learner plus a forced exploration epoch every 50 slots. | primary comparator |
| SCO-reset-CE | Since-reset selected observations; point estimate; scheduling-induced exploration only. | proposed |
| SCO-reset-UCB | Since-reset selected observations; confidence-aware index; scheduling-induced exploration only. | proposed |

Sources: tables/method_design.json, results/frozen/tmc_v16_baseline_expansion.json, results/frozen/tmc_ts_baseline_expansion.json.
