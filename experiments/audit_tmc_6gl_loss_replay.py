"""Audit the exploratory 6GL measured-loss replay outputs."""
import csv
import json
import math
from pathlib import Path

R=Path("results")
raw=list(csv.DictReader((R/"tmc_6gl_loss_replay_raw.csv").open(encoding="utf-8")))
summary=list(csv.DictReader((R/"tmc_6gl_loss_replay_summary.csv").open(encoding="utf-8")))
meta=json.loads((R/"tmc_6gl_loss_replay_meta.json").read_text(encoding="utf-8"))
assert len(raw)==80, len(raw)
assert len(summary)==8, len(summary)
assert meta["status"]=="exploratory_not_paper_facing"
assert meta["H"]==91 and meta["K"]==10 and meta["N"]==2
for row in raw:
    for key in ("post_cost","total_cost","delivery_rate","redundant_attempt_rate","stale_arrival_rate"):
        assert math.isfinite(float(row[key]))
by={(int(r["assignment"]),r["profile"],r["method"]):r for r in raw}
reductions=[]
for a in range(10):
    sco=float(by[a,"fixed_1_1","sco"]["post_cost"])
    pa=float(by[a,"fixed_1_1","pa16"]["post_cost"])
    reductions.append(100.0*(sco-pa)/sco)
assert all(x>0 for x in reductions), reductions
report={"raw_rows":len(raw),"summary_rows":len(summary),"pa_vs_sco_wins":sum(x>0 for x in reductions),"assignments":len(reductions),"pa_vs_sco_reduction_mean_pct":sum(reductions)/len(reductions),"pa_vs_sco_reduction_min_pct":min(reductions),"pa_vs_sco_reduction_max_pct":max(reductions),"claim_boundary":meta["claim_boundary"]}
(R/"tmc_6gl_loss_replay_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
