"""Exploratory channel-only replay with measured 6GL-CLD26_v2 loss traces.

The physical CV banks remain synthetic and the 1+1-slot latency condition is
explicitly synthetic. Results must not be described as measured-delay replay.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_random_delay import run_policy_random_delay

RESULTS = Path("results")
TRACE = RESULTS / "tmc_6gl_aligned_trace.csv"
SEEDS = tuple(20267000 + i for i in range(10))
METHODS = (
    ("true", "true", None),
    ("sco", "sco_reset_ucb", None),
    ("forced", "ps_forced_reset_ucb", None),
    ("pa16", "inflight_sco_ucb", 16.0),
    ("stopwait", "inflight_sco_ucb", 1e6),
)
PROFILES = {"zero_rtt": (0, 0), "fixed_1_1": (1, 1)}


def load_probabilities():
    groups = {}
    with TRACE.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["test"], []).append(float(row["app_success_rate"]))
    names = sorted(groups, key=lambda x: int(x[4:]))
    horizon = min(len(groups[name]) for name in names)
    return names, np.asarray([groups[name][:horizon] for name in names]), horizon


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    trace_names, p_trace, H = load_probabilities()
    K, N, B, n0, change_t, block_length = len(trace_names), 2, 1, 8, H // 2, 64
    raw=[]
    for assignment, seed in enumerate(SEEDS):
        (_, _, theta0, theta1, c00, c01, changed, pre_bank, post_bank, ages0) = make_problem(seed,B,K,change_t,H,n0,block_length)
        perm=np.random.default_rng(seed+91).permutation(K)
        probs=p_trace[perm][None,:,:]
        uniforms=np.random.default_rng(seed+50000).random((B,K,H))
        # Binary outcomes are encoded as 0/1 and consumed by the unmodified
        # runner with threshold 0.5.
        outcome_bank=np.where(uniforms < probs, 0.0, 1.0)
        for profile,(df,db) in PROFILES.items():
            forward=np.full((B,K,H),df,dtype=int); feedback=np.full((B,K,H),db,dtype=int)
            common=(theta0,theta1,c00,c01,changed,pre_bank,post_bank,outcome_bank,forward,feedback,ages0,N,n0,change_t,H)
            for label,policy,beta in METHODS:
                if profile=="zero_rtt" and label in {"pa16","stopwait"}: continue
                kwargs={"success_probability":0.5}
                if beta is not None: kwargs["inflight_beta"]=beta
                r=run_policy_random_delay(policy,*common,**kwargs)
                raw.append({
                    "assignment":assignment,"seed":seed,"profile":profile,"method":label,
                    "trace_order":"|".join(trace_names[i] for i in perm),
                    "post_cost":float(np.mean(r["post_cost"])),"total_cost":float(np.mean(r["avg_cost"])),
                    "delivery_rate":r["delivery_rate"],"redundant_attempt_rate":r["redundant_attempt_rate"],
                    "stale_arrival_rate":r["stale_arrival_rate"],"max_inflight_count":r["max_inflight_count"],
                })
        print("completed",assignment,seed,flush=True)
    write_csv(RESULTS/"tmc_6gl_loss_replay_raw.csv",raw)
    summary=[]
    for profile in PROFILES:
        for label,_,_ in METHODS:
            vals=[r for r in raw if r["profile"]==profile and r["method"]==label]
            if not vals: continue
            rec={"profile":profile,"method":label,"assignments":len(vals)}
            for metric in ("post_cost","total_cost","delivery_rate","redundant_attempt_rate","stale_arrival_rate"):
                x=np.asarray([float(r[metric]) for r in vals])
                rec[f"{metric}_mean"]=float(x.mean()); rec[f"{metric}_min"]=float(x.min()); rec[f"{metric}_max"]=float(x.max())
            summary.append(rec)
    write_csv(RESULTS/"tmc_6gl_loss_replay_summary.csv",summary)
    meta={
        "status":"exploratory_not_paper_facing","dataset":"6GL-CLD26_v2","source_doi":"10.5281/zenodo.21240929",
        "seeds":SEEDS,"K":K,"N":N,"B":B,"H":H,"change_t":change_t,"n0":n0,
        "trace_tests":trace_names,"profiles":{"zero_rtt":"measured application loss, zero synthetic RTT","fixed_1_1":"measured application loss, synthetic one-slot forward and feedback delay"},
        "physical_process":"frozen synthetic CV problems; channel-only isolation",
        "statistics":"assignment sensitivity only; assignments reuse the same ten source missions and are not independent channel replicates",
        "claim_boundary":"not a measured-delay replay and not yet manuscript evidence",
    }
    (RESULTS/"tmc_6gl_loss_replay_meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
