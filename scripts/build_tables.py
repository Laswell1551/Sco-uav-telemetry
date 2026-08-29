"""Build paper-facing Markdown tables from frozen machine-readable results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "results" / "frozen"
OUT = ROOT / "artifacts" / "tables"
TABLES = ROOT / "tables"

LABELS = {
    "cumulative_ce": "Cumulative CE",
    "cumulative_ucb_cv": "Cumulative UCB-CV",
    "sw_ce_32": "SW-CE (32)",
    "sw_ucb_cv_64": "SW-Whittle-CV (64)",
    "dts_whittle_cv": "DTS-Whittle-CV",
    "ts_whittle_cv": "TS-Whittle-CV",
    "de_cd_whittle_cv": "DE-CD-Whittle-CV",
    "max_age": "Max-Age",
    "ps_forced_reset_ucb": "Forced-reset-UCB",
    "forced_reset_ucb": "Forced-reset-UCB",
    "sco_reset_ce": "SCO-reset-CE",
    "sco_reset_ucb": "SCO-reset-UCB",
}


def load_json(name: str) -> Any:
    return json.loads((FROZEN / name).read_text(encoding="utf-8-sig"))


def load_csv(name: str) -> list[dict[str, str]]:
    with (FROZEN / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def interval(mean: float, low: float, high: float) -> str:
    return f"{mean:.2f} [{low:.2f}, {high:.2f}]"


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    def clean(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(clean, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_table(name: str, title: str, headers: list[str], rows: list[list[str]], sources: list[str]) -> None:
    text = f"# {title}\n\n" + markdown(headers, rows)
    rendered = [item if "/" in item else f"results/frozen/{item}" for item in sources]
    text += "\nSources: " + ", ".join(rendered) + ".\n"
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")


def controlled_rows() -> list[dict[str, Any]]:
    base = {row["method"]: row for row in load_csv("tmc_confirmatory_summary.csv")}
    rows: dict[str, dict[str, Any]] = {}
    for method, row in base.items():
        rows[method] = {
            "method_id": method,
            "method_label": LABELS[method],
            "full": [float(row["total_excess_mean"]), float(row["total_excess_ci_low"]), float(row["total_excess_ci_high"])],
            "post": [float(row["post_excess_mean"]), float(row["post_excess_ci_low"]), float(row["post_excess_ci_high"])],
            "evidence_status": "primary",
        }
    v16 = load_json("tmc_v16_baseline_expansion.json")
    summaries = v16["formal"]["seed_cluster_bootstrap"]["summaries"]
    for method in ("dts_whittle_cv", "de_cd_whittle_cv"):
        item = summaries[method]
        total = item["total_excess_cost_pct"]
        post = item["post_excess_cost_pct"]
        rows[method] = {
            "method_id": method, "method_label": LABELS[method],
            "full": [total["mean"], *total["ci95"]],
            "post": [post["mean"], *post["ci95"]],
            "evidence_status": "retrospective matched adaptation",
        }
    ts = load_json("tmc_ts_baseline_expansion.json")["formal_summary"]
    rows["ts_whittle_cv"] = {
        "method_id": "ts_whittle_cv", "method_label": LABELS["ts_whittle_cv"],
        "full": [ts["total_ex"]["mean"], *ts["total_ex"]["ci95"]],
        "post": [ts["post_ex"]["mean"], *ts["post_ex"]["ci95"]],
        "evidence_status": "retrospective matched adaptation",
    }
    max_age = load_json("tmc_external_baseline_addendum_v1.json")["summary"]
    rows["max_age"] = {
        "method_id": "max_age", "method_label": LABELS["max_age"],
        "full": [max_age["total_ex_mean"], *max_age["total_ex_cluster_ci"]],
        "post": [max_age["post_ex_mean"], *max_age["post_ex_cluster_ci"]],
        "evidence_status": "retrospective low-information comparator",
    }
    order = (
        "cumulative_ce", "cumulative_ucb_cv", "sw_ce_32", "sw_ucb_cv_64",
        "dts_whittle_cv", "ts_whittle_cv", "de_cd_whittle_cv", "max_age",
        "ps_forced_reset_ucb", "sco_reset_ce", "sco_reset_ucb",
    )
    return [rows[item] for item in order]


def trace_rows() -> list[dict[str, Any]]:
    base = {
        "uzh_fpv": load_json("uzh_trace_replay_v1.json")["summary_mean_ci95"],
        "m3ed_falcon": load_json("m3ed_trace_replay_v1.json")["summary_mean_ci95"],
    }
    extra = load_json("tmc_v16_trace_baseline_expansion.json")["datasets"]
    methods = (
        "cumulative_ce", "cumulative_ucb_cv", "sw_ce_32", "sw_ucb_cv_64",
        "dts_whittle_cv", "de_cd_whittle_cv", "forced_reset_ucb",
        "sco_reset_ce", "sco_reset_ucb",
    )
    out = []
    for method in methods:
        item: dict[str, Any] = {
            "method_id": method,
            "method_label": LABELS[method],
            "evidence_status": (
                "retrospective matched adaptation"
                if method in {"dts_whittle_cv", "de_cd_whittle_cv"}
                else "primary replay"
            ),
        }
        for dataset in ("uzh_fpv", "m3ed_falcon"):
            if method in {"dts_whittle_cv", "de_cd_whittle_cv"}:
                values = extra[dataset]["summary_mean_ci95"]["methods"][method]["excess_pct"]
            else:
                values = base[dataset][method]["excess_pct"]
            item[dataset] = values
        out.append(item)
    combined = {
        "method_id": "aoi_round_robin",
        "method_label": "AoI / round robin",
        "evidence_status": "primary replay",
    }
    for dataset in ("uzh_fpv", "m3ed_falcon"):
        aoi = base[dataset]["aoi"]["excess_pct"]
        rr = base[dataset]["round_robin"]["excess_pct"]
        if abs(float(aoi[0]) - float(rr[0])) > 1e-12:
            raise AssertionError(f"AoI and round-robin means differ for {dataset}")
        combined[dataset] = [aoi[0], min(aoi[1], rr[1]), max(aoi[2], rr[2])]
    out.insert(6, combined)
    return out


def random_delay_rows() -> list[dict[str, Any]]:
    paired = {
        row["profile"]: row
        for row in load_csv("tmc_random_delay_formal_paired_summary.csv")
    }
    he = {
        row["profile"]: row
        for row in load_csv("tmc_he_rm_formal_addendum_paired_summary.csv")
    }
    labels = {
        "fixed": "Fixed",
        "light_iid": "Light i.i.d.",
        "markov_burst": "Markov burst",
        "feedback_heavy": "Feedback-heavy",
        "forward_heavy": "Forward-heavy",
        "heavy_iid": "Heavy i.i.d.",
        "lognormal": "Log-normal",
    }
    out = []
    for profile in labels:
        primary = paired[profile]
        matched = he[profile]
        out.append({
            "profile_id": profile,
            "profile_label": labels[profile],
            "seeds": int(primary["seeds"]),
            "sco_post_cost": float(matched["sco_post_cost_mean"]),
            "rm_ack_post_cost": float(matched["he_post_cost_mean"]),
            "pa_post_cost": float(matched["pa_post_cost_mean"]),
            "gain_vs_sco": [
                float(primary["pa_reduction_vs_sco_pct_mean"]),
                float(primary["pa_reduction_vs_sco_pct_ci_low"]),
                float(primary["pa_reduction_vs_sco_pct_ci_high"]),
            ],
            "gain_vs_rm_ack": [
                float(matched["pa_reduction_vs_he_pct_mean"]),
                float(matched["pa_reduction_vs_he_pct_ci_low"]),
                float(matched["pa_reduction_vs_he_pct_ci_high"]),
            ],
        })
    return out


def static_manifest(name: str) -> list[dict[str, Any]]:
    rows = json.loads((TABLES / name).read_text(encoding="utf-8"))
    for row in rows:
        for relative in row["artifact_paths"] if "artifact_paths" in row else row["source_paths"]:
            if not (ROOT / relative).is_file():
                raise AssertionError(f"missing static-manifest source: {relative}")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    controlled = controlled_rows()
    traces = trace_rows()
    delay = random_delay_rows()
    design = static_manifest("method_design.json")
    claims = static_manifest("claim_evidence.json")

    v16 = load_json("tmc_v16_baseline_expansion.json")
    choices = v16["pilot"]["choices"]
    if float(choices["de_alpha"]) != 0.5 or float(choices["dts_gamma"]) != 0.99:
        raise AssertionError("method-design manifest disagrees with frozen v16 choices")
    if int(load_json("tmc_ts_baseline_expansion.json")["selected_episode_length"]) != 1:
        raise AssertionError("method-design manifest disagrees with frozen TS choice")

    best_full = min(row["full"][0] for row in controlled)
    best_post = min(row["post"][0] for row in controlled)
    table2 = []
    for row in controlled:
        full = interval(*row["full"])
        post = interval(*row["post"])
        if row["full"][0] == best_full:
            full = f"**{full}**"
        if row["post"][0] == best_post:
            post = f"**{post}**"
        table2.append([row["method_label"], full, post, row["evidence_status"]])
    write_table(
        "table_ii_controlled.md", "Table II - Controlled drift",
        ["Method", "Full-horizon excess cost (%)", "Post-change excess cost (%)", "Evidence"],
        table2,
        ["tmc_confirmatory_summary.csv", "tmc_v16_baseline_expansion.json", "tmc_ts_baseline_expansion.json", "tmc_external_baseline_addendum_v1.json"],
    )

    write_table(
        "table_iii_method_design.md", "Table III - Matched method design",
        ["Method", "Matched design", "Evidence"],
        [[row["method_label"], row["matched_design"], row["evidence_status"]] for row in design],
        ["tables/method_design.json", "tmc_v16_baseline_expansion.json", "tmc_ts_baseline_expansion.json"],
    )

    best_uzh = min(row["uzh_fpv"][0] for row in traces)
    best_m3ed = min(row["m3ed_falcon"][0] for row in traces)
    table4 = []
    for row in traces:
        uzh = interval(*row["uzh_fpv"])
        m3ed = interval(*row["m3ed_falcon"])
        if row["uzh_fpv"][0] == best_uzh:
            uzh = f"**{uzh}**"
        if row["m3ed_falcon"][0] == best_m3ed:
            m3ed = f"**{m3ed}**"
        table4.append([row["method_label"], uzh, m3ed, row["evidence_status"]])
    write_table(
        "table_iv_traces.md", "Table IV - External trace replay",
        ["Method", "UZH-FPV excess cost (%)", "M3ED Falcon excess cost (%)", "Evidence"],
        table4,
        ["uzh_trace_replay_v1.json", "m3ed_trace_replay_v1.json", "tmc_v16_trace_baseline_expansion.json"],
    )

    write_table(
        "table_v_random_delay.md", "Table V - Random two-way delay",
        ["Profile", "SCO post cost", "RM-ACK post cost", "PA-SCO post cost", "PA gain vs SCO (%)", "PA gain vs RM-ACK (%)"],
        [[
            row["profile_label"], f"{row['sco_post_cost']:.2f}",
            f"{row['rm_ack_post_cost']:.2f}", f"{row['pa_post_cost']:.2f}",
            interval(*row["gain_vs_sco"]), interval(*row["gain_vs_rm_ack"]),
        ] for row in delay],
        ["tmc_random_delay_formal_paired_summary.csv", "tmc_he_rm_formal_addendum_paired_summary.csv"],
    )

    write_table(
        "table_vi_claim_boundaries.md", "Table VI - Claim and evidence boundaries",
        ["Claim", "Main evidence", "Boundary"],
        [[row["claim"], row["main_evidence"], row["boundary"]] for row in claims],
        ["tables/claim_evidence.json"],
    )
    payload = {"table_ii": controlled, "table_iii": design, "table_iv": traces, "table_v": delay, "table_vi": claims}
    (OUT / "normalized_tables.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote five tables and normalized data to {OUT}")


if __name__ == "__main__":
    main()
