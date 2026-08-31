"""
Runs three systems over the IDENTICAL held-out cases and reports the comparison:

  BASELINE   trained model + static risk-band rules (no evidence)
  BASELINE+  same, plus the largest cohort deviation cited as a reason
  AGENT      the full agentic workflow

Baseline+ exists so the agent cannot win by construction. A metric that
rewards evidence would hand the agent a free victory over a baseline that
returns none, so we built the strongest cheap opponent we could — score, top
deviation, template — and ran it on the same cases under the same rubric.

Other fairness controls:
  - same trained model, feature set and risk thresholds for all three
  - same frozen holdout, from the seed used in training; unseen by all
  - the rubric imports nothing from the agent's Critic
  - every case reported, including failures

Disclosed resource difference: the agent may call tools (logit decomposition,
cohort percentile, intervention simulation) the baselines do not. Those tools
ARE the intervention under evaluation.
"""
import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced import llm  # noqa: E402
from advanced.orchestration.workflow import run_for_employee as agent_run  # noqa: E402
from advanced.tools import ToolBox  # noqa: E402
from baseline.pipeline import run_for_employee as baseline_run  # noqa: E402
from baseline.pipeline_plus import cohort_statistics  # noqa: E402
from baseline.pipeline_plus import run_for_employee as baseline_plus_run  # noqa: E402
from evaluation.rubric import MAX_SCORE, aggregate, score_case  # noqa: E402

RANDOM_STATE = 42
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")


def get_eval_cases() -> pd.DataFrame:
    """The frozen test split — the same rows train.py held out."""
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    _, test = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=df["attrition"]
    )
    return test.reset_index(drop=True)


def normalise_agent_result(res: dict) -> dict:
    detail = res.get("detail") or {}
    rc = detail.get("root_cause") or {}
    ic = detail.get("intervention") or {}
    rec = ic.get("recommendation") or {}
    return {
        "employee_id": res.get("employee_id"),
        "attrition_probability": (detail.get("risk") or {}).get("attrition_probability"),
        "risk_level": (detail.get("risk") or {}).get("risk_level"),
        "recommendation": rec.get("label"),
        "action_taken": ic.get("status") == "PROPOSED",
        "escalated": res.get("status") == "ESCALATED",
        "evidence": [c["statement"] for c in rc.get("confirmed_drivers", [])],
        "targets_drivers": rec.get("targets_drivers", []),
        "cited_features": [c["feature"] for c in rc.get("confirmed_drivers", [])],
        "status": res.get("status"),
        "tool_calls": res.get("tool_call_count", 0),
    }


def main():
    # The headline table must mean the same thing on every machine, so the
    # optional prose layer is off here regardless of whether a key is present.
    # It changes no decision and nothing the rubric scores; leaving it on would
    # only make an identical result slower and provider-dependent.
    llm.set_narration(False)

    cases = get_eval_cases()
    model = joblib.load(os.path.join(BASE_DIR, "baseline", "attrition_model.joblib"))
    features = [c for c in cases.columns if c not in ("employee_id", "attrition")]
    full = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    stats = cohort_statistics(full, features)
    toolbox = ToolBox()

    print(f"Evaluation cases (frozen holdout): {len(cases)}")
    print(f"  actual leavers: {int(cases['attrition'].sum())} | "
          f"actual stayers: {int((cases['attrition'] == 0).sum())}")
    print()

    systems = {}

    for name, runner in [
        ("baseline", lambda e: {**baseline_run(e, model, features),
                                "cited_features": [], "targets_drivers": [], "tool_calls": 0}),
        ("baseline_plus", lambda e: {**baseline_plus_run(e, model, features, stats),
                                     "tool_calls": 0}),
        ("agent", lambda e: normalise_agent_result(agent_run(e, toolbox, save=False))),
    ]:
        t0 = time.time()
        rows, scored = [], []
        for _, row in cases.iterrows():
            emp = row.to_dict()
            res = runner(emp)
            rows.append(res)
            scored.append(score_case(res, emp, features))
        agg = aggregate(scored)
        agg["wall_clock_seconds"] = round(time.time() - t0, 2)
        agg["avg_tool_calls"] = round(sum(r.get("tool_calls", 0) for r in rows) / len(rows), 2)
        agg["escalated_to_human"] = sum(1 for r in rows if r.get("escalated"))
        systems[name] = agg
        pd.DataFrame(rows).to_csv(
            os.path.join(EVIDENCE_DIR, f"{name}_case_results.csv"), index=False)
        pd.DataFrame(scored).to_csv(
            os.path.join(EVIDENCE_DIR, f"{name}_case_scores.csv"), index=False)

    report = {
        "eval_cases": len(cases),
        "split": "frozen holdout, test_size=0.25, seed=42, stratified",
        "primary_metric": "reviewer_case_quality_pct (rubric 0-5, independent of the Critic)",
        "counter_metric": "wasted_intervention_rate",
        "systems": systems,
        "headline_delta_vs_strongest_baseline": {
            "reviewer_case_quality_pct": round(
                systems["agent"]["reviewer_case_quality_pct"]
                - systems["baseline_plus"]["reviewer_case_quality_pct"], 4),
            "wasted_intervention_rate": round(
                systems["agent"]["wasted_intervention_rate"]
                - systems["baseline_plus"]["wasted_intervention_rate"], 4),
        },
    }

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    _print_table(systems)
    return report


def _print_table(s):
    b, bp, a = s["baseline"], s["baseline_plus"], s["agent"]
    w = 78
    print("=" * w)
    print("THREE SYSTEMS — identical held-out cases, identical rubric")
    print("=" * w)
    print(f"{'METRIC':<36}{'BASELINE':>13}{'BASELINE+':>13}{'AGENT':>13}")
    print("-" * w)

    def row(label, key, pct=True, best_high=True):
        vals = [b[key], bp[key], a[key]]
        fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v}")
        print(f"{label:<36}" + "".join(f"{fmt(v):>13}" for v in vals))

    print(f"{'Reviewer case quality (PRIMARY)':<36}"
          f"{b['reviewer_case_quality_pct']:>12.1%}"
          f"{bp['reviewer_case_quality_pct']:>13.1%}"
          f"{a['reviewer_case_quality_pct']:>13.1%}")
    print(f"{'  mean score (0-' + str(MAX_SCORE) + ')':<36}"
          f"{b['reviewer_case_quality_mean']:>12}"
          f"{bp['reviewer_case_quality_mean']:>13}"
          f"{a['reviewer_case_quality_mean']:>13}")
    print("-" * w)
    for label, key in [
        ("  d1 correct triage", "d1_correct_triage"),
        ("  d2 evidence present", "d2_evidence_present"),
        ("  d3 evidence verifiable", "d3_evidence_verifiable"),
        ("  d4 action has mechanism", "d4_action_has_mechanism"),
        ("  d5 proportionate", "d5_proportionate"),
    ]:
        row(label, key)
    print("-" * w)
    print(f"{'Wasted intervention (COUNTER)':<36}"
          f"{b['wasted_intervention_rate']:>12.1%}"
          f"{bp['wasted_intervention_rate']:>13.1%}"
          f"{a['wasted_intervention_rate']:>13.1%}")
    row("Catch w/ verified evidence", "catch_rate_with_verified_evidence")
    row("Raw flag rate on leavers", "raw_flag_rate_on_leavers")
    print(f"{'Unverifiable evidence claims':<36}"
          f"{b['unverifiable_evidence_claims']:>12}"
          f"{bp['unverifiable_evidence_claims']:>13}"
          f"{a['unverifiable_evidence_claims']:>13}")
    print(f"{'Interventions proposed':<36}"
          f"{b['total_interventions_proposed']:>12}"
          f"{bp['total_interventions_proposed']:>13}"
          f"{a['total_interventions_proposed']:>13}")
    print(f"{'Escalated to human':<36}{'—':>12}{'—':>13}{a['escalated_to_human']:>13}")
    print(f"{'Avg tool calls':<36}{0:>12}{0:>13}{a['avg_tool_calls']:>13}")
    print(f"{'Wall clock (s)':<36}"
          f"{b['wall_clock_seconds']:>12}{bp['wall_clock_seconds']:>13}"
          f"{a['wall_clock_seconds']:>13}")
    print("=" * w)


if __name__ == "__main__":
    main()
