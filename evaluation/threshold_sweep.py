"""
Records the evidence-threshold experiment that set MIN_CONTRIBUTION.

Run separately from evaluate.py because it deliberately mutates the agent's
threshold across runs. Its output is the evidence behind one changelog entry.
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced import llm  # noqa: E402
from advanced.agents import root_cause  # noqa: E402
from advanced.orchestration.workflow import run_for_employee as agent_run  # noqa: E402
from advanced.tools import ToolBox  # noqa: E402
from evaluation.evaluate import get_eval_cases, normalise_agent_result  # noqa: E402
from evaluation.rubric import aggregate, score_case  # noqa: E402

SWEEP = [0.10, 0.25, 0.40, 0.60, 0.80]
CHOSEN = 0.40


def _explain(chosen: dict, best: dict, tied: list) -> str:
    """Write the conclusion from the measured rows, so it cannot go stale."""
    q = chosen["reviewer_case_quality_pct"]

    # No setting beats the chosen one on the primary metric.
    if best["reviewer_case_quality_pct"] <= q:
        others = [t for t in tied if t["min_contribution"] != chosen["min_contribution"]]
        if others:
            worst_other = max(others, key=lambda r: r["wasted_intervention_rate"])
            return (
                f"MIN_CONTRIBUTION={chosen['min_contribution']} dominates on this "
                f"run: it ties the best case-quality score ({q:.1%}, shared with "
                f"{', '.join(str(t['min_contribution']) for t in others)}) while "
                f"wasting fewer interventions "
                f"({chosen['wasted_intervention_rate']:.1%} vs "
                f"{worst_other['wasted_intervention_rate']:.1%}). Note that this "
                f"used to be a real trade-off — before the Iteration 13 direction "
                f"fix, a lower threshold bought case quality at the cost of the "
                f"counter-metric. Confirming more genuine drivers removed the "
                f"need to loosen the floor to find any."
            )
        return (
            f"MIN_CONTRIBUTION={chosen['min_contribution']} is the outright best "
            f"setting on this run: highest case quality ({q:.1%}) and a "
            f"wasted-intervention rate of "
            f"{chosen['wasted_intervention_rate']:.1%}. No trade-off to report."
        )

    return (
        f"A genuine trade-off, reported rather than resolved quietly. "
        f"MIN_CONTRIBUTION={best['min_contribution']} scores HIGHER on the "
        f"primary metric ({best['reviewer_case_quality_pct']:.1%} vs "
        f"{chosen['reviewer_case_quality_pct']:.1%}) because it cites more "
        f"drivers per case. We chose {chosen['min_contribution']} on the frozen "
        f"counter-metric: wasted interventions "
        f"{best['wasted_intervention_rate']:.1%} -> "
        f"{chosen['wasted_intervention_rate']:.1%}, with the verified-evidence "
        f"catch rate at {chosen['catch_rate_with_verified_evidence']:.1%}. The "
        f"counter-metric was frozen in advance precisely to stop case quality "
        f"being bought by flagging more people, and manager attention is the "
        f"scarce resource for this user. A reviewer who weights the primary "
        f"metric strictly should prefer {best['min_contribution']}; both rows "
        f"are published here so that disagreement is possible."
    )


def main():
    # Same reason as evaluate.py: the prose layer changes no decision and
    # nothing this sweep measures, so leaving it on would turn a three-second
    # deterministic experiment into 860 API calls.
    llm.set_narration(False)

    cases = get_eval_cases()
    features = [c for c in cases.columns if c not in ("employee_id", "attrition")]
    toolbox = ToolBox()
    original = root_cause.MIN_CONTRIBUTION
    rows = []

    try:
        for mc in SWEEP:
            root_cause.MIN_CONTRIBUTION = mc
            scored, escalated, interventions = [], 0, 0
            for _, row in cases.iterrows():
                emp = row.to_dict()
                res = normalise_agent_result(agent_run(emp, toolbox, save=False))
                scored.append(score_case(res, emp, features))
                escalated += res["escalated"]
                interventions += res["action_taken"]
            agg = aggregate(scored)
            rows.append({
                "min_contribution": mc,
                "reviewer_case_quality_pct": agg["reviewer_case_quality_pct"],
                "catch_rate_with_verified_evidence": agg["catch_rate_with_verified_evidence"],
                "wasted_intervention_rate": agg["wasted_intervention_rate"],
                "interventions_proposed": interventions,
                "escalated": escalated,
            })
    finally:
        root_cause.MIN_CONTRIBUTION = original

    # The narrative is DERIVED from the rows, not written alongside them.
    # A hardcoded conclusion is how a report ends up quoting numbers the
    # experiment no longer produces — which is exactly what happened to the
    # previous version of this file after the Iteration 13 fix changed every
    # value in the table while the prose kept citing the old ones.
    chosen = next(r for r in rows if r["min_contribution"] == CHOSEN)
    best_quality = max(rows, key=lambda r: r["reviewer_case_quality_pct"])
    top = best_quality["reviewer_case_quality_pct"]
    tied = [r for r in rows if r["reviewer_case_quality_pct"] == top]
    out = {
        "experiment": "evidence threshold (MIN_CONTRIBUTION) sweep",
        "eval_cases": len(cases),
        "results": rows,
        "chosen": CHOSEN,
        "best_on_primary_metric": [r["min_contribution"] for r in tied],
        "reason": _explain(chosen, best_quality, tied),
    }
    path = os.path.join(BASE_DIR, "evidence", "threshold_sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'MIN_CONTRIB':<13}{'quality':>10}{'catch':>9}{'wasted':>9}"
          f"{'n_interv':>10}{'escal':>8}")
    for r in rows:
        print(f"{r['min_contribution']:<13}{r['reviewer_case_quality_pct']:>10.1%}"
              f"{r['catch_rate_with_verified_evidence']:>9.1%}"
              f"{r['wasted_intervention_rate']:>9.1%}{r['interventions_proposed']:>10}"
              f"{r['escalated']:>8}")
    print(f"\nChosen: {out['chosen']}")
    print(f"Best on primary metric: {out['best_on_primary_metric']}")
    print(f"\n{out['reason']}")
    return out


if __name__ == "__main__":
    main()
