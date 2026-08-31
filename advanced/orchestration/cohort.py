"""
Cohort triage — the artifact the reviewer actually receives.

Running one employee at a time is how you demo an agent. It is not how anyone
does this job. An HR business partner opens their week with a cohort, not an
employee ID, and what they need is a prioritised worklist where every row
already has its case attached, plus the handful of findings that only exist at
the cohort level.

    python -m advanced.orchestration.cohort                # whole cohort
    python -m advanced.orchestration.cohort --limit 60     # first 60
    python -m advanced.orchestration.cohort --no-memory    # annotation off

Writes:
    evidence/cohort_worklist.csv    one row per employee, review order
    evidence/cohort_briefing.md     the reviewer-facing briefing
    evidence/cohort_memory.json     the memory store the run built

TWO PASSES, ON PURPOSE. Pass 1 runs every case and populates memory. Pass 2
re-runs them with that memory available. The alternative — annotating as we go —
means employee #3 is judged against a two-person cohort and employee #300
against a full one, so two identical people get different write-ups based on
queue position. Prevalence is a property of the cohort, so it has to be computed
over the cohort before it is applied to anyone. The cost is a second pass over a
sub-second-per-case pipeline; the benefit is that the annotation means the same
thing on every row.

Nothing here executes an HR action. Every row is a case for a human to decide.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from advanced import llm
from advanced.memory import CohortMemory
from advanced.orchestration.workflow import run_for_employee
from advanced.tools import ToolBox

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

# Review order. Escalations first: those are the cases where the system could
# not resolve something and a human is the only way forward, so they are the
# ones that go stale worst if they sit behind routine approvals.
STATUS_PRIORITY = {"ESCALATED": 0, "HALTED_DATA_QUALITY": 1,
                   "APPROVED_FOR_REVIEW": 2, "NO_ACTION": 3}


def run_cohort(limit: int = None, use_memory: bool = True,
               data_path: str = None, save: bool = True,
               narrate: bool = False) -> dict:
    """
    Triage a whole cohort. Returns the report; writes the briefing when `save`.

    `narrate` is a parameter rather than something the caller sets beforehand,
    and it defaults OFF. This function runs the pipeline TWICE over every
    employee, and Mode 1's prose layer costs one completion per agent per case
    — so on 342 employees, narration turns a 2-second job into roughly 900 API
    calls and an hour of wall clock, throttled.

    That is not hypothetical: the switch used to live in this module's CLI
    `main()` instead of here, so the Streamlit page called straight through and
    got the hour-long path. A default that only holds for one caller is not a
    default. Nothing about narration changes a decision.
    """
    toolbox = ToolBox(data_path=data_path) if data_path else ToolBox()
    df = toolbox.cohort if limit is None else toolbox.cohort.head(limit)
    employees = [row.to_dict() for _, row in df.iterrows()]

    memory = CohortMemory(autoload=False) if use_memory else None
    t0 = time.time()

    previous_narration = llm.NARRATION
    llm.set_narration(narrate)
    try:
        # --- Pass 1: decide every case, building memory ---------------------
        # Collect-only: pass 1's annotations are discarded, so computing them
        # per employee is pure waste — and each one scans the store.
        if memory is not None:
            memory.collect_only()
        for emp in employees:
            run_for_employee(emp, toolbox, save=False, memory=memory)

        # --- Pass 2: same decisions, now with cohort-wide context -----------
        # Frozen: pass 2 reads the memory pass 1 built. Re-recording would write
        # data identical to what is already there and invalidate the statistics
        # cache once per employee, which is what made this quadratic.
        if memory is not None:
            memory.freeze()
        rows = []
        for emp in employees:
            res = run_for_employee(emp, toolbox, save=False, memory=memory)
            rows.append(_to_row(res, emp))
        if memory is not None:
            memory.unfreeze()
    finally:
        llm.set_narration(previous_narration)

    elapsed = time.time() - t0
    findings = memory.systemic_findings() if memory else []
    rows.sort(key=lambda r: (STATUS_PRIORITY.get(r["status"], 9),
                             -(r["attrition_probability"] or 0)))

    report = {
        "generated_by": "frontier-hr-agent cohort triage",
        "llm": llm.describe(),
        "memory": "enabled" if use_memory else "disabled",
        "n_employees": len(rows),
        "wall_clock_seconds": round(elapsed, 2),
        "seconds_per_case": round(elapsed / len(rows), 4) if rows else 0.0,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "risk_counts": dict(Counter(r["risk_level"] for r in rows if r["risk_level"])),
        "systemic_findings": findings,
        "llm_usage": llm.METER.snapshot(),
        "worklist": rows,
    }

    if save:
        _write_csv(rows)
        _write_briefing(report)
        if memory:
            memory.save()
    return report


def _to_row(res: dict, emp: dict) -> dict:
    detail = res.get("detail") or {}
    risk = detail.get("risk") or {}
    rc = detail.get("root_cause") or {}
    ic = detail.get("intervention") or {}
    rec = ic.get("recommendation") or {}
    mem = detail.get("memory") or {}
    gate = next((s for s in res["trajectory"] if s.get("step") == "human_gate"), {})

    return {
        "employee_id": res["employee_id"],
        "status": res["status"],
        "risk_level": risk.get("risk_level"),
        "attrition_probability": risk.get("attrition_probability"),
        "root_cause_status": rc.get("status"),
        "evidence": [c["statement"] for c in rc.get("confirmed_drivers", [])],
        "actionable_features": rc.get("actionable_features", []),
        "contextual_features": rc.get("contextual_features", []),
        "recommendation": rec.get("label"),
        "recommendation_key": rec.get("key"),
        "simulated_delta_pp": rec.get("simulated_delta_pp"),
        "unresolved_objections": gate.get("unresolved_objections", []),
        "reason": gate.get("reason"),
        "memory_notes": [n["note"] for n in mem.get("driver_context", [])],
        "precedent": mem.get("precedent") or {},
        "requires_human_decision": gate.get("requires_human_decision", False),
        "tool_calls": res.get("tool_call_count", 0),
    }


def _write_csv(rows: list):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(EVIDENCE_DIR, "cohort_worklist.csv")
    cols = ["employee_id", "status", "risk_level", "attrition_probability",
            "root_cause_status", "recommendation", "simulated_delta_pp",
            "requires_human_decision", "evidence", "actionable_features",
            "contextual_features", "memory_notes", "unresolved_objections",
            "tool_calls"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (" | ".join(map(str, v)) if isinstance(v, list) else v)
                        for k, v in r.items() if k in cols})
    return path


def _write_briefing(report: dict) -> str:
    """
    The reviewer-facing document.

    Written to be read by a person who has twenty minutes and a cohort of
    hundreds: what needs a decision, what the evidence is, and what the system
    could not resolve — in that order, with the caveats attached to the numbers
    they qualify rather than collected in a footer nobody reads.
    """
    rows = report["worklist"]
    L = []
    A = L.append

    A("# Onboarding risk review — cohort briefing")
    A("")
    A(f"{report['n_employees']} employees reviewed in "
      f"{report['wall_clock_seconds']}s "
      f"({report['seconds_per_case'] * 1000:.0f} ms per case). "
      f"Provider: `{report['llm']}`.")
    A("")
    A("**Nothing in this document has been actioned.** Every row is a case "
      "prepared for a human decision. The system does not contact anyone, "
      "change any record, or take any HR action.")
    A("")

    counts = report["status_counts"]
    A("## What needs you")
    A("")
    A("| Outcome | Count | What it means |")
    A("|---|---:|---|")
    A(f"| Escalated | {counts.get('ESCALATED', 0)} | The system could not "
      "resolve these. Read these first. |")
    A(f"| Approved for review | {counts.get('APPROVED_FOR_REVIEW', 0)} | A "
      "specific plan with evidence, awaiting your decision. |")
    A(f"| No action | {counts.get('NO_ACTION', 0)} | Low risk; intervening "
      "would not be proportionate. |")
    A(f"| Halted (data quality) | {counts.get('HALTED_DATA_QUALITY', 0)} | "
      "Record failed validation and was never scored. |")
    A("")

    if report["systemic_findings"]:
        A("## Cohort-level findings")
        A("")
        A("These are not visible from any individual case. They come from the "
          "pattern across everyone reviewed, and they are addressed to whoever "
          "owns the team rather than to the reviewer of any one person.")
        A("")
        for f in report["systemic_findings"]:
            A(f"### {f['label']} — {f['affected_count']} of the "
              f"{f['n_actioned']} employees receiving an intervention "
              f"({f['share_of_actioned']:.0%})")
            A("")
            A(f["finding"])
            A("")
            A(f"Sample: {', '.join(f['affected_sample'])}"
              + (" ..." if f["affected_count"] > len(f["affected_sample"]) else ""))
            A("")

    for heading, status, blurb in [
        ("Escalated — unresolved, needs a human", "ESCALATED",
         "The system reached a High-risk or unverifiable conclusion it would "
         "not sign off on. The objection is stated so you can see what it "
         "could not settle."),
        ("Approved for review — plan and evidence attached", "APPROVED_FOR_REVIEW",
         "Each recommendation targets a driver confirmed from the employee's "
         "own record. Simulated deltas are model what-ifs on correlational "
         "features, NOT causal effect estimates."),
    ]:
        subset = [r for r in rows if r["status"] == status]
        if not subset:
            continue
        A(f"## {heading} ({len(subset)})")
        A("")
        A(blurb)
        A("")
        for r in subset:
            A(f"### {r['employee_id']} — {r['risk_level']} risk "
              f"({(r['attrition_probability'] or 0):.1%})")
            A("")
            if r["evidence"]:
                A("**Evidence**")
                A("")
                for e in r["evidence"]:
                    A(f"- {e}")
                A("")
            else:
                A("**Evidence** — none confirmed. "
                  f"{r.get('reason') or 'No single driver clears the evidence floors.'}")
                A("")
            if r["recommendation"]:
                A(f"**Proposed** {r['recommendation']} "
                  f"(simulated {r['simulated_delta_pp']:+.1f} pp — model what-if, "
                  "not a causal estimate)")
                A("")
            if r["contextual_features"]:
                A(f"**Noted, no lever** {', '.join(r['contextual_features'])} — "
                  "real for this employee, but nothing in this system can "
                  "honestly claim to change it.")
                A("")
            for n in r["memory_notes"]:
                A(f"**Cohort context** {n}")
                A("")
            if r["precedent"].get("inconsistent"):
                p = r["precedent"]
                A(f"**Consistency check** {p['comparable_cases']} comparable "
                  f"case(s) with the same risk band and drivers received "
                  f"differing outcomes: {p['prior_outcomes']}. Worth a look "
                  "before you decide this one.")
                A("")
            for o in r["unresolved_objections"]:
                A(f"**Unresolved** {o}")
                A("")

    A("---")
    A("")
    A("## How to read this")
    A("")
    A("Every cited number is recomputed from the employee's record by the "
      "evaluation rubric independently of the agents that produced it "
      "(`evaluation/rubric.py`). Driver contributions are an exact "
      "decomposition of the trained model's logit, not a post-hoc narrative.")
    A("")
    A("Protected attributes (gender, age, marital status) are excluded from "
      "the model. Simulated intervention effects describe what the model would "
      "predict under changed inputs — they are not evidence that the "
      "intervention causes the change.")
    A("")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    path = os.path.join(EVIDENCE_DIR, "cohort_briefing.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(EVIDENCE_DIR, "cohort_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def main():
    ap = argparse.ArgumentParser(description="Run cohort triage.")
    ap.add_argument("--limit", type=int, default=None, help="only the first N employees")
    ap.add_argument("--no-memory", action="store_true", help="disable cohort memory")
    ap.add_argument("--data", default=None, help="path to an alternative cohort CSV")
    ap.add_argument("--narrate", action="store_true",
                    help="enable the optional LLM prose layer (costs one "
                         "completion per agent per case; changes no decision)")
    args = ap.parse_args()

    report = run_cohort(limit=args.limit, use_memory=not args.no_memory,
                        data_path=args.data, narrate=args.narrate)

    print(f"Reviewed {report['n_employees']} employees in "
          f"{report['wall_clock_seconds']}s "
          f"({report['seconds_per_case'] * 1000:.0f} ms/case)")
    print(f"  status: {report['status_counts']}")
    print(f"  risk:   {report['risk_counts']}")
    if report["systemic_findings"]:
        print(f"\n  {len(report['systemic_findings'])} cohort-level finding(s):")
        for f in report["systemic_findings"]:
            print(f"    - {f['label']}: {f['affected_count']} of "
                  f"{f['n_actioned']} actioned ({f['share_of_actioned']:.0%})")
    else:
        print("  no intervention reached the systemic-concentration threshold")
    print()
    print("  evidence/cohort_briefing.md   reviewer briefing")
    print("  evidence/cohort_worklist.csv  ranked worklist")
    print("  evidence/cohort_memory.json   memory store")


if __name__ == "__main__":
    main()
