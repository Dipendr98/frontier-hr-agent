"""
INDEPENDENT REVIEWER RUBRIC — the primary evaluation.

This file is isolated from advanced/agents/critic.py and imports no agent
logic. That separation is the point: a metric computed by the same Critic the
solution contains would mean the solution grades its own exam.

PRIMARY METRIC — Reviewer Case Quality Score (0-5 per case, reported as mean
and as % of the maximum). It scores the artifact a reviewer actually receives,
using five dimensions that EITHER system can score on. Nothing here is
reachable only by an agent.

  1. Correct triage           flagged if and only if the person actually left
  2. Evidence present         at least one case-specific fact is cited
  3. Evidence verifiable      every cited fact is INDEPENDENTLY RECOMPUTED here
                              from the employee record and must match
  4. Action has a mechanism   the proposed action can move a cited signal
  5. Proportionate            action intensity matches the risk band

Dimension 3 is what stops "attach a plausible sentence" from scoring. The
rubric recomputes the numbers itself and rejects any claim it cannot confirm,
so a fabricated or stale citation scores zero even though a string is present.

COUNTER-METRIC — Wasted Intervention Rate. Of employees who actually stayed,
the fraction given a formal intervention. Reported alongside so that flagging
everyone cannot win.

Frozen before the final comparison was run.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced.tools import FEATURE_LEVERS, FEATURE_META  # catalogues only, no agent logic

ACTIONABLE_FEATURES = set(FEATURE_LEVERS.keys())
MAX_SCORE = 5

# Tolerance when re-checking a cited numeric value against the record.
NUMERIC_TOLERANCE = 0.051


def _extract_numbers(text: str) -> list:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def _features_named_in(statement: str, features: list) -> list:
    """
    Match a statement to the feature(s) it refers to, by raw column name or by
    the human-readable label a system may legitimately use instead.

    The label lookup matters: the first version of this rubric matched only raw
    column names, so it marked every agent citation unverifiable purely because
    the agent writes "Job satisfaction" rather than "JobSatisfaction". That was
    a defect in the measurement, not in the system being measured. Recorded
    here because it is exactly the kind of error that silently decides a
    comparison.
    """
    low = statement.lower()
    named = []
    for f in features:
        label = FEATURE_META.get(f, {}).get("label", f)
        if f in statement or f.lower() in low or label.lower() in low:
            named.append(f)
    return named


def _verify_evidence(statement: str, employee: dict, features: list) -> bool:
    """
    Independently confirm a cited claim against the employee record.

    A statement passes only if it names a real feature AND the employee's
    actual value for that feature appears among the numbers in the statement.
    Deliberately strict: it checks the citation is true of THIS employee, not
    merely well-formed.
    """
    named = _features_named_in(statement, features)
    if not named:
        return False
    numbers = _extract_numbers(statement)
    if not numbers:
        return False
    for feat in named:
        actual = float(employee[feat])
        if any(abs(n - actual) <= NUMERIC_TOLERANCE for n in numbers):
            return True
    return False


def score_case(case_result: dict, employee: dict, features: list) -> dict:
    """Score one case. `employee` is the raw record, used to re-verify claims."""
    actual_attrition = int(employee["attrition"])
    flagged = bool(case_result.get("action_taken") or case_result.get("escalated"))
    evidence = case_result.get("evidence") or []

    # 1. Correct triage
    d1 = int(flagged == (actual_attrition == 1))

    # 2. Evidence present
    d2 = int(len(evidence) > 0)

    # 3. Evidence verifiable — recomputed here, not trusted
    verified = [e for e in evidence if _verify_evidence(e, employee, features)]
    d3 = int(len(evidence) > 0 and len(verified) == len(evidence))

    # 4. Action has a mechanism on a cited signal
    targets = set(case_result.get("targets_drivers") or [])
    cited = set(case_result.get("cited_features") or [])
    for e in evidence:
        cited.update(_features_named_in(e, features))
    if not flagged:
        d4 = 1 if actual_attrition == 0 else 0   # correctly doing nothing counts
    else:
        d4 = int(bool(targets & ACTIONABLE_FEATURES & cited))

    # 5. Proportionality
    level = case_result.get("risk_level")
    if level == "Low":
        d5 = int(not case_result.get("action_taken"))
    elif level == "High":
        d5 = int(flagged)
    else:
        d5 = 1

    total = d1 + d2 + d3 + d4 + d5

    return {
        "employee_id": case_result.get("employee_id"),
        "actual_attrition": actual_attrition,
        "flagged": flagged,
        "d1_correct_triage": d1,
        "d2_evidence_present": d2,
        "d3_evidence_verifiable": d3,
        "d4_action_has_mechanism": d4,
        "d5_proportionate": d5,
        "case_score": total,
        "unverifiable_evidence": len(evidence) - len(verified),
        "wasted_intervention": flagged and actual_attrition == 0,
        "caught_leaver": flagged and actual_attrition == 1,
        "caught_with_verified_evidence": (
            flagged and actual_attrition == 1 and d3 == 1),
    }


def aggregate(scored: list) -> dict:
    n = len(scored)
    leavers = [s for s in scored if s["actual_attrition"] == 1]
    stayers = [s for s in scored if s["actual_attrition"] == 0]
    total_score = sum(s["case_score"] for s in scored)

    return {
        "n_cases": n,
        "n_leavers": len(leavers),
        "n_stayers": len(stayers),
        # PRIMARY
        "reviewer_case_quality_mean": round(total_score / n, 3) if n else 0.0,
        "reviewer_case_quality_pct": round(total_score / (n * MAX_SCORE), 4) if n else 0.0,
        # Per-dimension, so a reviewer can see where the difference comes from
        "d1_correct_triage": round(sum(s["d1_correct_triage"] for s in scored) / n, 4),
        "d2_evidence_present": round(sum(s["d2_evidence_present"] for s in scored) / n, 4),
        "d3_evidence_verifiable": round(sum(s["d3_evidence_verifiable"] for s in scored) / n, 4),
        "d4_action_has_mechanism": round(sum(s["d4_action_has_mechanism"] for s in scored) / n, 4),
        "d5_proportionate": round(sum(s["d5_proportionate"] for s in scored) / n, 4),
        # Supporting
        "catch_rate_with_verified_evidence": round(
            sum(s["caught_with_verified_evidence"] for s in leavers) / len(leavers), 4
        ) if leavers else 0.0,
        "raw_flag_rate_on_leavers": round(
            sum(s["flagged"] for s in leavers) / len(leavers), 4) if leavers else 0.0,
        "unverifiable_evidence_claims": sum(s["unverifiable_evidence"] for s in scored),
        # COUNTER
        "wasted_intervention_rate": round(
            sum(s["wasted_intervention"] for s in stayers) / len(stayers), 4) if stayers else 0.0,
        "total_interventions_proposed": sum(s["flagged"] for s in scored),
    }
