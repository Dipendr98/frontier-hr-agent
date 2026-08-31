"""
BASELINE+ — a deliberately strengthened baseline.

The first baseline (pipeline.py) attaches no evidence, so any metric that
rewards evidence hands the agent a win by construction. That is a weak
comparison and we do not want to rest on it.

Baseline+ closes that gap the way a competent engineer would without building
an agent: after scoring, take the employee's largest raw deviation from the
cohort mean in the concerning direction, state it as the reason, and propose
the intervention that has a lever on that feature. This is roughly "score + top
feature + targeted template" — a strong, cheap, entirely reasonable approach,
and the one a competent engineer would most likely build if told "add a reason
to each flag and act on it".

It is a fair opponent. It cites real numbers, it is checkable, and it costs no
tool calls. What it does NOT do is verify that the cited signal is the one the
model actually used, check whether the proposed action has any mechanism on
that signal, or refuse when the evidence is thin. Those are the specific
capabilities the agent adds, and the evaluation is designed to isolate them.
"""
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced.tools import (FEATURE_LEVERS, FEATURE_META,  # noqa: E402
                            INTERVENTION_CATALOG)  # catalogues only
from baseline.pipeline import STATIC_RULES, classify  # noqa: E402

# Same fixed action table as the simple baseline.
# The template picks the single biggest raw z-deviation as "the reason".


def run_for_employee(employee: dict, model, features: list, cohort_stats: dict) -> dict:
    X = pd.DataFrame([employee])[features]
    prob = float(model.predict_proba(X)[0, 1])
    level = classify(prob)

    # Largest standardised deviation from the cohort, in the CONCERNING
    # direction.
    #
    # This used to take the largest deviation by absolute value, which is not
    # what the comment above it claimed and not what a competent engineer would
    # ship: it could cite "job satisfaction is 4 vs a mean of 2.7" as the reason
    # someone is a flight risk. Direction is read from FEATURE_META, the same
    # table the agent uses, so the baseline gets the same information — the
    # comparison is only fair if the opponent is the strongest cheap version of
    # itself. (The same defect in the agent's own gate is Iteration 13.)
    worst_feature, worst_z, worst_val = None, 0.0, None
    for feat in features:
        mean, std = cohort_stats[feat]
        if std == 0:
            continue
        z = (float(employee[feat]) - mean) / std
        bad_when = FEATURE_META.get(feat, {}).get("bad_when", "low")
        concerning = -z if bad_when == "low" else z
        if concerning > worst_z:
            worst_feature, worst_z, worst_val = feat, z, float(employee[feat])

    evidence, targets, recommendation = [], [], None
    if level != "Low":
        recommendation = STATIC_RULES[level]
    if worst_feature is not None and level != "Low":
        evidence.append(
            f"{worst_feature} is {worst_val:g} vs cohort mean "
            f"{cohort_stats[worst_feature][0]:.2f} (z={worst_z:+.2f})."
        )
        # Point the action at the feature that was cited, where a lever exists.
        #
        # This baseline previously cited a driver and then proposed an unrelated
        # action from the risk-band table, declaring `targets_drivers: []`
        # because no link had been established. Honest about itself — and an
        # under-powered opponent, since it scored zero on "action has a
        # mechanism" for every flagged case. Mapping the cited feature to its
        # lever is one dictionary lookup and obviously what someone would build,
        # so leaving it out was handicapping the comparison, not protecting it.
        # It lifts BASELINE+ from 63.3% to 68.4% and shrinks the agent's margin
        # from +29.8pp to +24.6pp. That is the number worth reporting.
        levers = FEATURE_LEVERS.get(worst_feature, [])
        if levers:
            recommendation = INTERVENTION_CATALOG[levers[0]]["label"]
            targets = [worst_feature]

    return {
        "employee_id": employee.get("employee_id"),
        "attrition_probability": round(prob, 4),
        "risk_level": level,
        "recommendation": recommendation,
        "action_taken": level != "Low",
        "evidence": evidence,
        # Still no verification that the cited feature is what the model used,
        # no check that the intervention would help, and no ability to refuse.
        # Those remain the agent's contribution, and the evaluation isolates them.
        "targets_drivers": targets,
        "cited_features": [worst_feature] if worst_feature and level != "Low" else [],
        "escalated": False,
    }


def cohort_statistics(df: pd.DataFrame, features: list) -> dict:
    return {f: (float(df[f].mean()), float(df[f].std())) for f in features}
