"""
BASELINE+ — a deliberately strengthened baseline.

The first baseline (pipeline.py) attaches no evidence, so any metric that
rewards evidence hands the agent a win by construction. That is a weak
comparison and we do not want to rest on it.

Baseline+ closes that gap the way a competent engineer would without building
an agent: after scoring, take the employee's largest raw deviation from the
cohort mean, state it as the reason, and keep the same fixed risk-band action
table. This is roughly "score + top feature + template" — a strong, cheap,
entirely reasonable approach, and the one a reviewer would most likely build
if told "add a reason to each flag".

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

from baseline.pipeline import STATIC_RULES, classify  # noqa: E402

# Same fixed action table as the simple baseline.
# The template picks the single biggest raw z-deviation as "the reason".


def run_for_employee(employee: dict, model, features: list, cohort_stats: dict) -> dict:
    X = pd.DataFrame([employee])[features]
    prob = float(model.predict_proba(X)[0, 1])
    level = classify(prob)

    # Largest standardised deviation from the cohort, in the "worse" direction.
    worst_feature, worst_z, worst_val = None, 0.0, None
    for feat in features:
        mean, std = cohort_stats[feat]
        if std == 0:
            continue
        z = (float(employee[feat]) - mean) / std
        if abs(z) > abs(worst_z):
            worst_feature, worst_z, worst_val = feat, z, float(employee[feat])

    evidence = []
    if worst_feature is not None and level != "Low":
        evidence.append(
            f"{worst_feature} is {worst_val:g} vs cohort mean "
            f"{cohort_stats[worst_feature][0]:.2f} (z={worst_z:+.2f})."
        )

    return {
        "employee_id": employee.get("employee_id"),
        "attrition_probability": round(prob, 4),
        "risk_level": level,
        "recommendation": None if level == "Low" else STATIC_RULES[level],
        "action_taken": level != "Low",
        "evidence": evidence,
        # The action comes from the risk band, not from the cited feature, so
        # there is no claim that it targets that driver. Stating this honestly
        # rather than asserting a link the method does not establish.
        "targets_drivers": [],
        "cited_features": [worst_feature] if worst_feature and level != "Low" else [],
        "escalated": False,
    }


def cohort_statistics(df: pd.DataFrame, features: list) -> dict:
    return {f: (float(df[f].mean()), float(df[f].std())) for f in features}
