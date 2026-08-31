"""
Agent 1: Data Quality Agent

A real gate, not a formality: if the record cannot be trusted, the pipeline
halts for that employee instead of scoring it anyway. The original pipeline
had no validation at all before scoring, which meant a malformed record still
produced a confident-looking risk number.
"""
from advanced.tools import FEATURE_BOUNDS


def run(employee: dict, required_features: list) -> dict:
    issues = []

    for field in required_features:
        if field not in employee or employee[field] is None:
            issues.append(f"missing_field:{field}")
            continue
        try:
            float(employee[field])
        except (TypeError, ValueError):
            issues.append(f"non_numeric:{field}={employee[field]!r}")

    for field, (lo, hi) in FEATURE_BOUNDS.items():
        if field in employee and employee[field] is not None:
            try:
                val = float(employee[field])
            except (TypeError, ValueError):
                continue
            if not (lo <= val <= hi):
                issues.append(f"out_of_range:{field}={val} (expected {lo}-{hi})")

    return {
        "agent": "data_quality",
        "verdict": "PASS" if not issues else "FAIL",
        "issues": issues,
        "fields_checked": len(required_features),
    }
