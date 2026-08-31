"""
BASELINE — the reasonable basic way to handle this task today.

Trained model + static threshold rules. This is what an HR analytics team
would actually build first, and it is what the agent solution must beat on
identical cases.

Deliberately fair: the baseline uses the SAME trained model, the SAME data
and the SAME risk thresholds as the agent. The only difference is what
happens after the score — static rules here, investigation and verification
in the agent. That isolates the agentic contribution rather than confounding
it with a better model.

The rules below are not a strawman. They are the standard pattern: rank by
risk, map risk band to a fixed action list, hand the list to a manager.
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.35

# Fixed risk-band -> action mapping. No case-specific reasoning.
STATIC_RULES = {
    "High": "Assign an onboarding mentor and schedule a 1-on-1 within 48 hours",
    "Medium": "Increase manager check-in frequency to weekly",
    "Low": "Continue routine check-ins",
}


def classify(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "High"
    if prob >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def run_for_employee(employee: dict, model, features: list) -> dict:
    X = pd.DataFrame([employee])[features]
    prob = float(model.predict_proba(X)[0, 1])
    level = classify(prob)
    action = STATIC_RULES[level]

    return {
        "employee_id": employee.get("employee_id"),
        "attrition_probability": round(prob, 4),
        "risk_level": level,
        "recommendation": None if level == "Low" else action,
        "action_taken": level != "Low",
        # The baseline cites no per-case evidence — this is precisely the
        # bottleneck the agent addresses, not an artificial handicap.
        "evidence": [],
        "escalated": False,
    }


if __name__ == "__main__":
    import joblib

    model = joblib.load(os.path.join(BASE_DIR, "baseline", "attrition_model.joblib"))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    features = [c for c in df.columns if c not in ("employee_id", "attrition")]

    results = [run_for_employee(r.to_dict(), model, features) for _, r in df.iterrows()]
    out = pd.DataFrame(results)
    print(f"Baseline run over {len(out)} cases")
    print(out["risk_level"].value_counts().to_string())
    print(f"Actions proposed: {out['action_taken'].sum()}")
