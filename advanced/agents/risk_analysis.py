"""
Agent 2: Risk Analysis Agent

Scope is deliberately narrow now: score the employee and classify the level.
Explanation belongs to the Root Cause agent, action belongs to the
Intervention agent. Keeping these separate is what lets the Critic check one
against the other — a single agent that both decides and justifies its own
decision cannot be audited that way.
"""

HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.35


def classify(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "High"
    if prob >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def run(employee: dict, toolbox) -> dict:
    prediction = toolbox.predict_risk(employee)
    level = classify(prediction["attrition_probability"])
    return {
        "agent": "risk_analysis",
        "employee_id": employee.get("employee_id"),
        "attrition_probability": prediction["attrition_probability"],
        "risk_level": level,
    }
