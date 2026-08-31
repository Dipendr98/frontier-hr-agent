"""
Agent 6: Human Approval Gate

Per the hackathon rulebook: consequential actions stay in simulation and a
human approves before anything happens. Nothing here executes an HR action.

Three outcomes, and the middle one matters most:
  APPROVED_FOR_REVIEW  - plan is coherent, queued for a human decision
  ESCALATED            - critic never cleared it; a human must look, and the
                         unresolved objections travel with the case
  NO_ACTION            - low risk, nothing to review
"""

HIGH_RISK_THRESHOLD = 0.6


def run(risk: dict, root_cause: dict, intervention: dict, critic_result: dict) -> dict:
    prob = risk["attrition_probability"]

    if critic_result["verdict"] != "APPROVE":
        return {
            "agent": "human_gate",
            "status": "ESCALATED",
            "employee_id": risk["employee_id"],
            "reason": "Critic did not clear the plan after the allowed revisions.",
            "unresolved_objections": critic_result["problems"],
            "proposed_recommendation": (
                intervention["recommendation"]["label"]
                if intervention.get("recommendation") else None
            ),
            "requires_human_decision": True,
        }

    if intervention["status"] == "NO_ACTION_PROPOSED":
        status = "ESCALATED" if prob >= HIGH_RISK_THRESHOLD else "NO_ACTION"
        return {
            "agent": "human_gate",
            "status": status,
            "employee_id": risk["employee_id"],
            "reason": intervention["reason"],
            "requires_human_decision": status == "ESCALATED",
        }

    return {
        "agent": "human_gate",
        "status": "APPROVED_FOR_REVIEW",
        "employee_id": risk["employee_id"],
        "risk_level": risk["risk_level"],
        "proposed_recommendation": intervention["recommendation"]["label"],
        "evidence": [c["statement"] for c in root_cause["confirmed_drivers"]],
        "simulated_delta_pp": intervention["recommendation"]["simulated_delta_pp"],
        "caveat": intervention["recommendation"]["caveat"],
        "requires_human_decision": True,
        "note": "No action is executed by this system. A human reviewer decides.",
    }
