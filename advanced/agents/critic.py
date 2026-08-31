"""
Agent 5: Critic

Every check here is DECIDABLE from the structured outputs — no LLM opinion is
required to fail a plan. That matters because a critic that only produces
prose can be talked out of its objection; this one cannot. Its verdict is
reproducible, which is what makes the baseline-vs-advanced comparison
measurable rather than anecdotal.
"""

HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.35


def run(risk: dict, root_cause: dict, intervention: dict) -> dict:
    problems = []
    prob = risk["attrition_probability"]

    # C1: elevated risk with a proposed action but no confirmed driver behind it.
    if prob >= MEDIUM_RISK_THRESHOLD and root_cause["status"] == "UNEXPLAINED":
        if intervention["status"] == "PROPOSED":
            problems.append(
                "unsupported_recommendation: risk is elevated but no driver was "
                "confirmed, so the proposed intervention rests on the score alone."
            )

    # C2: the recommendation must target a driver that was actually confirmed.
    if intervention["status"] == "PROPOSED":
        targets = intervention["recommendation"]["targets_drivers"]
        confirmed = set(root_cause["actionable_features"])
        if not targets or not set(targets).issubset(confirmed):
            problems.append(
                f"driver_mismatch: recommendation targets {targets} which is not a "
                f"subset of confirmed drivers {sorted(confirmed)}."
            )

    # C3: don't spend an intervention where the simulation shows no benefit.
    if intervention["status"] == "PROPOSED":
        delta = intervention["recommendation"]["simulated_delta_pp"]
        if delta >= 0:
            problems.append(
                f"no_simulated_benefit: chosen intervention shows {delta:+.1f} pp, "
                "i.e. no modelled improvement."
            )

    # C4: high risk must not end with nothing happening.
    if prob >= HIGH_RISK_THRESHOLD and intervention["status"] == "NO_ACTION_PROPOSED":
        problems.append(
            "high_risk_no_action: risk is high but no intervention was proposed; "
            "this must be escalated to a human, not closed."
        )

    # C5: low risk should not trigger heavy intervention.
    if prob < MEDIUM_RISK_THRESHOLD and intervention["status"] == "PROPOSED":
        problems.append(
            "over_intervention: risk is low; proposing a formal intervention is "
            "not proportionate."
        )

    verdict = "APPROVE" if not problems else "REVISE"
    return {
        "agent": "critic",
        "verdict": verdict,
        "problems": problems,
        "checks_run": ["C1", "C2", "C3", "C4", "C5"],
    }
