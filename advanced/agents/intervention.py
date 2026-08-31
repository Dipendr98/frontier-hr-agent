"""
Agent 4: Intervention Planner

Hard constraint: an intervention may only be proposed if it has a lever on a
driver the Root Cause agent actually CONFIRMED. This is the fix for the
classic failure in the original pipeline, where a fixed rule fired
"assign mentor" regardless of whether mentoring had anything to do with the
employee's situation.

Candidates are ranked by simulated risk reduction, which is a model what-if,
not a causal estimate — labelled as such at every layer.
"""
from advanced.llm import reason_with_llm_or_fallback
from advanced.tools import FEATURE_LEVERS, INTERVENTION_CATALOG


def run(employee: dict, risk: dict, root_cause: dict, toolbox) -> dict:
    # Proportionality gate. Confirmed drivers exist for almost every employee
    # — someone is always below the cohort mean on something — so driver
    # evidence alone is not grounds to intervene. Without this check the
    # planner proposed formal interventions for employees at ~1% risk, which
    # the Critic correctly flagged as over-intervention. Caught in testing;
    # see Improvement Changelog.
    if risk["risk_level"] == "Low":
        return {
            "agent": "intervention",
            "status": "NO_ACTION_PROPOSED",
            "reason": (
                f"Risk is Low ({risk['attrition_probability']:.1%}). Intervening here "
                "would not be proportionate and would consume manager attention "
                "needed by higher-risk cases."
            ),
            "options_considered": [],
            "recommendation": None,
        }

    actionable = root_cause["actionable_features"]

    # Build the allowed candidate set from confirmed drivers only.
    candidates = []
    for feat in actionable:
        for lever in FEATURE_LEVERS.get(feat, []):
            if lever not in candidates:
                candidates.append(lever)

    if not candidates:
        return {
            "agent": "intervention",
            "status": "NO_ACTION_PROPOSED",
            "reason": (
                "No confirmed driver has an available lever. Escalating for human "
                "review rather than proposing an unsupported intervention."
            ),
            "options_considered": [],
            "recommendation": None,
        }

    # Score each candidate with the counterfactual simulator.
    options = []
    for key in candidates:
        sim = toolbox.simulate_intervention(employee, key)
        options.append({
            "key": key,
            "label": INTERVENTION_CATALOG[key]["label"],
            "simulated_delta_pp": sim["delta_pp"],
            "targets_drivers": [f for f in actionable if key in FEATURE_LEVERS.get(f, [])],
        })

    # Most negative delta = largest simulated risk reduction.
    options.sort(key=lambda o: o["simulated_delta_pp"])
    best = options[0]

    rationale = reason_with_llm_or_fallback(
        task=(
            "In 2 sentences, justify choosing this intervention over the alternatives. "
            "Reference only the drivers and simulated deltas given. Do not claim the "
            "intervention will cause the reduction — it is a model what-if."
        ),
        context=(
            f"Confirmed drivers: {[c['label'] for c in root_cause['confirmed_drivers']]}\n"
            f"Options: {options}\nChosen: {best['label']}"
        ),
        fallback=(
            f"{best['label']} targets "
            f"{', '.join(best['targets_drivers'])} and shows the largest simulated "
            f"risk reduction ({best['simulated_delta_pp']:+.1f} pp) among available options."
        ),
    )

    return {
        "agent": "intervention",
        "status": "PROPOSED",
        "options_considered": options,
        "recommendation": {
            "key": best["key"],
            "label": best["label"],
            "targets_drivers": best["targets_drivers"],
            "simulated_delta_pp": best["simulated_delta_pp"],
            "caveat": "Simulated model what-if, not a causal effect estimate.",
        },
        "rationale": rationale,
    }
