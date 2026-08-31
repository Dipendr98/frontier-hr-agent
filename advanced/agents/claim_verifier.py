"""
Agent 7: Claim Verifier — the check that makes Mode 3 different from Mode 2.

Mode 2 hands the model tools and lets it decide everything, including what to
conclude. That is genuinely agentic and it is also unaccountable: the model's
`finalize` call is accepted verbatim, so a confident summary and a correct one
are indistinguishable to everything downstream.

This agent closes that. It takes the model's finalize payload as an ALLEGATION
and re-derives every part of it from the tool layer, using the same evidence
floors the deterministic Mode 1 pipeline applies. Nothing the model said is
trusted; the tools are called again and the answers must agree.

Six decidable checks. Every one is a comparison between a claim and a number:

  V1 risk_level_mismatch      claimed band != band from predict_risk
  V2 unverified_driver        claimed driver fails the contribution or the
                              cohort-severity floor when recomputed
  V3 unknown_intervention     recommended key is not in the catalogue
  V4 no_mechanism             recommended intervention has no lever on any
                              VERIFIED driver
  V5 no_simulated_benefit     re-simulating the recommendation gives delta >= 0
  V6 action_without_evidence  an intervention is recommended with no driver
                              surviving verification
  V7 unverifiable_rationale   the free-text rationale states a number about a
                              feature that the employee's record contradicts

V7 exists because V1-V6 check the STRUCTURED payload, and the reviewer does not
read the structured payload — they read the rationale. An agent whose fields are
all correct and whose sentence says "job satisfaction of 2" about someone who
scored 4 has still misinformed the person making the decision. Verifying the
struct and shipping the prose unchecked verifies the part nobody looks at.

Why re-derive rather than ask the model to double-check itself: a model asked
to review its own output tends to agree with it, and the resulting loop looks
like verification while functioning as agreement — the failure this project's
Critic was built to avoid. See the Hot Take in CHANGELOG.md.

Note the division of labour. The Critic (agent 5) checks a plan the
DETERMINISTIC agents built, where the inputs are already structured and
trustworthy. This agent checks a plan a LANGUAGE MODEL asserted, so it must
first establish whether the inputs are true at all. Same philosophy, different
threat model.
"""
import re

from advanced.agents.risk_analysis import classify
from advanced.agents.root_cause import MAX_SEVERITY_PERCENTILE, MIN_CONTRIBUTION
from advanced.tools import FEATURE_LEVERS, FEATURE_META, INTERVENTION_CATALOG, evidence_statement

# Words that mean the following number is a statistic ABOUT the feature rather
# than the employee's value FOR it. "Overtime is at the 70th percentile" is not
# a claim that overtime equals 70.
_STAT_WORDS = (
    r"percentile|logit|contribution|contributes|delta|pp\b|percent|%|"
    r"mean|median|average|cohort|risk|probability|score|floor|threshold|rank"
)
# "<feature> is 2", "<feature> of 3", "<feature> = 1", "<feature> at 4".
_VALUE_AFTER = re.compile(
    r"^\W{0,3}(?:is|of|at|=|:|was|scores?|rated?|value)\W{0,3}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_TOLERANCE = 0.051


def unverifiable_numeric_claims(text: str, employee: dict, features: list) -> list:
    """
    Find statements in free text that assert a feature value the record contradicts.

    Deliberately conservative. It only fires on an explicit "<feature> is <number>"
    construction, and it stays silent when the number is qualified by a statistic
    word or when the feature is never named. A verifier that false-positives sends
    the model back to correct a claim that was fine, which burns budget and
    teaches it to distrust true statements — so when this is unsure, it says
    nothing. Under-reporting here is recoverable; over-reporting is not.

    Returns [(feature, stated_value, actual_value), ...].
    """
    if not text:
        return []
    found, seen = [], set()
    for feat in features:
        if feat in seen or feat not in employee:
            continue
        label = FEATURE_META.get(feat, {}).get("label", feat)
        for alias in {feat, label}:
            for m in re.finditer(re.escape(alias), text, re.IGNORECASE):
                tail = text[m.end():m.end() + 40]
                vm = _VALUE_AFTER.match(tail)
                if not vm:
                    continue
                # Skip when the number is a statistic about the feature.
                after_number = tail[vm.end():vm.end() + 18]
                if re.match(rf"\s*(?:{_STAT_WORDS})", after_number, re.IGNORECASE):
                    continue
                stated = float(vm.group(1))
                try:
                    actual = float(employee[feat])
                except (TypeError, ValueError):
                    continue
                if abs(stated - actual) > _TOLERANCE:
                    found.append((feat, stated, actual))
                    seen.add(feat)
                break
            if feat in seen:
                break
    return found


def verify(employee: dict, claim: dict, toolbox) -> dict:
    """
    Re-derive the model's finalize payload from the tools.

    `claim` is the raw finalize arguments. Returns a verdict plus the evidence
    that survived, so an approved Mode 3 case carries recomputed statements
    rather than the model's own prose.
    """
    problems = []
    claimed_drivers = [d for d in (claim.get("confirmed_drivers") or []) if isinstance(d, str)]
    recommended = (claim.get("recommended_intervention") or "").strip() or None

    # --- V1: the risk band is not the model's to decide -------------------
    truth = toolbox.predict_risk(employee)
    actual_prob = truth["attrition_probability"]
    actual_level = classify(actual_prob)
    claimed_level = (claim.get("risk_level") or "").strip().title()
    if claimed_level != actual_level:
        problems.append(
            f"risk_level_mismatch: you stated {claimed_level or 'nothing'} but "
            f"predict_risk gives {actual_prob:.1%}, which is {actual_level}."
        )

    # --- V2: each claimed driver must clear both evidence floors ----------
    drivers = {d["feature"]: d for d in
               toolbox.get_risk_drivers(employee, top_n=len(toolbox.features))}
    verified, rejected = [], []
    for feat in claimed_drivers:
        d = drivers.get(feat)
        if d is None:
            rejected.append({"feature": feat, "reason": "unknown_feature"})
            problems.append(
                f"unverified_driver: '{feat}' is not a feature of this model.")
            continue
        if d["logit_contribution"] < MIN_CONTRIBUTION:
            rejected.append({"feature": feat, "reason": "contribution_below_floor",
                             "logit_contribution": d["logit_contribution"]})
            problems.append(
                f"unverified_driver: {d['label']} contributes only "
                f"{d['logit_contribution']:+.2f} to the logit, below the "
                f"{MIN_CONTRIBUTION} floor required to call it a driver."
            )
            continue
        pct = toolbox.get_cohort_percentile(feat, d["value"])
        if pct.get("error") or pct["severity_percentile"] > MAX_SEVERITY_PERCENTILE:
            sev = pct.get("severity_percentile")
            rejected.append({"feature": feat, "reason": "not_unusual_vs_cohort",
                             "severity_percentile": sev})
            problems.append(
                f"unverified_driver: {d['label']} is not unusual for this cohort "
                f"(worst {sev:.0f}% threshold is {MAX_SEVERITY_PERCENTILE:.0f}%); "
                f"a feature everyone shares is not this person's driver."
            )
            continue
        verified.append({
            "feature": feat,
            "label": d["label"],
            "value": d["value"],
            "cohort_mean": d["cohort_mean"],
            "percentile": pct["percentile"],
            "severity_percentile": pct["severity_percentile"],
            "has_lever": bool(FEATURE_LEVERS.get(feat)),
            "logit_contribution": d["logit_contribution"],
            "statement": evidence_statement(
                d["label"], d["value"], d["cohort_mean"], pct["percentile"],
                pct["severity_percentile"], d["logit_contribution"]),
        })

    verified_features = {v["feature"] for v in verified}
    simulated = None

    if recommended:
        # --- V3: the intervention must exist ------------------------------
        if recommended not in INTERVENTION_CATALOG:
            problems.append(
                f"unknown_intervention: '{recommended}' is not in the catalogue. "
                f"Valid keys: {sorted(INTERVENTION_CATALOG)}."
            )
        else:
            # --- V4: it must have a lever on something that survived ------
            movable = {f for f in verified_features if recommended in FEATURE_LEVERS.get(f, [])}
            if not movable:
                problems.append(
                    f"no_mechanism: '{recommended}' has no lever on any verified "
                    f"driver {sorted(verified_features) or '[]'}. It cannot move "
                    f"what you cited."
                )

            # --- V5: re-simulate; the benefit is not the model's word -----
            sim = toolbox.simulate_intervention(employee, recommended)
            if "error" not in sim:
                simulated = sim
                if sim["delta_pp"] >= 0:
                    problems.append(
                        f"no_simulated_benefit: re-simulating '{recommended}' gives "
                        f"{sim['delta_pp']:+.1f} pp, i.e. no modelled improvement."
                    )

        # --- V6: no action on evidence that did not survive ---------------
        if not verified:
            problems.append(
                "action_without_evidence: you recommended an intervention but no "
                "driver you cited survived verification."
            )

    # --- V7: the sentence the reviewer reads must also be true ------------
    bad_claims = unverifiable_numeric_claims(claim.get("rationale") or "", employee,
                                             toolbox.features)
    for feat, stated, actual in bad_claims:
        problems.append(
            f"unverifiable_rationale: your rationale says {feat} is {stated:g}, "
            f"but this employee's record has {actual:g}. Cite the record."
        )

    return {
        "agent": "claim_verifier",
        "verdict": "VERIFIED" if not problems else "REJECTED",
        "problems": problems,
        "checks_run": ["V1", "V2", "V3", "V4", "V5", "V6", "V7"],
        "recomputed_risk": {"attrition_probability": actual_prob,
                            "risk_level": actual_level},
        "verified_drivers": verified,
        "rejected_claims": rejected,
        "claimed_drivers": claimed_drivers,
        "recommended_intervention": recommended,
        "simulated": simulated,
    }


def objection_prompt(result: dict) -> str:
    """Turn a rejection into a correction the model can act on, not a scolding."""
    lines = [
        "Your finalize call was checked against the tools and REJECTED. "
        "Every claim below was recomputed from the model and the cohort — these "
        "are not opinions about your answer, they are measurements that "
        "disagree with it.",
        "",
    ]
    lines += [f"  - {p}" for p in result["problems"]]
    lines += [
        "",
        f"Recomputed risk: {result['recomputed_risk']['attrition_probability']:.1%} "
        f"({result['recomputed_risk']['risk_level']}).",
    ]
    if result["verified_drivers"]:
        lines.append("Drivers that DID survive verification: "
                     + ", ".join(v["feature"] for v in result["verified_drivers"]))
    else:
        lines.append("No driver you cited survived verification.")
    lines += [
        "",
        "Investigate further with the tools if you need to, then call finalize "
        "again with only claims you can support. If nothing survives and the "
        "risk is not Low, finalize with an empty confirmed_drivers list and no "
        "intervention — escalating an unexplained case to a human is a correct "
        "answer here, and a better one than an unsupported recommendation.",
    ]
    return "\n".join(lines)
