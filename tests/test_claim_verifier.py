"""
Tests for the Mode 3 claim verifier.

These are seeded-failure tests: each one hands the verifier a claim that is
wrong in exactly one way and asserts it is caught by the intended check. A
verifier is only worth having if it fails things, so the tests are mostly about
what it rejects rather than what it accepts.

The last group is the other half of the contract, and matters just as much: the
verifier must NOT reject a true claim. A false positive sends the model back to
correct something that was right, which costs budget and, over a loop, trains
it away from correct answers.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pytest  # noqa: E402

from advanced.agents import claim_verifier  # noqa: E402
from advanced.agents.risk_analysis import classify  # noqa: E402


@pytest.fixture(scope="module")
def high_risk_employee(toolbox, cohort_df):
    """An employee the model scores High, with at least one confirmable driver."""
    for _, row in cohort_df.iterrows():
        emp = row.to_dict()
        if toolbox.predict_risk(emp)["attrition_probability"] >= 0.6:
            return emp
    pytest.skip("no high-risk employee in the cohort")


def _truthful_claim(employee, toolbox):
    """Build the claim a perfect agent would make for this employee."""
    from advanced.agents import risk_analysis, root_cause
    risk = risk_analysis.run(employee, toolbox)
    rc = root_cause.run(employee, risk, toolbox)
    key = None
    if rc["actionable_features"]:
        from advanced.tools import FEATURE_LEVERS
        for feat in rc["actionable_features"]:
            for lever in FEATURE_LEVERS[feat]:
                if toolbox.simulate_intervention(employee, lever)["delta_pp"] < 0:
                    key = lever
                    break
            if key:
                break
    return {
        "risk_level": risk["risk_level"],
        "confirmed_drivers": [c["feature"] for c in rc["confirmed_drivers"]],
        "recommended_intervention": key or "",
        "rationale": "Drivers confirmed against the cohort.",
    }


# --- V1: the model does not get to decide the risk band --------------------

def test_v1_catches_wrong_risk_band(high_risk_employee, toolbox):
    claim = _truthful_claim(high_risk_employee, toolbox)
    claim["risk_level"] = "Low"
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert result["verdict"] == "REJECTED"
    assert any("risk_level_mismatch" in p for p in result["problems"])


def test_v1_recomputes_the_band_itself(high_risk_employee, toolbox):
    result = claim_verifier.verify(high_risk_employee, {"risk_level": "High",
                                                        "confirmed_drivers": []}, toolbox)
    prob = result["recomputed_risk"]["attrition_probability"]
    assert result["recomputed_risk"]["risk_level"] == classify(prob)


# --- V2: claimed drivers must clear the same floors Mode 1 applies ---------

def test_v2_rejects_a_driver_that_is_not_a_model_feature(high_risk_employee, toolbox):
    claim = {"risk_level": classify(toolbox.predict_risk(high_risk_employee)
                                    ["attrition_probability"]),
             "confirmed_drivers": ["Astrology"], "rationale": "x"}
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert any("unverified_driver" in p for p in result["problems"])


def test_v2_rejects_a_feature_the_employee_is_not_unusual_on(toolbox, cohort_df):
    """
    The failure the deterministic pipeline had for its whole life: citing a
    feature that is technically positive but ordinary for this cohort.
    """
    emp = cohort_df.iloc[0].to_dict()
    drivers = toolbox.get_risk_drivers(emp, top_n=len(toolbox.features))
    ordinary = None
    for d in drivers:
        pct = toolbox.get_cohort_percentile(d["feature"], d["value"])
        if pct.get("severity_percentile", 0) > claim_verifier.MAX_SEVERITY_PERCENTILE:
            ordinary = d["feature"]
            break
    if ordinary is None:
        pytest.skip("this employee is unusual on everything")
    claim = {"risk_level": classify(toolbox.predict_risk(emp)["attrition_probability"]),
             "confirmed_drivers": [ordinary], "rationale": "x"}
    result = claim_verifier.verify(emp, claim, toolbox)
    assert any("unverified_driver" in p for p in result["problems"])


# --- V3/V4/V5: the recommendation must be real, relevant and beneficial ----

def test_v3_rejects_an_invented_intervention(high_risk_employee, toolbox):
    claim = _truthful_claim(high_risk_employee, toolbox)
    claim["recommended_intervention"] = "send_a_fruit_basket"
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert any("unknown_intervention" in p for p in result["problems"])


def test_v4_rejects_an_intervention_with_no_lever_on_a_verified_driver(toolbox, cohort_df):
    """The exact error observed in a real Mode 2 run: right catalogue key, wrong problem."""
    emp = cohort_df.iloc[0].to_dict()
    claim = {"risk_level": classify(toolbox.predict_risk(emp)["attrition_probability"]),
             "confirmed_drivers": ["StockOptionLevel"],
             "recommended_intervention": "structured_training_time",
             "rationale": "x"}
    result = claim_verifier.verify(emp, claim, toolbox)
    assert any("no_mechanism" in p or "unverified_driver" in p
               for p in result["problems"])


# --- V6: no action without evidence that survived --------------------------

def test_v6_rejects_action_when_nothing_survived(high_risk_employee, toolbox):
    claim = {"risk_level": classify(toolbox.predict_risk(high_risk_employee)
                                    ["attrition_probability"]),
             "confirmed_drivers": [],
             "recommended_intervention": "assign_mentor",
             "rationale": "Felt right."}
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert any("action_without_evidence" in p for p in result["problems"])


# --- V7: the sentence the reviewer reads is checked too --------------------

def test_v7_catches_a_false_number_in_the_rationale(high_risk_employee, toolbox):
    claim = _truthful_claim(high_risk_employee, toolbox)
    actual = float(high_risk_employee["JobSatisfaction"])
    claim["rationale"] = f"Job satisfaction is {actual + 2:g}, which is concerning."
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert any("unverifiable_rationale" in p for p in result["problems"])


def test_v7_does_not_fire_on_statistics_about_a_feature(high_risk_employee, toolbox):
    """'overtime at the 70th percentile' is not a claim that overtime equals 70."""
    bad = claim_verifier.unverifiable_numeric_claims(
        "Working overtime sits at the 70th percentile and contributes +1.21 to "
        "the logit, a delta of -7.1 pp.", high_risk_employee, list(toolbox.features))
    assert bad == []


def test_v7_does_not_fire_on_a_true_statement(high_risk_employee, toolbox):
    actual = float(high_risk_employee["WorkLifeBalance"])
    bad = claim_verifier.unverifiable_numeric_claims(
        f"Work-life balance is {actual:g}.", high_risk_employee, list(toolbox.features))
    assert bad == []


# --- the other half of the contract: do not reject the truth ---------------

def test_a_truthful_claim_is_verified(high_risk_employee, toolbox):
    claim = _truthful_claim(high_risk_employee, toolbox)
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert result["verdict"] == "VERIFIED", result["problems"]


def test_verified_statements_match_the_deterministic_agent(high_risk_employee, toolbox):
    """
    Mode 1 and Mode 3 must cite a driver identically. They use one shared
    formatter precisely so the rubric cannot score the same fact differently
    depending on which mode produced it.
    """
    from advanced.agents import risk_analysis, root_cause
    risk = risk_analysis.run(high_risk_employee, toolbox)
    rc = root_cause.run(high_risk_employee, risk, toolbox)
    claim = _truthful_claim(high_risk_employee, toolbox)
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    assert ({c["statement"] for c in rc["confirmed_drivers"]}
            == {v["statement"] for v in result["verified_drivers"]})


def test_objection_prompt_names_every_problem(high_risk_employee, toolbox):
    claim = _truthful_claim(high_risk_employee, toolbox)
    claim["risk_level"] = "Low"
    result = claim_verifier.verify(high_risk_employee, claim, toolbox)
    text = claim_verifier.objection_prompt(result)
    for p in result["problems"]:
        assert p in text
    assert "escalating an unexplained case to a human is a correct answer" in text
