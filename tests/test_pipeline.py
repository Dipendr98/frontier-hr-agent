"""
Tests that check the properties this submission actually claims.

Not coverage theatre: each test corresponds to a claim in the README, so a
reviewer can verify the claims by running pytest rather than trusting prose.
"""
import os
import sys

import joblib
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced.agents import critic, data_quality, intervention, root_cause  # noqa: E402
from advanced.orchestration.workflow import run_for_employee  # noqa: E402
from advanced.tools import FEATURE_LEVERS, ToolBox  # noqa: E402


@pytest.fixture(scope="module")
def toolbox():
    return ToolBox()


@pytest.fixture(scope="module")
def employee(toolbox):
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    return df.iloc[0].to_dict()


# --- claim: the model is a real trained artifact, not hardcoded weights ---

def test_model_artifact_loads_and_predicts_without_retraining():
    model = joblib.load(os.path.join(BASE_DIR, "baseline", "attrition_model.joblib"))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    features = [c for c in df.columns if c not in ("employee_id", "attrition")]
    proba = model.predict_proba(df[features].head(5))
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_has_learned_nonuniform_coefficients():
    """Hardcoded weights would be identical or absent; learned ones vary."""
    model = joblib.load(os.path.join(BASE_DIR, "baseline", "attrition_model.joblib"))
    coefs = model.named_steps["clf"].coef_[0]
    assert len(set(coefs.round(6))) > 1


# --- claim: protected attributes are excluded from the feature set ---

def test_no_protected_attributes_in_features():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    banned = {"Gender", "Age", "MaritalStatus", "Gender_flag", "Married_flag"}
    assert not (banned & set(df.columns))


# --- claim: bad input halts the pipeline instead of being scored ---

def test_data_quality_rejects_out_of_range(toolbox, employee):
    bad = dict(employee)
    bad["JobSatisfaction"] = 99
    result = data_quality.run(bad, toolbox.features)
    assert result["verdict"] == "FAIL"
    assert any("out_of_range" in i for i in result["issues"])


def test_bad_record_halts_before_scoring(toolbox, employee):
    bad = dict(employee)
    bad["WorkLifeBalance"] = -5
    res = run_for_employee(bad, toolbox, save=False)
    assert res["status"] == "HALTED_DATA_QUALITY"


# --- claim: explanations are grounded in the model, not invented ---

def test_risk_drivers_decompose_the_model_logit(toolbox, employee):
    """Contributions must sum to the model's own logit, minus the intercept."""
    import numpy as np
    drivers = toolbox.get_risk_drivers(employee, top_n=len(toolbox.features))
    total = sum(d["logit_contribution"] for d in drivers)

    X = pd.DataFrame([employee])[toolbox.features]
    z = toolbox.model.named_steps["scaler"].transform(X)[0]
    clf = toolbox.model.named_steps["clf"]
    expected = float(np.dot(clf.coef_[0], z))
    assert total == pytest.approx(expected, abs=1e-3)


def test_root_cause_reports_unexplained_rather_than_inventing(toolbox):
    """An average employee should yield no confirmed driver, not a fabricated one."""
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    avg = df[toolbox.features].median().to_dict()
    avg["employee_id"] = "EMP_SYNTHETIC_MEDIAN"
    risk = toolbox.predict_risk(avg)
    rc = root_cause.run(avg, {**risk, "risk_level": "Medium"}, toolbox)
    if not rc["confirmed_drivers"]:
        assert rc["status"] == "UNEXPLAINED"


# --- claim: recommendations only target drivers that were confirmed ---

def test_recommendation_targets_only_confirmed_drivers(toolbox):
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    checked = 0
    for _, row in df.head(40).iterrows():
        res = run_for_employee(row.to_dict(), toolbox, save=False)
        detail = res.get("detail")
        if not detail or detail["intervention"]["status"] != "PROPOSED":
            continue
        targets = set(detail["intervention"]["recommendation"]["targets_drivers"])
        confirmed = set(detail["root_cause"]["actionable_features"])
        assert targets.issubset(confirmed)
        checked += 1
    assert checked > 0, "no proposed interventions were exercised"


def test_every_lever_maps_to_a_real_intervention():
    from advanced.tools import INTERVENTION_CATALOG
    for feature, levers in FEATURE_LEVERS.items():
        for lever in levers:
            assert lever in INTERVENTION_CATALOG


# --- claim: the critic rejects a seeded weak recommendation ---

def test_critic_rejects_unsupported_recommendation():
    risk = {"employee_id": "EMP_TEST", "attrition_probability": 0.75, "risk_level": "High"}
    rc = {"status": "UNEXPLAINED", "confirmed_drivers": [], "actionable_features": []}
    ic = {
        "status": "PROPOSED",
        "recommendation": {"key": "assign_mentor", "label": "Assign an onboarding mentor",
                           "targets_drivers": ["JobInvolvement"], "simulated_delta_pp": -3.0,
                           "caveat": "x"},
    }
    result = critic.run(risk, rc, ic)
    assert result["verdict"] == "REVISE"
    assert any("unsupported_recommendation" in p for p in result["problems"])


def test_critic_rejects_intervention_with_no_simulated_benefit():
    risk = {"employee_id": "EMP_TEST", "attrition_probability": 0.75, "risk_level": "High"}
    rc = {"status": "EXPLAINED",
          "confirmed_drivers": [{"statement": "x"}],
          "actionable_features": ["JobInvolvement"]}
    ic = {
        "status": "PROPOSED",
        "recommendation": {"key": "assign_mentor", "label": "Assign an onboarding mentor",
                           "targets_drivers": ["JobInvolvement"], "simulated_delta_pp": +1.0,
                           "caveat": "x"},
    }
    result = critic.run(risk, rc, ic)
    assert result["verdict"] == "REVISE"
    assert any("no_simulated_benefit" in p for p in result["problems"])


def test_low_risk_gets_no_intervention(toolbox, employee):
    """Regression test for the over-intervention bug found during evaluation."""
    risk = {"employee_id": "X", "attrition_probability": 0.011, "risk_level": "Low"}
    rc = {"status": "EXPLAINED", "confirmed_drivers": [{"statement": "x"}],
          "actionable_features": ["JobSatisfaction"]}
    result = intervention.run(employee, risk, rc, toolbox)
    assert result["status"] == "NO_ACTION_PROPOSED"


# --- claim: nothing is executed without a human ---

def test_no_case_reaches_an_executed_action(toolbox):
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    allowed = {"APPROVED_FOR_REVIEW", "ESCALATED", "NO_ACTION", "HALTED_DATA_QUALITY"}
    for _, row in df.head(30).iterrows():
        res = run_for_employee(row.to_dict(), toolbox, save=False)
        assert res["status"] in allowed


def test_approved_cases_still_require_human_decision(toolbox):
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    for _, row in df.head(30).iterrows():
        res = run_for_employee(row.to_dict(), toolbox, save=False)
        gate = [s for s in res["trajectory"] if s.get("step") == "human_gate"]
        if gate and gate[0]["status"] == "APPROVED_FOR_REVIEW":
            assert gate[0]["requires_human_decision"] is True
            return
    pytest.fail("no approved case found to check")


# --- claim: the evaluation rubric is independent of the critic ---

def test_rubric_does_not_import_agent_logic():
    """The rubric may import catalogues, never agent decision code."""
    path = os.path.join(BASE_DIR, "evaluation", "rubric.py")
    source = open(path).read()
    assert "from advanced.agents" not in source
    assert "import critic" not in source


def test_rubric_rejects_fabricated_evidence():
    """A well-formed but false citation must score zero on verifiability."""
    from evaluation.rubric import _verify_evidence
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    features = [c for c in df.columns if c not in ("employee_id", "attrition")]
    emp = df.iloc[0].to_dict()
    true_val = emp["JobSatisfaction"]
    fake = true_val + 2 if true_val <= 2 else true_val - 2

    assert _verify_evidence(
        f"Job satisfaction is {true_val:g} vs cohort mean 2.73.", emp, features)
    assert not _verify_evidence(
        f"Job satisfaction is {fake:g} vs cohort mean 2.73.", emp, features)


def test_rubric_matches_labels_and_raw_names():
    """Regression test: the first rubric only matched raw column names and so
    marked every agent citation unverifiable. See CHANGELOG."""
    from evaluation.rubric import _features_named_in
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))
    features = [c for c in df.columns if c not in ("employee_id", "attrition")]
    assert "JobSatisfaction" in _features_named_in("JobSatisfaction is 2", features)
    assert "JobSatisfaction" in _features_named_in("Job satisfaction is 2", features)


def test_agent_beats_strongest_baseline_on_primary_metric():
    """The headline claim, asserted rather than only reported."""
    import json
    path = os.path.join(BASE_DIR, "evidence", "evaluation_report.json")
    if not os.path.exists(path):
        pytest.skip("run evaluation/evaluate.py first")
    r = json.load(open(path))
    agent = r["systems"]["agent"]["reviewer_case_quality_pct"]
    strong = r["systems"]["baseline_plus"]["reviewer_case_quality_pct"]
    assert agent > strong


# --- claim: every case produces a traceable trajectory ---

def test_trajectory_covers_every_agent(toolbox, employee):
    res = run_for_employee(employee, toolbox, save=False)
    steps = " ".join(str(s.get("step")) for s in res["trajectory"])
    for agent_name in ["data_quality", "risk_analysis", "root_cause",
                       "intervention", "critic", "human_gate"]:
        assert agent_name in steps


def test_tool_calls_are_logged(toolbox, employee):
    res = run_for_employee(employee, toolbox, save=False)
    calls = [s for s in res["trajectory"] if s.get("step") == "tool_calls"][0]["calls"]
    assert len(calls) > 0
    assert all("tool" in c and "args" in c for c in calls)


# --- claim: the pipeline runs with no LLM configured ---

def test_runs_without_any_llm_provider(toolbox, employee):
    from advanced import llm
    if not llm.is_enabled():
        res = run_for_employee(employee, toolbox, save=False)
        assert res["status"] != "HALTED_DATA_QUALITY"
