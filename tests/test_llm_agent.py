"""
Tests for the genuine LLM tool-calling agent (advanced/orchestration/llm_agent.py).

Most of these require a real provider and skip otherwise — and that skip is
itself the point: this file is honest that it cannot verify model-driven
behaviour without a model. The deterministic pipeline's tests do NOT skip
without a key, because that pipeline does not need one.

Note that `tests/conftest.py` loads .env, so these run whenever a key is
configured anywhere the rest of the project would find one. Before that existed
they skipped even on a machine with a working provider, which made the suite
report less than it knew.
"""
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced import llm  # noqa: E402
from advanced.orchestration import llm_agent  # noqa: E402
from tests.conftest import requires_llm  # noqa: E402


# --- no-key behaviour: verifiable without a provider -----------------------

def test_llm_agent_refuses_to_run_without_a_provider(monkeypatch):
    """The whole point of this file: no rule-based fallback pretending to be an agent."""
    monkeypatch.setattr(llm, "PROVIDER", "none")
    monkeypatch.setattr(llm, "BASE_URL", "")
    monkeypatch.setattr(llm, "API_KEY", "")
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        llm_agent.run_llm_agent("EMP4")


def test_unknown_employee_fails_fast(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "BASE_URL", "http://localhost:1")
    monkeypatch.setattr(llm, "MODEL", "x")
    with pytest.raises(ValueError, match="not found"):
        llm_agent.run_llm_agent("NOT_A_REAL_ID")


def test_tool_bridge_returns_structured_errors_not_exceptions(toolbox):
    """
    A model that guesses a feature name is behaving normally. Raising would
    abort the whole investigation over a recoverable mistake, so every bad
    argument comes back as something the model can read and correct.
    """
    eid = toolbox.cohort["employee_id"].iloc[0]
    cases = [
        ("get_cohort_percentile", {"employee_id": eid, "feature": "Vibes"}, "unknown_feature"),
        ("simulate_intervention", {"employee_id": eid, "intervention_key": "nope"},
         "unknown_intervention"),
        ("predict_risk", {}, "missing_argument"),
        ("predict_risk", {"employee_id": "GHOST"}, "employee_not_found"),
        ("teleport", {"employee_id": eid}, "unknown_tool"),
    ]
    for name, args, expected in cases:
        out = llm_agent._execute_tool(toolbox, name, args)
        assert out.get("error") == expected, (name, args, out)


def test_tool_bridge_survives_non_dict_arguments(toolbox):
    assert llm_agent._execute_tool(toolbox, "predict_risk", None)["error"] == "bad_arguments"


def test_malformed_tool_arguments_parse_to_empty_dict():
    assert llm_agent._parse_args({"function": {"arguments": "{not json"}}) == {}
    assert llm_agent._parse_args({"function": {"arguments": "[1,2]"}}) == {}
    assert llm_agent._parse_args({"function": {"arguments": '{"a":1}'}}) == {"a": 1}


def test_system_prompt_states_the_thresholds_it_is_judged_against():
    """
    Measuring a model against rules it was never told is a prompt bug reported
    as a model failure. See Iteration 15 in CHANGELOG.md.
    """
    from advanced.agents.risk_analysis import HIGH_RISK_THRESHOLD
    prompt = llm_agent.SYSTEM_PROMPT.format(
        max_calls=12, min_contribution=0.4, max_severity=35.0,
        high=HIGH_RISK_THRESHOLD, medium=0.35, levers=llm_agent._lever_reference())
    assert f"{HIGH_RISK_THRESHOLD:.2f}" in prompt
    assert "WorkLifeBalance: workload_review" in prompt
    assert "NO INTERVENTION EXISTS for" in prompt


# --- live behaviour --------------------------------------------------------

def _completed(res):
    """
    A run the provider never finished tells us about the free tier, not the
    agent. Skip rather than fail — otherwise a rate limit reads as a behaviour
    regression and the suite stops meaning anything on a shared key.
    """
    if res["status"] in ("LLM_ERROR", "NO_FINALIZE_CALLED"):
        pytest.skip(f"provider did not complete the run ({res['status']})")
    return res


@requires_llm
def test_mode3_verifies_before_approving():
    res = _completed(llm_agent.run_llm_agent("EMP1180", verify=True))
    assert res["status"] in ("APPROVED_FOR_REVIEW", "ESCALATED", "NO_ACTION")
    if res["status"] == "APPROVED_FOR_REVIEW":
        assert res["verification"]["verdict"] == "VERIFIED"
        assert res["verification"]["verified_drivers"]


@requires_llm
def test_mode3_never_approves_an_unverified_claim():
    """The guarantee: an approved Mode 3 case has passed every check."""
    res = _completed(llm_agent.run_llm_agent("EMP1180", verify=True))
    if res["status"] == "APPROVED_FOR_REVIEW":
        assert res["verification"]["problems"] == []


@requires_llm
def test_mode2_respects_the_tool_budget():
    """
    Regression: the budget used to count model TURNS, so a model emitting four
    tool calls in one turn spent four and was charged one. An EMP1180 run
    advertised as 'max 8' executed 17.

    The toolbox log is asserted too, not just the counter — the counter agreeing
    with itself proves nothing, and the log was separately inflated by internal
    profile re-fetches until those were cached.
    """
    res = _completed(llm_agent.run_llm_agent("EMP1180", verify=False))
    assert res["tool_calls_used"] <= llm_agent.MAX_TOOL_CALLS
    assert len(res["raw_tool_log"]) <= llm_agent.MAX_TOOL_CALLS


@requires_llm
def test_mode3_respects_the_tool_budget():
    res = _completed(llm_agent.run_llm_agent("EMP1180", verify=True))
    assert res["tool_calls_used"] <= llm_agent.MAX_TOOL_CALLS
    assert len(res["raw_tool_log"]) <= llm_agent.MAX_TOOL_CALLS


@requires_llm
def test_agent_calls_predict_risk_before_deciding():
    res = _completed(llm_agent.run_llm_agent("EMP1180", verify=True))
    assert "predict_risk" in [c["tool"] for c in res["raw_tool_log"]]


@requires_llm
def test_nothing_is_executed_in_either_mode():
    allowed = {"APPROVED_FOR_REVIEW", "ESCALATED", "NO_ACTION",
               "COMPLETED", "LLM_ERROR", "NO_FINALIZE_CALLED"}
    for verify in (False, True):
        res = llm_agent.run_llm_agent("EMP4", verify=verify)
        assert res["status"] in allowed
        assert res["decision"] is None or "executed" not in str(res["decision"]).lower()


@requires_llm
def test_usage_is_accounted_per_run():
    res = _completed(llm_agent.run_llm_agent("EMP4", verify=True))
    u = res["usage"]
    assert u["llm_calls"] > 0 and u["total_tokens"] > 0
    assert u["estimated_cost_usd"] >= 0
