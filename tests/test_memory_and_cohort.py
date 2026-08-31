"""
Tests for cross-case memory and cohort triage.

The load-bearing assertion in this file is `test_memory_never_changes_a_decision`.
Memory that can quietly alter who gets escalated is memory you cannot audit, and
it would break the reproducibility claim: the headline evaluation runs with
memory off, so if memory changed decisions the reported numbers would describe a
system nobody actually runs.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pytest  # noqa: E402

from advanced.memory import (COMMON_DRIVER_SHARE, MIN_CASES_FOR_PREVALENCE,  # noqa: E402
                             SYSTEMIC_SHARE, CohortMemory)
from advanced.orchestration.cohort import run_cohort  # noqa: E402
from advanced.orchestration.workflow import run_for_employee  # noqa: E402


@pytest.fixture
def memory(tmp_path):
    return CohortMemory(path=str(tmp_path / "mem.json"), autoload=False)


def _record(mem, eid, level, drivers, key, status="APPROVED_FOR_REVIEW"):
    mem.record(eid, {"risk_level": level, "attrition_probability": 0.7},
               {"confirmed_drivers": [{"feature": d, "label": d} for d in drivers],
                "actionable_features": drivers},
               {"recommendation": {"key": key} if key else None}, status)


# --- the boundary that makes memory safe -----------------------------------

def test_memory_never_changes_a_decision(toolbox, cohort_df, memory):
    """Same employees, memory on and off — identical terminal states."""
    employees = [r.to_dict() for _, r in cohort_df.head(40).iterrows()]
    without = [run_for_employee(e, toolbox, save=False)["status"] for e in employees]
    for e in employees:                       # warm the store
        run_for_employee(e, toolbox, save=False, memory=memory)
    with_mem = [run_for_employee(e, toolbox, save=False, memory=memory)["status"]
                for e in employees]
    assert without == with_mem


def test_memory_annotations_are_labelled_as_annotations(toolbox, cohort_df, memory):
    employees = [r.to_dict() for _, r in cohort_df.head(30).iterrows()]
    for e in employees:
        run_for_employee(e, toolbox, save=False, memory=memory)
    res = run_for_employee(employees[0], toolbox, save=False, memory=memory)
    step = next(s for s in res["trajectory"] if s["step"] == "cohort_memory")
    assert "does not affect status" in step["note"]


# --- prevalence -------------------------------------------------------------

def test_no_prevalence_claims_below_the_minimum_sample(memory):
    """A 'rare driver' computed from four people is not a finding."""
    for i in range(MIN_CASES_FOR_PREVALENCE - 1):
        _record(memory, f"E{i}", "High", ["WorkLifeBalance"], "workload_review")
    confirmed = [{"feature": "WorkLifeBalance", "label": "Work-life balance"}]
    assert memory.contextualise_drivers(confirmed) == []
    assert memory.systemic_findings() == []


def test_a_universal_driver_is_reported_as_environmental(memory):
    n = MIN_CASES_FOR_PREVALENCE + 10
    for i in range(n):
        _record(memory, f"E{i}", "High", ["JobLevel"], None)
    notes = memory.contextualise_drivers([{"feature": "JobLevel", "label": "Job level"}])
    assert notes and notes[0]["kind"] == "environmental"
    assert notes[0]["cohort_share"] >= COMMON_DRIVER_SHARE


def test_a_rare_driver_is_reported_as_distinguishing(memory):
    n = MIN_CASES_FOR_PREVALENCE + 30
    for i in range(n):
        _record(memory, f"E{i}", "High", ["JobLevel"], None)
    _record(memory, "RARE", "High", ["WorkLifeBalance"], "workload_review")
    notes = memory.contextualise_drivers(
        [{"feature": "WorkLifeBalance", "label": "Work-life balance"}])
    assert notes and notes[0]["kind"] == "distinguishing"


# --- systemic findings ------------------------------------------------------

def test_concentration_is_measured_over_actioned_cases_not_everyone(memory):
    """
    Ten actioned employees all getting the same plan is 100% concentration,
    whether or not ninety Low-risk people were also reviewed. Diluting by the
    people you are not acting on hides exactly the pattern being looked for.
    """
    for i in range(10):
        _record(memory, f"A{i}", "High", ["OverTime_flag"], "workload_review")
    for i in range(90):
        _record(memory, f"L{i}", "Low", [], None, status="NO_ACTION")
    assert memory.n_cases == 100
    assert memory.n_actioned == 10
    assert memory.intervention_concentration()["workload_review"] == 1.0
    finding = memory.systemic_findings()[0]
    assert finding["share_of_actioned"] == 1.0
    assert finding["affected_count"] == 10


def test_no_systemic_finding_when_interventions_are_spread_out(memory):
    keys = ["workload_review", "assign_mentor", "team_environment_review",
            "compensation_review", "ownership_assignment"]
    for i in range(MIN_CASES_FOR_PREVALENCE + 10):
        _record(memory, f"E{i}", "High", ["JobSatisfaction"], keys[i % len(keys)])
    assert all(f["share_of_actioned"] >= SYSTEMIC_SHARE
               for f in memory.systemic_findings())


def test_recording_the_same_employee_twice_does_not_double_count(memory):
    for _ in range(5):
        _record(memory, "EMP1", "High", ["OverTime_flag"], "workload_review")
    assert memory.n_cases == 1


# --- precedent --------------------------------------------------------------

def test_precedent_flags_inconsistent_treatment_of_comparable_cases(memory):
    _record(memory, "E1", "High", ["OverTime_flag"], "workload_review",
            status="APPROVED_FOR_REVIEW")
    _record(memory, "E2", "High", ["OverTime_flag"], "workload_review",
            status="ESCALATED")
    p = memory.precedent_for("E3", "High", ["OverTime_flag"])
    assert p["comparable_cases"] == 2
    assert p["inconsistent"] is True


def test_precedent_excludes_the_employee_being_reviewed(memory):
    _record(memory, "E1", "High", ["OverTime_flag"], "workload_review")
    assert memory.precedent_for("E1", "High", ["OverTime_flag"]) == {}


# --- persistence ------------------------------------------------------------

def test_memory_survives_a_round_trip(memory):
    _record(memory, "E1", "High", ["OverTime_flag"], "workload_review")
    memory.save()
    assert CohortMemory(path=memory.path).n_cases == 1


def test_a_corrupt_store_does_not_take_the_pipeline_down(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json at all")
    assert CohortMemory(path=str(path)).n_cases == 0


# --- cohort triage ----------------------------------------------------------

def test_cohort_run_produces_a_worklist_and_no_executed_action():
    report = run_cohort(limit=40, use_memory=True, save=False)
    allowed = {"APPROVED_FOR_REVIEW", "ESCALATED", "NO_ACTION", "HALTED_DATA_QUALITY"}
    assert report["n_employees"] == 40
    assert set(report["status_counts"]) <= allowed
    assert len(report["worklist"]) == 40


def test_worklist_puts_escalations_first():
    report = run_cohort(limit=120, use_memory=True, save=False)
    order = [r["status"] for r in report["worklist"]]
    if "ESCALATED" in order:
        assert order.index("ESCALATED") < order.index("NO_ACTION")


def test_cohort_never_calls_the_llm_unless_asked(monkeypatch):
    """
    Regression, and the expensive kind.

    The narration switch used to live in this module's CLI `main()`, so the CLI
    was fast and every other caller — the Streamlit page — silently took the
    slow path: two passes over 342 employees at two completions each, about 900
    throttled API calls and an hour instead of two seconds. It looked like the
    triage was broken. It was working, 900 times.

    A default that only holds for one entry point is not a default, so this
    asserts it at the function that does the work.
    """
    from advanced import llm

    calls = []
    monkeypatch.setattr(llm, "chat",
                        lambda *a, **k: calls.append(1) or {"content": "x"})
    monkeypatch.setattr(llm, "is_enabled", lambda: True)
    monkeypatch.setattr(llm, "NARRATION", True)   # as a fresh process sees it

    run_cohort(limit=15, use_memory=True, save=False)
    assert calls == [], f"cohort triage made {len(calls)} LLM calls uninvited"


def test_cohort_restores_the_narration_setting_it_found(monkeypatch):
    """Toggling a global and leaving it toggled would break the next caller."""
    from advanced import llm
    monkeypatch.setattr(llm, "NARRATION", True)
    run_cohort(limit=5, use_memory=False, save=False)
    assert llm.NARRATION is True


def test_cohort_is_deterministic_across_runs():
    a = run_cohort(limit=60, use_memory=True, save=False)
    b = run_cohort(limit=60, use_memory=True, save=False)
    assert ([r["employee_id"] for r in a["worklist"]]
            == [r["employee_id"] for r in b["worklist"]])
    assert a["status_counts"] == b["status_counts"]
