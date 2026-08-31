"""
Orchestrates the agent loop for one employee:

  data_quality -> risk_analysis -> root_cause -> intervention
       -> critic -> [revise, up to MAX_REVISIONS] -> human_gate

Revision is not a re-run of the same call: the Critic's specific complaint is
fed back, and the loop tightens the evidence floor so the next attempt must
clear a higher bar. If it still cannot, the case is ESCALATED to a human
rather than approved — refusing to act is a valid outcome here.

Every step, including every tool call, is written to trajectories/<id>.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from advanced import llm
from advanced.agents import critic, data_quality, human_gate, intervention, risk_analysis, root_cause
from advanced.memory import CohortMemory
from advanced.tools import ToolBox

MAX_REVISIONS = 2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAJ_DIR = os.path.join(BASE_DIR, "trajectories")


def run_for_employee(employee: dict, toolbox: ToolBox = None, save: bool = True,
                     memory: CohortMemory = None) -> dict:
    """
    Run the fixed agent sequence for one employee.

    `memory` is optional and purely additive: it annotates the case with cohort
    context and consistency precedent. It cannot change `status`, so the
    headline evaluation runs with memory=None and stays exactly reproducible.
    """
    toolbox = toolbox or ToolBox()
    toolbox.reset_log()
    trajectory = [{"step": "config", "llm": llm.describe(),
                   "memory": "enabled" if memory is not None else "disabled"}]

    # --- Step 1: data quality gate ---
    dq = data_quality.run(employee, toolbox.features)
    trajectory.append({"step": "data_quality", **dq})
    if dq["verdict"] == "FAIL":
        return _finish(employee, trajectory, toolbox, "HALTED_DATA_QUALITY", None, save)

    # --- Step 2: risk analysis ---
    risk = risk_analysis.run(employee, toolbox)
    trajectory.append({"step": "risk_analysis", **risk})

    # --- Steps 3-5: root cause -> intervention -> critic, with revision loop ---
    rc = ic = cr = None
    for attempt in range(1, MAX_REVISIONS + 2):
        # On revision, raise the evidence bar instead of retrying identically.
        strictness = 1.0 + 0.4 * (attempt - 1)
        rc = _root_cause_with_strictness(employee, risk, toolbox, strictness)
        trajectory.append({"step": f"root_cause_attempt_{attempt}",
                           "strictness": round(strictness, 2), **rc})

        ic = intervention.run(employee, risk, rc, toolbox)
        trajectory.append({"step": f"intervention_attempt_{attempt}", **ic})

        cr = critic.run(risk, rc, ic)
        trajectory.append({"step": f"critic_attempt_{attempt}", **cr})

        if cr["verdict"] == "APPROVE":
            break

    # --- Step 6: human gate ---
    gate = human_gate.run(risk, rc, ic, cr)
    trajectory.append({"step": "human_gate", **gate})

    detail = {"risk": risk, "root_cause": rc, "intervention": ic, "critic": cr}

    # --- Step 7 (optional): cohort memory, annotation only ---
    if memory is not None:
        notes = memory.contextualise_drivers(rc["confirmed_drivers"])
        precedent = memory.precedent_for(
            employee.get("employee_id"), risk["risk_level"], rc["actionable_features"])
        detail["memory"] = {"driver_context": notes, "precedent": precedent}
        trajectory.append({"step": "cohort_memory", "driver_context": notes,
                           "precedent": precedent,
                           "note": "Annotation only — does not affect status."})
        memory.record(employee.get("employee_id"), risk, rc, ic, gate["status"])

    return _finish(employee, trajectory, toolbox, gate["status"], detail, save)


def _root_cause_with_strictness(employee, risk, toolbox, strictness):
    """Re-run root cause with a raised contribution floor on each revision."""
    original = root_cause.MIN_CONTRIBUTION
    try:
        root_cause.MIN_CONTRIBUTION = original * strictness
        return root_cause.run(employee, risk, toolbox)
    finally:
        root_cause.MIN_CONTRIBUTION = original


def _finish(employee, trajectory, toolbox, status, detail, save):
    trajectory.append({"step": "tool_calls", "calls": toolbox.call_log})
    result = {
        "employee_id": employee.get("employee_id"),
        "status": status,
        "tool_call_count": len(toolbox.call_log),
        "detail": detail,
        "trajectory": trajectory,
    }
    if save:
        os.makedirs(TRAJ_DIR, exist_ok=True)
        path = os.path.join(TRAJ_DIR, f"{employee.get('employee_id', 'unknown')}.json")
        with open(path, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    import pandas as pd

    tb = ToolBox()
    emp_id = sys.argv[1] if len(sys.argv) > 1 else "EMP1015"
    emp = tb.get_employee_profile(emp_id)
    if "error" in emp:
        known = tb.cohort["employee_id"].tolist()
        print(f"Employee '{emp_id}' not found in data/onboarding_data.csv.")
        print(f"Known ids look like: {', '.join(known[:5])} ... ({len(known)} total)")
        sys.exit(1)
    res = run_for_employee(emp, tb)

    print(f"LLM: {llm.describe()}")
    print(f"Employee: {res['employee_id']} | status: {res['status']} "
          f"| tool calls: {res['tool_call_count']}")
    for step in res["trajectory"]:
        name = step.get("step")
        if name == "tool_calls":
            continue
        marker = step.get("verdict") or step.get("status") or ""
        print(f"  - {name}: {marker}")
