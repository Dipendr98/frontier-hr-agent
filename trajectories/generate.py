"""
Regenerate the representative trajectories.

    python trajectories/generate.py            # Mode 1 only, no key needed
    python trajectories/generate.py --llm      # also Mode 2 and Mode 3

Hand-curated trajectory files go stale the moment a threshold changes, and a
stale trajectory is worse than none: it documents a system that no longer
exists. This script picks the representative cases by SEARCHING the cohort for
each outcome rather than hardcoding employee IDs, so the set stays honest
across changes to the evidence floors.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from advanced import llm
from advanced.memory import CohortMemory
from advanced.orchestration import llm_agent
from advanced.orchestration.workflow import run_for_employee
from advanced.tools import ToolBox

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)

# The case the CHANGELOG discusses at length; always included by name.
CHALLENGING_CASE = "EMP1180"


def _write(name: str, payload):
    with open(os.path.join(HERE, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {name}.json")


def generate_mode1(toolbox, df):
    print("Mode 1 — deterministic pipeline")
    memory = CohortMemory(autoload=False)
    for _, row in df.iterrows():                       # warm memory first
        run_for_employee(row.to_dict(), toolbox, save=False, memory=memory)

    wanted = {"APPROVED_FOR_REVIEW": 2, "ESCALATED": 2, "NO_ACTION": 2}
    seen = {k: 0 for k in wanted}

    # Always include the challenging case, whatever it resolves to now.
    special = toolbox.get_employee_profile(CHALLENGING_CASE)
    if "error" not in special:
        res = run_for_employee(special, toolbox, save=False, memory=memory)
        _write(f"MODE1_{res['status']}_{CHALLENGING_CASE}_challenging", res["trajectory"])
        seen[res["status"]] = seen.get(res["status"], 0) + 1

    for _, row in df.iterrows():
        emp = row.to_dict()
        if emp["employee_id"] == CHALLENGING_CASE:
            continue
        if all(seen.get(k, 0) >= v for k, v in wanted.items()):
            break
        res = run_for_employee(emp, toolbox, save=False, memory=memory)
        status = res["status"]
        if seen.get(status, 0) >= wanted.get(status, 0):
            continue
        seen[status] = seen.get(status, 0) + 1
        _write(f"MODE1_{status}_{emp['employee_id']}", res["trajectory"])

    # A record that must never be scored at all.
    bad = dict(df.iloc[0].to_dict())
    bad["employee_id"] = "EMP_MALFORMED"
    bad["JobSatisfaction"] = 99
    res = run_for_employee(bad, toolbox, save=False)
    _write(f"MODE1_{res['status']}_EMP_MALFORMED", res["trajectory"])

    for status, want in wanted.items():
        got = seen.get(status, 0)
        if got == 0:
            print(f"  note: no case in the cohort currently reaches {status}")
        elif got < want:
            print(f"  note: only {got} case(s) reach {status} (wanted {want}) — "
                  f"the cohort has no more")


def _run_with_retry(emp_id, toolbox, verify, tag, attempts=3):
    """
    A trajectory that records a rate limit is not a representative trajectory.

    It documents the provider's free tier, not what the agent does, and
    committing it would put a misleading file in front of a judge. So a run the
    provider never completed is retried and then skipped, loudly — rather than
    written out as though it were a result.
    """
    for attempt in range(1, attempts + 1):
        try:
            res = llm_agent.run_llm_agent(emp_id, toolbox, verify=verify,
                                          memory=CohortMemory())
        except Exception as e:
            print(f"  {emp_id} {tag}: {type(e).__name__}: {e}")
            return None
        if res["status"] != "LLM_ERROR":
            return res
        if attempt < attempts:
            print(f"  {emp_id} {tag}: provider error, retrying "
                  f"({attempt}/{attempts - 1})")
            time.sleep(10 * attempt)
    print(f"  {emp_id} {tag}: SKIPPED — provider never completed the run")
    return None


def generate_llm_modes(toolbox, df):
    if not llm.is_enabled():
        print("Modes 2/3 — skipped, no provider configured "
              "(set LLM_PRESET + LLM_API_KEY)")
        return
    print(f"Modes 2 and 3 — {llm.describe()}")

    ids = [CHALLENGING_CASE] + [
        e for e in df["employee_id"].tolist()[:4] if e != CHALLENGING_CASE]

    corrected_written = False
    for emp_id in ids:
        for verify, tag in [(False, "MODE2_RAW"), (True, "MODE3_VERIFIED")]:
            res = _run_with_retry(emp_id, toolbox, verify, tag)
            if res is None:
                continue
            # Keep the whole run, not just the trace: the brief asks for the
            # feedback that shaped the next step, which is the verification
            # payload, and for the human checkpoint, which is the final status.
            _write(f"{tag}_{res['status']}_{emp_id}", {
                "employee_id": emp_id,
                "mode": res["mode"],
                "status": res["status"],
                "requires_human_decision": res["requires_human_decision"],
                "tool_calls_used": res["tool_calls_used"],
                "verify_retries": res["verify_retries"],
                "usage": res["usage"],
                "final_decision": res["decision"],
                "verification": res["verification"],
                "trace": res["trace"],
            })
            if verify and res["verify_retries"] > 0:
                corrected_written = True

    if not corrected_written:
        print("  note: no Mode 3 run needed a correction this time; the "
              "retry path is exercised by tests/test_claim_verifier.py and "
              "measured in evidence/llm_agent_report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="also generate Mode 2 and Mode 3 trajectories (needs a key)")
    args = ap.parse_args()

    # Deterministic prose so regenerating does not produce a spurious diff.
    llm.set_narration(False)

    toolbox = ToolBox()
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))

    for f in os.listdir(HERE):
        if f.endswith(".json"):
            os.remove(os.path.join(HERE, f))

    generate_mode1(toolbox, df)
    if args.llm:
        generate_llm_modes(toolbox, df)
    print("\nDone. See trajectories/README.md for what each file shows.")


if __name__ == "__main__":
    main()
