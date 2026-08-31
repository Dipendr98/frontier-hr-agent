"""
GENUINE LLM-driven agent: the model chooses which tool to call, with what
arguments, and when it has enough evidence to stop — not a fixed Python
sequence. This is the "agent decides" pattern the deterministic pipeline in
advanced/agents/ deliberately does NOT use.

Two modes live here, and the difference between them is the experiment:

  MODE 2 (verify=False) — the model's finalize call is accepted as written.
  MODE 3 (verify=True)  — the model's finalize call is treated as an
                          allegation. Every claim is re-derived from the tools
                          by advanced/agents/claim_verifier.py, and a rejected
                          claim goes back to the model with the specific
                          measurement that contradicts it. Bounded retries;
                          an unresolved case is ESCALATED, never approved.

Running both on the same cases is how we measure what verification is worth,
rather than asserting it. See evaluation/evaluate_llm_agent.py.

Both modes require a real LLM_API_KEY. There is no rule-based fallback here on
purpose: if there is no model making the decisions, there is no agent to run,
and this file says so rather than silently downgrading to the deterministic
pipeline and calling that "the agent decided".
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from advanced import llm
from advanced.agents import claim_verifier
from advanced.agents.risk_analysis import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD
from advanced.memory import CohortMemory
from advanced.tools import FEATURE_LEVERS, FEATURE_META, INTERVENTION_CATALOG, ToolBox

# Budget is counted in ACTUAL TOOL EXECUTIONS, not model turns.
#
# The earlier version counted loop iterations. A model that emits four tool
# calls in one turn therefore spent four units of budget and was charged one,
# and an EMP1180 run advertised as "max 8" executed 17 tools. A budget that
# does not count the thing it limits is not a budget.
MAX_TOOL_CALLS = 12
# How many times a rejected finalize may be sent back with the objection.
MAX_VERIFY_RETRIES = 2
# Hard ceiling on model round trips, so a model that never calls a tool and
# never finalizes still terminates.
MAX_TURNS = 16

VALID_INTERVENTIONS = sorted(INTERVENTION_CATALOG)

# --- Tool schema the model is allowed to choose from -----------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_profile",
            "description": "Fetch the raw onboarding record for one employee.",
            "parameters": {
                "type": "object",
                "properties": {"employee_id": {"type": "string"}},
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_risk",
            "description": "Score the employee with the trained attrition model.",
            "parameters": {
                "type": "object",
                "properties": {"employee_id": {"type": "string"}},
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_drivers",
            "description": (
                "Get the model's exact per-feature logit contributions for "
                "this employee, ranked. Positive = raises risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {"employee_id": {"type": "string"},
                               "top_n": {"type": "integer", "default": 4}},
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cohort_percentile",
            "description": (
                "Where this employee's value for a feature sits versus the cohort. "
                "Returns severity_percentile, which is direction-aware: 0 means "
                "worst in the cohort on that feature, 100 means best. Use "
                "severity_percentile, NOT the raw percentile, to judge whether a "
                "value is unusual — for features where high is bad (overtime, "
                "commute) a high raw percentile is the bad end."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "feature": {"type": "string"},
                },
                "required": ["employee_id", "feature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_intervention",
            "description": (
                "Re-score the employee under a what-if intervention. Returns "
                "the simulated risk delta. NOT a causal estimate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "intervention_key": {"type": "string", "enum": VALID_INTERVENTIONS},
                },
                "required": ["employee_id", "intervention_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": (
                "Call this ONLY when you have enough evidence to stop. Ends the "
                "investigation and submits your recommendation for human review."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level": {"type": "string", "enum": ["Low", "Medium", "High"]},
                    "confirmed_drivers": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Exact feature names you verified as real drivers, e.g. "
                            "WorkLifeBalance. Empty list if none survived."
                        ),
                    },
                    "recommended_intervention": {
                        "type": "string", "enum": VALID_INTERVENTIONS + [""],
                        "description": "Catalogue key, or empty string for no action.",
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["risk_level", "confirmed_drivers", "rationale"],
            },
        },
    },
]

# The model is told the operational definitions it is being judged against.
#
# It is not optional politeness. The first measured Mode 2 run failed
# verification on 83% of cases, and most of those failures were the model
# applying its OWN idea of what "Medium risk" means (it called 10.4% Medium)
# and its own idea of which intervention addresses which problem (it proposed
# an ownership assignment for a job-satisfaction driver). Neither the
# thresholds nor the lever map were in the prompt. Measuring a model against
# rules it was never given tells you nothing about verification — it tells you
# the prompt was incomplete. Same discipline as BASELINE+: beat the strongest
# version of the opponent, not a convenient one.
def _lever_reference() -> str:
    lines = []
    for feat, levers in sorted(FEATURE_LEVERS.items()):
        lines.append(f"     {feat}: {', '.join(levers)}")
    unlevered = sorted(set(FEATURE_META) - set(FEATURE_LEVERS))
    lines.append("     NO INTERVENTION EXISTS for: " + ", ".join(unlevered))
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an HR onboarding-risk investigator. You have tools,
not a fixed script — decide yourself which to call, in what order, and when
you have enough evidence to stop.

Definitions you are held to. These are the system's, not yours:
  Risk bands: High >= {high:.2f}, Medium >= {medium:.2f}, otherwise Low.
    Apply these numerically to the probability predict_risk returns. Do not
    substitute your own sense of what "medium risk" feels like.
  Which intervention can move which driver:
{levers}
    An intervention that does not appear against a driver cannot address it.

Rules you must follow:
1. Never state a risk level before calling predict_risk.
2. Never claim a driver is real without calling get_risk_drivers AND
   get_cohort_percentile on it. A driver must be BOTH a material contributor
   (logit_contribution >= {min_contribution}) AND unusual for this employee
   versus the cohort (severity_percentile <= {max_severity}). Use
   severity_percentile, not the raw percentile: it already accounts for whether
   high or low is the bad direction. A feature nobody is unusual on is not a
   driver just because its contribution is positive.
3. Before recommending an intervention, call simulate_intervention on your top
   1-2 candidates and prefer the one with the larger negative risk delta. Only
   recommend an intervention that can actually move a driver you verified.
4. If risk is Low, call finalize immediately with an empty confirmed_drivers
   list and no intervention — do not spend tool calls investigating someone
   who is not at risk.
5. You have at most {max_calls} tool calls. Call finalize as soon as you have
   enough evidence — do not call tools you do not need.
6. finalize is mandatory. You must call it to end the investigation.
7. Recommending nothing is a legitimate answer. If no driver survives, say so
   and finalize with an empty list rather than reaching for a plausible action.
   Nothing you decide is executed; a human reviews every case."""


# --------------------------------------------------------------- tool bridge


def _execute_tool(toolbox: ToolBox, name: str, args: dict, cache: dict = None) -> dict:
    """
    Run one model-chosen tool call.

    Every failure is returned as a structured `error` the model can read and
    correct from. Raising here would abort the whole investigation because the
    model guessed a feature name, which is a normal thing for it to do and not
    a reason to lose the case.

    `cache` holds employee records already fetched during this run. Without it
    every scoring tool re-fetched the profile it needs, so `toolbox.call_log`
    carried two entries per model-chosen call and no longer matched what the
    agent actually did — the same class of miscount as the turn-based budget.
    """
    if not isinstance(args, dict):
        return {"error": "bad_arguments", "detail": "arguments must be a JSON object"}

    emp_id = args.get("employee_id")
    if not emp_id:
        return {"error": "missing_argument", "detail": "employee_id is required"}

    cache = cache if cache is not None else {}
    key = str(emp_id)
    if key in cache:
        employee = cache[key]
    else:
        employee = toolbox.get_employee_profile(key)
        if "error" not in employee:
            cache[key] = employee
    if "error" in employee:
        return employee

    if name == "get_employee_profile":
        return employee

    if name == "predict_risk":
        return toolbox.predict_risk(employee)

    if name == "get_risk_drivers":
        try:
            top_n = int(args.get("top_n", 4))
        except (TypeError, ValueError):
            top_n = 4
        return toolbox.get_risk_drivers(employee, top_n=max(1, min(top_n, len(toolbox.features))))

    if name == "get_cohort_percentile":
        feature = args.get("feature")
        if feature not in toolbox.features:
            return {"error": "unknown_feature", "feature": feature,
                    "known_features": toolbox.features}
        return toolbox.get_cohort_percentile(feature, float(employee[feature]))

    if name == "simulate_intervention":
        key = args.get("intervention_key")
        if key not in INTERVENTION_CATALOG:
            return {"error": "unknown_intervention", "intervention_key": key,
                    "valid_interventions": VALID_INTERVENTIONS}
        return toolbox.simulate_intervention(employee, key)

    return {"error": "unknown_tool", "tool": name,
            "valid_tools": [t["function"]["name"] for t in TOOLS]}


def _parse_args(tc: dict) -> dict:
    raw = (tc.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------- the agent


def run_llm_agent(employee_id: str, toolbox: ToolBox = None, verify: bool = True,
                  memory: CohortMemory = None) -> dict:
    """
    Run the model in a real tool-use loop.

    verify=True  -> Mode 3: finalize is re-derived and may be sent back.
    verify=False -> Mode 2: finalize is accepted as written.

    Raises RuntimeError if no LLM provider is configured — this mode has no
    deterministic fallback, because a fallback here would mean the "agent"
    isn't actually deciding anything.
    """
    if not llm.is_enabled():
        raise RuntimeError(
            "No LLM provider configured. This mode requires a real key — set "
            "LLM_PRESET + LLM_API_KEY (see .env.example), then check it with "
            "`python -m advanced.doctor`. The deterministic pipeline in "
            "advanced/orchestration/workflow.py is the no-key path; this file "
            "is specifically the genuine model-driven agent and does not "
            "degrade to rules, or it would not be demonstrating anything."
        )

    toolbox = toolbox or ToolBox()
    toolbox.reset_log()
    employee = toolbox.get_employee_profile(str(employee_id))
    if "error" in employee:
        raise ValueError(f"Employee '{employee_id}' not found in the cohort.")
    # The lookup above is scaffolding, not agent behaviour; don't bill it.
    toolbox.reset_log()

    mode = "MODE_3_VERIFIED" if verify else "MODE_2_RAW"
    before = llm.METER.snapshot()
    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            max_calls=MAX_TOOL_CALLS,
            min_contribution=claim_verifier.MIN_CONTRIBUTION,
            max_severity=claim_verifier.MAX_SEVERITY_PERCENTILE,
            high=HIGH_RISK_THRESHOLD, medium=MEDIUM_RISK_THRESHOLD,
            levers=_lever_reference())},
        {"role": "user", "content": f"Investigate employee {employee_id}."},
    ]
    trace = [{"step": "start", "employee_id": employee_id, "mode": mode,
              "llm": llm.describe(), "tool_budget": MAX_TOOL_CALLS}]

    tools_used = 0
    verify_attempts = 0
    last_verification = None
    profile_cache = {str(employee_id): employee}

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = llm.chat(messages, tools=TOOLS, max_tokens=1024)
        except Exception as e:
            trace.append({"step": "llm_error", "turn": turn,
                          "error": f"{type(e).__name__}: {e}"})
            return _finish(employee_id, mode, "LLM_ERROR", None, None, trace,
                           toolbox, tools_used, verify_attempts, before, t0, memory,
                           employee)
        messages.append(response)

        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            # Model spoke without calling a tool — nudge it, don't crash the run.
            trace.append({"step": "model_text", "turn": turn,
                          "content": (response.get("content") or "")[:500]})
            messages.append({"role": "user",
                             "content": "Call a tool, or call finalize to stop."})
            continue

        finished = None
        for tc in tool_calls:
            name = (tc.get("function") or {}).get("name", "")
            args = _parse_args(tc)

            if name == "finalize":
                finished = (tc, args)
                break

            if tools_used >= MAX_TOOL_CALLS:
                result = {"error": "budget_exhausted",
                          "detail": (f"You have used all {MAX_TOOL_CALLS} tool calls. "
                                     "Call finalize now with what you have.")}
            else:
                tools_used += 1
                result = _execute_tool(toolbox, name, args, profile_cache)

            trace.append({"step": "tool_call", "turn": turn, "n": tools_used,
                          "tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                             "content": json.dumps(result, default=str)})

        if finished is None:
            continue

        tc, claim = finished

        # ---- MODE 2: take the model at its word --------------------------
        if not verify:
            trace.append({"step": "finalize", "turn": turn, "decision": claim})
            return _finish(employee_id, mode, "COMPLETED", claim, None, trace,
                           toolbox, tools_used, verify_attempts, before, t0, memory,
                           employee)

        # ---- MODE 3: the claim is an allegation until re-derived ---------
        verification = claim_verifier.verify(employee, claim, toolbox)
        last_verification = verification
        trace.append({"step": "verification", "turn": turn,
                      "attempt": verify_attempts + 1, **verification})

        if verification["verdict"] == "VERIFIED":
            trace.append({"step": "finalize", "turn": turn, "decision": claim})
            # Same terminal vocabulary as Mode 1, so the two are comparable and
            # so "no action needed" is not filed as "awaiting your decision" —
            # a reviewer's queue length is the resource this project is about.
            verified_status = (
                "APPROVED_FOR_REVIEW" if verification["recommended_intervention"]
                else "NO_ACTION")
            return _finish(employee_id, mode, verified_status, claim,
                           verification, trace, toolbox, tools_used,
                           verify_attempts, before, t0, memory, employee)

        verify_attempts += 1
        if verify_attempts > MAX_VERIFY_RETRIES:
            # Refusing to approve what we could not verify is the correct
            # outcome. The objections travel with the case to the human.
            trace.append({"step": "escalate", "turn": turn,
                          "reason": "claims did not survive verification after "
                                    f"{MAX_VERIFY_RETRIES} corrections"})
            return _finish(employee_id, mode, "ESCALATED", claim, verification,
                           trace, toolbox, tools_used, verify_attempts, before, t0,
                           memory, employee)

        messages.append({"role": "tool", "tool_call_id": tc.get("id", "finalize"),
                         "content": claim_verifier.objection_prompt(verification)})

    trace.append({"step": "turn_limit_reached", "turns": MAX_TURNS})
    return _finish(employee_id, mode, "NO_FINALIZE_CALLED", None, last_verification,
                   trace, toolbox, tools_used, verify_attempts, before, t0, memory,
                   employee)


def _finish(employee_id, mode, status, claim, verification, trace, toolbox,
            tools_used, verify_attempts, before, t0, memory, employee):
    after = llm.METER.snapshot()
    usage = {
        "llm_calls": after["llm_calls"] - before["llm_calls"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
        "estimated_cost_usd": round(
            after["estimated_cost_usd"] - before["estimated_cost_usd"], 6),
        "wall_clock_seconds": round(time.time() - t0, 2),
    }

    # Memory annotates; it never changes `status`. See advanced/memory.py.
    memory_notes = {}
    if memory is not None and verification:
        drivers = [v["feature"] for v in verification.get("verified_drivers", [])]
        notes = memory.contextualise_drivers(verification.get("verified_drivers", []))
        precedent = memory.precedent_for(
            employee_id, verification["recomputed_risk"]["risk_level"], drivers)
        memory_notes = {"driver_context": notes, "precedent": precedent}
        memory.record(
            employee_id,
            verification["recomputed_risk"],
            {"confirmed_drivers": verification.get("verified_drivers", []),
             "actionable_features": [v["feature"] for v in
                                     verification.get("verified_drivers", [])
                                     if v.get("has_lever")]},
            {"recommendation": {"key": verification.get("recommended_intervention")}},
            status,
        )

    trace.append({"step": "usage", **usage})
    return {
        "employee_id": employee_id,
        "mode": mode,
        "status": status,
        "decision": claim,
        "verification": verification,
        "memory": memory_notes,
        "requires_human_decision": status in ("APPROVED_FOR_REVIEW", "ESCALATED"),
        "tool_calls_used": tools_used,
        "verify_retries": verify_attempts,
        "usage": usage,
        "trace": trace,
        "raw_tool_log": toolbox.call_log,
    }


# --------------------------------------------------------------- CLI


def _print_run(result: dict):
    print(f"LLM: {llm.describe()}")
    print(f"Mode: {result['mode']}")
    print(f"Employee: {result['employee_id']} | status: {result['status']}")
    print(f"Tool calls: {result['tool_calls_used']}/{MAX_TOOL_CALLS} | "
          f"verify retries: {result['verify_retries']}")
    u = result["usage"]
    print(f"Usage: {u['llm_calls']} LLM calls, {u['total_tokens']} tokens, "
          f"{u['wall_clock_seconds']}s, est. ${u['estimated_cost_usd']:.5f}")
    print()
    for step in result["trace"]:
        s = step.get("step")
        if s == "tool_call":
            err = step["result"].get("error") if isinstance(step["result"], dict) else None
            print(f"  [{step['n']:>2}] {step['tool']}({json.dumps(step['args'])})"
                  + (f"  -> error: {err}" if err else ""))
        elif s == "verification":
            print(f"  VERIFY attempt {step['attempt']}: {step['verdict']}")
            for p in step["problems"]:
                print(f"        x {p}")
        elif s == "finalize":
            print("  FINALIZE")
        elif s == "escalate":
            print(f"  ESCALATED: {step['reason']}")
        elif s == "model_text":
            print(f"  (model spoke without a tool call)")

    if result.get("verification") and result["verification"]["verified_drivers"]:
        print("\nVERIFIED EVIDENCE (recomputed, not the model's wording):")
        for v in result["verification"]["verified_drivers"]:
            print(f"  - {v['statement']}")
    if result["decision"]:
        print("\nFINAL DECISION:")
        print(json.dumps(result["decision"], indent=2))
    if result.get("memory", {}).get("driver_context"):
        print("\nCOHORT MEMORY:")
        for n in result["memory"]["driver_context"]:
            print(f"  - {n['note']}")
    print(f"\nrequires_human_decision: {result['requires_human_decision']} "
          "— nothing here is executed.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    raw = "--raw" in args or "--no-verify" in args
    ids = [a for a in args if not a.startswith("-")]
    emp_id = ids[0] if ids else "EMP1180"
    _print_run(run_llm_agent(emp_id, verify=not raw, memory=CohortMemory()))
