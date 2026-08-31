"""
Provider self-test.  `python -m advanced.doctor`

Answers the question you actually have before a demo: *is my key working, and
is this model usable for the agent modes?* Those are two different questions —
a key can authenticate perfectly against a model that ignores tool schemas, or
against one that takes ten minutes per case. Both make Mode 2/3 useless, and
neither shows up as an error.

Checks, in order:
  1. Config detected (no secret is ever printed)
  2. Mode 1 works with NO provider          — the reproducibility guarantee
  3. Provider reachable + authenticated     — a real completion
  4. Model emits tool calls                 — required by Modes 2 and 3
  5. Round-trip latency vs the agent budget — projected seconds per case

Exit code 0 if Modes 1-3 are all usable, 1 otherwise. Safe to run in CI.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from advanced import llm  # noqa: E402

# Mode 2 issues roughly this many sequential round trips on a contested case.
TURNS_PER_CASE = 8
# Above this projected per-case wall clock the model is technically working but
# not demoable. Chosen from experience: a reviewer will not wait 10 minutes.
SLOW_CASE_SECONDS = 120

OK, WARN, BAD = "PASS", "WARN", "FAIL"


def _line(status, title, detail=""):
    mark = {OK: "  ok  ", WARN: " warn ", BAD: " FAIL "}[status]
    print(f"[{mark}] {title}")
    if detail:
        for chunk in str(detail).splitlines():
            print(f"         {chunk}")


def _masked_key_state() -> str:
    """Report only whether a key is present and its length. Never the value."""
    key = os.environ.get("LLM_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return "not set"
    return f"set ({len(key)} chars)"


def check_config():
    print("=" * 72)
    print("FRONTIER HR AGENT — provider self-test")
    print("=" * 72)
    print()
    print(f"  LLM_PRESET    {os.environ.get('LLM_PRESET', '(unset)')}")
    print(f"  LLM_PROVIDER  {llm.PROVIDER}")
    print(f"  LLM_BASE_URL  {llm.BASE_URL or '(n/a)'}")
    print(f"  LLM_MODEL     {llm.active_model() or '(unset)'}")
    print(f"  LLM_API_KEY   {_masked_key_state()}")
    print(f"  LLM_TIMEOUT   {llm.TIMEOUT}s   retries: {llm.MAX_RETRIES}")
    print()


def check_mode1_without_provider() -> bool:
    """
    Mode 1 must reach a decision with no key at all. This is the guarantee the
    whole reproducibility claim rests on, so it is checked first and it is
    checked by actually running a case, not by reading a flag.
    """
    try:
        from advanced.orchestration.workflow import run_for_employee
        from advanced.tools import ToolBox
        tb = ToolBox()
        emp = tb.get_employee_profile(tb.cohort["employee_id"].iloc[0])
        res = run_for_employee(emp, tb, save=False)
        allowed = {"APPROVED_FOR_REVIEW", "ESCALATED", "NO_ACTION", "HALTED_DATA_QUALITY"}
        if res["status"] in allowed:
            _line(OK, "Mode 1 (deterministic) reaches a terminal state",
                  f"{res['employee_id']} -> {res['status']}  "
                  f"({res['tool_call_count']} tool calls, no key required)")
            return True
        _line(BAD, "Mode 1 produced an unexpected status", res["status"])
        return False
    except Exception as e:
        _line(BAD, "Mode 1 failed to run", f"{type(e).__name__}: {e}")
        return False


def check_provider_reachable() -> bool:
    if not llm.is_enabled():
        _line(WARN, "No LLM provider configured",
              "Mode 1 and every headline number still reproduce.\n"
              "Modes 2 and 3 need a key: set LLM_PRESET + LLM_API_KEY "
              "(see .env.example).")
        return False
    t0 = time.time()
    try:
        msg = llm.chat([{"role": "user",
                         "content": "Reply with exactly the word: READY"}], max_tokens=20)
    except Exception as e:
        _line(BAD, "Provider unreachable or rejected the key", f"{type(e).__name__}: {e}")
        return False
    dt = time.time() - t0
    body = (msg.get("content") or "").strip()
    if not body:
        _line(WARN, f"Provider answered in {dt:.1f}s but returned empty content",
              "Usable for tool calling, but Mode 1's prose layer will fall back.")
        return True
    _line(OK, f"Provider reachable and authenticated ({dt:.1f}s)", f"replied: {body[:60]!r}")
    return True


def check_tool_calling() -> bool:
    ok, detail = llm.supports_tool_calling()
    if ok:
        _line(OK, "Model emits tool calls (Modes 2 and 3 available)", detail)
    else:
        _line(BAD, "Model did NOT emit a tool call",
              f"{detail}\nModes 2/3 will not work on this model. Pick a "
              f"tool-calling model via LLM_MODEL.")
    return ok


def check_latency() -> bool:
    """Project per-case wall clock from measured round trips."""
    snap = llm.METER.snapshot()
    if snap["llm_calls"] == 0:
        return True
    per_turn = snap["llm_seconds"] / snap["llm_calls"]
    projected = per_turn * TURNS_PER_CASE
    detail = (f"{per_turn:.1f}s per round trip x ~{TURNS_PER_CASE} turns "
              f"= ~{projected:.0f}s per Mode 2 case")
    if projected > SLOW_CASE_SECONDS:
        _line(WARN, "Model is slow inside an agent loop",
              f"{detail}\nWorks, but too slow to demo. Try a faster "
              f"tool-calling model via LLM_MODEL.")
        return True
    _line(OK, "Latency is workable for the agent loop", detail)
    return True


def main() -> int:
    check_config()
    mode1 = check_mode1_without_provider()
    reachable = check_provider_reachable()
    tools_ok = check_tool_calling() if reachable else False
    if reachable:
        check_latency()

    snap = llm.METER.snapshot()
    print()
    print("-" * 72)
    print(f"  Self-test used {snap['llm_calls']} LLM calls, "
          f"{snap['total_tokens']} tokens, "
          f"est. ${snap['estimated_cost_usd']:.6f} ({snap['retries']} retries)")
    print("-" * 72)
    print()
    print(f"  Mode 1  deterministic pipeline   {'READY' if mode1 else 'BROKEN'}   (no key needed)")
    print(f"  Mode 2  LLM tool-calling agent   {'READY' if tools_ok else 'UNAVAILABLE'}")
    print(f"  Mode 3  verified LLM agent       {'READY' if tools_ok else 'UNAVAILABLE'}")
    print()

    if not mode1:
        print("  Mode 1 is broken — this is the path every reported number uses.")
        return 1
    if not reachable:
        print("  No provider: Mode 1 reproduces everything. Modes 2/3 are off.")
        return 0
    return 0 if tools_ok else 1


if __name__ == "__main__":
    sys.exit(main())
