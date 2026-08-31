"""
MODE 2 vs MODE 3 — what does verification actually buy?

The deterministic pipeline (Mode 1) is where every headline number comes from,
because it reproduces with no key. But the interesting agentic question is the
one Mode 1 cannot answer: when a language model genuinely drives the
investigation, how often is the thing it concludes *wrong*, and does checking
it help?

This measures that on the same cases, with the same tools, same prompt, same
model. The only difference is whether finalize is believed.

  MODE 2  the model's finalize call is accepted as written.
  MODE 3  finalize is re-derived from the tools by claim_verifier; a rejected
          claim goes back with the measurement that contradicts it; unresolved
          cases are ESCALATED rather than approved.

Two things are reported, and the first matters more:

  1. RAW CLAIM ERROR RATE. Mode 2 runs are also passed through the verifier
     PASSIVELY — checked, never corrected, never told. That gives an unbiased
     count of how often an unsupervised tool-calling agent asserts something
     the tools do not support, without the measurement changing the behaviour.

  2. Reviewer Case Quality, scored by the SAME independent rubric used for
     Mode 1 (evaluation/rubric.py), over the artifact each mode hands the
     reviewer: Mode 2's own prose, Mode 3's recomputed statements.

Cases are the frozen holdout, sampled with a fixed seed so a rerun hits the
same employees. Requires a real key; costs a few cents. Runtime scales with
your provider's latency — check it first with `python -m advanced.doctor`.

    python evaluation/evaluate_llm_agent.py --n 30
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd  # noqa: E402

from advanced import llm  # noqa: E402
from advanced.agents import claim_verifier  # noqa: E402
from advanced.orchestration.llm_agent import run_llm_agent  # noqa: E402
from advanced.tools import ToolBox  # noqa: E402
from evaluation.evaluate import get_eval_cases  # noqa: E402
from evaluation.rubric import aggregate, score_case  # noqa: E402

EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
SEED = 42


def _normalise(res: dict, employee: dict) -> dict:
    """
    Reduce an LLM-agent run to the artifact a reviewer receives, so the same
    rubric can score it.

    BOTH modes hand the reviewer the model's rationale, so both are scored on
    it. Mode 3 additionally attaches the recomputed driver statements. Anything
    else would compare different artifacts: dropping Mode 2's rationale would
    hide the prose the reviewer actually reads, and dropping Mode 3's would
    give it a free pass on the sentence its verifier now checks (V7).
    """
    decision = res.get("decision") or {}
    verification = res.get("verification") or {}
    verified = verification.get("verified_drivers") or []
    rationale = (decision.get("rationale") or "").strip()

    if res["mode"] == "MODE_3_VERIFIED" and res["status"] == "ESCALATED":
        evidence, cited = [], []           # nothing was signed off
    elif res["mode"] == "MODE_3_VERIFIED" and res["status"] == "NO_ACTION":
        evidence = [rationale] if rationale else []
        cited = []
    elif res["mode"] == "MODE_3_VERIFIED":
        evidence = [v["statement"] for v in verified] + ([rationale] if rationale else [])
        cited = [v["feature"] for v in verified]
    else:
        evidence = [rationale] if rationale else []
        cited = [d for d in (decision.get("confirmed_drivers") or [])
                 if isinstance(d, str)]

    key = (decision.get("recommended_intervention") or "").strip() or None
    from advanced.tools import FEATURE_LEVERS
    targets = [f for f in cited if key and key in FEATURE_LEVERS.get(f, [])]

    return {
        "employee_id": res["employee_id"],
        "risk_level": (verification.get("recomputed_risk") or {}).get(
            "risk_level") or decision.get("risk_level"),
        "recommendation": key,
        "action_taken": bool(key) and res["status"] != "ESCALATED",
        "escalated": res["status"] == "ESCALATED",
        "evidence": evidence,
        "cited_features": cited,
        "targets_drivers": targets,
        "status": res["status"],
        "tool_calls": res["tool_calls_used"],
    }


def _normalise_verified_only(res: dict) -> dict:
    """
    Mode 3 scored on its RECOMPUTED evidence alone, with the model's prose
    excluded.

    This exists so the choice of artifact cannot quietly decide the comparison.
    Including the rationale is the conservative reading and is what the headline
    uses; excluding it is the reading where "the evidence" means only the part
    the verifier actually re-derived. The two disagree, so both are reported.
    Publishing only the one that suits us is the exact move this project's
    changelog criticises elsewhere.
    """
    row = dict(res["_row"])
    verified = (res.get("verification") or {}).get("verified_drivers") or []
    row["evidence"] = [v["statement"] for v in verified]
    return row


def _run_one(args):
    emp, verify = args
    tb = ToolBox()                       # one per worker: call_log is per-run state
    try:
        res = run_llm_agent(emp["employee_id"], tb, verify=verify, memory=None)
    except Exception as e:
        return emp, None, {"error": f"{type(e).__name__}: {e}"}

    passive = None
    if not verify and res.get("decision"):
        # Check Mode 2's answer WITHOUT telling it. The point is to measure the
        # unsupervised error rate, so the observation must not change the run.
        passive = claim_verifier.verify(emp, res["decision"], tb)
    return emp, res, passive


def main(n: int, workers: int):
    if not llm.is_enabled():
        print("No LLM provider configured. Modes 2 and 3 need a real key.")
        print("Set LLM_PRESET + LLM_API_KEY, then: python -m advanced.doctor")
        return None

    cases = get_eval_cases().sample(n=n, random_state=SEED).reset_index(drop=True)
    employees = [r.to_dict() for _, r in cases.iterrows()]
    features = [c for c in cases.columns if c not in ("employee_id", "attrition")]

    print(f"MODE 2 vs MODE 3 — {len(employees)} cases from the frozen holdout")
    print(f"Provider: {llm.describe()}")
    print(f"Workers: {workers}\n")

    llm.METER.reset()
    completed, meta, raw_failures = {}, {}, []

    # --- run both modes, keeping every result keyed by employee -------------
    for label, verify in [("mode2_raw", False), ("mode3_verified", True)]:
        t0 = time.time()
        before = llm.METER.snapshot()
        ok, errors, retries, passive_rejects = {}, [], 0, 0
        check_counts = Counter()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, [(e, verify) for e in employees]))

        for emp, res, passive in results:
            eid = emp["employee_id"]
            # A run the provider never completed is an infrastructure failure,
            # not a bad answer. Scoring it would report a rate limit as a
            # quality difference between the two modes.
            if res is None or res["status"] in ("LLM_ERROR", "NO_FINALIZE_CALLED"):
                errors.append(eid)
                continue
            res["_row"] = _normalise(res, emp)
            ok[eid] = res
            retries += res.get("verify_retries", 0)
            if passive and passive["verdict"] == "REJECTED":
                passive_rejects += 1
                for p in passive["problems"]:
                    check_counts[p.split(":")[0]] += 1
                raw_failures.append({"employee_id": eid, "problems": passive["problems"]})
            if verify:
                for step in res["trace"]:
                    if step.get("step") == "verification" and step["verdict"] == "REJECTED":
                        for p in step["problems"]:
                            check_counts[p.split(":")[0]] += 1

        after = llm.METER.snapshot()
        completed[label] = ok
        meta[label] = {
            "run_errors": len(errors),
            "failed_employee_ids": errors,
            "wall_clock_seconds": round(time.time() - t0, 1),
            "llm_calls": after["llm_calls"] - before["llm_calls"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "estimated_cost_usd": round(
                after["estimated_cost_usd"] - before["estimated_cost_usd"], 5),
            "verify_retries": retries,
            "failed_checks": dict(check_counts),
            "passive_rejects": passive_rejects,
        }

    # --- score ONLY the cases both modes completed --------------------------
    #
    # Paired, deliberately. A provider that rate-limits five Mode 3 runs and no
    # Mode 2 runs would otherwise compare 20 cases against 15 and report the
    # difference as an effect of verification. The cases are the unit of
    # comparison, so a case only counts when both modes produced an answer for it.
    paired = sorted(set(completed["mode2_raw"]) & set(completed["mode3_verified"]))
    by_id = {e["employee_id"]: e for e in employees}
    systems = {}

    for label, verify in [("mode2_raw", False), ("mode3_verified", True)]:
        rows = [completed[label][eid]["_row"] for eid in paired]
        scored = [score_case(r, by_id[r["employee_id"]], features) for r in rows]
        agg = aggregate(scored)
        agg.update(meta[label])
        agg.update({
            "n_scored": len(rows),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / len(rows), 2)
            if rows else 0,
            "escalated": sum(1 for r in rows if r["escalated"]),
        })
        if not verify:
            agg["raw_claim_error_rate"] = round(
                meta[label]["passive_rejects"] / len(rows), 4) if rows else 0.0
            agg["raw_claim_errors"] = meta[label]["passive_rejects"]
        else:
            alt_rows = [_normalise_verified_only(completed[label][eid]) for eid in paired]
            alt = aggregate([score_case(r, by_id[r["employee_id"]], features)
                             for r in alt_rows])
            agg["variant_verified_evidence_only"] = {
                "note": ("Mode 3 scored on the recomputed statements alone, with "
                         "the model's prose excluded. Reported alongside the "
                         "headline so the choice of artifact is visible rather "
                         "than decisive."),
                "reviewer_case_quality_pct": alt["reviewer_case_quality_pct"],
                "d2_evidence_present": alt["d2_evidence_present"],
                "d3_evidence_verifiable": alt["d3_evidence_verifiable"],
                "unverifiable_evidence_claims": alt["unverifiable_evidence_claims"],
            }
        systems[label] = agg
        pd.DataFrame(rows).to_csv(
            os.path.join(EVIDENCE_DIR, f"llm_{label}_case_results.csv"), index=False)

    report = {
        "n_cases_attempted": len(employees),
        "n_cases_scored": len(paired),
        "scored_employee_ids": paired,
        "sample": f"frozen holdout, .sample(n={n}, random_state={SEED})",
        "provider": llm.describe(),
        "pairing": ("Scored only on cases BOTH modes completed. A provider that "
                    "rate-limits one mode more than the other would otherwise "
                    "show up as a quality difference."),
        "note": ("Mode 2 runs are additionally checked by claim_verifier "
                 "PASSIVELY — observed, never corrected — to measure the "
                 "unsupervised claim error rate without changing behaviour."),
        "variance_warning": (
            "The model is non-deterministic and n is small, so the composite "
            "case-quality score moves by several points between runs. The claim "
            "error rate and the unverifiable-claim count are the stable signals; "
            "treat a few points of case quality as noise, not as a result."),
        "systems": systems,
        "mode2_raw_claim_failures": raw_failures,
    }
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, "llm_agent_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    _print(systems, len(paired), len(employees))
    return report


def _print(s, n, attempted):
    m2, m3 = s["mode2_raw"], s["mode3_verified"]
    w = 74
    print("=" * w)
    print(f"GENUINE TOOL-CALLING AGENT — {n} identical cases, same model")
    if n < attempted:
        print(f"  ({attempted} attempted; scored on the {n} both modes completed)")
    print("=" * w)
    print(f"{'METRIC':<40}{'MODE 2 raw':>16}{'MODE 3 verified':>18}")
    print("-" * w)

    def row(label, key, fmt="{:.1%}"):
        a, b = m2.get(key), m3.get(key)
        f = lambda v: "—" if v is None else fmt.format(v)
        print(f"{label:<40}{f(a):>16}{f(b):>18}")

    print(f"{'Claims failing verification':<40}"
          f"{m2['raw_claim_error_rate']:>15.1%}{'0.0% (by construction)':>18}")
    print(f"{'  (Mode 2 observed, not corrected)':<40}"
          f"{str(m2['raw_claim_errors']) + ' cases':>16}{'':>18}")
    print("-" * w)
    row("Reviewer case quality (rubric)", "reviewer_case_quality_pct")
    row("  d2 evidence present", "d2_evidence_present")
    row("  d3 evidence verifiable", "d3_evidence_verifiable")
    row("  d4 action has mechanism", "d4_action_has_mechanism")
    print(f"{'Unverifiable evidence claims':<40}"
          f"{m2['unverifiable_evidence_claims']:>16}"
          f"{m3['unverifiable_evidence_claims']:>18}")
    print("-" * w)
    row("Wasted intervention (counter)", "wasted_intervention_rate")
    print(f"{'Escalated to a human':<40}{m2['escalated']:>16}{m3['escalated']:>18}")
    print(f"{'Verifier corrections sent back':<40}{'—':>16}{m3['verify_retries']:>18}")
    print(f"{'Avg tool calls per case':<40}"
          f"{m2['avg_tool_calls']:>16}{m3['avg_tool_calls']:>18}")
    print(f"{'LLM calls':<40}{m2['llm_calls']:>16}{m3['llm_calls']:>18}")
    print(f"{'Est. cost per case (USD)':<40}"
          f"{m2['estimated_cost_usd'] / n:>16.5f}{m3['estimated_cost_usd'] / n:>18.5f}")
    print(f"{'Wall clock (s, parallel)':<40}"
          f"{m2['wall_clock_seconds']:>16}{m3['wall_clock_seconds']:>18}")
    print(f"{'Runs the provider never completed':<40}"
          f"{m2['run_errors']:>16}{m3['run_errors']:>18}")
    print("=" * w)
    print("Composite case quality is noisy at this n on a non-deterministic "
          "model.\nThe claim error rate and unverifiable-claim count are the "
          "stable signals.")
    alt = m3.get("variant_verified_evidence_only")
    if alt:
        print(f"\nSensitivity — Mode 3 scored on recomputed evidence only "
              f"(prose excluded):")
        print(f"  reviewer case quality  {alt['reviewer_case_quality_pct']:.1%}"
              f"   (headline: {m3['reviewer_case_quality_pct']:.1%})")
        print(f"  d3 evidence verifiable {alt['d3_evidence_verifiable']:.1%}"
              f"   unverifiable claims: {alt['unverifiable_evidence_claims']}")
        print("  Both readings are published; neither is chosen for being kinder.")
    if m3["failed_checks"]:
        print("\nWhich checks caught something (both modes):")
        for k, v in sorted(m3["failed_checks"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<32}{v:>4}")
    if m2["failed_checks"]:
        print("\nMode 2 unsupervised failures by check:")
        for k, v in sorted(m2["failed_checks"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<32}{v:>4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="cases to sample")
    # Kept low deliberately: advanced/llm.py throttles globally, and free tiers
    # 429 long before CPU is the constraint. Raise it with LLM_MAX_CONCURRENCY
    # too if your account allows more.
    ap.add_argument("--workers", type=int, default=2, help="parallel runs")
    args = ap.parse_args()
    main(args.n, args.workers)
