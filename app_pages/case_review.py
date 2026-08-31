"""One employee, one investigation, all three modes."""
import json

import streamlit as st

from advanced import llm
from advanced.memory import CohortMemory
from advanced.orchestration import llm_agent, workflow
from app_shared import (get_toolbox, provider_status, render_employee_profile,
                        render_evidence, render_risk, render_sidebar_provider,
                        render_status)

MODES = {
    "Mode 1 · Deterministic": (
        "Fixed agent sequence, no key required. Every headline number in the "
        "Evidence page comes from this mode."),
    "Mode 2 · LLM agent (raw)": (
        "The model picks the tools and the stopping point, and its conclusion "
        "is accepted as written. Requires a provider."),
    "Mode 3 · LLM agent (verified)": (
        "Same agent, but every claim is re-derived from the tools before it "
        "reaches you. Unverifiable cases are escalated, not approved."),
}

st.title("Case review")
st.caption("Turn an onboarding risk score into a case a reviewer can act on.")

# --- inputs ----------------------------------------------------------------

with st.sidebar:
    st.subheader("Data")
    uploaded = st.file_uploader("Cohort CSV", type=["csv"],
                                help="Optional. Defaults to the prepared IBM "
                                     "onboarding cohort.")

toolbox = get_toolbox(uploaded.getvalue() if uploaded else None)
df = toolbox.cohort
render_sidebar_provider()

with st.sidebar:
    st.subheader("Investigation")
    employee_id = st.selectbox("Employee", df["employee_id"].tolist())
    mode = st.radio("Mode", list(MODES), captions=list(MODES.values()))
    provider_ok, _ = provider_status()
    needs_key = not mode.startswith("Mode 1")
    run = st.button("Run investigation", type="primary", width="stretch",
                    icon=":material/play_arrow:",
                    disabled=needs_key and not provider_ok)
    if needs_key and not provider_ok:
        st.caption(":material/lock: Set a provider in `.env` to enable this mode.")

emp = df[df["employee_id"] == employee_id].iloc[0].to_dict()

st.subheader(f"{employee_id}")
render_employee_profile(emp)

if not run:
    st.stop()

# --- Mode 1 ----------------------------------------------------------------

if mode.startswith("Mode 1"):
    with st.status("Running the deterministic pipeline...", expanded=True) as status:
        memory = CohortMemory()
        res = workflow.run_for_employee(emp, toolbox, save=False, memory=memory)
        for step in res["trajectory"]:
            name = step.get("step")
            if name in ("tool_calls", "config"):
                continue
            marker = step.get("verdict") or step.get("status") or "done"
            st.write(f"**{name.replace('_', ' ')}** — `{marker}`")
        status.update(label=f"Complete — {res['status']}", state="complete",
                      expanded=False)

    detail = res.get("detail") or {}
    rc, ic = detail.get("root_cause") or {}, detail.get("intervention") or {}
    risk = detail.get("risk") or {}

    left, right = st.columns([2, 1])
    with right:
        with st.container(border=True):
            st.markdown("**Outcome**")
            render_status(res["status"])
            if risk:
                render_risk(risk["attrition_probability"], risk["risk_level"])
            st.caption(f"{res['tool_call_count']} tool calls")

    with left:
        with st.container(border=True):
            st.markdown("**Evidence**")
            render_evidence(
                [c["statement"] for c in rc.get("confirmed_drivers", [])],
                "No driver clears both the contribution and cohort floors. "
                "The risk is reported as unexplained rather than given a "
                "manufactured cause.")
            if rc.get("contextual_features"):
                st.caption(
                    ":material/info: Confirmed but not actionable: "
                    + ", ".join(rc["contextual_features"])
                    + " — real for this employee, but nothing here can "
                      "honestly claim to change it.")

        if ic.get("recommendation"):
            with st.container(border=True):
                st.markdown("**Proposed action**")
                st.write(ic["recommendation"]["label"])
                st.metric("Simulated change",
                          f"{ic['recommendation']['simulated_delta_pp']:+.1f} pp")
                st.caption("Model what-if on correlational features. NOT a "
                           "causal effect estimate.")
                if ic.get("rationale"):
                    st.write(ic["rationale"])
        elif ic.get("reason"):
            with st.container(border=True):
                st.markdown("**No action proposed**")
                st.write(ic["reason"])

    mem = detail.get("memory") or {}
    if mem.get("driver_context") or mem.get("precedent"):
        with st.container(border=True):
            st.markdown("**Cohort memory**")
            for n in mem.get("driver_context", []):
                st.markdown(f"- {n['note']}")
            p = mem.get("precedent") or {}
            if p.get("inconsistent"):
                st.warning(
                    f"{p['comparable_cases']} comparable case(s) with the same "
                    f"risk band and drivers received differing outcomes: "
                    f"{p['prior_outcomes']}.", icon=":material/balance:")
            st.caption("Annotation only — memory never changes the outcome.")

    with st.expander("Full trajectory", icon=":material/route:"):
        st.json(res["trajectory"])

# --- Modes 2 and 3 ---------------------------------------------------------

else:
    verify = mode.startswith("Mode 3")
    label = "verified" if verify else "raw"
    try:
        with st.status(f"The model is investigating ({label})...",
                       expanded=True) as status:
            res = llm_agent.run_llm_agent(employee_id, toolbox, verify=verify,
                                          memory=CohortMemory())
            for step in res["trace"]:
                s = step.get("step")
                if s == "tool_call":
                    err = step["result"].get("error") if isinstance(
                        step["result"], dict) else None
                    st.write(f":material/build: `{step['tool']}`"
                             + (f" — error: `{err}`" if err else ""))
                elif s == "verification":
                    if step["verdict"] == "VERIFIED":
                        st.write(":material/verified: All claims re-derived and confirmed")
                    else:
                        st.write(f":material/report: Attempt {step['attempt']} "
                                 f"rejected — sending the measurements back")
                        for p in step["problems"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;• {p}")
                elif s == "model_text":
                    st.write(":material/chat: model replied without a tool call")
                elif s == "llm_error":
                    st.write(f":material/cloud_off: {step['error']}")
            status.update(label=f"Complete — {res['status']}", state="complete",
                          expanded=False)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}", icon=":material/error:")
        st.stop()

    decision = res.get("decision") or {}
    verification = res.get("verification") or {}

    left, right = st.columns([2, 1])
    with right:
        with st.container(border=True):
            st.markdown("**Outcome**")
            render_status(res["status"], res["requires_human_decision"])
            recomputed = verification.get("recomputed_risk")
            if recomputed:
                render_risk(recomputed["attrition_probability"],
                            recomputed["risk_level"])
            u = res["usage"]
            st.caption(
                f"{res['tool_calls_used']}/{llm_agent.MAX_TOOL_CALLS} tool calls · "
                f"{res['verify_retries']} corrections · {u['llm_calls']} LLM calls\n\n"
                f"{u['total_tokens']} tokens · {u['wall_clock_seconds']}s · "
                f"est. ${u['estimated_cost_usd']:.5f}")

    with left:
        if verify:
            with st.container(border=True):
                st.markdown("**Verified evidence**")
                st.caption("Recomputed from the model and the cohort — not the "
                           "model's own wording.")
                render_evidence(
                    [v["statement"] for v in verification.get("verified_drivers", [])],
                    "No claim survived verification.")
        else:
            st.warning(
                "Mode 2 accepts the model's conclusion as written. Nothing "
                "below has been checked against the tools — that is the point "
                "of the comparison, not an oversight.",
                icon=":material/warning:")

        if decision.get("rationale"):
            with st.container(border=True):
                st.markdown("**The model's rationale**")
                st.write(decision["rationale"])

        if verification.get("problems"):
            with st.container(border=True):
                st.markdown("**Unresolved objections**")
                for p in verification["problems"]:
                    st.markdown(f"- {p}")

    with st.expander("Final payload and trace", icon=":material/route:"):
        st.json({"decision": decision, "trace": res["trace"]})
