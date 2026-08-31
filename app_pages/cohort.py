"""The whole cohort as a reviewer's worklist, plus what only the cohort shows."""
import os

import pandas as pd
import streamlit as st

from advanced.orchestration.cohort import run_cohort
from app_shared import BASE_DIR, render_sidebar_provider

st.title("Cohort triage")
st.caption("A prioritised worklist with every case already built — and the "
           "findings that no single case can show.")

render_sidebar_provider()

with st.sidebar:
    st.subheader("Run")
    limit = st.slider("Employees to review", 20, 342, 342, step=20)
    use_memory = st.toggle("Cross-case memory", value=True,
                           help="Adds cohort context and systemic findings. "
                                "Never changes an outcome.")
    narrate = st.toggle(
        "LLM-written explanations", value=False,
        help="Off by default. Rewrites each explanation with the language "
             "model. It changes no decision and costs ~2 API calls per "
             "employee per pass — on the full cohort that is roughly 900 "
             "calls and the better part of an hour instead of two seconds.")
    if narrate:
        st.warning(
            f"~{limit * 2 * 2} API calls, several minutes. Decisions and every "
            "number below will be identical either way.",
            icon=":material/hourglass_top:")
    go = st.button("Run triage", type="primary", width="stretch",
                   icon=":material/play_arrow:")

if not go:
    st.info("Set the size on the left and run. The full 342-employee cohort "
            "takes about two seconds and needs no API key.",
            icon=":material/groups:")
    st.stop()

with st.spinner(f"Reviewing {limit} employees..."):
    report = run_cohort(limit=limit, use_memory=use_memory, save=True,
                        narrate=narrate)

counts = report["status_counts"]
with st.container(horizontal=True):
    st.metric("Reviewed", report["n_employees"], border=True)
    st.metric("Escalated", counts.get("ESCALATED", 0), border=True,
              help="The system would not sign these off. Read them first.")
    st.metric("Awaiting decision", counts.get("APPROVED_FOR_REVIEW", 0), border=True)
    st.metric("No action", counts.get("NO_ACTION", 0), border=True)
    st.metric("Per case", f"{report['seconds_per_case'] * 1000:.0f} ms", border=True)

# --- what only the cohort shows -------------------------------------------

if report["systemic_findings"]:
    st.subheader("Cohort-level findings")
    for f in report["systemic_findings"]:
        with st.container(border=True):
            st.markdown(f"**{f['label']}** — {f['affected_count']} of the "
                        f"{f['n_actioned']} employees receiving an intervention "
                        f"({f['share_of_actioned']:.0%})")
            st.write(f["finding"])
            st.caption(f"Addressed to: {f['addressed_to']} · "
                       f"e.g. {', '.join(f['affected_sample'])}")
elif use_memory:
    st.caption("No intervention reached the concentration threshold that would "
               "make it a structural finding rather than individual ones.")

# --- the worklist ----------------------------------------------------------

st.subheader("Worklist")
st.caption("Escalations first, then by risk. Nothing here has been actioned.")

rows = pd.DataFrame([{
    "Employee": r["employee_id"],
    "Status": r["status"].replace("_", " ").title(),
    "Risk": r["risk_level"],
    # Stored 0-100, not 0-1. ProgressColumn applies the printf format to the
    # raw value, so a probability of 0.989 under "%.1f%%" renders as "1.0%" —
    # a 99% risk displayed as 1%, next to a full-width bar contradicting it.
    "Probability": (r["attrition_probability"] or 0) * 100,
    "Proposed": r["recommendation"] or "—",
    "Simulated Δ (pp)": r["simulated_delta_pp"],
    "Evidence": len(r["evidence"]),
    "Needs a decision": r["requires_human_decision"],
} for r in report["worklist"]])

st.dataframe(
    rows, hide_index=True, width="stretch",
    column_config={
        "Probability": st.column_config.ProgressColumn(
            "Attrition risk", format="%.1f%%", min_value=0, max_value=100),
        "Simulated Δ (pp)": st.column_config.NumberColumn(
            format="%+.1f", help="Model what-if, not a causal effect estimate."),
        "Evidence": st.column_config.NumberColumn(
            help="Confirmed drivers cited on the case."),
    },
)

# --- the deliverable -------------------------------------------------------

briefing_path = os.path.join(BASE_DIR, "evidence", "cohort_briefing.md")
if os.path.exists(briefing_path):
    with open(briefing_path) as f:
        briefing = f.read()
    st.subheader("Reviewer briefing")
    with st.container(horizontal=True):
        st.download_button("Download briefing", briefing, "cohort_briefing.md",
                           "text/markdown", icon=":material/download:")
        st.download_button("Download worklist (CSV)", rows.to_csv(index=False),
                           "cohort_worklist.csv", "text/csv",
                           icon=":material/download:")
    with st.expander("Preview", icon=":material/description:"):
        st.markdown(briefing)
