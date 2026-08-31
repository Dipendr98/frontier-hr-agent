"""Every number this project claims, next to the file it came from."""
import pandas as pd
import streamlit as st

from app_shared import load_evidence, no_evidence_yet, render_sidebar_provider

st.title("Evidence")
st.caption("Each table below is read straight from `evidence/`. Regenerate any "
           "of them with the command shown.")

render_sidebar_provider()

tab_main, tab_llm, tab_sweep, tab_model = st.tabs(
    ["Baseline vs agent", "Verification", "Threshold sweep", "Model"])

# --- the headline ----------------------------------------------------------

with tab_main:
    report = load_evidence("evaluation_report.json")
    if not report:
        no_evidence_yet("evaluation report", "python evaluation/evaluate.py")
    else:
        s = report["systems"]
        st.subheader("Three systems, identical cases, identical rubric")
        st.caption(f"{report['eval_cases']} cases · {report['split']} · "
                   f"primary metric: {report['primary_metric']}")

        delta = report["headline_delta_vs_strongest_baseline"]["reviewer_case_quality_pct"]
        with st.container(horizontal=True):
            st.metric("Baseline", f"{s['baseline']['reviewer_case_quality_pct']:.1%}",
                      border=True)
            st.metric("Baseline+", f"{s['baseline_plus']['reviewer_case_quality_pct']:.1%}",
                      border=True,
                      help="Score + top cohort deviation + template. The "
                           "strongest thing you can build without agents.")
            st.metric("Agent", f"{s['agent']['reviewer_case_quality_pct']:.1%}",
                      f"{delta:+.1%} vs Baseline+", border=True)

        st.caption("The headline is the gap over Baseline+, not the larger gap "
                   "over the naive baseline. We report the harder comparison.")

        labels = [
            ("Reviewer case quality (primary)", "reviewer_case_quality_pct", "pct"),
            ("d1 correct triage", "d1_correct_triage", "pct"),
            ("d2 evidence present", "d2_evidence_present", "pct"),
            ("d3 evidence verifiable", "d3_evidence_verifiable", "pct"),
            ("d4 action has mechanism", "d4_action_has_mechanism", "pct"),
            ("d5 proportionate", "d5_proportionate", "pct"),
            ("Wasted intervention (counter)", "wasted_intervention_rate", "pct"),
            ("Catch rate w/ verified evidence", "catch_rate_with_verified_evidence", "pct"),
            ("Unverifiable evidence claims", "unverifiable_evidence_claims", "num"),
            ("Interventions proposed", "total_interventions_proposed", "num"),
            ("Avg tool calls per case", "avg_tool_calls", "num"),
        ]
        # Every cell is a string. Mixing "42.8%" and 44 in one column gives an
        # object dtype that Arrow refuses to serialise, which crashes the page
        # rather than degrading it — caught by tests/test_dashboard.py.
        def cell(system, key, fmt):
            v = s[system][key]
            return f"{v:.1%}" if fmt == "pct" else str(v)

        table = pd.DataFrame([{
            "Metric": name,
            "Baseline": cell("baseline", k, fmt),
            "Baseline+": cell("baseline_plus", k, fmt),
            "Agent": cell("agent", k, fmt),
        } for name, k, fmt in labels])
        st.dataframe(table, hide_index=True, width="stretch")

        st.caption("Dimension 3 is what stops a plausible sentence from "
                   "scoring: the rubric recomputes every cited number itself "
                   "and rejects anything it cannot confirm. It imports no agent "
                   "logic — the solution does not grade its own exam.")
        st.code("python evaluation/evaluate.py", language="bash")

# --- what verification buys ------------------------------------------------

with tab_llm:
    report = load_evidence("llm_agent_report.json")
    if not report:
        no_evidence_yet("Mode 2 / Mode 3 comparison",
                        "python evaluation/evaluate_llm_agent.py --n 20")
    else:
        m2, m3 = report["systems"]["mode2_raw"], report["systems"]["mode3_verified"]
        n = report.get("n_cases_scored") or report.get("n_cases") or 1
        attempted = report.get("n_cases_attempted", n)
        st.subheader("Does checking the model's answer help?")
        st.caption(f"{n} cases · same model, same tools, same prompt · "
                   f"{report['provider']}")
        if attempted > n:
            st.caption(f":material/info: {attempted} attempted; scored on the "
                       f"{n} both modes completed, so a rate limit in one mode "
                       f"cannot look like a quality difference.")

        with st.container(horizontal=True):
            st.metric("Mode 2 claims that failed verification",
                      f"{m2['raw_claim_error_rate']:.0%}", border=True,
                      help="Observed passively — checked but never corrected, "
                           "so the measurement does not change the behaviour.")
            st.metric("Case quality, Mode 2",
                      f"{m2['reviewer_case_quality_pct']:.1%}", border=True)
            st.metric("Case quality, Mode 3",
                      f"{m3['reviewer_case_quality_pct']:.1%}",
                      f"{m3['reviewer_case_quality_pct'] - m2['reviewer_case_quality_pct']:+.1%}",
                      border=True)

        alt = m3.get("variant_verified_evidence_only") or {}
        rows = [
            ("Unverifiable claims (incl. the model's prose)",
             m2["unverifiable_evidence_claims"], m3["unverifiable_evidence_claims"]),
            ("Unverifiable claims (recomputed evidence only)",
             "n/a", alt.get("unverifiable_evidence_claims", "—")),
            ("Wasted intervention rate", f"{m2['wasted_intervention_rate']:.1%}",
             f"{m3['wasted_intervention_rate']:.1%}"),
            ("Escalated to a human", m2["escalated"], m3["escalated"]),
            ("Corrections sent back", "—", m3["verify_retries"]),
            ("Avg tool calls per case", m2["avg_tool_calls"], m3["avg_tool_calls"]),
            ("LLM calls", m2["llm_calls"], m3["llm_calls"]),
            ("Est. cost per case (USD)",
             f"{m2['estimated_cost_usd'] / n:.5f}",
             f"{m3['estimated_cost_usd'] / n:.5f}"),
            ("Runs the provider never completed", m2["run_errors"], m3["run_errors"]),
        ]
        st.dataframe(
            pd.DataFrame([(a, str(b), str(c)) for a, b, c in rows],
                         columns=["Metric", "Mode 2 raw", "Mode 3 verified"]),
            hide_index=True, width="stretch")

        st.warning(
            "**What this does not show.** Verification does not reliably lift "
            "the composite case-quality score — at this sample size on a "
            "non-deterministic model it moves several points in both directions "
            "between runs. What moves consistently is the claim error rate and "
            "the unverifiable claims in the recomputed evidence. Reporting the "
            "noisy composite as the win would be the same mistake this project "
            "made once already.",
            icon=":material/balance:")

        if alt:
            st.caption(
                f"Sensitivity: scoring Mode 3 on its recomputed evidence alone, "
                f"with the model's prose excluded, gives "
                f"{alt['reviewer_case_quality_pct']:.1%} instead of "
                f"{m3['reviewer_case_quality_pct']:.1%}. Both are published; "
                f"neither is chosen for being kinder.")

        if m2.get("failed_checks"):
            st.subheader("What the verifier caught in unsupervised runs")
            st.dataframe(
                pd.DataFrame(sorted(m2["failed_checks"].items(),
                                    key=lambda kv: -kv[1]),
                             columns=["Check", "Times triggered"]),
                hide_index=True, width="stretch")

        failures = report.get("mode2_raw_claim_failures") or []
        if failures:
            with st.expander(f"The {len(failures)} unsupervised failures in full",
                             icon=":material/report:"):
                for f in failures:
                    st.markdown(f"**{f['employee_id']}**")
                    for p in f["problems"]:
                        st.markdown(f"- {p}")

        st.code("python evaluation/evaluate_llm_agent.py --n 20", language="bash")

# --- the threshold we did not tune to flatter ourselves --------------------

with tab_sweep:
    sweep = load_evidence("threshold_sweep.json")
    if not sweep:
        no_evidence_yet("threshold sweep", "python evaluation/threshold_sweep.py")
    else:
        st.subheader("Evidence threshold sweep")
        st.caption("`MIN_CONTRIBUTION` decides how strong a signal must be "
                   "before a driver may be cited.")
        rows = sweep.get("results", sweep) if isinstance(sweep, dict) else sweep
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info(
            "This sweep does not have a clean winner, and the row that beats "
            "our chosen setting is published above rather than omitted. A "
            "reviewer who weights the primary metric strictly should prefer the "
            "lower threshold; we chose the one that wastes less manager "
            "attention, because that is the scarce resource for this user.",
            icon=":material/balance:")
        st.code("python evaluation/threshold_sweep.py", language="bash")

# --- the model is a component, not the contribution ------------------------

with tab_model:
    metrics = load_evidence("baseline_metrics.json")
    fairness = load_evidence("fairness_cost.json")
    if not metrics:
        no_evidence_yet("model metrics", "python baseline/train.py")
    else:
        st.subheader("Model performance")
        st.caption("A supporting component. The contribution of this project is "
                   "the case built around the score, not the score.")
        st.json(metrics, expanded=True)
    if fairness:
        st.subheader("Cost of excluding protected attributes")
        st.caption("Gender, age and marital status are excluded from the "
                   "feature set. This measures what that costs.")
        st.json(fairness, expanded=True)
        st.code("python baseline/fairness_cost.py", language="bash")
