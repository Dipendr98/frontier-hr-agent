"""
Frontier HR Agent — dashboard entry point.

    streamlit run app.py

Three pages, matching the three things a reviewer actually does: work one case,
work the cohort, and check whether the system is worth trusting.
"""
import streamlit as st

st.set_page_config(
    page_title="Frontier HR Agent",
    page_icon=":material/groups:",
    layout="wide",
)

pages = [
    st.Page("app_pages/case_review.py", title="Case review",
            icon=":material/person_search:", default=True),
    st.Page("app_pages/cohort.py", title="Cohort triage",
            icon=":material/groups:"),
    st.Page("app_pages/evidence.py", title="Evidence",
            icon=":material/query_stats:"),
]

st.navigation(pages).run()
