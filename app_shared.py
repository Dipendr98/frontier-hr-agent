"""
Shared loaders and helpers for the dashboard pages.

Kept out of the page scripts so each page stays a straight-line script, and so
the one genuinely expensive thing — the trained model plus the cohort — is
loaded once per dataset rather than once per rerun.

Note what is NOT cached: the generated files in evidence/. They are small, and
caching them is how the evidence page ended up showing numbers from whenever the
server happened to start. See `load_evidence`.
"""
import json
import os

import pandas as pd
import streamlit as st

from advanced import llm
from advanced.tools import FEATURE_META, ToolBox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

RISK_COLOUR = {"High": "red", "Medium": "orange", "Low": "green"}
STATUS_BADGE = {
    "APPROVED_FOR_REVIEW": ("blue", ":material/how_to_reg:"),
    "ESCALATED": ("orange", ":material/priority_high:"),
    "NO_ACTION": ("green", ":material/check:"),
    "HALTED_DATA_QUALITY": ("red", ":material/block:"),
    "LLM_ERROR": ("red", ":material/cloud_off:"),
    "COMPLETED": ("blue", ":material/done_all:"),
    "NO_FINALIZE_CALLED": ("red", ":material/timer_off:"),
}


@st.cache_resource(show_spinner="Loading model and cohort...")
def get_toolbox(data_bytes: bytes = None) -> ToolBox:
    """
    One ToolBox per distinct dataset.

    Cached as a resource, not data: it holds the loaded scikit-learn pipeline,
    which is a shared object rather than something to serialise per session.
    """
    if data_bytes is None:
        return ToolBox()
    import io
    return ToolBox(dataframe=pd.read_csv(io.BytesIO(data_bytes)))


def load_evidence(name: str):
    """
    Read a generated evidence file. Deliberately NOT cached.

    It was cached, on the filename alone, which meant a dashboard left open
    across an evaluation kept serving the numbers it had when the server
    started — the worst possible staleness, on the one page whose entire job is
    to show current evidence, and invisible because the old numbers look
    perfectly plausible.

    The first fix put the file's mtime in the cache key. That works until two
    writes land in the same second, which filesystems with 1-second mtime
    granularity make easy and a regenerate-then-refresh makes likely. The tell
    was in the test: it had to call os.utime() to push mtime forward before the
    assertion would pass. A test that manipulates the clock to satisfy the
    implementation is testing the implementation, not the behaviour.

    So: no cache. These are a handful of small JSON files (the largest is a few
    KB) read once per page render, and caching them was never worth anything —
    it was reflex, and it bought a correctness bug. If something genuinely large
    lands in evidence/ later, cache THAT call site with an explicit key, rather
    than making every reader of this function pay for it.
    """
    path = os.path.join(EVIDENCE_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        if name.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        return pd.read_csv(path)
    except (json.JSONDecodeError, OSError, pd.errors.ParserError):
        return None


def provider_status():
    """(ok_for_agent_modes, human readable label)."""
    if not llm.is_enabled():
        return False, "No provider — Mode 1 only"
    return True, llm.active_model()


def render_sidebar_provider():
    with st.sidebar:
        st.subheader("Provider")
        ok, label = provider_status()
        if ok:
            st.badge(label, icon=":material/cloud_done:", color="green")
            st.caption("Modes 2 and 3 available. Verify with "
                       "`python -m advanced.doctor`.")
        else:
            st.badge("Deterministic only", icon=":material/cloud_off:", color="grey")
            st.caption("Mode 1 reproduces every headline number with no key. "
                       "Set `LLM_PRESET` and `LLM_API_KEY` in `.env` for Modes 2/3.")


def render_employee_profile(emp: dict):
    """The four signals a reviewer looks at first, plus the rest on demand."""
    with st.container(horizontal=True):
        st.metric("Job satisfaction", f"{emp.get('JobSatisfaction', '—')} / 4", border=True)
        st.metric("Work-life balance", f"{emp.get('WorkLifeBalance', '—')} / 4", border=True)
        st.metric("Overtime", "Yes" if emp.get("OverTime_flag") == 1 else "No", border=True)
        st.metric("Years with manager", emp.get("YearsWithCurrManager", "—"), border=True)

    with st.expander("Full record", icon=":material/table_rows:"):
        rows = [{"Feature": FEATURE_META.get(k, {}).get("label", k), "Value": v}
                for k, v in emp.items()
                if k not in ("employee_id", "attrition")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if "attrition" in emp:
            st.caption(
                "This dataset carries a ground-truth `attrition` label used only "
                "for evaluation. No agent reads it — they would score perfectly "
                "if they did.")


def render_risk(probability: float, level: str):
    """
    Show the score and its band without implying a direction of travel.

    `st.metric(..., delta=level)` was wrong here: Streamlit renders a string
    delta as a green up-arrow, so a 79% attrition risk displayed as a green
    "↑ High" — which reads as good news about the worst case in the queue. A
    risk band is a category, not a change, so it gets a badge coloured by
    severity instead.
    """
    st.metric("Attrition risk", f"{probability:.1%}")
    st.badge(f"{level} risk", color=RISK_COLOUR.get(level, "grey"))


def render_status(status: str, requires_human: bool = True):
    colour, icon = STATUS_BADGE.get(status, ("grey", ":material/help:"))
    st.badge(status.replace("_", " ").title(), icon=icon, color=colour)
    if requires_human:
        st.caption("Nothing is executed by this system. A human reviewer decides.")


def render_evidence(statements: list, empty_note: str = None):
    if statements:
        for s in statements:
            st.markdown(f"- {s}")
    elif empty_note:
        st.caption(empty_note)


def no_evidence_yet(what: str, command: str):
    st.info(f"No {what} found yet.", icon=":material/science:")
    st.code(command, language="bash")
