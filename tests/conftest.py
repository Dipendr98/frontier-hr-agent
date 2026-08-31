"""
Shared test fixtures.

Importing `advanced` loads .env (see advanced/__init__.py), which is what makes
the LLM-dependent tests actually run on a machine that has a provider
configured. Before this existed only app.py called load_dotenv(), so those
tests skipped silently even with a working key sitting in .env — a test suite
that reports "3 skipped" when it could have reported "3 passed" is quietly
telling you less than it knows.
"""
import os
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import advanced  # noqa: F401,E402  (loads .env as a side effect)
from advanced import llm  # noqa: E402
from advanced.tools import ToolBox  # noqa: E402

# Live-agent tests skip when there is no provider, and can also be skipped
# explicitly with FRONTIER_SKIP_LIVE=1.
#
# The explicit switch exists because run_all.sh advertises a ~60 second
# reproduction, and that promise quietly broke for anyone with a key in .env:
# the seven live tests each drive a full agent loop, so the same script took ten
# minutes and looked like a hang. No headline number depends on a provider, so
# the reproduction script should not either. It now sets this flag, and the live
# tests get their own command.
SKIP_LIVE = os.environ.get("FRONTIER_SKIP_LIVE", "").strip().lower() in ("1", "true", "yes")

requires_llm = pytest.mark.skipif(
    SKIP_LIVE or not llm.is_enabled(),
    reason=("FRONTIER_SKIP_LIVE=1 — run without it to exercise the live agent"
            if SKIP_LIVE else
            "LLM_PRESET/LLM_API_KEY not set — this test requires a real provider"),
)


@pytest.fixture(scope="session")
def cohort_df():
    return pd.read_csv(os.path.join(BASE_DIR, "data", "onboarding_data.csv"))


@pytest.fixture(scope="module")
def toolbox():
    return ToolBox()


@pytest.fixture(scope="module")
def employee(toolbox, cohort_df):
    return cohort_df.iloc[0].to_dict()


@pytest.fixture(autouse=True)
def _no_narration():
    """
    Tests assert on DECISIONS, which must not depend on a provider being up.

    Mode 1's prose layer is disabled for every test so a slow or rate-limited
    provider cannot turn a logic assertion into a flaky network assertion. Tests
    that specifically exercise the LLM re-enable it themselves.
    """
    before = llm.NARRATION
    llm.set_narration(False)
    yield
    llm.set_narration(before)
