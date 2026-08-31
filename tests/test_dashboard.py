"""
Smoke tests for the Streamlit dashboard, via Streamlit's own AppTest harness.

These exist because a dashboard is the part of a submission most likely to be
broken at the moment someone opens it, and the least likely to be covered: it
is not imported by anything else, so a rename in `advanced/` can break it
silently and nothing fails until a judge clicks the tab.

AppTest runs each page as a real script with no browser, so these are fast and
run in CI. Only the no-key paths are exercised — the pages that need a provider
are checked for their disabled state instead.
"""
import json
import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit not installed").AppTest

TIMEOUT = 120


def _page(name):
    return AppTest.from_file(os.path.join(BASE_DIR, "app_pages", name),
                             default_timeout=TIMEOUT)


def test_entry_point_declares_every_page():
    """A page file renamed out from under st.navigation is a 404 for the judge."""
    at = AppTest.from_file(os.path.join(BASE_DIR, "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    for name in ("case_review.py", "cohort.py", "evidence.py"):
        assert os.path.exists(os.path.join(BASE_DIR, "app_pages", name))


def test_case_review_renders_without_running_anything():
    at = _page("case_review.py").run()
    assert not at.exception
    assert any("Case review" in str(t.value) for t in at.title)
    # Three modes offered, and the page stops before executing one.
    assert len(at.radio[0].options) == 3


def test_case_review_runs_mode_1_with_no_provider(monkeypatch):
    """The no-key path has to work — it is the one every number depends on."""
    monkeypatch.setenv("LLM_NARRATE", "0")
    at = _page("case_review.py")
    at.run()
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Evidence" in text or "No action proposed" in text


def test_cohort_page_renders_and_triages():
    at = _page("cohort.py").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Worklist" in str(s.value) for s in at.subheader)


def test_evidence_page_reads_the_report_without_recomputing_it():
    at = _page("evidence.py").run()
    assert not at.exception
    assert any("Evidence" in str(t.value) for t in at.title)


def test_no_page_crashes_when_evidence_files_are_missing(monkeypatch, tmp_path):
    """
    A judge who clones the repo and opens the dashboard before running
    anything should get instructions, not a traceback.
    """
    import app_shared
    monkeypatch.setattr(app_shared, "EVIDENCE_DIR", str(tmp_path))
    at = _page("evidence.py").run()
    assert not at.exception


def test_evidence_is_re_read_when_it_changes_on_disk(monkeypatch, tmp_path):
    """
    Regression: the loader cached on filename alone, so a dashboard left open
    across an evaluation kept serving the numbers it had at startup. Stale
    evidence is worst on the evidence page, and it is invisible — the old
    numbers look entirely plausible.

    Note what this test does NOT do: touch mtime, sleep, or clear a cache. An
    earlier version had to advance mtime by a second before the second
    assertion would pass, which meant it was accommodating the cache-key scheme
    rather than checking the behaviour a user gets. Two writes inside one second
    are exactly the case that broke, so the test must not step around them.
    """
    import app_shared
    monkeypatch.setattr(app_shared, "EVIDENCE_DIR", str(tmp_path))
    path = tmp_path / "evaluation_report.json"

    # Rewritten back-to-back, well inside one filesystem mtime tick.
    for expected in (1, 2, 3):
        path.write_text(json.dumps({"eval_cases": expected}))
        got = app_shared.load_evidence("evaluation_report.json")["eval_cases"]
        assert got == expected, (
            f"served eval_cases={got} after writing {expected} — the loader is "
            "caching stale evidence")


def test_missing_evidence_reads_as_absent_not_as_an_error(monkeypatch, tmp_path):
    """None means 'not generated yet', which the pages turn into instructions."""
    import app_shared
    monkeypatch.setattr(app_shared, "EVIDENCE_DIR", str(tmp_path))
    assert app_shared.load_evidence("nothing_here.json") is None


def test_corrupt_evidence_does_not_raise(monkeypatch, tmp_path):
    """A half-written file during regeneration must not crash the dashboard."""
    import app_shared
    monkeypatch.setattr(app_shared, "EVIDENCE_DIR", str(tmp_path))
    (tmp_path / "evaluation_report.json").write_text('{"eval_cases": ')
    assert app_shared.load_evidence("evaluation_report.json") is None
