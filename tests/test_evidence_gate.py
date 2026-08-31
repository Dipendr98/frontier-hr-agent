"""
Regression tests for the direction-aware evidence gate (Iteration 13).

The defect these lock down: the cohort-percentile floor was applied to the RAW
percentile, which only means "unusually bad" for features where LOW is the
concerning direction. For overtime, commute distance, business travel and
previous-employer count — where HIGH is the concerning direction — a bad value
sits at a HIGH percentile and was therefore rejected as "not unusual".

Measured before the fix, across all 342 employees: bad-when-high features were
confirmed as drivers exactly 0 times and rejected as `not_unusual_vs_cohort`
251 times. Overtime alone accounted for 104 of those, while a lever for it sat
in the catalogue unusable. The documentation said "worst 35% of the cohort" the
whole time; the code said something else.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pytest  # noqa: E402

from advanced.agents import risk_analysis, root_cause  # noqa: E402
from advanced.tools import FEATURE_META, ordinal  # noqa: E402


def test_severity_percentile_inverts_for_bad_when_high(toolbox):
    """Overtime=1 is unusual-BAD even though it sits at a high raw percentile."""
    pct = toolbox.get_cohort_percentile("OverTime_flag", 1)
    assert pct["bad_when"] == "high"
    assert pct["percentile"] > 50, "most of the cohort does not work overtime"
    assert pct["severity_percentile"] == pytest.approx(100.0 - pct["percentile"])
    assert pct["severity_percentile"] < root_cause.MAX_SEVERITY_PERCENTILE


def test_severity_percentile_is_identity_for_bad_when_low(toolbox):
    pct = toolbox.get_cohort_percentile("JobSatisfaction", 1)
    assert pct["bad_when"] == "low"
    assert pct["severity_percentile"] == pct["percentile"]


def test_bad_when_high_features_can_now_be_confirmed(toolbox, cohort_df):
    """
    The regression itself: at least one bad-when-high feature must be
    confirmable somewhere in the cohort. Before the fix this was impossible by
    construction, for every employee and every such feature.
    """
    high_bad = {f for f, m in FEATURE_META.items() if m["bad_when"] == "high"}
    confirmed_high_bad = set()
    for _, row in cohort_df.head(120).iterrows():
        emp = row.to_dict()
        risk = risk_analysis.run(emp, toolbox)
        rc = root_cause.run(emp, risk, toolbox)
        confirmed_high_bad |= {c["feature"] for c in rc["confirmed_drivers"]} & high_bad
    assert confirmed_high_bad, (
        "no bad-when-high feature was ever confirmed — the percentile gate is "
        "direction-blind again")


def test_good_value_on_bad_when_high_feature_is_not_a_driver(toolbox):
    """Not working overtime must never be evidence of attrition risk."""
    pct = toolbox.get_cohort_percentile("OverTime_flag", 0)
    assert pct["severity_percentile"] > root_cause.MAX_SEVERITY_PERCENTILE


def test_confirmed_drivers_are_split_by_whether_a_lever_exists(toolbox, cohort_df):
    """
    A driver with no honest intervention is reported but never actioned, so the
    two lists must stay disjoint and must cover every confirmed driver.
    """
    for _, row in cohort_df.head(60).iterrows():
        emp = row.to_dict()
        risk = risk_analysis.run(emp, toolbox)
        rc = root_cause.run(emp, risk, toolbox)
        confirmed = {c["feature"] for c in rc["confirmed_drivers"]}
        actionable = set(rc["actionable_features"])
        contextual = set(rc["contextual_features"])
        assert actionable | contextual == confirmed
        assert not (actionable & contextual)


def test_ordinal_suffixes():
    """'81th percentile' in a document a person signs is a tell, not a typo."""
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 81, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
        "81st", "100th"]


def test_unknown_feature_returns_error_not_exception(toolbox):
    out = toolbox.get_cohort_percentile("NotAFeature", 1)
    assert out["error"] == "unknown_feature"


def test_percentile_matches_the_naive_scan(toolbox, cohort_df):
    """
    The binary-search percentile must equal `(series < value).mean()` exactly.

    That scan is what the gate was originally written against and what every
    reported number was measured with, so an optimisation that shifts a value
    by a fraction of a percentile could silently move a driver across the 35%
    floor. Checked for every feature at every value the cohort actually
    contains, plus the boundaries.
    """
    for feat in toolbox.features:
        series = cohort_df[feat]
        values = set(series.unique()) | {series.min() - 1, series.max() + 1}
        for value in values:
            expected = float((series < value).mean() * 100)
            got = toolbox.get_cohort_percentile(feat, value)["percentile"]
            assert got == pytest.approx(round(expected, 1)), (feat, value)


def test_cohort_mean_and_median_match_pandas(toolbox, cohort_df):
    """Precomputed summaries must equal what they replaced."""
    for feat in toolbox.features:
        assert toolbox._means[feat] == pytest.approx(
            round(float(cohort_df[feat].mean()), 2))
        assert toolbox._medians[feat] == pytest.approx(float(cohort_df[feat].median()))
