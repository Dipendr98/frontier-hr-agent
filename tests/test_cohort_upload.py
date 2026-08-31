"""
Tests for the custom-cohort upload path.

The dashboard lets anyone drop in a CSV, and the cohort runner takes `--data`.
Both accept arbitrary files, so both are the place where a stranger's first
interaction with this project happens — and before these tests every wrong file
produced a raw traceback from inside pandas or scikit-learn:

    KeyError: 'employee_id'
    TypeError: Cannot perform reduction 'median' with string dtype
    ValueError: The feature names should match those that were passed during fit

Each of those is accurate and tells the person nothing about what to fix. A
validation layer is only worth having if it catches every one of them, so each
gets its own case here.

The schema is read from the trained model (`feature_names_in_`) rather than
hardcoded, so retraining on a different feature set cannot leave the validator
checking for columns the model no longer wants.
"""
import os
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from advanced.tools import (MIN_COHORT_ROWS, CohortSchemaError,  # noqa: E402
                            ToolBox, validate_cohort)


@pytest.fixture(scope="module")
def expected(toolbox):
    return list(getattr(toolbox.model, "feature_names_in_", []))


# --- the happy path --------------------------------------------------------

def test_the_shipped_cohort_validates(cohort_df, expected):
    assert validate_cohort(cohort_df, expected) == []


def test_extra_columns_are_ignored_not_rejected(cohort_df):
    """
    A real HR export has payroll, department, manager name and much else. The
    model uses 13 columns; everything else should be tolerated silently rather
    than forcing people to strip their file down first.
    """
    tb = ToolBox(dataframe=cohort_df.assign(Salary=50000, Dept="Sales"))
    emp = tb.get_employee_profile(tb.cohort["employee_id"].iloc[0])
    assert 0.0 <= tb.predict_risk(emp)["attrition_probability"] <= 1.0
    assert "Salary" not in tb.features


# --- each raw exception that used to escape --------------------------------

def test_missing_feature_is_named(cohort_df, expected):
    problems = validate_cohort(cohort_df.drop(columns=["OverTime_flag"]), expected)
    assert any("OverTime_flag" in p for p in problems)


def test_missing_employee_id_is_explained(cohort_df, expected):
    problems = validate_cohort(cohort_df.drop(columns=["employee_id"]), expected)
    assert any("employee_id" in p for p in problems)


def test_non_numeric_feature_names_the_column_and_the_value(cohort_df, expected):
    problems = validate_cohort(cohort_df.assign(JobSatisfaction="high"), expected)
    assert any("JobSatisfaction" in p and "high" in p for p in problems)


def test_an_unrelated_csv_is_rejected(expected):
    problems = validate_cohort(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), expected)
    assert problems


def test_an_empty_file_says_it_expects_a_csv(expected):
    """What someone who uploaded a PDF needs to be told."""
    problems = validate_cohort(pd.DataFrame(), expected)
    assert len(problems) == 1
    assert "CSV" in problems[0] and "PDF" in problems[0]


def test_a_tiny_cohort_is_rejected(cohort_df, expected):
    """
    Percentiles are the evidence. Over three people they are not a statement
    about anyone, so a file that small is refused rather than scored.
    """
    problems = validate_cohort(cohort_df.head(3), expected)
    assert any(str(MIN_COHORT_ROWS) in p for p in problems)


# --- the contract the dashboard relies on ----------------------------------

def test_toolbox_raises_the_typed_error_not_a_raw_one(cohort_df):
    """
    The dashboard catches CohortSchemaError specifically. If a bad cohort
    escapes as KeyError or ValueError instead, the user gets a traceback.
    """
    for bad in (cohort_df.drop(columns=["employee_id"]),
                cohort_df.drop(columns=["OverTime_flag"]),
                cohort_df.assign(WorkLifeBalance="low"),
                pd.DataFrame(),
                cohort_df.head(2)):
        with pytest.raises(CohortSchemaError):
            ToolBox(dataframe=bad)


def test_every_problem_message_is_actionable(cohort_df, expected):
    """
    Each message must name the thing to fix. "Invalid input" is a message that
    tells someone only that they have failed.
    """
    cases = [cohort_df.drop(columns=["employee_id"]),
             cohort_df.drop(columns=["JobLevel"]),
             cohort_df.assign(JobInvolvement="yes"),
             pd.DataFrame()]
    for df in cases:
        for problem in validate_cohort(df, expected):
            assert len(problem) > 40, f"too terse to act on: {problem!r}"


def test_validator_schema_comes_from_the_model(toolbox):
    """Retraining on new features must not leave the validator behind."""
    assert list(toolbox.model.feature_names_in_) == toolbox.features


# --- wrong file type entirely ----------------------------------------------

def _load(raw: bytes):
    import app_shared
    app_shared.get_toolbox.clear()
    return app_shared.get_toolbox(raw)


@pytest.mark.parametrize("raw,expect", [
    (b"%PDF-1.4\n1 0 obj\n", "PDF"),
    (b"PK\x03\x04\x14\x00\x00\x00", "Excel"),
    (b'{"employees": []}', "JSON"),
    (b"<html><body>hi</body></html>", "HTML"),
])
def test_wrong_file_types_are_named_not_misdiagnosed(raw, expect):
    """
    Regression: a PDF was parsed by pandas as a one-column CSV and reported as
    "No employee_id column" — true, and it sends the person to edit a
    spreadsheet that was never the problem. The message must name the format.
    """
    with pytest.raises(CohortSchemaError, match=expect):
        _load(raw)
