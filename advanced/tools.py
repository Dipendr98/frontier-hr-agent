"""
Tool layer. Agents reach data and models ONLY through here.

Design rule that matters for judging: these tools return measured facts
(model coefficients, cohort percentiles, counterfactual re-scores). No tool
invents a narrative. Any explanation appearing in a trajectory is traceable
to a number produced in this file.
"""
import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "baseline")
DATA_PATH = os.path.join(BASE_DIR, "data", "onboarding_data.csv")

# Human-readable labels and whether a LOW value is the concerning direction.
FEATURE_META = {
    "JobSatisfaction":          {"label": "Job satisfaction",             "bad_when": "low"},
    "EnvironmentSatisfaction":  {"label": "Environment satisfaction",     "bad_when": "low"},
    "JobInvolvement":           {"label": "Job involvement",              "bad_when": "low"},
    "WorkLifeBalance":          {"label": "Work-life balance",            "bad_when": "low"},
    "RelationshipSatisfaction": {"label": "Relationship satisfaction",    "bad_when": "low"},
    "TrainingTimesLastYear":    {"label": "Training sessions attended",   "bad_when": "low"},
    "OverTime_flag":            {"label": "Working overtime",             "bad_when": "high"},
    "DistanceFromHome":         {"label": "Commute distance",             "bad_when": "high"},
    "BusinessTravel_freq":      {"label": "Business travel frequency",    "bad_when": "high"},
    "JobLevel":                 {"label": "Job level",                    "bad_when": "low"},
    "StockOptionLevel":         {"label": "Stock option level",           "bad_when": "low"},
    "NumCompaniesWorked":       {"label": "Previous employers",           "bad_when": "high"},
    "YearsWithCurrManager":     {"label": "Years with current manager",   "bad_when": "low"},
}

# Which interventions have a plausible mechanism on which feature.
# An intervention may only be proposed for a driver it can actually move.
FEATURE_LEVERS = {
    "JobSatisfaction":          ["role_fit_conversation", "assign_mentor"],
    "EnvironmentSatisfaction":  ["team_environment_review"],
    "JobInvolvement":           ["ownership_assignment", "assign_mentor"],
    "WorkLifeBalance":          ["workload_review"],
    "RelationshipSatisfaction": ["increase_manager_checkins", "assign_mentor"],
    "TrainingTimesLastYear":    ["structured_training_time"],
    "OverTime_flag":            ["workload_review"],
    "YearsWithCurrManager":     ["increase_manager_checkins"],
    "StockOptionLevel":         ["compensation_review"],
    # Deliberately NO levers: DistanceFromHome, BusinessTravel_freq, JobLevel,
    # NumCompaniesWorked. These are either fixed facts about the person or
    # things an onboarding intervention has no honest mechanism to change.
}

INTERVENTION_CATALOG = {
    "increase_manager_checkins": {
        "label": "Increase manager check-in frequency to weekly",
        "simulated_effect": {"RelationshipSatisfaction": +1, "YearsWithCurrManager": 0},
    },
    "structured_training_time": {
        "label": "Allocate protected time for outstanding training",
        "simulated_effect": {"TrainingTimesLastYear": +2},
    },
    "assign_mentor": {
        "label": "Assign an onboarding mentor",
        "simulated_effect": {"JobInvolvement": +1, "RelationshipSatisfaction": +1},
    },
    "workload_review": {
        "label": "Review workload and reduce overtime",
        "simulated_effect": {"OverTime_flag": -1, "WorkLifeBalance": +1},
    },
    "role_fit_conversation": {
        "label": "Hold a role-fit conversation and adjust responsibilities",
        "simulated_effect": {"JobSatisfaction": +1},
    },
    "team_environment_review": {
        "label": "Review team environment and working conditions",
        "simulated_effect": {"EnvironmentSatisfaction": +1},
    },
    "ownership_assignment": {
        "label": "Assign clear ownership of a meaningful workstream",
        "simulated_effect": {"JobInvolvement": +1},
    },
    "compensation_review": {
        "label": "Review compensation and equity package",
        "simulated_effect": {"StockOptionLevel": +1},
    },
}

# Ranges the survey-style features are valid within, so simulations stay
# inside the space the model was trained on.
FEATURE_BOUNDS = {
    "JobSatisfaction": (1, 4),
    "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4),
    "WorkLifeBalance": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "TrainingTimesLastYear": (0, 6),
    "OverTime_flag": (0, 1),
    "StockOptionLevel": (0, 3),
}


def ordinal(n: float) -> str:
    """1 -> 1st, 2 -> 2nd, 11 -> 11th, 81 -> 81st."""
    i = int(round(n))
    if 11 <= (i % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


def evidence_statement(label: str, value: float, cohort_mean: float,
                       percentile: float, severity_percentile: float,
                       contribution: float) -> str:
    """
    The single canonical wording for a confirmed driver.

    Both the deterministic Root Cause agent and the Mode 3 Claim Verifier cite
    evidence, and the evaluation rubric re-parses what they write. Three copies
    of this sentence would eventually disagree about rounding or phrasing and
    the rubric would start failing one of them for reasons that have nothing to
    do with whether the claim is true. One function, one sentence.
    """
    if severity_percentile <= 0.5:
        standing = "the worst in the cohort"
    elif severity_percentile >= 99.5:
        standing = "the best in the cohort"
    else:
        standing = f"worst {severity_percentile:.0f}% of the cohort"
    return (
        f"{label} is {value:g} vs cohort mean {cohort_mean:g} "
        f"({ordinal(percentile)} percentile, {standing}), contributing "
        f"{contribution:+.2f} to the risk logit."
    )


class ToolBox:
    """All tools available to the agents."""

    def __init__(self, model_dir: str = MODEL_DIR, data_path: str = DATA_PATH, dataframe: pd.DataFrame = None):
        self.model = joblib.load(os.path.join(model_dir, "attrition_model.joblib"))
        if dataframe is not None:
            self.cohort = dataframe
        else:
            self.cohort = pd.read_csv(data_path)
        
        self.features = [c for c in self.cohort.columns
                         if c not in ("employee_id", "attrition")]
        self.call_log = []

        # Sorted copy of each feature column, built once.
        #
        # `get_cohort_percentile` is called several times per employee and the
        # obvious implementation — (series < value).mean() — scans the whole
        # column every time. That is O(n) per call and O(n^2) over a cohort:
        # fine at 342 employees, 3.2 billion comparisons at 20,000. Sorting up
        # front turns each lookup into a binary search.
        self._sorted = {f: np.sort(self.cohort[f].to_numpy()) for f in self.features}
        self._medians = {f: float(self.cohort[f].median()) for f in self.features}
        self._means = {f: round(float(self.cohort[f].mean()), 2) for f in self.features}

    # -- bookkeeping ------------------------------------------------------

    def _log(self, name, args, result):
        self.call_log.append({"tool": name, "args": args, "result": result})
        return result

    def reset_log(self):
        self.call_log = []

    # -- tools ------------------------------------------------------------

    def get_employee_profile(self, employee_id: str) -> dict:
        """Fetch one employee's onboarding record."""
        row = self.cohort[self.cohort["employee_id"] == employee_id]
        if row.empty:
            return self._log("get_employee_profile", {"employee_id": employee_id},
                             {"error": "employee_not_found"})
        return self._log("get_employee_profile", {"employee_id": employee_id},
                         row.iloc[0].to_dict())

    def predict_risk(self, employee: dict) -> dict:
        """Score one employee with the trained attrition model."""
        X = pd.DataFrame([employee])[self.features]
        proba = float(self.model.predict_proba(X)[0, 1])
        out = {"attrition_probability": round(proba, 4)}
        return self._log("predict_risk", {"employee_id": employee.get("employee_id")}, out)

    def get_cohort_percentile(self, feature: str, value: float) -> dict:
        """
        Where this employee sits versus the onboarding cohort.

        Returns BOTH a raw percentile and a direction-aware `severity_percentile`.
        The distinction is not cosmetic and it cost us a real defect:

          `percentile` is "% of the cohort below this value". For a feature where
          LOW is the concerning direction (job satisfaction) a low percentile means
          unusual-bad. For a feature where HIGH is concerning (overtime, commute) a
          low percentile means unusual-GOOD, and a gate written as
          `percentile <= 35` silently rejects every high-bad feature that exists.

        `severity_percentile` normalises that: 0 = worst in the cohort on this
        feature, 100 = best, regardless of direction. A single threshold on it
        means "in the worst X% of the cohort", which is what the rule was always
        supposed to say. See Iteration 13 in CHANGELOG.md.
        """
        if feature not in self.cohort.columns:
            return self._log("get_cohort_percentile", {"feature": feature, "value": value},
                             {"error": "unknown_feature", "feature": feature,
                              "known_features": self.features})
        # searchsorted(..., "left") counts strictly-smaller values, matching the
        # original (series < value).mean() exactly — verified by
        # test_percentile_matches_the_naive_scan over every feature and value.
        arr = self._sorted[feature]
        pct = float(np.searchsorted(arr, value, side="left") / len(arr) * 100)
        bad_when = FEATURE_META.get(feature, {}).get("bad_when", "low")
        severity = pct if bad_when == "low" else 100.0 - pct
        out = {"feature": feature, "value": value, "percentile": round(pct, 1),
               "bad_when": bad_when, "severity_percentile": round(severity, 1),
               "cohort_median": self._medians[feature]}
        return self._log("get_cohort_percentile", {"feature": feature, "value": value}, out)

    def get_risk_drivers(self, employee: dict, top_n: int = 4) -> list:
        """
        Rank the features driving THIS employee's score.

        For a scaled logistic regression the logit is a linear sum, so each
        feature contributes coef * z where z is its standardised deviation.
        That is an exact decomposition of the model's own output, not a
        post-hoc story. Positive contribution pushes risk up.
        """
        scaler = self.model.named_steps["scaler"]
        clf = self.model.named_steps["clf"]
        X = pd.DataFrame([employee])[self.features]
        z = scaler.transform(X)[0]
        coefs = clf.coef_[0]

        drivers = []
        for feat, coef, zval in zip(self.features, coefs, z):
            contribution = float(coef * zval)
            drivers.append({
                "feature": feat,
                "label": FEATURE_META.get(feat, {}).get("label", feat),
                "value": float(employee[feat]),
                "cohort_mean": self._means[feat],
                "logit_contribution": round(contribution, 4),
                "raises_risk": contribution > 0,
            })
        drivers.sort(key=lambda d: d["logit_contribution"], reverse=True)
        top = drivers[:top_n]
        return self._log("get_risk_drivers",
                         {"employee_id": employee.get("employee_id"), "top_n": top_n}, top)

    def simulate_intervention(self, employee: dict, intervention_key: str) -> dict:
        """
        Re-score the employee after applying an intervention's assumed effect.

        This is a MODEL-BASED WHAT-IF over correlational features, NOT a causal
        effect estimate. It answers 'what would the model predict if these
        inputs changed', which is not 'what would happen if HR did this'.
        Labelled as such everywhere downstream.
        """
        spec = INTERVENTION_CATALOG.get(intervention_key)
        if spec is None:
            return self._log("simulate_intervention", {"intervention": intervention_key},
                             {"error": "unknown_intervention"})

        before = float(self.model.predict_proba(pd.DataFrame([employee])[self.features])[0, 1])
        modified = dict(employee)
        for feat, delta in spec["simulated_effect"].items():
            if delta == 0:
                continue
            new_val = modified[feat] + delta
            lo, hi = FEATURE_BOUNDS.get(feat, (None, None))
            if lo is not None:
                new_val = min(hi, max(lo, new_val))
            modified[feat] = new_val
        after = float(self.model.predict_proba(pd.DataFrame([modified])[self.features])[0, 1])

        out = {
            "intervention": intervention_key,
            "label": spec["label"],
            "risk_before": round(before, 4),
            "risk_after_simulated": round(after, 4),
            "delta_pp": round((after - before) * 100, 2),
            "caveat": "SIMULATED model what-if on correlational features; NOT a causal effect.",
        }
        return self._log("simulate_intervention",
                         {"employee_id": employee.get("employee_id"),
                          "intervention": intervention_key}, out)
