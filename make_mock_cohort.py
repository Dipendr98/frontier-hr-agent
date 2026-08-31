"""
Generate a mock cohort for testing the upload path.

Deliberately NOT a copy of the shipped data. This invents a different company —
"Northwind" — with distributions that are clearly unlike the IBM cohort:
overtime is common rather than rare, satisfaction skews lower, commutes are
shorter. That difference is the point of the test. Evidence in this system is
relative to the cohort ("worst 30% of the cohort"), so if an uploaded file were
scored against the *built-in* cohort's statistics instead of its own, every
percentile in the output would be quietly wrong and nothing would crash.

Two employees are planted with known extremes so the output can be checked by
hand rather than only in aggregate.

    python make_mock_cohort.py       -> /tmp/northwind_cohort.csv
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 200


def main():
    df = pd.DataFrame({
        "employee_id": [f"NW{i:04d}" for i in range(N)],
        # Northwind is a harder place to work than the IBM cohort: lower
        # satisfaction, far more overtime, shorter commutes.
        "JobSatisfaction":          RNG.choice([1, 2, 3, 4], N, p=[.30, .30, .25, .15]),
        "EnvironmentSatisfaction":  RNG.choice([1, 2, 3, 4], N, p=[.25, .30, .25, .20]),
        "JobInvolvement":           RNG.choice([1, 2, 3, 4], N, p=[.10, .25, .45, .20]),
        "WorkLifeBalance":          RNG.choice([1, 2, 3, 4], N, p=[.20, .35, .30, .15]),
        "RelationshipSatisfaction": RNG.choice([1, 2, 3, 4], N, p=[.20, .25, .30, .25]),
        "TrainingTimesLastYear":    RNG.integers(0, 7, N),
        "OverTime_flag":            RNG.choice([0, 1], N, p=[.35, .65]),   # IBM: ~28%
        "DistanceFromHome":         RNG.integers(1, 15, N),                # IBM: up to 29
        "BusinessTravel_freq":      RNG.choice([0, 1, 2], N, p=[.20, .55, .25]),
        "JobLevel":                 RNG.choice([1, 2, 3], N, p=[.55, .35, .10]),
        "StockOptionLevel":         RNG.choice([0, 1, 2, 3], N, p=[.45, .35, .15, .05]),
        "NumCompaniesWorked":       RNG.integers(0, 8, N),
        "YearsWithCurrManager":     RNG.integers(0, 3, N),
        # Present only so the evaluation rubric can score; no agent reads it.
        "attrition":                RNG.choice([0, 1], N, p=[.70, .30]),
    })

    # Two planted cases, so the output can be checked against known extremes.
    df.loc[0, ["employee_id", "JobSatisfaction", "WorkLifeBalance",
               "OverTime_flag", "DistanceFromHome"]] = ["NW_WORST", 1, 1, 1, 14]
    df.loc[1, ["employee_id", "JobSatisfaction", "WorkLifeBalance",
               "OverTime_flag", "DistanceFromHome"]] = ["NW_BEST", 4, 4, 0, 1]

    path = "/tmp/northwind_cohort.csv"
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} employees")
    print(f"  overtime rate     {df['OverTime_flag'].mean():.0%}  (IBM cohort: 28%)")
    print(f"  mean job satisf.  {df['JobSatisfaction'].mean():.2f}  (IBM cohort: 2.76)")
    print(f"  mean commute      {df['DistanceFromHome'].mean():.1f}  (IBM cohort: 9.1)")
    return path


if __name__ == "__main__":
    main()
