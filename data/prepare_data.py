"""
Prepares the IBM HR Analytics Employee Attrition dataset for the
early-tenure (onboarding) risk task.

Source: IBM Watson Analytics sample set, distributed on Kaggle. It is a
FICTIONAL dataset created by IBM data scientists — no real employee records
are used anywhere in this project.

Three decisions are made here, all deliberate and all defensible:

1. TENURE FILTER. The raw file covers the full employee lifecycle. This
   project is about the onboarding window, so we keep YearsAtCompany <= 2.

2. PROTECTED ATTRIBUTES ARE DROPPED. Gender, Age and MaritalStatus are
   removed from the feature set even though they carry signal. A model that
   routes retention interventions by protected class is not deployable
   whatever its AUC, and the hackathon rulebook requires an ethical use of
   people's data. The measured cost of this exclusion is reported in
   evidence/fairness_note.md rather than hidden.

3. NO FABRICATED PRODUCTIVITY TARGET. The raw data has no time-to-
   productivity column. Rather than invent one, the regression head is
   dropped and the pipeline predicts attrition risk only. See README.
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "onboarding_data.csv")

MAX_TENURE_YEARS = 2

# Excluded from features on ethical grounds, not statistical ones.
PROTECTED_ATTRIBUTES = ["Gender", "Age", "MaritalStatus"]

# Constant columns carry no information.
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]

# Columns that leak tenure/outcome or are identifiers.
LEAKY_OR_ID = ["EmployeeNumber"]

# The feature set the models actually see. Chosen to be things an HR team can
# observe during onboarding and, where possible, act on.
FEATURES = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "WorkLifeBalance",
    "RelationshipSatisfaction",
    "TrainingTimesLastYear",
    "OverTime_flag",
    "DistanceFromHome",
    "BusinessTravel_freq",
    "JobLevel",
    "StockOptionLevel",
    "NumCompaniesWorked",
    "YearsWithCurrManager",
]

BUSINESS_TRAVEL_MAP = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2,
}


def prepare() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    df = df[df["YearsAtCompany"] <= MAX_TENURE_YEARS].copy()

    df["employee_id"] = "EMP" + df["EmployeeNumber"].astype(str)
    df["attrition"] = (df["Attrition"] == "Yes").astype(int)
    df["OverTime_flag"] = (df["OverTime"] == "Yes").astype(int)
    df["BusinessTravel_freq"] = df["BusinessTravel"].map(BUSINESS_TRAVEL_MAP).fillna(1).astype(int)

    keep = ["employee_id", "attrition"] + FEATURES
    out = df[keep].dropna().reset_index(drop=True)
    return out


if __name__ == "__main__":
    out = prepare()
    out.to_csv(OUT_PATH, index=False)
    print(f"Rows: {len(out)}")
    print(f"Attrition rate: {out['attrition'].mean():.1%}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"Excluded (protected): {PROTECTED_ATTRIBUTES}")
    print(f"Excluded (constant): {CONSTANT_COLUMNS}")
    print(f"Excluded (identifier): {LEAKY_OR_ID}")
    print(f"\nWritten to {OUT_PATH}")
    print(out.head())
