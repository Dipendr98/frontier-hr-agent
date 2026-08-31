# Data

## Source

**IBM HR Analytics Employee Attrition & Performance**
`raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 rows, 35 columns)

- Created by IBM data scientists as a **fictional** sample dataset — it
  contains no real employee records.
- Distributed publicly via Kaggle
  (`pavansubhasht/ibm-hr-analytics-attrition-dataset`) and originally through
  IBM Watson Analytics sample data.
- Kaggle lists the dataset under the **Database Contents License (DbCL) v1.0**,
  which permits use, modification and redistribution of the contents.
  Included here unmodified for reproducibility, with this attribution.

This satisfies the brief's ground rule 07 ("public or synthetic data are
usually the easiest options") — this dataset is both.

## Prepared file

`onboarding_data.csv` is produced by `prepare_data.py` (deterministic, no
randomness). Three decisions, argued in the script header and README:

1. **Tenure filter** `YearsAtCompany <= 2` → 342 rows, 29.8% attrition —
   the onboarding window this project is about.
2. **Protected attributes removed** — Gender, Age, MaritalStatus are excluded
   from the features. Measured cost: +0.028 CV AUC forgone, inside the ±0.047
   CV standard deviation (`../evidence/fairness_cost.json`).
3. **No fabricated targets** — the raw data has no time-to-productivity
   column, so none was invented and no regressor is shipped.

Integrity: `baseline/train.py` records the prepared file's SHA-256 prefix in
`../evidence/baseline_metrics.json`, so the artifact and the data it was
trained on are mutually verifiable.
