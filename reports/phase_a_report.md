# Phase A Report — Chronic Homelessness Risk Model
**Waterloo Region | Data for Good | April 2026**

---

## Objective

Build a machine learning model to identify clients at risk of becoming chronically homeless **at the moment of intake**, so caseworkers can intervene early — before chronic homelessness occurs.

---

## Data

| Source | Records | Clients |
|--------|---------|---------|
| D4G_CAM_Final.xlsx | 59,624 rows | 10,118 unique clients |
| After cut-off (Feb 2025) | 58,656 rows | **9,576 valid clients** |

- **Cut-off:** 6 months before dataset end — ensures each client had observable follow-up
- **Target:** `became_chronic = 1` if any CAM record shows `Chronic Homelessness Y/N = Y`
- **Class distribution:** 34% chronic (3,259) / 66% non-chronic (6,317)

---

## Methodology

**Algorithm:** LightGBM (handles NaN and categorical features natively — no one-hot encoding)

**Train/test split:** Temporal — clients before Feb 2024 (train: 75.5%) vs after (test: 24.5%)

**19 Phase A features** — demographics only:

| Category | Features |
|----------|----------|
| Temporal | entry_year, entry_season, days_before_first_snapshot |
| Demographics | age, gender, indigenous_status, veteran_status, immigration_status, household_type |
| Income | has_income, income_source, yearly_income, annual_income_range, has_very_low_income + missing flags |
| Clinical | tri_morbidity, returned_from_housing, first_homeless_episode |

**Hyperparameter tuning:** Optuna — 50 trials, maximizing ROC-AUC

---

## Results

| Metric | Default (threshold 0.50) | Optimal (threshold 0.20) |
|--------|--------------------------|--------------------------|
| ROC-AUC | **0.779** ✓ | 0.779 |
| Recall (Chronic) | 0.39 | **0.68** ✓ |
| Precision (Chronic) | 0.46 | 0.40 |
| F1 (Chronic) | 0.42 | **0.50** |

**Threshold 0.20 selected** — in this context, missing a chronic client (false negative) is more costly than a false alarm (false positive).

---

## Fairness Audit (fairlearn MetricFrame)

| Group | Result | Recall Gap |
|-------|--------|------------|
| Indigenous Status | **PASS** ✓ | 0.025 (threshold: 0.10) |
| Age Group | **PASS** ✓ | 0.083 (threshold: 0.10) |
| Gender | ALERT ⚠️ | 1.000 — driven by very small groups (Other/Unknown) |

**Critical check:** Indigenous clients do NOT have the highest False Negative Rate — Non-Indigenous FNR (0.324) > Indigenous FNR (0.308). No deploy block.

> ⚠️ Higher chronic rates among Indigenous clients reflect **systemic inequity**, not individual risk.

---

## Demo Application

A Streamlit web app (`app.py`) was built for caseworker testing:

- **Input:** 19 fields across 4 sections (Temporal, Demographics, Income, Clinical)
- **Output:** Risk score (%) + risk level (High / Moderate / Low) + recommendation
- **Auto-computed:** entry_year, entry_season, days_before_first_snapshot, income flags
- **Run:** `streamlit run app.py`

---

## Limitations & Next Steps

| Limitation | Plan |
|------------|------|
| Only demographic features | **Phase B:** add CAM interaction features (case management, SPDAT, housing placement) |
| Precision 0.40 below target (>0.50) | Expected to improve significantly in Phase B |
| Geography features excluded | Pending clarification with project leads |
| No causal inference | Model predicts association, not causation |

**Phase B** will add ~10 interaction features from `Recent Interaction Module` in CAM, expected to push ROC-AUC > 0.82 and Recall > 0.75.

---

*Model artifact:* `models/phase_a_baseline_20260414.pkl` | *Notebooks:* `01_EDA`, `02_baseline_model`, `03_fairness_audit`
