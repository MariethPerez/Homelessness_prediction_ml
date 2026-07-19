# Chronic Homelessness Risk Model — Waterloo Region

**Data for Good | Waterloo Region | Phase B — July 2026**

A machine learning system to identify clients at risk of becoming chronically homeless **at the moment of intake**, so caseworkers can intervene early — before chronic homelessness occurs.

---

## Project Overview

This project builds a predictive model using intake data from the Community Assessment Module (CAM) system. Given a client's demographic, clinical, and service profile at first contact with the shelter system, the model estimates the probability that the client will become chronically homeless.

The tool is designed to **support**, not replace, caseworker judgment.

---

## Results

### Phase B — Model 1 (23 features, current)

| Metric | Phase A (19 feat) | Phase B (23 feat) | Target |
|--------|:-----------------:|:-----------------:|:------:|
| ROC-AUC | 0.776 | **0.874** | >0.82 ✅ |
| Recall (Chronic) | 0.606 | **0.812** | >0.75 ✅ |
| Precision (Chronic) | 0.398 | **0.420** | >0.50 ⚠️ |
| F1 (Chronic) | 0.480 | **0.554** | >0.65 ⚠️ |
| Decision threshold | 0.20 | **0.32** | recalibrated |

> Threshold recalibrated from 0.20 to 0.32 in Phase B — the 4 new service features shifted the model's score distribution upward. At 0.32, the model flags ~65% of clients (vs ~90% at 0.20) while maintaining Recall > 0.80.

---

## Two-Model Architecture

```
Model 1   → 02_baseline_model.ipynb               → first snapshot only   → risk score at intake (caseworkers)
Model 2   → 04_intervention_explainability.ipynb  → first 12 months       → effectiveness analysis (PECH)
                ├── Component A: Kaplan-Meier survival curves
                └── Component B: LightGBM retrospective (12-month window)
```

| | Model 1 | Model 2 |
|---|---|---|
| **Purpose** | Risk score at intake | Explain intervention effectiveness |
| **Data** | First snapshot only | First 12 months |
| **Key features** | 23 Phase B features | Housing stability + interaction rates |
| **Leakage** | None | Intentional — retrospective analysis only |
| **Used in** | Streamlit app | PECH reports |

> **Note:** An intervention simulation feature is under development in `05_whatif_model.ipynb`. The initial approach reflected caseworker selection bias and has been redesigned.

---

## Project Structure

```
waterloo_homelessness_ml/
├── app.py                                  ← Streamlit app for caseworkers
├── requirements.txt
│
├── data/                                   ← NOT included in this repository
│   ├── raw/                                ← Original client data files (confidential)
│   └── processed/                          ← Derived from raw data (also confidential)
│
├── notebooks/
│   ├── 01_EDA.ipynb                        ← EDA + Phase B feature exploration
│   ├── 02_baseline_model.ipynb             ← Model 1: LightGBM training + Optuna + SHAP
│   ├── 03_fairness_audit.ipynb             ← Fairness evaluation by demographic group
│   ├── 04_intervention_explainability.ipynb ← Model 2: Kaplan-Meier + retrospective LightGBM
│   ├── 05_whatif_model.ipynb               ← Model What-if: intervention simulation
│   └── 06_intervention_outcomes.ipynb      ← Intervention trajectory analysis for PECH
│
├── src/
│   ├── config.py                           ← Feature lists, mappings, constants
│   ├── data_loader.py                      ← Load CAM data, apply cut-off filter
│   └── feature_engineering.py             ← Build features, derive targets, encode
│
├── reports/
│   └── phase_b_report.html                 ← Phase B report (model + intervention outcomes)
│
└── models/                                 ← NOT included in this repository
    ├── phase_a_baseline_*.pkl              ← Phase A model (April 2026)
    ├── phase_b_model1_*.pkl               ← Phase B Model 1 (July 2026)
    └── phase_b_whatif_*.pkl               ← Phase B What-if model (July 2026)
```

> **Data and models are not included in this repository.** Raw data contains confidential client information. Access to the original `D4G_CAM_Final.xlsx` file is required to reproduce results.

---

## Data

- **Source:** `D4G_CAM_Final.xlsx` — 59,624 rows, 10,118 unique clients, Jan 2023 – Aug 2025
- **After cut-off filter (Feb 2025):** 9,576 valid clients (ensures 6+ months of observable follow-up)
- **Target:** `became_chronic = 1` if any CAM record shows `Chronic Homelessness Y/N = Y`
- **Class distribution:** 34% chronic / 66% non-chronic

---

## Features

### Phase B — 23 features (Model 1)

| Category | Features |
|----------|----------|
| Temporal (3) | entry_year, entry_season, days_before_first_snapshot |
| Demographics (6) | age, gender, indigenous_status, veteran_status, immigration_status, household_type |
| Income (8) | has_income, income_source, yearly_income, annual_income_range, has_very_low_income + missing flags |
| Clinical (2) | tri_morbidity, returned_from_housing, first_homeless_episode |
| **Service — Phase B (4)** | **last_known_housing_category, last_known_housing_missing, first_intervention_type, days_since_last_activity** |

### Housing Type Grouping (9 categories)

63 raw values from `Last Known Housing Type` grouped into 9 categories. `Cambridge Shelter` treated as standalone category (>10% data volume per technical lead recommendation).

| Category | Description |
|---|---|
| Emergency Shelter | Shelters, hostels, halfway houses |
| Cambridge Shelter | Stay at: ES - Cambridge Shelter (10.4% of records) |
| Street/Encampment | Makeshift/street, encampments, vehicles |
| Motel | Hotels, motels |
| Couch Surfing/Family | Staying with friends/family, temporarily |
| Transitional Housing | Structured transitional programs |
| Stable Housing | Apartments, rentals, social housing, home ownership |
| Institutional | Hospital, correctional, detox, treatment |
| Unknown | NaN or "None Specified" |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd waterloo_homelessness_ml
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On macOS, LightGBM requires OpenMP:
> ```bash
> brew install libomp
> ```

### 4. Add the data file

Place `D4G_CAM_Final.xlsx` in `data/raw/`. This file is not included in the repository.

---

## How to Run

Run notebooks in order:

```bash
jupyter notebook
```

| Step | Notebook | Output |
|------|----------|--------|
| 1 | `01_EDA.ipynb` | `data/processed/training_dataset.parquet` |
| 2 | `02_baseline_model.ipynb` | `models/phase_b_model1_*.pkl` |
| 3 | `03_fairness_audit.ipynb` | Fairness report |
| 4 | `04_intervention_explainability.ipynb` | KM curves, SHAP, permutation importance |
| 5 | `05_whatif_model.ipynb` | Intervention simulation — in development |
| 6 | `06_intervention_outcomes.ipynb` | Intervention trajectory analysis for PECH |

### Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires `phase_b_model1_*.pkl` in `models/`.

---

## Test Cases (real clients from test set)

Use these to verify the app is working correctly. All clients are from the post-Feb 2024 test set. Enter the values exactly as shown and compare the predicted score.

---

### Test Case 1 — HIGH RISK (client became chronic ✅)
**Dummy Client ID:** `A102A28`

| Field | Value |
|---|---|
| Date client record was created | 2024-11-02 |
| Date of first monthly snapshot | 2025-07-31 |
| Age | 39 |
| Gender | Male |
| Indigenous Status | Non-Indigenous |
| Veteran Status | Not a Veteran |
| Immigration Status | Canadian Citizen |
| Household Type | Single |
| Income | Yes — Ontario Works (OW), $700/month |
| Tri-Morbidity | No |
| Returned from Housing | No |
| First Homeless Episode | Yes |
| Last Known Housing Type | Emergency Shelter |
| First Interaction Module | Admissions |
| Days Since Last Activity | 28 |

**Expected result:** ~86.9% → 🔴 HIGH RISK — client did become chronically homeless.

---

### Test Case 2 — LOW RISK (client did not become chronic ✅)
**Dummy Client ID:** `A116Q99`

| Field | Value |
|---|---|
| Date client record was created | 2024-08-22 |
| Date of first monthly snapshot | 2024-08-31 |
| Age | 50 |
| Gender | Female |
| Indigenous Status | Indigenous |
| Veteran Status | Not a Veteran |
| Immigration Status | Canadian Citizen |
| Household Type | Single |
| Income | No |
| Tri-Morbidity | No |
| Returned from Housing | No |
| First Homeless Episode | No |
| Last Known Housing Type | Unknown |
| First Interaction Module | Goods and Services |
| Days Since Last Activity | 24 |

**Expected result:** ~0.2% → 🟢 LOW RISK — client did not become chronically homeless.

---

### Test Case 3 — YOUTH WARNING (client became chronic, model underestimates ⚠️)
**Dummy Client ID:** `A794Z21`

| Field | Value |
|---|---|
| Date client record was created | 2025-01-24 |
| Date of first monthly snapshot | 2025-07-31 |
| Age | 17 |
| Gender | Female |
| Indigenous Status | Non-Indigenous |
| Veteran Status | Not a Veteran |
| Immigration Status | Canadian Citizen |
| Household Type | Family |
| Income | No |
| Tri-Morbidity | No |
| Returned from Housing | No |
| First Homeless Episode | Yes |
| Last Known Housing Type | Motel |
| First Interaction Module | Case Management |
| Days Since Last Activity | 180 |

**Expected result:** ~39.3% → 🟡 LOW-MODERATE RISK + ⚠️ youth warning. Client did become chronically homeless — illustrates the model's lower recall for youth (<25), documented in the fairness audit.

---

## Fairness Audit (Phase B — threshold 0.32)

| Group | Result | Notes |
|-------|--------|-------|
| Indigenous Status | ⚠️ Gap 0.179 | **Favorable direction** — Indigenous recall (0.962) > Non-Indigenous (0.783). FNR lowest for Indigenous clients. |
| Gender | ⚠️ Gap 1.000 | **Statistical artifact** — driven by tiny Unknown/Other groups. Female vs Male gap is only 0.043. |
| Age Group | 🔴 Gap 0.401 | **Real concern** — Youth (<25) recall 0.551 vs Adult 0.877. App shows manual review warning for clients under 25. |

**Critical check:** Indigenous clients do NOT have the highest False Negative Rate.

> ⚠️ Higher chronic rates among Indigenous clients reflect **systemic inequity**, not individual risk factors. This note is displayed in the app for every Indigenous client assessed.

---

## Key Findings from Intervention Outcomes Analysis (Notebook 06)

Descriptive trajectory analysis comparing high-risk clients (Model 1 score > 0.50) who avoided chronic homelessness vs those who became chronic. Base: 3,850 high-risk clients — 1,147 never became chronic (29.8%), 2,703 did (70.2%).

- **Goods and Services signal:** appears +9 percentage points more at intake (28.9% vs 19.9%) and +11 points at last recorded intervention (31.3% vs 20.2%) in clients who never became chronic
- **Case Management and selection bias:** more frequent in clients who became chronic, but mean risk score (0.869) is the highest of all interventions — caseworkers assign it to the most vulnerable cases, not because it is harmful
- **Trajectory:** clients who became chronic had more snapshots and more intervention changes, consistent with longer and more complex cases

> ⚠️ All findings are **descriptive associations, not causal evidence**. Observational data cannot prove that any intervention caused a better or worse outcome.

---

## Key Findings from Intervention Analysis (Model 2)

- **Housing type is the strongest predictor** — `housing_most_frequent` over 12 months has 5× higher permutation importance than any intervention feature
- **Housing instability** (`housing_changes_count`) strongly increases chronic risk
- **Intervention signal is weak** — specific interventions have minimal permutation importance, likely due to caseworker selection bias (higher-risk clients receive more intensive interventions)
- **Statistical significance:** Goods and Services, Case Management, Group Activities, Service Restrictions, and SPDAT show significant Kaplan-Meier differences (p<0.05) — but direction reflects client risk level at assignment, not causal effect

> ⚠️ All intervention analysis reflects **association, not causation**. Clients who received SPDAT or Case Management show higher chronic rates because those interventions are assigned to higher-risk individuals.

---

## Limitations

| Limitation | Status |
|------------|--------|
| Precision 0.42 below target (>0.50) | Inherent to 34% base rate; threshold trade-off |
| Youth (<25) recall gap | Documented in app with manual review warning |
| No causal inference | Association only — declared throughout |
| Geography excluded | 47-69% missing — excluded from all models |
| What-if model uses full history | Intentional leakage — declared, for simulation only |

---

## Important Disclaimer

> This tool informs, it does not decide. The caseworker's professional judgment always takes precedence over any model output. Risk scores reflect historical patterns in aggregate data and do not determine individual outcomes.

---

*Built with Python, LightGBM, scikit-learn, fairlearn, lifelines, Optuna, SHAP, and Streamlit.*
