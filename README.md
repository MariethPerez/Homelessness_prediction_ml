# Chronic Homelessness Risk Model — Waterloo Region

**Data for Good | Waterloo Region | Phase A — April 2026**

A machine learning system to identify clients at risk of becoming chronically homeless **at the moment of intake**, so caseworkers can intervene early — before chronic homelessness occurs.

---

## Project Overview

This project builds a predictive model using intake data from the Community Assessment Module (CAM) system. Given a client's demographic and clinical profile at first contact with the shelter system, the model estimates the probability that the client will become chronically homeless.

The tool is designed to **support**, not replace, caseworker judgment.

---

## Results (Phase A Baseline)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.779** |
| Recall (Chronic clients) | **0.68** |
| Precision (Chronic clients) | 0.40 |
| F1 (Chronic clients) | 0.50 |
| Decision threshold | 0.20 |

> Threshold set at 0.20 because missing a chronic client (false negative) is more costly than a false alarm (false positive) in this context.

---

## Project Structure

```
waterloo_homelessness_ml/
├── app.py                          ← Streamlit demo app for caseworkers
├── requirements.txt
│
├── data/                           ← NOT included in this repository
│   ├── raw/                        ← Original client data files (confidential)
│   └── processed/                  ← Derived from raw data (also confidential)
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory data analysis + dataset save
│   ├── 02_baseline_model.ipynb     ← LightGBM model training + threshold tuning
│   └── 03_fairness_audit.ipynb     ← Fairness evaluation by demographic group
│
├── src/
│   ├── config.py                   ← Feature lists, mappings, constants
│   ├── data_loader.py              ← Load CAM data, apply cut-off
│   └── feature_engineering.py     ← Build intake record, derive targets, encode features
│
├── models/                         ← NOT included in this repository (large files)
│   └── phase_a_baseline_*.pkl      ← Trained model artifact
│
└── reports/
    ├── phase_a_report.md           ← Phase A summary report
    ├── 01_EDA.html                 ← Rendered EDA notebook
    ├── 02_baseline_model.html      ← Rendered model notebook
    └── 03_fairness_audit.html      ← Rendered fairness audit notebook
```

> **Data and models are not included in this repository.** Raw data contains confidential client information. The processed dataset and model artifacts are derived from that data. Access to the original data files is required to reproduce results.

---

## Data

- **Source:** `D4G_CAM_Final.xlsx` — 59,624 rows, 10,118 unique clients
- **After cut-off (Feb 2025):** 9,576 valid clients (ensures 6+ months of observable follow-up)
- **Target:** `became_chronic = 1` if any CAM record shows `Chronic Homelessness Y/N = Y`
- **Class distribution:** 34% chronic (3,259) / 66% non-chronic (6,317)

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

### 4. Add the data files

Place `D4G_CAM_Final.xlsx` in `data/raw/`. This file is not included in the repository and must be obtained from the project data custodian.

---

## How to Run

### Step 1 — EDA and data processing

Open and run all cells in `notebooks/01_EDA.ipynb`.

This reads the raw Excel file, explores the data, and saves the processed dataset to `data/processed/training_dataset.parquet`.

```bash
jupyter notebook
```

### Step 2 — Train the model

Open and run `notebooks/02_baseline_model.ipynb`.

Loads the parquet dataset, trains a LightGBM classifier with Optuna hyperparameter tuning, tunes the decision threshold, and saves the model to `models/`.

### Step 3 — Fairness audit

Open and run `notebooks/03_fairness_audit.ipynb`.

Evaluates model equity across Indigenous Status, Gender, and Age groups using `fairlearn.MetricFrame`.

### Step 4 — Demo app

```bash
streamlit run app.py
```

Opens a web interface where you can enter client intake information and receive a risk score. Requires the trained model in `models/`.

---

## Fairness Audit Summary

| Group | Result | Recall Gap |
|-------|--------|------------|
| Indigenous Status | **PASS** | 0.025 (threshold: 0.10) |
| Age Group | **PASS** | 0.083 (threshold: 0.10) |
| Gender | ALERT ⚠️ | 1.000 — driven by very small groups (Other/Unknown) |

**Critical check:** Indigenous clients do NOT have the highest False Negative Rate — Non-Indigenous FNR (0.324) > Indigenous FNR (0.308).

> ⚠️ Higher chronic rates among Indigenous clients reflect **systemic inequity**, not individual risk. This note is displayed in the app whenever an Indigenous client is assessed.

---

## Model Details

- **Algorithm:** LightGBM (handles missing values and categorical features natively)
- **Train/test split:** Temporal — clients before Feb 2024 (train: 75.5%) vs after (test: 24.5%)
- **Hyperparameter tuning:** Optuna — 50 trials, maximizing ROC-AUC
- **19 features:** demographics, income, clinical history, temporal control

| Category | Features |
|----------|----------|
| Temporal | entry_year, entry_season, days_before_first_snapshot |
| Demographics | age, gender, indigenous_status, veteran_status, immigration_status, household_type |
| Income | has_income, income_source, yearly_income, annual_income_range, has_very_low_income + missing flags |
| Clinical | tri_morbidity, returned_from_housing, first_homeless_episode |

---

## Limitations

| Limitation | Plan |
|------------|------|
| Demographic features only | **Phase B:** add CAM interaction features (case management, SPDAT, housing placement) |
| Precision 0.40 below target (>0.50) | Expected to improve significantly in Phase B |
| Geography features excluded | Pending clarification with project leads |
| No causal inference | Model predicts association, not causation |

---

## Important Disclaimer

> This tool informs, it does not decide. The caseworker's professional judgment always takes precedence over any model output. Risk scores reflect historical patterns in aggregate data and do not determine individual outcomes.

---

## Phase B — Planned

Phase B will add ~10 interaction features from the `Recent Interaction Module` in CAM (case management, SPDAT assessments, housing placement, follow-ups), expected to push ROC-AUC > 0.82 and Recall > 0.75.

---

*Built with Python, LightGBM, scikit-learn, fairlearn, Optuna, and Streamlit.*
