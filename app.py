"""
app.py — Streamlit demo app for Waterloo Homelessness Risk Prediction
Phase A Baseline Model
"""

import sys
sys.path.append('.')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
from datetime import date

from src.config import PHASE_A_FEATURES, CAT_FEATURES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Homelessness Risk Tool — Waterloo Region",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_files = sorted(glob.glob('models/phase_a_baseline_*.pkl'))
    if not model_files:
        st.error("No model found in models/ folder.")
        st.stop()
    artifact = joblib.load(model_files[-1])
    return artifact['model']

model = load_model()

THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Income source options
# ---------------------------------------------------------------------------
INCOME_SOURCES = [
    'Canada Pension Plan (CPP)',
    'Canada-Ontario Housing Benefit (COHB)',
    'Child Support',
    'Child Tax Benefits',
    'Continued Care and Support for Youth (CCSY)',
    'Disability Benefits',
    'Employment - Full-Time',
    'Employment - Part-Time',
    'Employment Benefits / Insurance (EI)',
    'Employment Wage / Salary',
    'Family and Friends',
    'Guaranteed Income Supplement (GIS)',
    'Insurance Settlement',
    'Long Term Disability (private)',
    'OSAP',
    'Old Age Security',
    'Ontario Disability Support Program (ODSP)',
    'Ontario Works (OW)',
    'Other',
    'Other Canada Pension Plan Benefits',
    'Partner Support',
    'Pension',
    'Portable Rent Assistance (PRA)',
    'Provincial Disability Benefits',
    'Provincial Social Assistance',
    'Public / Social Assistance',
    'Rent Supplement',
    'Savings',
    'Self-Employment',
    'Student Loan(s)',
    'Veterans Affairs Canada Financial Benefit',
    "Workers' Compensation Benefits",
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏠 Homelessness Risk Assessment Tool")
st.caption("Waterloo Region — Phase A Baseline Model")
st.markdown(
    "> ⚠️ **This tool informs, it does not decide.** "
    "The caseworker's judgment always takes precedence."
)
st.divider()

# --- Section 1: Temporal ---
st.subheader("📅 Temporal Information")

col1, col2 = st.columns(2)
with col1:
    record_created = st.date_input(
        "Date client record was created",
        value=date.today(),
        help="Date when the client was first registered in the system (Date Client Record Was Created).",
    )
with col2:
    first_snapshot = st.date_input(
        "Date of first monthly snapshot",
        value=date.today(),
        help="Date of the first monthly observable record in CAM. If unknown, use today's date.",
    )

entry_year = record_created.year
entry_month = record_created.month
if entry_month in [12, 1, 2]:
    entry_season = 'Winter'
elif entry_month in [3, 4, 5]:
    entry_season = 'Spring'
elif entry_month in [6, 7, 8]:
    entry_season = 'Summer'
else:
    entry_season = 'Fall'

days_before_first_snapshot = max((first_snapshot - record_created).days, 0)
st.caption(f"Entry year: **{entry_year}** | Season: **{entry_season}** | Days before first snapshot: **{days_before_first_snapshot}**")

st.divider()

# --- Section 2: Demographics ---
st.subheader("👤 Demographics")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
with col2:
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other', 'Unknown'])
with col3:
    indigenous_status = st.selectbox(
        "Indigenous Status",
        ['Non-Indigenous', 'Indigenous', 'Unknown']
    )

col4, col5, col6 = st.columns(3)
with col4:
    veteran_status = st.selectbox("Veteran Status", ['Not a Veteran', 'Veteran', 'Unknown'])
with col5:
    immigration_status = st.selectbox(
        "Immigration Status",
        ['Canadian Citizen', 'Immigrant', 'Refugee', 'No Permanent Resident', 'Unknown']
    )
with col6:
    household_type = st.selectbox(
        "Household Type",
        ['Single', 'Family', 'Youth', 'Family Head']
    )

st.divider()

# --- Section 3: Income ---
st.subheader("💰 Income")

has_income_input = st.radio("Does the client have reported income?", ["No", "Yes"], horizontal=True)

income_source = None
monthly_income = None

if has_income_input == "Yes":
    col1, col2 = st.columns(2)
    with col1:
        income_source = st.selectbox("Primary Income Source", INCOME_SOURCES)
    with col2:
        monthly_income = st.number_input("Monthly Income Amount ($)", min_value=0.0, value=0.0, step=50.0)
else:
    st.caption("No income reported — income fields will be marked as missing.")

# Derive income features
has_income = 1 if has_income_input == "Yes" else 0
yearly_income = monthly_income * 12 if monthly_income is not None else np.nan
yearly_income_missing = 0 if (monthly_income is not None and monthly_income > 0) else 1
income_source_val = income_source if income_source else np.nan
income_source_missing = 0 if income_source else 1

if not np.isnan(yearly_income) and yearly_income == 0:
    annual_income_range = 'No Income / Not Reported'
elif np.isnan(yearly_income):
    annual_income_range = 'No Income / Not Reported'
elif yearly_income < 20_000:
    annual_income_range = 'Very Low (0-19,999)'
elif yearly_income < 40_000:
    annual_income_range = 'Low (20,000-39,999)'
elif yearly_income < 80_000:
    annual_income_range = 'Medium (40,000-79,999)'
else:
    annual_income_range = 'High (80,000+)'

has_very_low_income = 1 if (not np.isnan(yearly_income) and 0 < yearly_income < 20_000) else 0

st.divider()

# --- Section 4: Clinical / History ---
st.subheader("🏥 Clinical / History")

col1, col2, col3 = st.columns(3)
with col1:
    tri_morbidity = st.checkbox("Tri-Morbidity", help="Client has tri-morbidity (mental health + substance use + physical health issues).")
with col2:
    returned_from_housing = st.checkbox("Returned from Housing", help="Client returned from a housing placement.")
with col3:
    first_homeless_episode = st.checkbox("First Homeless Episode", help="This is the client's first homeless episode.")

st.divider()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Calculate Risk", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        'entry_year'                : entry_year,
        'entry_season'              : entry_season,
        'days_before_first_snapshot': days_before_first_snapshot,
        'age'                       : age,
        'gender'                    : gender,
        'indigenous_status'         : indigenous_status,
        'veteran_status'            : veteran_status,
        'immigration_status'        : immigration_status,
        'household_type'            : household_type,
        'has_income'                : has_income,
        'income_source'             : income_source_val,
        'income_source_missing'     : income_source_missing,
        'yearly_income'             : yearly_income if not yearly_income_missing else np.nan,
        'yearly_income_missing'     : yearly_income_missing,
        'annual_income_range'       : annual_income_range,
        'has_very_low_income'       : has_very_low_income,
        'tri_morbidity'             : int(tri_morbidity),
        'returned_from_housing'     : int(returned_from_housing),
        'first_homeless_episode'    : int(first_homeless_episode),
    }])

    # Cast categoricals
    for col in CAT_FEATURES:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype('category')

    # Predict
    proba = model.predict_proba(input_data[PHASE_A_FEATURES])[0][1]
    is_risk = proba >= THRESHOLD

    # Display result
    st.divider()
    st.subheader("Assessment Result")

    if proba >= 0.70:
        level = "HIGH RISK"
        color = "red"
        recommendation = "Immediate follow-up recommended."
    elif proba >= 0.40:
        level = "MODERATE RISK"
        color = "orange"
        recommendation = "Follow-up recommended within the next intake."
    elif proba >= THRESHOLD:
        level = "LOW-MODERATE RISK"
        color = "orange"
        recommendation = "Monitor — flag for periodic check-in."
    else:
        level = "LOW RISK"
        color = "green"
        recommendation = "No immediate action required. Continue regular monitoring."

    st.metric(label="Predicted Risk of Chronic Homelessness", value=f"{proba:.1%}")

    if color == "red":
        st.error(f"🔴 **{level}** — {recommendation}")
    elif color == "orange":
        st.warning(f"🟡 **{level}** — {recommendation}")
    else:
        st.success(f"🟢 **{level}** — {recommendation}")

    st.caption(f"Decision threshold: {THRESHOLD} | Risk score: {proba:.4f}")

    st.divider()
    st.caption(
        "⚠️ This score is based on demographic information only (Phase A model). "
        "It reflects historical patterns and does not determine individual outcomes. "
        "The caseworker's professional judgment always takes precedence."
    )
    if indigenous_status == 'Indigenous':
        st.caption(
            "ℹ️ Note: Higher chronic rates among Indigenous clients reflect systemic inequity, "
            "not individual risk factors."
        )
