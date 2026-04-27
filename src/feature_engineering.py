"""
feature_engineering.py — Build intake records, derive targets, encode features
Waterloo Region Homelessness ML — PHASE A
"""

import pandas as pd
import numpy as np
from src.config import (
    GENDER_MAP, INDIGENOUS_MAP, VETERAN_MAP, IMMIGRATION_MAP,
    INCOME_SOURCE_MAP, NO_INCOME_VALUES,
    WATERLOO_CITIES, LOCAL_GEO_VALUES,
    LEAKY_COLS, ID_COLS, PHASE_A_FEATURES, CAT_FEATURES,
)


# ---------------------------------------------------------------------------
# Step 1: Derive targets (one row per client)
# ---------------------------------------------------------------------------

def derive_targets(cam: pd.DataFrame) -> pd.DataFrame:
    """
    Derive target variables for each client from full CAM history.

    Returns a DataFrame with one row per client:
        - became_chronic      : 1 if any record has Chronic Homelessness Y/N = 'Y'
        - date_became_chronic : date of first record with Y/N = 'Y' (NaT if never)
        - days_to_chronic     : days from record creation to first chronic date
                                (0 if already chronic at intake, NaN if never)
    """
    def client_target(group):
        chronic_rows = group[group['Chronic Homelessness Y/N'] == 'Y']
        if len(chronic_rows) == 0:
            return pd.Series({
                'became_chronic'     : 0,
                'date_became_chronic': pd.NaT,
                'days_to_chronic'    : np.nan,
            })
        first_chronic_date = chronic_rows['Date'].min()
        entry_date = group['Date Client Record Was Created'].iloc[0]
        days = (first_chronic_date - entry_date).days
        days = max(days, 0)  # 0 if already chronic at intake
        return pd.Series({
            'became_chronic'     : 1,
            'date_became_chronic': first_chronic_date,
            'days_to_chronic'    : days,
        })

    targets = cam.groupby('Dummy Client ID').apply(client_target).reset_index()
    print(f"Targets derived: {targets['became_chronic'].sum():,} chronic "
          f"({targets['became_chronic'].mean():.1%}) / "
          f"{(targets['became_chronic'] == 0).sum():,} non-chronic")
    return targets


# ---------------------------------------------------------------------------
# Step 2: Build intake record (first snapshot per client)
# ---------------------------------------------------------------------------

def build_intake_record(cam: pd.DataFrame) -> pd.DataFrame:
    """
    Take the first monthly snapshot per client (sorted by Date).
    Derive temporal control features from Date Client Record Was Created.

    Returns one row per client with raw intake columns + temporal features.
    """
    intake = (
        cam.sort_values('Date')
           .groupby('Dummy Client ID', as_index=False)
           .first()
    )

    # Temporal control features (from Date Client Record Was Created — constant per client)
    intake['entry_year'] = intake['Date Client Record Was Created'].dt.year
    intake['entry_season'] = intake['Date Client Record Was Created'].dt.month.map({
        12: 'Winter', 1: 'Winter',  2: 'Winter',
        3: 'Spring',  4: 'Spring',  5: 'Spring',
        6: 'Summer',  7: 'Summer',  8: 'Summer',
        9: 'Fall',   10: 'Fall',   11: 'Fall',
    })

    # Days between record creation and first observable snapshot
    intake['days_before_first_snapshot'] = (
        intake['Date'] - intake['Date Client Record Was Created']
    ).dt.days

    print(f"Intake records built: {len(intake):,} clients")
    return intake


# ---------------------------------------------------------------------------
# Step 3: Encode features
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standardization mappings and derive engineered features.
    All features taken from the intake record (first snapshot).

    Returns DataFrame with all Phase A features added.
    """
    df = df.copy()

    # --- Demographics ---
    df['gender'] = df['Gender Identity'].map(GENDER_MAP).fillna('Unknown')

    df['indigenous_status'] = df['Indigenous Status'].map(INDIGENOUS_MAP).fillna('Unknown')

    df['veteran_status'] = df['Veteran Status'].map(VETERAN_MAP).fillna('Unknown')

    df['immigration_status'] = (
        df['Citizenship and Immigration Status'].map(IMMIGRATION_MAP).fillna('Unknown')
    )

    df['household_type'] = df['Household Type']  # already clean: Single/Family/Youth/Family Head

    df['age'] = df['Age']

    # --- Geography ---
    df['last_housed_location'] = df['City Name Where Last Housed'].apply(
        lambda x: 'Waterloo Region'    if x in WATERLOO_CITIES
        else      'Other Region'       if pd.notna(x)
        else      'Never Housed/Unknown'
    )

    df['is_local_region'] = df['Geographic Region'].apply(
        lambda x: 1 if x in LOCAL_GEO_VALUES else (0 if pd.notna(x) else np.nan)
    )
    df['geo_missing'] = df['Geographic Region'].isna().astype(int)

    # --- Income ---
    df['has_income'] = df['Primary Income Type'].apply(
        lambda x: 0 if pd.isna(x) or x in NO_INCOME_VALUES else 1
    )

    df['income_source'] = (
        df['Primary Income Type']
        .map(INCOME_SOURCE_MAP)
        .fillna(df['Primary Income Type'])
    )
    # Treat no-income values as NaN for income_source
    df.loc[df['has_income'] == 0, 'income_source'] = np.nan
    df['income_source_missing'] = df['income_source'].isna().astype(int)

    df['yearly_income'] = df['Monthly Primary Income Amount'] * 12
    df['yearly_income_missing'] = df['yearly_income'].isna().astype(int)

    df['annual_income_range'] = df['yearly_income'].apply(_income_range)

    df['has_very_low_income'] = (
        (df['yearly_income'] > 0) & (df['yearly_income'] < 20_000)
    ).astype(int)

    # --- Clinical / History ---
    df['tri_morbidity']       = (df['Tri-Morbidity'] == 'Y').astype(int)
    df['returned_from_housing'] = (df['Returned From Housing Y/N'] == 'Y').astype(int)
    df['first_homeless_episode'] = (df['First Homeless Episode Y/N'] == 'Y').astype(int)

    return df


def _income_range(yearly):
    if pd.isna(yearly) or yearly == 0:
        return 'No Income / Not Reported'
    elif yearly < 20_000:
        return 'Very Low (0-19,999)'
    elif yearly < 40_000:
        return 'Low (20,000-39,999)'
    elif yearly < 80_000:
        return 'Medium (40,000-79,999)'
    else:
        return 'High (80,000+)'


# ---------------------------------------------------------------------------
# Step 4: Build full training dataset
# ---------------------------------------------------------------------------

def build_training_dataset(cam: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: targets + intake record + encoded features.

    Returns one row per client (9,576 expected) with:
        - ID / reference columns (kept for auditing)
        - Encoded Phase A features
        - Target: became_chronic
    """
    targets  = derive_targets(cam)
    intake   = build_intake_record(cam)
    intake   = encode_features(intake)

    # Join targets onto intake
    training_df = intake.merge(targets, on='Dummy Client ID', how='left')

    print(f"\nTraining dataset: {training_df.shape[0]:,} rows x {training_df.shape[1]} cols")
    print(f"Phase A features available: {sum(f in training_df.columns for f in PHASE_A_FEATURES)}/{len(PHASE_A_FEATURES)}")
    return training_df


def get_X_y(training_df: pd.DataFrame):
    """
    Split training dataset into feature matrix X and target vector y.
    Categorical columns are cast to pandas 'category' dtype as required by LightGBM >= 4.0.
    """
    X = training_df[PHASE_A_FEATURES].copy()
    y = training_df['became_chronic'].copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype('category')
    return X, y
