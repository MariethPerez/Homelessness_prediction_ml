"""
config.py — Constants, mappings and column lists for the project
Waterloo Region Homelessness ML — PHASE A
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Key dates
# ---------------------------------------------------------------------------
CUTOFF_DATE = pd.Timestamp('2025-02-28')   # 6 months before end of dataset
TRAIN_TEST_DATE = '2024-02-01'             # Temporal split ~75/25 (verified with data)

# ---------------------------------------------------------------------------
# Columns to exclude from feature matrix X
# ---------------------------------------------------------------------------

# Leaky: directly define the target or are future outcomes
LEAKY_COLS = [
    'Days Homeless In Past Year',
    'Days Homeless in Lifetime',
    'Days Homeless in the Last 3 Years',
    'Current Episode Days Homeless',
    'Days Unsheltered Homeless in Lifetime',
    'Chronic Homelessness Y/N',
    'Housing Secured Date',
    'Expected Move In Date',
    'Housing Search Started Date',
]

# IDs and dates: kept in dataset for auditing, not fed to the model
ID_COLS = [
    'Dummy Client ID',
    'Date',
    'Date Client Record Was Created',
    'date_became_chronic',
    'days_to_chronic',
]

# ---------------------------------------------------------------------------
# Categorical columns for LightGBM (no one-hot encoding needed)
# ---------------------------------------------------------------------------
CAT_FEATURES = [
    'entry_season',
    'gender',
    'indigenous_status',
    'veteran_status',
    'immigration_status',
    'household_type',
    'last_housed_location',
    'income_source',
    'annual_income_range',
]

# ---------------------------------------------------------------------------
# Phase A features (19 conceptual features)
# ---------------------------------------------------------------------------
PHASE_A_FEATURES = [
    # Temporal control
    'entry_year',
    'entry_season',
    'days_before_first_snapshot',
    # Demographics
    'age',
    'gender',
    'indigenous_status',
    'veteran_status',
    'immigration_status',
    'household_type',
    # Geography — excluded pending clarification with project leads
    # 'last_housed_location',  # 47% missing — needs verification
    # 'is_local_region',       # 69% missing — too many nulls
    # 'geo_missing',           # flag for is_local_region — excluded with parent
    # Income
    'has_income',
    'income_source',
    'income_source_missing',
    'yearly_income',
    'yearly_income_missing',
    'annual_income_range',
    'has_very_low_income',
    # Clinical / History
    'tri_morbidity',
    'returned_from_housing',
    'first_homeless_episode',
]

# ---------------------------------------------------------------------------
# Standardization mappings
# ---------------------------------------------------------------------------

GENDER_MAP = {
    'Male'                    : 'Male',
    'Female'                  : 'Female',
    'Transgender'             : 'Other',
    'Trans Man'               : 'Other',
    'Non-Binary (Genderqueer)': 'Other',
    'Other'                   : 'Other',
    'Other (Not Listed)'      : 'Other',
    'Unknown'                 : 'Unknown',
    "Don't Know"              : 'Unknown',
}

INDIGENOUS_MAP = {
    'Non-Indigenous'            : 'Non-Indigenous',
    'First Nations (Status)'    : 'Indigenous',
    'First Nations (Non-Status)': 'Indigenous',
    'First Nations: On-reserve' : 'Indigenous',
    'First Nations: Off-reserve': 'Indigenous',
    'Non-Status'                : 'Indigenous',
    'Métis'                     : 'Indigenous',
    'Inuit'                     : 'Indigenous',
    'Unknown'                   : 'Unknown',
    "Don't know"                : 'Unknown',
    'Not Specified'             : 'Unknown',
    'Not Specfied'              : 'Unknown',   # typo in original data
    '-->Not Specified'          : 'Unknown',
    '-->not specified'          : 'Unknown',
    'Select an option'          : 'Unknown',
}

VETERAN_MAP = {
    'Not a Veteran'                  : 'Not a Veteran',
    'Veteran - Canadian Armed Forces': 'Veteran',   # collapsed: only 71 veterans total (0.7%)
    'Veteran - Civilian'             : 'Veteran',
    'Veteran - Allies'               : 'Veteran',
    'Unknown / Not Asked'            : 'Unknown',
    'Undeclared / Refused'           : 'Unknown',
    'Select an option'               : 'Unknown',
}

IMMIGRATION_MAP = {
    'Canadian Citizen - Born in Canada'         : 'Canadian Citizen',
    'Born in Canada'                            : 'Canadian Citizen',
    'Canadian Citizen - Born Outside of Canada' : 'Canadian Citizen',
    'Permanent Resident / Immigrant'            : 'Immigrant',
    'Established Immigrant/Refugee (>5 Years)'  : 'Immigrant',
    'Recent Immigrant (<5 years)'               : 'Immigrant',
    'Refugee'                                   : 'Refugee',
    'Refugee Claimant'                          : 'Refugee',
    'Recent Refugee (<5 years)'                 : 'Refugee',
    'Work Visa'                                 : 'No Permanent Resident',
    'Student Visa'                              : 'No Permanent Resident',
    'Visitor Visa'                              : 'No Permanent Resident',
    'Undeclared'                                : 'Unknown',
}

INCOME_SOURCE_MAP = {
    '-->None Specified': 'Not Specified',
    'Select an option' : 'Not Specified',
    '-->ODSP'          : 'Ontario Disability Support Program (ODSP)',
    'ODSP'             : 'Ontario Disability Support Program (ODSP)',
    # All other values kept as-is
}

NO_INCOME_VALUES = {'Select an option', '-->None Specified'}

WATERLOO_CITIES = {
    'Kitchener', 'Cambridge', 'Waterloo', 'Elmira',
    'New Hamburg', 'Baden', 'Wellesley', 'St. Jacobs',
}

LOCAL_GEO_VALUES = {
    'This Community', 'Kitchener', 'Cambridge',
    'This Region / County', 'Townships',
}
