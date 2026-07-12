"""
config.py — Constants, mappings and column lists for the project
Waterloo Region Homelessness ML — PHASE A / PHASE B
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
    'last_known_housing_category',   # Phase B
    'first_intervention_type',       # Phase B
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

# ---------------------------------------------------------------------------
# Phase B — Service / intervention features
# ---------------------------------------------------------------------------

ADMIN_MODULES = ['Admissions', 'Client Details']

HOUSING_TYPE_MAPPING = {
    # Cambridge Shelter (standalone — >10% individual volume)
    'Stay at: ES - Cambridge Shelter'                              : 'Cambridge Shelter',
    # Emergency Shelter
    'Emergency Shelter'                                            : 'Emergency Shelter',
    'Halfway House'                                                : 'Emergency Shelter',
    'Hostel'                                                       : 'Emergency Shelter',
    'Stay at: ES - HOF ShelterCare'                                : 'Emergency Shelter',
    'Stay at: ES - House of Friendship'                            : 'Emergency Shelter',
    'Stay at: ES - SHIP'                                           : 'Emergency Shelter',
    'Stay at: ES - SHIP (Edith Mac)'                               : 'Emergency Shelter',
    'Stay at: ES - SHIP 84 Frederick'                              : 'Emergency Shelter',
    'Stay at: ES - Safe Haven Shelter'                             : 'Emergency Shelter',
    'Stay at: ES - TWC Erbs Road Shelter'                          : 'Emergency Shelter',
    'Stay at: ES - The Working Centre'                             : 'Emergency Shelter',
    'Stay at: ES - YW Shelter'                                     : 'Emergency Shelter',
    'Stay at: ES - YWCA Cambridge'                                 : 'Emergency Shelter',
    'Stay at: ES - oneROOF'                                        : 'Emergency Shelter',
    'Stay at: ES – Kitchener TSO'                             : 'Emergency Shelter',
    'Stay at: ES - Argus Shelter'                                  : 'Emergency Shelter',
    'Stay at: Argus Residence for Young Men'                       : 'Emergency Shelter',
    'Violence Against Women – Emergency Shelter'              : 'Emergency Shelter',
    # Transitional Housing
    'Transitional Housing'                                         : 'Transitional Housing',
    'Stay at: IH - TWC University Ave'                             : 'Transitional Housing',
    'Stay at: IH - SHIP University Ave'                            : 'Transitional Housing',
    'Stay at: TH - Marillac Place'                                 : 'Transitional Housing',
    'Stay at: TH - KWUNWP'                                         : 'Transitional Housing',
    'Stay at: TH - SHIP University Ave'                            : 'Transitional Housing',
    'Stay at: TH - Porchlight'                                     : 'Transitional Housing',
    'Domestic Violence – Transition House'                    : 'Transitional Housing',
    'Violence Against Women – Transition House'               : 'Transitional Housing',
    # Motel
    'Hotel / Motel'                                                : 'Motel',
    'Stay at: Motels - Cambridge Shelter'                          : 'Motel',
    'Stay at: Motels - RoW'                                        : 'Motel',
    'Stay at: Motels - The Working Centre'                         : 'Motel',
    'Stay at: Motels - YW'                                         : 'Motel',
    'Stay at: Motels - YW Families'                                : 'Motel',
    'Stay at: Motels'                                              : 'Motel',
    # Street / Encampment
    'Makeshift / Street'                                           : 'Street/Encampment',
    'Encampment/Campsite'                                          : 'Street/Encampment',
    'Vehicle'                                                      : 'Street/Encampment',
    # Couch Surfing / Family
    'Couch Surfing – Staying with Friends / Family / Acquaintances': 'Couch Surfing/Family',
    'Couch Surfing – Staying Temporarily with Others'         : 'Couch Surfing/Family',
    "Housed in Family's House / Apartment"                         : 'Couch Surfing/Family',
    'Child/Youth With Parent/Guardian'                             : 'Couch Surfing/Family',
    '-->Child/Youth with Parent/Guardian'                          : 'Couch Surfing/Family',
    # Stable Housing
    'Apartment/House Rental'                                       : 'Stable Housing',
    'Apartment / House Rental'                                     : 'Stable Housing',
    'Rental at Market Price'                                       : 'Stable Housing',
    'Rental at Market Price with Rent Subsidy'                     : 'Stable Housing',
    'Co-op Housing'                                                : 'Stable Housing',
    'Home Ownership'                                               : 'Stable Housing',
    'Housed On-Reserve'                                            : 'Stable Housing',
    'Room in a House'                                              : 'Stable Housing',
    'Rooming House'                                                : 'Stable Housing',
    'Single Room Occupancy'                                        : 'Stable Housing',
    'Social / Community Housing'                                   : 'Stable Housing',
    'Supportive Housing'                                           : 'Stable Housing',
    'Group Home'                                                   : 'Stable Housing',
    'Residential Care Facility'                                    : 'Stable Housing',
    # Institutional
    'Correctional Facility'                                        : 'Institutional',
    'Detoxification Facility'                                      : 'Institutional',
    'Hospital - Medical'                                           : 'Institutional',
    'Hospital - Psychiatric'                                       : 'Institutional',
    'Recovery / Treatment Facility'                                : 'Institutional',
    # Unknown
    'None Specified'                                               : 'Unknown',
}

PHASE_B_SERVICE_FEATURES = [
    'last_known_housing_category',
    'last_known_housing_missing',
    'first_intervention_type',
    'days_since_last_activity',
]

PHASE_B_ALL_FEATURES = PHASE_A_FEATURES + PHASE_B_SERVICE_FEATURES  # 23 total
