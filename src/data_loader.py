"""
data_loader.py — Load CAM data and apply cut-off filter
Waterloo Region Homelessness ML — PHASE A
"""

import pandas as pd
from pathlib import Path
from src.config import CUTOFF_DATE

# Path to raw data (relative to project root)
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
CAM_FILE = DATA_DIR / 'D4G_CAM_Final.xlsx'


def load_cam(filepath=None) -> pd.DataFrame:
    """
    Load D4G_CAM_Final.xlsx and parse date columns.

    Returns
    -------
    pd.DataFrame
        Raw CAM data with date columns parsed.
    """
    path = filepath or CAM_FILE
    cam = pd.read_excel(
        path,
        parse_dates=['Date', 'Date Client Record Was Created'],
    )
    print(f"Loaded CAM: {cam.shape[0]:,} rows, {cam['Dummy Client ID'].nunique():,} unique clients")
    return cam


def apply_cutoff(cam: pd.DataFrame, cutoff=CUTOFF_DATE) -> pd.DataFrame:
    """
    Keep only clients whose record was created on or before the cut-off date.
    This ensures each client had at least 6 months of observable follow-up.

    Parameters
    ----------
    cam : pd.DataFrame
        Raw CAM data.
    cutoff : pd.Timestamp
        Cut-off date (default: 2025-02-28).

    Returns
    -------
    pd.DataFrame
        Filtered CAM data with valid clients only.
    """
    valid = cam[cam['Date Client Record Was Created'] <= cutoff].copy()
    excluded = cam['Dummy Client ID'].nunique() - valid['Dummy Client ID'].nunique()
    print(f"After cut-off ({cutoff.date()}): {valid['Dummy Client ID'].nunique():,} valid clients "
          f"({excluded} excluded — entered system too recently)")
    return valid


def load_valid_cam(filepath=None) -> pd.DataFrame:
    """
    Convenience function: load CAM and apply cut-off in one step.

    Returns
    -------
    pd.DataFrame
        CAM data filtered to valid clients (9,576 expected).
    """
    cam = load_cam(filepath)
    return apply_cutoff(cam)
