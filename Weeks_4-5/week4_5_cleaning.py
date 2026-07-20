"""
Week 4-5 Data Cleaning — HALF 1
Inputs : ../Week 2-3/outputs/sold_with_rates.csv
         ../Week 2-3/outputs/listings_with_rates.csv
Outputs: ./outputs/

IMPORTANT: This script does NOT delete any rows or columns.
It only (1) fixes data types, (2) reports possible issues, and
(3) adds new "flag" columns that mark values worth a closer look.
Actually removing bad data is left for Half 2.
"""

import os
import sys
import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════
# os.path.dirname(__file__) = the folder this script lives in ("Week 4-5").
# We build every other path relative to that so the script works no matter
# where it is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The Week 2-3 script saved its results in a sibling folder: "..\Week 2-3\outputs"
WEEK23_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Week 2-3", "outputs")
SOLD_FILE = os.path.join(WEEK23_OUTPUT_DIR, "sold_with_rates.csv")
LISTINGS_FILE = os.path.join(WEEK23_OUTPUT_DIR, "listings_with_rates.csv")

for path in (SOLD_FILE, LISTINGS_FILE):
    if not os.path.exists(path):
        sys.exit(f"ERROR: required input file not found: {path}")

# Columns this script needs to work with (from the task instructions).
DATE_COLS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]
NUMERIC_COLS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
]

deliverables = []  # keeps track of every file this script saves, for the final summary


def save_csv(df, name):
    """Small helper so we don't repeat the same 3 lines every time we save a file."""
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    deliverables.append(path)
    return path


# ════════════════════════════════════════════════════════════════════════
# STEP 1 — CONVERT DATE COLUMNS
# ════════════════════════════════════════════════════════════════════════
def convert_dates(df, cols, label):
    """
    Converts the given columns from plain text to real pandas datetime values.
    errors="coerce" means: if a value can't be understood as a date, turn it
    into NaT (pandas' version of a missing date) instead of crashing.
    """
    print(f"\n[{label}] Converting date columns: {cols}")
    for col in cols:
        if col not in df.columns:
            print(f"    WARNING: column '{col}' not found - skipped.")
            continue

        # How many values were already blank BEFORE we touch anything.
        missing_before = df[col].isna().sum()

        df[col] = pd.to_datetime(df[col], errors="coerce")

        # Any value that is missing now but was NOT missing before must have
        # failed to parse as a date (e.g. a typo or bad format).
        missing_after = df[col].isna().sum()
        unparseable = missing_after - missing_before
        print(f"    {col}: dtype -> {df[col].dtype}, "
              f"unparseable values found: {unparseable}")

    return df


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — REVIEW COLUMNS FOR REDUNDANCY (report only, nothing is dropped)
# ════════════════════════════════════════════════════════════════════════
def report_redundant_columns(df, label):
    """
    Flags columns that LOOK redundant or unnecessary so a human can review
    them later. Nothing is removed here — this is a report only.

    Three kinds of checks:
      A) Constant columns   - every row has the exact same value (no info).
      B) Fully-empty columns - 100% missing, provide nothing to analyze.
      C) Known duplicate-meaning groups - columns that describe the same
         real-world thing in more than one way (e.g. an ID stored three
         different times, or a full name that's just First + Last combined).
    """
    print(f"\n[{label}] Reviewing columns for redundancy...")

    # --- A) Constant columns (only one unique value across all rows) -----
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"    Constant columns (same value every row): {constant_cols}")
    else:
        print("    Constant columns: none found.")

    # --- B) Fully empty columns -------------------------------------------
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        print(f"    100% empty columns: {empty_cols}")
    else:
        print("    100% empty columns: none found.")

    # --- C) Known groups of columns that likely overlap in meaning --------
    # Each entry: (short reason, list of column names to check for).
    # We only print a group if 2+ of its columns actually exist in this
    # dataframe (sold and listings don't have the exact same columns).
    candidate_groups = [
        ("Same listing identified 3 different ways",
         ["ListingKey", "ListingKeyNumeric", "ListingId"]),
        ("Agent full name vs. first/last name (full name is derivable)",
         ["ListAgentFullName", "ListAgentFirstName", "ListAgentLastName"]),
        ("Fireplace count vs. fireplace yes/no flag",
         ["FireplacesTotal", "FireplaceYN"]),
        ("Lot size expressed 4 different ways",
         ["LotSizeAcres", "LotSizeSquareFeet", "LotSizeArea", "LotSizeDimensions"]),
        ("Living/building area overlap",
         ["LivingArea", "BuildingAreaTotal", "AboveGradeFinishedArea", "BelowGradeFinishedArea"]),
        ("Parking capacity overlap",
         ["ParkingTotal", "GarageSpaces", "CoveredSpaces", "AttachedGarageYN"]),
        ("Stories vs. Levels (both describe number of floors)",
         ["Stories", "Levels"]),
        ("Geography captured at multiple, overlapping levels",
         ["City", "PostalCode", "CountyOrParish", "MLSAreaMajor", "SubdivisionName", "StateOrProvince"]),
        ("Street number duplicated inside the full address string",
         ["StreetNumberNumeric", "UnparsedAddress"]),
        ("Buyer agency compensation stored as amount + type",
         ["BuyerAgencyCompensation", "BuyerAgencyCompensationType"]),
        ("Internal MLS system metadata, not useful for price analysis",
         ["OriginatingSystemName", "OriginatingSystemSubName"]),
        ("year_month was only needed to merge in mortgage rates",
         ["year_month", "rate_30yr_fixed"]),
    ]

    print("    Possible redundant / low-value groups (for your review):")
    any_group_found = False
    for reason, cols in candidate_groups:
        present = [c for c in cols if c in df.columns]
        if len(present) >= 2:
            any_group_found = True
            print(f"      - {reason}: {present}")
    if not any_group_found:
        print("      None of the known candidate groups apply to this dataset.")


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — MISSING VALUE REPORT
# ════════════════════════════════════════════════════════════════════════
def missing_value_report(df, label, out_name):
    """Builds a count + percentage of missing values for every column and saves it."""
    print(f"\n[{label}] Building missing value report...")

    report = pd.DataFrame({
        "Column": df.columns,
        "MissingCount": df.isna().sum().values,
        "MissingPct": (df.isna().mean() * 100).round(2).values,
    })
    report = report.sort_values("MissingPct", ascending=False).reset_index(drop=True)

    save_csv(report, out_name)
    print(f"    Saved: {out_name}")
    print("    Top 5 columns with the most missing data:")
    print(report.head(5).to_string(index=False))

    return report


# ════════════════════════════════════════════════════════════════════════
# STEP 4 — ENFORCE NUMERIC TYPES
# ════════════════════════════════════════════════════════════════════════
def enforce_numeric(df, cols, label):
    """
    Makes sure each column in `cols` is actually stored as a number.
    pd.to_numeric with errors="coerce" turns any value that isn't a valid
    number (e.g. stray text) into NaN instead of crashing the script.
    """
    print(f"\n[{label}] Enforcing numeric types: {cols}")
    for col in cols:
        if col not in df.columns:
            print(f"    WARNING: column '{col}' not found - skipped.")
            continue

        missing_before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_after = df[col].isna().sum()
        newly_invalid = missing_after - missing_before

        print(f"    {col}: dtype -> {df[col].dtype}, "
              f"values that weren't valid numbers: {newly_invalid}")

    return df


# ════════════════════════════════════════════════════════════════════════
# STEP 5 — ADD INVALID-VALUE FLAG COLUMNS (does not delete anything)
# ════════════════════════════════════════════════════════════════════════
def add_invalid_flags(df, label):
    """
    Adds True/False flag columns marking rows with values that don't make
    sense (e.g. a negative bedroom count). The original data is left as-is;
    these flags just make it easy to filter/inspect/drop later in Half 2.

    Note: pandas comparisons with a missing value (NaN) automatically
    evaluate to False, so rows with missing data are NOT flagged as invalid
    here — "missing" and "invalid" are tracked separately on purpose.
    """
    print(f"\n[{label}] Adding invalid-value flag columns...")

    flags = {
        "invalid_close_price_flag": ("ClosePrice", lambda s: s <= 0),
        "invalid_living_area_flag": ("LivingArea", lambda s: s <= 0),
        "invalid_dom_flag": ("DaysOnMarket", lambda s: s < 0),
        "invalid_beds_flag": ("BedroomsTotal", lambda s: s < 0),
        "invalid_baths_flag": ("BathroomsTotalInteger", lambda s: s < 0),
    }

    for flag_name, (col, condition) in flags.items():
        if col not in df.columns:
            print(f"    WARNING: column '{col}' not found - '{flag_name}' skipped.")
            continue

        df[flag_name] = condition(df[col])
        flagged_count = df[flag_name].sum()
        print(f"    {flag_name}: {flagged_count} row(s) flagged")

    return df


# ════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION — runs all 5 steps on one dataset
# ════════════════════════════════════════════════════════════════════════
def process_dataset(input_path, label, missing_report_name, cleaned_out_name):
    print(f"\n{'=' * 70}")
    print(f"PROCESSING: {label}  ({os.path.basename(input_path)})")
    print(f"{'=' * 70}")

    df = pd.read_csv(input_path, low_memory=False)
    rows_before = df.shape[0]
    print(f"Rows before cleaning: {rows_before:,}")
    print(f"Columns: {df.shape[1]}")

    df = convert_dates(df, DATE_COLS, label)
    report_redundant_columns(df, label)
    missing_value_report(df, label, missing_report_name)
    df = enforce_numeric(df, NUMERIC_COLS, label)
    df = add_invalid_flags(df, label)

    rows_after = df.shape[0]
    print(f"\n[{label}] Rows before: {rows_before:,}  |  Rows after: {rows_after:,}  "
          f"(should match - this script never drops rows)")

    print(f"\n[{label}] Final dtype confirmation for key columns:")
    for col in DATE_COLS + NUMERIC_COLS:
        if col in df.columns:
            print(f"    {col}: {df[col].dtype}")

    save_csv(df, cleaned_out_name)
    print(f"\n[{label}] Saved cleaned file: {cleaned_out_name} "
          f"({df.shape[0]:,} rows x {df.shape[1]} cols)")

    return df


# ════════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════════
sold_df = process_dataset(
    SOLD_FILE, "SOLD",
    missing_report_name="missing_value_report_sold.csv",
    cleaned_out_name="sold_cleaned_half1.csv",
)

listings_df = process_dataset(
    LISTINGS_FILE, "LISTINGS",
    missing_report_name="missing_value_report_listings.csv",
    cleaned_out_name="listings_cleaned_half1.csv",
)

# ════════════════════════════════════════════════════════════════════════
# DELIVERABLES SUMMARY
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ALL DELIVERABLES SAVED TO: outputs/")
print(f"{'=' * 70}")
for i, path in enumerate(deliverables, 1):
    print(f"  {i:>2}. {os.path.basename(path)}")
print(f"\nTotal: {len(deliverables)} files")
print("Week 4-5 Half 1 cleaning complete.\n")
