"""
Week 4-5 Data Cleaning — HALF 2
Inputs : ./outputs/sold_cleaned_half1.csv
         ./outputs/listings_cleaned_half1.csv
Outputs: ./outputs/

IMPORTANT: This script does NOT delete any rows or columns, and it does not
modify any existing column produced by Half 1. It only ADDS new "flag"
columns (date-consistency + geographic-validation) and produces summary
reports. Row counts in must equal row counts out.
"""

import os
import sys
import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOLD_FILE = os.path.join(OUTPUT_DIR, "sold_cleaned_half1.csv")
LISTINGS_FILE = os.path.join(OUTPUT_DIR, "listings_cleaned_half1.csv")

for path in (SOLD_FILE, LISTINGS_FILE):
    if not os.path.exists(path):
        sys.exit(f"ERROR: required input file not found: {path}")

# Date columns involved in the listing -> purchase -> close sequence check.
DATE_SEQUENCE_COLS = ["ListingContractDate", "PurchaseContractDate", "CloseDate"]

# California's approximate bounding box. Used to catch geocoding errors
# (e.g. a lookup that returns a coordinate from another state/country) and
# sentinel/placeholder values like (0, 0). Bounds are intentionally a bit
# loose ("roughly") so real border-adjacent CA properties aren't flagged:
#   Latitude:  32.5 (south, near the Mexico border)  to 42.0 (north, OR border)
#   Longitude: -124.0 (west, Pacific coast)           to -114.0 (east, NV/AZ border)
# Any coordinate outside this box cannot physically be a California property.
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.0
CA_LON_MIN, CA_LON_MAX = -124.0, -114.0

deliverables = []  # tracks every file this script saves, for the final summary


def save_csv(df, name):
    """Small helper so we don't repeat the same 3 lines every time we save a file."""
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    deliverables.append(path)
    return path


# ════════════════════════════════════════════════════════════════════════
# STEP 1 — DATE CONSISTENCY FLAGS
# ════════════════════════════════════════════════════════════════════════
def add_date_consistency_flags(df, label):
    """
    Adds 3 True/False columns that catch logically-impossible date orderings.

    A property should move through its lifecycle in this order:
        ListingContractDate  <=  PurchaseContractDate  <=  CloseDate

    - listing_after_close_flag : the listing date is AFTER the close date
                                  (impossible — can't close before it was listed)
    - purchase_after_close_flag: the purchase contract date is AFTER the close date
                                  (impossible — can't close before the contract)
    - negative_timeline_flag   : True if ANY date in the sequence is out of
                                  logical order (listing > purchase, purchase > close,
                                  or listing > close). This is the "catch-all" flag.

    Rows with missing dates are NOT flagged (a comparison against NaT is
    False in pandas), since we can't prove a violation without both dates.
    Dates are parsed as datetime here in case they came in as strings.
    """
    print(f"\n[{label}] Adding date consistency flags...")

    for col in DATE_SEQUENCE_COLS:
        if col not in df.columns:
            sys.exit(f"ERROR: required column '{col}' not found in {label} dataset.")
        df[col] = pd.to_datetime(df[col], errors="coerce")

    listing = df["ListingContractDate"]
    purchase = df["PurchaseContractDate"]
    close = df["CloseDate"]

    listing_after_close = listing > close
    purchase_after_close = purchase > close
    listing_after_purchase = listing > purchase

    df["listing_after_close_flag"] = listing_after_close
    df["purchase_after_close_flag"] = purchase_after_close
    df["negative_timeline_flag"] = (
        listing_after_close | purchase_after_close | listing_after_purchase
    )

    for flag_name in [
        "listing_after_close_flag",
        "purchase_after_close_flag",
        "negative_timeline_flag",
    ]:
        count = int(df[flag_name].sum())
        pct = round(count / len(df) * 100, 2)
        print(f"    {flag_name}: {count:,} row(s) flagged ({pct}%)")

    return df


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — GEOGRAPHIC VALIDATION FLAGS
# ════════════════════════════════════════════════════════════════════════
def add_geographic_flags(df, label):
    """
    Adds 4 True/False columns that catch bad or suspicious coordinates.

    - missing_coords_flag   : Latitude or Longitude is null
    - zero_coords_flag      : Latitude or Longitude is exactly 0
                               (a common "sentinel" value used by some
                               geocoders/imports to mean "unknown", masquerading
                               as a real coordinate near Africa's coast)
    - invalid_longitude_flag: Longitude > 0 (California longitudes must be
                               negative — a positive value means the sign was
                               dropped or the point is in the wrong hemisphere)
    - out_of_state_flag     : coordinate falls outside California's
                               approximate bounding box (see CA_LAT_*/CA_LON_*
                               constants above) — catches geocoding errors that
                               land in another state/country

    These checks are independent and can overlap (e.g. a zero-coordinate row
    is also out_of_state). Missing values never trigger the numeric checks
    (zero/longitude/out_of_state) since a null can't be compared meaningfully.
    """
    print(f"\n[{label}] Adding geographic validation flags...")

    for col in ["Latitude", "Longitude"]:
        if col not in df.columns:
            sys.exit(f"ERROR: required column '{col}' not found in {label} dataset.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    lat = df["Latitude"]
    lon = df["Longitude"]

    missing_coords = lat.isna() | lon.isna()
    zero_coords = (lat == 0) | (lon == 0)
    invalid_longitude = lon > 0
    out_of_state = (
        (lat < CA_LAT_MIN) | (lat > CA_LAT_MAX) |
        (lon < CA_LON_MIN) | (lon > CA_LON_MAX)
    )
    # NaN comparisons already evaluate to False, so missing rows are
    # naturally excluded from zero/invalid/out_of_state above.

    df["missing_coords_flag"] = missing_coords
    df["zero_coords_flag"] = zero_coords
    df["invalid_longitude_flag"] = invalid_longitude
    df["out_of_state_flag"] = out_of_state

    for flag_name in [
        "missing_coords_flag",
        "zero_coords_flag",
        "invalid_longitude_flag",
        "out_of_state_flag",
    ]:
        count = int(df[flag_name].sum())
        pct = round(count / len(df) * 100, 2)
        print(f"    {flag_name}: {count:,} row(s) flagged ({pct}%)")

    return df


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — GEOGRAPHIC DATA QUALITY SUMMARY
# ════════════════════════════════════════════════════════════════════════
def geographic_quality_summary(df, label, out_name):
    """
    Builds a one-table summary of coordinate quality:
      - total records
      - records with VALID coordinates (non-null, non-zero, in-CA-bounds)
      - records missing, zero, invalid-longitude, or out-of-bounds, each
        as a count and a percentage of the total.
    Saves the summary as a CSV and prints it.
    """
    print(f"\n[{label}] Building geographic data quality summary...")

    total = len(df)
    valid_coords = (
        ~df["missing_coords_flag"]
        & ~df["zero_coords_flag"]
        & ~df["invalid_longitude_flag"]
        & ~df["out_of_state_flag"]
    )

    rows = [
        ("Total records", total, 100.00),
        ("Valid coordinates", int(valid_coords.sum()),
         round(valid_coords.sum() / total * 100, 2)),
        ("Missing coordinates", int(df["missing_coords_flag"].sum()),
         round(df["missing_coords_flag"].sum() / total * 100, 2)),
        ("Zero coordinates", int(df["zero_coords_flag"].sum()),
         round(df["zero_coords_flag"].sum() / total * 100, 2)),
        ("Invalid longitude (positive)", int(df["invalid_longitude_flag"].sum()),
         round(df["invalid_longitude_flag"].sum() / total * 100, 2)),
        ("Out of California bounds", int(df["out_of_state_flag"].sum()),
         round(df["out_of_state_flag"].sum() / total * 100, 2)),
    ]

    summary = pd.DataFrame(rows, columns=["Metric", "Count", "Percent"])
    save_csv(summary, out_name)

    print(f"    Saved: {out_name}")
    print(summary.to_string(index=False))

    return summary


# ════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION — runs all steps on one dataset
# ════════════════════════════════════════════════════════════════════════
def process_dataset(input_path, label, geo_summary_name, cleaned_out_name):
    print(f"\n{'=' * 70}")
    print(f"PROCESSING: {label}  ({os.path.basename(input_path)})")
    print(f"{'=' * 70}")

    df = pd.read_csv(input_path, low_memory=False)
    rows_before = df.shape[0]
    cols_before = df.shape[1]
    print(f"Rows before: {rows_before:,}")
    print(f"Columns before: {cols_before}")

    df = add_date_consistency_flags(df, label)
    df = add_geographic_flags(df, label)
    geographic_quality_summary(df, label, geo_summary_name)

    rows_after = df.shape[0]
    cols_after = df.shape[1]
    print(f"\n[{label}] Rows before: {rows_before:,}  |  Rows after: {rows_after:,}  "
          f"(should match - this script never drops rows)")
    print(f"[{label}] Columns before: {cols_before}  |  Columns after: {cols_after}  "
          f"(should be +7 new flag columns)")

    new_flag_cols = [
        "listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag",
        "missing_coords_flag", "zero_coords_flag", "invalid_longitude_flag", "out_of_state_flag",
    ]
    print(f"\n[{label}] Final dtype confirmation for new flag columns:")
    for col in new_flag_cols:
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
    geo_summary_name="geographic_quality_summary_sold.csv",
    cleaned_out_name="sold_cleaned_half2.csv",
)

listings_df = process_dataset(
    LISTINGS_FILE, "LISTINGS",
    geo_summary_name="geographic_quality_summary_listings.csv",
    cleaned_out_name="listings_cleaned_half2.csv",
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
print("Week 4-5 Half 2 cleaning complete.\n")
