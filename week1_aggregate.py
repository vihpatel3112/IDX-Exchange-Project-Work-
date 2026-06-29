"""
Week 1 – Monthly Dataset Aggregation
Concatenates all monthly MLS files from January 2024 through the most recently
completed calendar month into two combined datasets (listings and sold),
filters both to PropertyType == 'Residential', and saves as new CSVs.
"""

import os
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script
START_YEAR, START_MONTH = 2024, 1

# Most recently completed calendar month (one month before today)
today = date.today()
end = date(today.year, today.month, 1) - relativedelta(months=1)
END_YEAR, END_MONTH = end.year, end.month

print(f"Date range: {START_YEAR}-{START_MONTH:02d} through {END_YEAR}-{END_MONTH:02d}")
print(f"Working directory: {DATA_DIR}\n")

# ---------------------------------------------------------------------------
# Helper: build an ordered list of (year, month) tuples
# ---------------------------------------------------------------------------

def month_range(start_year, start_month, end_year, end_month):
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

periods = month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)

# ---------------------------------------------------------------------------
# Load and concatenate Sold files
# ---------------------------------------------------------------------------

print("=" * 60)
print("SOLD FILES")
print("=" * 60)

sold_frames = []
sold_skipped = []

for y, m in periods:
    filename = f"CRMLSSold{y}{m:02d}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, low_memory=False)
        sold_frames.append(df)
        print(f"  Loaded {filename}: {len(df):,} rows")
    else:
        sold_skipped.append(filename)
        print(f"  Skipped (not found): {filename}")

print(f"\nFiles loaded : {len(sold_frames)}")
print(f"Files skipped: {len(sold_skipped)}")

# Row count before concatenation
total_sold_before = sum(len(f) for f in sold_frames)
print(f"\nRow count before concatenation (sum of individual files): {total_sold_before:,}")

sold = pd.concat(sold_frames, ignore_index=True)

# Row count after concatenation (confirms no rows lost/duplicated)
print(f"Row count after  concatenation                          : {len(sold):,}")
assert len(sold) == total_sold_before, "Row count mismatch after concatenation!"

# Filter to Residential
sold_before_filter = len(sold)
sold = sold[sold["PropertyType"] == "Residential"].reset_index(drop=True)
sold_after_filter = len(sold)

print(f"\nRow count before Residential filter: {sold_before_filter:,}")
print(f"Row count after  Residential filter: {sold_after_filter:,}")
print(f"Rows removed by filter             : {sold_before_filter - sold_after_filter:,}")

# Save
sold_out = os.path.join(DATA_DIR, "combined_sold_residential.csv")
sold.to_csv(sold_out, index=False)
print(f"\nSaved: {sold_out}")

# ---------------------------------------------------------------------------
# Load and concatenate Listing files
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("LISTING FILES")
print("=" * 60)

listing_frames = []
listing_skipped = []

for y, m in periods:
    filename = f"CRMLSListing{y}{m:02d}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, low_memory=False)
        listing_frames.append(df)
        print(f"  Loaded {filename}: {len(df):,} rows")
    else:
        listing_skipped.append(filename)
        print(f"  Skipped (not found): {filename}")

print(f"\nFiles loaded : {len(listing_frames)}")
print(f"Files skipped: {len(listing_skipped)}")

# Row count before concatenation
total_listing_before = sum(len(f) for f in listing_frames)
print(f"\nRow count before concatenation (sum of individual files): {total_listing_before:,}")

listings = pd.concat(listing_frames, ignore_index=True)

# Row count after concatenation (confirms no rows lost/duplicated)
print(f"Row count after  concatenation                          : {len(listings):,}")
assert len(listings) == total_listing_before, "Row count mismatch after concatenation!"

# Filter to Residential
listings_before_filter = len(listings)
listings = listings[listings["PropertyType"] == "Residential"].reset_index(drop=True)
listings_after_filter = len(listings)

print(f"\nRow count before Residential filter: {listings_before_filter:,}")
print(f"Row count after  Residential filter: {listings_after_filter:,}")
print(f"Rows removed by filter             : {listings_before_filter - listings_after_filter:,}")

# Save
listings_out = os.path.join(DATA_DIR, "combined_listings_residential.csv")
listings.to_csv(listings_out, index=False)
print(f"\nSaved: {listings_out}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Period            : {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
print(f"Sold rows (final) : {len(sold):,} Residential records")
print(f"Listing rows (final): {len(listings):,} Residential records")
print(f"Output files      : combined_sold_residential.csv")
print(f"                    combined_listings_residential.csv")
