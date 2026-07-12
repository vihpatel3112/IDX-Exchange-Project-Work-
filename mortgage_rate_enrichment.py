"""
Mortgage Rate Enrichment
Fetches 30-year fixed mortgage rates from FRED (MORTGAGE30US),
resamples to monthly averages, and merges onto both:
  - combined_sold_residential_updated.csv   (using CloseDate)
  - combined_listings_residential_updated.csv (using ListingContractDate)

Date parsing for CloseDate uses a two-pass approach to handle both
M/D/YYYY and YYYY-MM-DD formats found in the sold dataset.

Outputs saved to: ./outputs/
"""

import os
import sys
import warnings
import urllib.request
import io
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOLD_FILE     = os.path.join(BASE_DIR, "combined_sold_residential_updated.csv")
LISTINGS_FILE = os.path.join(BASE_DIR, "combined_listings_residential_updated.csv")
FRED_URL      = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

deliverables = []

def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    deliverables.append(path)
    return path


def parse_mixed_dates(series):
    """
    Two-pass date parser that handles mixed formats in one column:
      Pass 1 — infer_datetime_format=True (handles M/D/YYYY and ISO 8601)
      Pass 2 — explicit format="%Y-%m-%d" for any nulls that remain
    Falls back to format="mixed" on pandas 2.0+ if infer_datetime_format
    is unavailable.
    """
    raw = series.copy()

    # Pass 1
    try:
        parsed = pd.to_datetime(raw, errors="coerce", infer_datetime_format=True)
    except TypeError:
        # pandas 2.0+ deprecated infer_datetime_format; use format="mixed"
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=False)

    # Pass 2: retry remaining nulls with explicit YYYY-MM-DD
    still_null = parsed.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
    retry_count = int(still_null.sum())
    if retry_count > 0:
        retried = pd.to_datetime(raw[still_null], format="%Y-%m-%d", errors="coerce")
        parsed = parsed.copy()
        parsed.loc[still_null] = retried
        recovered = int(retried.notna().sum())
        print(f"  Two-pass date fix: {retry_count:,} nulls retried with YYYY-MM-DD, "
              f"{recovered:,} recovered, {retry_count - recovered:,} still null")

    return parsed


# ════════════════════════════════════════════════════════════════════════════════
# 1. FETCH MORTGAGE RATES FROM FRED
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1 — Fetching MORTGAGE30US from FRED")
print(f"{'='*60}")

try:
    req = urllib.request.Request(FRED_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    rates_raw = pd.read_csv(io.StringIO(raw))
    # FRED CSV uses 'observation_date' and 'MORTGAGE30US' as column names
    rates_raw.columns = ["date", "rate_30yr_fixed"]
    rates_raw["date"] = pd.to_datetime(rates_raw["date"], errors="coerce")
    # FRED marks missing values as "." — coerce to NaN
    rates_raw["rate_30yr_fixed"] = pd.to_numeric(rates_raw["rate_30yr_fixed"], errors="coerce")
    rates_raw = rates_raw.dropna(subset=["rate_30yr_fixed"])
    print(f"Fetched {len(rates_raw):,} weekly observations")
    print(f"Date range: {rates_raw['date'].min().date()} to {rates_raw['date'].max().date()}")
    print(f"Rate range: {rates_raw['rate_30yr_fixed'].min():.2f}% to "
          f"{rates_raw['rate_30yr_fixed'].max():.2f}%")
except Exception as e:
    sys.exit(f"ERROR fetching FRED data: {e}\n"
             "Check internet connection or try again later.")


# ════════════════════════════════════════════════════════════════════════════════
# 2. RESAMPLE TO MONTHLY AVERAGES
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 2 — Resampling to monthly averages")
print(f"{'='*60}")

rates_raw = rates_raw.set_index("date")
monthly_rates = (
    rates_raw["rate_30yr_fixed"]
    .resample("MS")
    .mean()
    .round(4)
    .reset_index()
)
monthly_rates.columns = ["year_month", "rate_30yr_fixed"]
monthly_rates["year_month"] = monthly_rates["year_month"].dt.to_period("M").astype(str)

save_csv(monthly_rates, "mortgage_rates_monthly.csv")
print(f"Monthly rate observations: {len(monthly_rates)}")
print(f"\nMost recent 12 months:")
print(monthly_rates.tail(12).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 3. MERGE ONTO SOLD DATASET  (two-pass date fix)
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3 — Merging onto SOLD dataset (CloseDate — two-pass parsing)")
print(f"{'='*60}")

if not os.path.exists(SOLD_FILE):
    print(f"WARNING: {os.path.basename(SOLD_FILE)} not found — skipping sold merge.")
else:
    sold = pd.read_csv(SOLD_FILE, low_memory=False)
    print(f"Sold dataset loaded: {sold.shape[0]:,} rows x {sold.shape[1]} columns")

    if "CloseDate" not in sold.columns:
        print("ERROR: 'CloseDate' column missing from sold dataset — cannot merge.")
    else:
        print(f"  Raw CloseDate sample values:")
        sample = sold["CloseDate"].dropna().head(6).tolist()
        for s in sample:
            print(f"    {s!r}")

        # Two-pass parse: handles both M/D/YYYY and YYYY-MM-DD
        sold["CloseDate"] = parse_mixed_dates(sold["CloseDate"])
        sold["year_month"] = sold["CloseDate"].dt.to_period("M").astype(str)

        total_null = int(sold["CloseDate"].isna().sum())
        print(f"  CloseDate nulls after two-pass parse: {total_null:,}")

        sold_enriched = sold.merge(monthly_rates, on="year_month", how="left")

        null_rate = int(sold_enriched["rate_30yr_fixed"].isna().sum())
        print(f"\nMerge result: {sold_enriched.shape[0]:,} rows")
        print(f"  Rows WITH rate_30yr_fixed populated : {sold_enriched.shape[0] - null_rate:,}")
        print(f"  Rows WITHOUT rate_30yr_fixed (nulls): {null_rate:,}")

        if null_rate == 0:
            print("  VALIDATION PASSED: 0 nulls in rate_30yr_fixed after merge")
        else:
            print(f"  WARNING: {null_rate:,} rows still have no rate — check CloseDate values")
            missing_months = (
                sold_enriched[sold_enriched["rate_30yr_fixed"].isna()]["year_month"]
                .value_counts()
                .head(10)
            )
            print(f"  Top year_month values with no rate match:\n{missing_months.to_string()}")

        preview_cols = ["year_month", "ClosePrice", "rate_30yr_fixed"]
        preview_cols = [c for c in preview_cols if c in sold_enriched.columns]
        print(f"\nPreview (first 5 rows):")
        print(sold_enriched[preview_cols].head(5).to_string(index=False))

        save_csv(sold_enriched, "sold_with_rates.csv")
        print(f"\nSaved: sold_with_rates.csv  ({sold_enriched.shape[0]:,} rows x {sold_enriched.shape[1]} cols)")


# ════════════════════════════════════════════════════════════════════════════════
# 4. MERGE ONTO LISTINGS DATASET  (ListingContractDate — already clean)
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 4 — Merging onto LISTINGS dataset (ListingContractDate)")
print(f"{'='*60}")

if not os.path.exists(LISTINGS_FILE):
    print(f"WARNING: {os.path.basename(LISTINGS_FILE)} not found — skipping listings merge.")
else:
    listings = pd.read_csv(LISTINGS_FILE, low_memory=False)
    dot1_cols = [c for c in listings.columns if c.endswith(".1")]
    if dot1_cols:
        listings.drop(columns=dot1_cols, inplace=True)
    print(f"Listings dataset loaded: {listings.shape[0]:,} rows x {listings.shape[1]} columns")

    if "ListingContractDate" not in listings.columns:
        print("ERROR: 'ListingContractDate' column missing — cannot merge.")
    else:
        listings["ListingContractDate"] = pd.to_datetime(
            listings["ListingContractDate"], errors="coerce"
        )
        listings["year_month"] = listings["ListingContractDate"].dt.to_period("M").astype(str)

        unparsed = int(listings["ListingContractDate"].isna().sum())
        if unparsed:
            print(f"  Note: {unparsed:,} rows have unparseable ListingContractDate")

        listings_enriched = listings.merge(monthly_rates, on="year_month", how="left")

        null_rate = int(listings_enriched["rate_30yr_fixed"].isna().sum())
        print(f"\nMerge result: {listings_enriched.shape[0]:,} rows")
        print(f"  Rows WITH rate_30yr_fixed populated : {listings_enriched.shape[0] - null_rate:,}")
        print(f"  Rows WITHOUT rate_30yr_fixed (nulls): {null_rate:,}")

        if null_rate == 0:
            print("  VALIDATION PASSED: 0 nulls in rate_30yr_fixed after merge")
        else:
            print(f"  WARNING: {null_rate:,} rows still have no rate")
            missing_months = (
                listings_enriched[listings_enriched["rate_30yr_fixed"].isna()]["year_month"]
                .value_counts()
                .head(10)
            )
            print(f"  Top year_month values with no rate match:\n{missing_months.to_string()}")

        preview_cols = ["year_month", "ListPrice", "rate_30yr_fixed"]
        preview_cols = [c for c in preview_cols if c in listings_enriched.columns]
        print(f"\nPreview (first 5 rows):")
        print(listings_enriched[preview_cols].head(5).to_string(index=False))

        save_csv(listings_enriched, "listings_with_rates.csv")
        print(f"\nSaved: listings_with_rates.csv  ({listings_enriched.shape[0]:,} rows x {listings_enriched.shape[1]} cols)")


# ════════════════════════════════════════════════════════════════════════════════
# 5. DELIVERABLES SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALL DELIVERABLES SAVED TO: outputs/")
print(f"{'='*60}")
for i, path in enumerate(deliverables, 1):
    print(f"  {i:>2}. {os.path.basename(path)}")
print(f"\nTotal: {len(deliverables)} files")
print("Mortgage rate enrichment complete.\n")
