"""
Week 6 - Feature Engineering and Market Metrics
IDX Exchange Data Analyst Internship

Loads the Week 4-5 cleaned sold dataset, engineers market metrics
(price ratios, price per sqft, timeline deltas), performs a spatial
join against CA unified school districts, builds segmented summary
tables, and saves all deliverables to the Week 6 output folder.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
INPUT_CSV = r"C:\Users\vidhi\Desktop\IDX Exchange\IDX Project\Week 4-5\outputs\sold_cleaned_half2.csv"
GEOJSON_PATH = r"C:\Users\vidhi\Desktop\IDX Exchange\IDX Project\ca_school_districts.geojson"
OUTPUT_DIR = r"C:\Users\vidhi\Desktop\IDX Exchange\IDX Project\Week 6"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ========================================================================
# STEP 1: LOAD DATA
# ========================================================================
section("STEP 1: LOAD DATA")

df = pd.read_csv(INPUT_CSV, low_memory=False)
row_count_start = len(df)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {df.shape[1]:,}")

print(
    "\nNote: Latitude/Longitude have some missing values (~16,103 rows). "
    "This is expected -- those rows will simply not receive a DistrictName "
    "later in the school district spatial join (Step 3)."
)
missing_coords = df["Latitude"].isna() | df["Longitude"].isna()
print(f"Actual rows with missing Latitude or Longitude: {missing_coords.sum():,}")


# ========================================================================
# STEP 2: ENGINEER MARKET METRICS
# ========================================================================
section("STEP 2: ENGINEER MARKET METRICS")

before_rows = len(df)

# --- 1 & 2. price_ratio / close_to_original_list_ratio ------------------
# Both metrics are the same calculation (ClosePrice / OriginalListPrice) --
# the handbook lists them as two separate named deliverables, so both
# column names are created here pointing to identical values.
# OriginalListPrice is left untouched; rows where it is 0 would divide to
# inf and silently poison mean() calculations downstream, so those rows are
# flagged and the ratio is set to NaN instead (NaN is excluded by mean()).
df["invalid_original_list_price_flag"] = df["OriginalListPrice"] == 0

df["price_ratio"] = df["ClosePrice"] / df["OriginalListPrice"]
df.loc[df["invalid_original_list_price_flag"], "price_ratio"] = pd.NA

df["close_to_original_list_ratio"] = df["ClosePrice"] / df["OriginalListPrice"]
df.loc[df["invalid_original_list_price_flag"], "close_to_original_list_ratio"] = pd.NA

# --- 3. price_per_sqft ----------------------------------------------------
# LivingArea is left untouched; rows where it is 0 would divide to inf.
# 'invalid_living_area_flag' already exists from Weeks 4-5 and covers
# exactly these zero-LivingArea rows, so it is reused rather than
# duplicated with a new flag column.
df["price_per_sqft"] = df["ClosePrice"] / df["LivingArea"]
df.loc[df["invalid_living_area_flag"], "price_per_sqft"] = pd.NA

# --- 4. days_on_market ------------------------------------------------
# DaysOnMarket already exists as a raw field. Confirm presence/numeric type
# rather than recreating it.
df["DaysOnMarket"] = pd.to_numeric(df["DaysOnMarket"], errors="coerce")
df["days_on_market"] = df["DaysOnMarket"]
print("Confirmed 'DaysOnMarket' is present and coerced to numeric; "
      "'days_on_market' mirrors it for this deliverable.")

# --- 5. YrMo --------------------------------------------------------------
# Clean YYYYMM string derived from CloseDate, separate from the existing
# 'year_month' Period column carried over from Weeks 2-3 mortgage enrichment.
df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
df["YrMo"] = df["CloseDate"].dt.strftime("%Y%m")

# --- 6. listing_to_contract_days -----------------------------------------
df["PurchaseContractDate"] = pd.to_datetime(df["PurchaseContractDate"], errors="coerce")
df["ListingContractDate"] = pd.to_datetime(df["ListingContractDate"], errors="coerce")

df["listing_to_contract_days"] = (
    df["PurchaseContractDate"] - df["ListingContractDate"]
).dt.days
df["invalid_listing_to_contract_flag"] = df["listing_to_contract_days"] < 0

# --- 7. contract_to_close_days --------------------------------------------
df["contract_to_close_days"] = (
    df["CloseDate"] - df["PurchaseContractDate"]
).dt.days
df["invalid_contract_to_close_flag"] = df["contract_to_close_days"] < 0

after_rows = len(df)
print(f"\nRow count before Step 2 transformations: {before_rows:,}")
print(f"Row count after Step 2 transformations:  {after_rows:,}")
assert before_rows == after_rows, "Row count changed during Step 2!"

# --- Sanity check: summary stats for all 7 new numeric columns ------------
numeric_new_cols = [
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "days_on_market",
    "listing_to_contract_days",
    "contract_to_close_days",
]

print("\n--- Summary statistics for new numeric columns ---")
summary_stats = df[numeric_new_cols].agg(["min", "max", "mean", "median"]).T
print(summary_stats)

print("\n--- YrMo value counts (top 20) ---")
print(df["YrMo"].value_counts(dropna=False).sort_index().head(20))

print("\n--- invalid_listing_to_contract_flag value counts ---")
print(df["invalid_listing_to_contract_flag"].value_counts(dropna=False))

print("\n--- invalid_contract_to_close_flag value counts ---")
print(df["invalid_contract_to_close_flag"].value_counts(dropna=False))

print("\n--- invalid_original_list_price_flag value counts (OriginalListPrice == 0) ---")
print(df["invalid_original_list_price_flag"].value_counts(dropna=False))
print("price_ratio / close_to_original_list_ratio set to NaN for these rows "
      "so mean() is no longer skewed by inf.")

print("\n--- invalid_living_area_flag value counts (reused from Weeks 4-5, LivingArea == 0) ---")
print(df["invalid_living_area_flag"].value_counts(dropna=False))
print("price_per_sqft set to NaN for these rows so mean() is no longer skewed by inf.")


# ========================================================================
# STEP 3: SCHOOL DISTRICT SPATIAL JOIN (GeoPandas)
# ========================================================================
section("STEP 3: SCHOOL DISTRICT SPATIAL JOIN")

before_rows_join = len(df)

# 1. Read the school district GeoJSON
districts = gpd.read_file(GEOJSON_PATH)
print(f"School districts loaded: {len(districts):,}")

# 2. Confirm DistrictType values before filtering
print("\nUnique DistrictType values:")
print(districts["DistrictType"].value_counts(dropna=False))

# 3. Filter to Unified districts only
unified_districts = districts[districts["DistrictType"] == "Unified"].copy()

# 4. Confirm remaining count
print(f"\nUnified districts remaining after filter: {len(unified_districts):,}")

# 6. CRS handling -- print district CRS, reproject to EPSG:4326 (WGS84) if needed
print(f"\nSchool district GeoJSON CRS: {unified_districts.crs}")
if unified_districts.crs is None:
    unified_districts = unified_districts.set_crs("EPSG:4326")
elif unified_districts.crs.to_string() != "EPSG:4326":
    print("Reprojecting school districts to EPSG:4326 to match Lat/Long points...")
    unified_districts = unified_districts.to_crs("EPSG:4326")
print(f"School district CRS after alignment: {unified_districts.crs}")

# 5. Build GeoDataFrame from sold dataset points; exclude rows with missing coords
has_coords = df["Latitude"].notna() & df["Longitude"].notna()
print(f"\nRows with valid coordinates (eligible for spatial join): {has_coords.sum():,}")
print(f"Rows without valid coordinates (excluded from join): {(~has_coords).sum():,}")

df_with_coords = df[has_coords].copy()
geometry = [Point(xy) for xy in zip(df_with_coords["Longitude"], df_with_coords["Latitude"])]
gdf_points = gpd.GeoDataFrame(df_with_coords, geometry=geometry, crs="EPSG:4326")

# 7. Spatial join: predicate='within' -- which unified district polygon
# contains each property point
joined = gpd.sjoin(
    gdf_points,
    unified_districts[["DistrictName", "geometry"]],
    how="left",
    predicate="within",
)

# If a point falls within multiple overlapping polygons, sjoin can produce
# duplicate rows -- drop duplicates on the original index, keeping the first match.
joined = joined[~joined.index.duplicated(keep="first")]

# 8. Map DistrictName back onto the full dataset (preserves full row count)
district_map = joined["DistrictName"]
df["DistrictName"] = df.index.map(district_map)

after_rows_join = len(df)
print(f"\nRow count before spatial join: {before_rows_join:,}")
print(f"Row count after spatial join:  {after_rows_join:,}")
assert before_rows_join == after_rows_join, "Row count changed during Step 3!"

# 9. Match diagnostics
matched = df["DistrictName"].notna()
print(f"\nRows matched to a district (DistrictName present): {matched.sum():,}")
print(f"Rows with DistrictName = NaN: {(~matched).sum():,}")
print(f"  -- of which missing coordinates (could not attempt join): {(~has_coords).sum():,}")
outside_polygons = (~matched) & has_coords
print(f"  -- of which had coordinates but fell outside all unified district polygons: {outside_polygons.sum():,}")


# ========================================================================
# STEP 4: SEGMENTED SUMMARY ANALYSIS
# ========================================================================
section("STEP 4: SEGMENTED SUMMARY ANALYSIS")


def build_summary(group_df, group_col):
    summary = group_df.groupby(group_col, dropna=False).agg(
        record_count=("ClosePrice", "count"),
        median_close_price=("ClosePrice", "median"),
        median_price_per_sqft=("price_per_sqft", "median"),
        avg_days_on_market=("DaysOnMarket", "mean"),
        avg_price_ratio=("price_ratio", "mean"),
    )
    return summary


# --- Summary 1: by PropertyType --------------------------------------------
summary_propertytype = build_summary(df, "PropertyType").sort_values(
    "record_count", ascending=False
)
print("\n--- Segment Summary: by PropertyType ---")
print(summary_propertytype)

# --- Summary 2: by CountyOrParish, top 15 by transaction count -------------
county_counts = df["CountyOrParish"].value_counts()
top15_counties = county_counts.head(15).index

summary_county_full = build_summary(df, "CountyOrParish")
summary_county = summary_county_full.loc[top15_counties].sort_values(
    "median_close_price", ascending=False
)
print("\n--- Segment Summary: Top 15 Counties by Transaction Count (sorted by median ClosePrice desc) ---")
print(summary_county)


# ========================================================================
# STEP 5: SAVE OUTPUTS
# ========================================================================
section("STEP 5: SAVE OUTPUTS")

# 1. Full enriched dataset
enriched_path = os.path.join(OUTPUT_DIR, "engineered_features_sold.csv")
df.to_csv(enriched_path, index=False)
print(f"Saved full enriched dataset: {enriched_path}  ({len(df):,} rows, {df.shape[1]} cols)")

# 2 & 3. Sample output table (15 representative rows)
sample_cols = [
    "CloseDate",
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "price_ratio",
    "price_per_sqft",
    "YrMo",
    "listing_to_contract_days",
    "contract_to_close_days",
    "DistrictName",
    "PropertyType",
    "CountyOrParish",
]

# Prefer rows that actually matched a district so the sample is representative
sample_source = df[df["DistrictName"].notna()]
if len(sample_source) < 15:
    sample_source = df
sample_df = sample_source[sample_cols].sample(n=15, random_state=42).reset_index(drop=True)

print("\n--- Sample Output Table (15 rows) ---")
print(sample_df)

sample_path = os.path.join(OUTPUT_DIR, "sample_output_table.csv")
sample_df.to_csv(sample_path, index=False)
print(f"\nSaved sample output table: {sample_path}")

# 4. PropertyType segment summary
propertytype_summary_path = os.path.join(OUTPUT_DIR, "segment_summary_propertytype.csv")
summary_propertytype.to_csv(propertytype_summary_path)
print(f"Saved PropertyType segment summary: {propertytype_summary_path}")

# 5. CountyOrParish segment summary
county_summary_path = os.path.join(OUTPUT_DIR, "segment_summary_county.csv")
summary_county.to_csv(county_summary_path)
print(f"Saved CountyOrParish segment summary: {county_summary_path}")


# ========================================================================
# FINAL SUMMARY
# ========================================================================
section("FINAL SUMMARY")

print(f"Row count at start of script:  {row_count_start:,}")
print(f"Row count at end of script:    {len(df):,}")
assert row_count_start == len(df), "Row count changed somewhere in the script!"
print("No rows were added or dropped during Week 6 processing.")

new_columns = [
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "days_on_market",
    "YrMo",
    "listing_to_contract_days",
    "invalid_listing_to_contract_flag",
    "contract_to_close_days",
    "invalid_contract_to_close_flag",
    "invalid_original_list_price_flag",
    "DistrictName",
]

print("\nNew columns added in Week 6:")
for col in new_columns:
    present = "OK" if col in df.columns else "MISSING"
    print(f"  - {col}  [{present}]")

print("\nNote: 'invalid_living_area_flag' (Weeks 4-5) was reused rather than "
      "duplicated -- it already flags LivingArea == 0 and is used in Week 6 "
      "to null out price_per_sqft for those rows.")

print("\nWeek 6 feature engineering script complete.")
