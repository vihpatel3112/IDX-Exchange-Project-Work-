"""
Weeks 2-3 EDA: Active/Combined Listings Residential Dataset
Outputs saved to: ./outputs/
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(BASE_DIR, "combined_listings_residential_updated.csv")
if not os.path.exists(DATA_FILE):
    sys.exit("ERROR: combined_listings_residential_updated.csv not found.")

deliverables = []

# ── Helpers ────────────────────────────────────────────────────────────────────
def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    deliverables.append(path)
    return path

def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    deliverables.append(path)
    return path


# ════════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1 — Loading data")
print(f"{'='*60}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop duplicate .1 columns produced by the combine step
dot1_cols = [c for c in df.columns if c.endswith(".1")]
if dot1_cols:
    df.drop(columns=dot1_cols, inplace=True)
    print(f"Dropped {len(dot1_cols)} duplicate '.1' columns -> {df.shape[1]} columns remain")


# ════════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURE INSPECTION
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 2 — Structure inspection")
print(f"{'='*60}")

# Build MissingCount BEFORE sorting to keep alignment correct
dtypes_df = pd.DataFrame({
    "Column":       df.columns.tolist(),
    "DType":        [str(df[c].dtype) for c in df.columns],
    "NonNull":      df.notna().sum().values,
    "MissingCount": df.isna().sum().values,
    "MissingPct":   (df.isna().mean() * 100).round(2).values,
})

save_csv(dtypes_df, "structure_dtypes_listings.csv")
print(f"Columns: {df.shape[1]}  |  Rows: {df.shape[0]:,}")
print(f"Saved: structure_dtypes_listings.csv")
print("\nColumn dtypes (first 15):")
print(dtypes_df.head(15).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 3. PROPERTYTYPE VALIDATION
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3 — PropertyType validation")
print(f"{'='*60}")

pt_counts = pd.DataFrame()
if "PropertyType" not in df.columns:
    print("WARNING: 'PropertyType' column not found — skipping.")
else:
    pt_counts = (
        df["PropertyType"]
        .value_counts(dropna=False)
        .rename_axis("PropertyType")
        .reset_index(name="Count")
    )
    pt_counts["Share%"] = (pt_counts["Count"] / len(df) * 100).round(2)
    save_csv(pt_counts, "propertytype_distribution_listings.csv")
    print(pt_counts.to_string(index=False))
    only_res = (df["PropertyType"].dropna() == "Residential").all()
    print(f"\nAll rows are Residential: {only_res}")

# MlsStatus breakdown — useful context for listings
if "MlsStatus" in df.columns:
    mls_counts = (
        df["MlsStatus"]
        .value_counts(dropna=False)
        .rename_axis("MlsStatus")
        .reset_index(name="Count")
    )
    mls_counts["Share%"] = (mls_counts["Count"] / len(df) * 100).round(2)
    save_csv(mls_counts, "mlsstatus_distribution_listings.csv")
    deliverables.append(os.path.join(OUTPUT_DIR, "mlsstatus_distribution_listings.csv"))
    print("\nMlsStatus breakdown:")
    print(mls_counts.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 4. MISSING VALUES REPORT
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 4 — Missing values report")
print(f"{'='*60}")

# Assign MissingCount BEFORE sorting to avoid column misalignment
missing = pd.DataFrame({
    "Column":       df.columns.tolist(),
    "MissingCount": df.isna().sum().values,
    "MissingPct":   (df.isna().mean() * 100).round(2).values,
})
missing.sort_values("MissingPct", ascending=False, inplace=True)
missing["Flag90"] = missing["MissingPct"] > 90

save_csv(missing, "missing_values_report_listings.csv")
flagged = missing[missing["Flag90"]]
print(f"Columns >90% missing ({len(flagged)}):")
if len(flagged):
    print(flagged[["Column", "MissingPct"]].to_string(index=False))
else:
    print("  None.")

print(f"\nTop 10 most missing:")
print(missing.head(10)[["Column", "MissingPct"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 5. NUMERIC DISTRIBUTION SUMMARIES
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 5 — Numeric distribution summaries")
print(f"{'='*60}")

CANDIDATE_NUMERIC = [
    "ListPrice", "OriginalListPrice", "LivingArea", "DaysOnMarket",
    "YearBuilt", "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeAcres", "LotSizeSquareFeet", "AssociationFee",
    "ParkingTotal", "GarageSpaces", "ClosePrice",
]
available = [c for c in CANDIDATE_NUMERIC if c in df.columns]
missing_cols = [c for c in CANDIDATE_NUMERIC if c not in df.columns]
if missing_cols:
    print(f"Columns not in dataset (skipped): {missing_cols}")

stats_rows = []
for col in available:
    s = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(s):
        continue
    stats_rows.append({
        "Column":   col,
        "Count":    int(s.count()),
        "Mean":     round(s.mean(), 2),
        "Median":   round(s.median(), 2),
        "Std":      round(s.std(), 2),
        "Min":      round(s.min(), 2),
        "P25":      round(s.quantile(0.25), 2),
        "P75":      round(s.quantile(0.75), 2),
        "P95":      round(s.quantile(0.95), 2),
        "Max":      round(s.max(), 2),
        "Skewness": round(s.skew(), 4),
    })

stats_df = pd.DataFrame(stats_rows)
save_csv(stats_df, "numeric_distribution_summary_listings.csv")
print(stats_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 6. HISTOGRAMS & BOXPLOTS
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 6 — Generating charts")
print(f"{'='*60}")

PLOT_COLS = [c for c in ["ListPrice", "LivingArea", "DaysOnMarket", "YearBuilt", "LotSizeSquareFeet"]
             if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

sns.set_theme(style="whitegrid", palette="muted")

# --- Histograms ---------------------------------------------------------------
n = len(PLOT_COLS)
cols_grid = 3
rows_grid = (n + cols_grid - 1) // cols_grid
fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(16, 5 * rows_grid))
axes = axes.flatten()

for i, col in enumerate(PLOT_COLS):
    data = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(data):
        axes[i].set_visible(False)
        continue
    cap = data.quantile(0.99)
    data_capped = data[data <= cap]
    axes[i].hist(data_capped, bins=60, color="#2171B5", edgecolor="white", linewidth=0.3)
    axes[i].set_title(f"{col}\n(capped at 99th pct = {cap:,.0f})", fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("Count", fontsize=8)
    axes[i].tick_params(labelsize=7)
    axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

for j in range(n, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribution of Key Numeric Fields (Listings — Residential)", fontsize=13, y=1.01)
plt.tight_layout()
save_fig("histograms_key_fields_listings.png")
print("Saved: histograms_key_fields_listings.png")

# --- Boxplots -----------------------------------------------------------------
fig, axes = plt.subplots(1, len(PLOT_COLS), figsize=(4 * len(PLOT_COLS), 6))
if len(PLOT_COLS) == 1:
    axes = [axes]
for i, col in enumerate(PLOT_COLS):
    data = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(data):
        axes[i].set_visible(False)
        continue
    cap = data.quantile(0.99)
    data_capped = data[data <= cap]
    axes[i].boxplot(data_capped, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#2171B5", color="#084594"),
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].tick_params(labelsize=7)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.suptitle("Boxplots of Key Numeric Fields (Listings — Residential, capped 99th pct)", fontsize=12)
plt.tight_layout()
save_fig("boxplots_key_fields_listings.png")
print("Saved: boxplots_key_fields_listings.png")


# ════════════════════════════════════════════════════════════════════════════════
# 7. ANALYTICAL QUESTIONS
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 7 — Answering analytical questions")
print(f"{'='*60}")

answers = {}

# Q1 — Residential share
if not pt_counts.empty:
    res_row = pt_counts[pt_counts["PropertyType"] == "Residential"]
    res_share = float(res_row["Share%"].values[0]) if len(res_row) else 0.0
    answers["Residential share %"] = round(res_share, 2)
    print(f"\nQ1 — Residential share: {res_share:.2f}%")

# Q2 — Median and average list price
if "ListPrice" in df.columns:
    lp = df["ListPrice"].dropna()
    lp = lp[lp > 0]
    median_lp = lp.median()
    mean_lp   = lp.mean()
    answers["Median ListPrice"] = round(median_lp, 2)
    answers["Mean ListPrice"]   = round(mean_lp, 2)
    print(f"\nQ2 — List Price:")
    print(f"     Median : ${median_lp:,.0f}")
    print(f"     Average: ${mean_lp:,.0f}")

# Q3 — Days on Market (active listings — may be sparse)
if "DaysOnMarket" in df.columns:
    dom = df["DaysOnMarket"].dropna()
    dom = dom[dom >= 0]
    if len(dom) > 0:
        dom_median = dom.median()
        dom_mean   = dom.mean()
        dom_p75    = dom.quantile(0.75)
        dom_p95    = dom.quantile(0.95)
        answers["Median DaysOnMarket"] = round(dom_median, 1)
        answers["Mean DaysOnMarket"]   = round(dom_mean, 1)
        print(f"\nQ3 — Days on Market (listings with DOM populated: {len(dom):,}):")
        print(f"     Median : {dom_median:.0f} days")
        print(f"     Average: {dom_mean:.1f} days")
        print(f"     75th pct: {dom_p75:.0f} days  |  95th pct: {dom_p95:.0f} days")

        # DOM histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        dom_capped = dom[dom <= dom.quantile(0.98)]
        ax.hist(dom_capped, bins=60, color="#E6550D", edgecolor="white", linewidth=0.3)
        ax.axvline(dom_median, color="navy", linewidth=2, linestyle="--",
                   label=f"Median = {dom_median:.0f}")
        ax.axvline(dom_mean, color="red", linewidth=1.5, linestyle=":",
                   label=f"Mean = {dom_mean:.1f}")
        ax.set_title("Days on Market Distribution (Listings — Residential)", fontsize=12)
        ax.set_xlabel("Days on Market")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        save_fig("days_on_market_histogram_listings.png")
        print("     Saved: days_on_market_histogram_listings.png")

# Q4 — List price vs original list price (price reduction analysis)
if "ListPrice" in df.columns and "OriginalListPrice" in df.columns:
    valid = df[["ListPrice", "OriginalListPrice"]].dropna()
    valid = valid[(valid["ListPrice"] > 0) & (valid["OriginalListPrice"] > 0)]
    valid["PriceChangePct"] = ((valid["ListPrice"] - valid["OriginalListPrice"])
                               / valid["OriginalListPrice"] * 100)
    pct_reduced  = (valid["PriceChangePct"] < -0.5).mean() * 100
    pct_increased = (valid["PriceChangePct"] > 0.5).mean() * 100
    pct_same     = 100 - pct_reduced - pct_increased
    median_change = valid["PriceChangePct"].median()
    answers["% Listings with Price Reduction"] = round(pct_reduced, 2)
    answers["% Listings with Price Increase"]  = round(pct_increased, 2)
    answers["Median Price Change %"]           = round(median_change, 4)
    print(f"\nQ4 — List vs Original List Price ({len(valid):,} records):")
    print(f"     Price reduced  : {pct_reduced:.1f}%")
    print(f"     Price unchanged: {pct_same:.1f}%")
    print(f"     Price increased: {pct_increased:.1f}%")
    print(f"     Median price change: {median_change:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    changes_capped = valid["PriceChangePct"]
    changes_capped = changes_capped[changes_capped.between(-30, 30)]
    ax.hist(changes_capped, bins=80, color="#74C476", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linewidth=2, linestyle="--", label="No change")
    ax.axvline(median_change, color="navy", linewidth=1.5, linestyle=":",
               label=f"Median = {median_change:.2f}%")
    ax.set_title("Price Change % vs Original List (Listings — Residential)", fontsize=12)
    ax.set_xlabel("(ListPrice - OriginalListPrice) / OriginalListPrice  ×100")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    save_fig("price_change_vs_original_listings.png")
    print("     Saved: price_change_vs_original_listings.png")

# Q5 — Date consistency
print(f"\nQ5 — Date consistency checks:")
date_cols = [c for c in ["ListingContractDate", "CloseDate", "PurchaseContractDate",
                          "ContractStatusChangeDate"] if c in df.columns]
date_issues = []
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    bad    = int(df[col].isna().sum())
    future = int((df[col] > pd.Timestamp("today")).sum())
    old    = int((df[col] < pd.Timestamp("2000-01-01")).sum())
    date_issues.append({"DateColumn": col, "UnparsedRows": bad,
                        "FutureDate": future, "Before2000": old})
    print(f"     {col}: unparsed={bad}, future={future}, before-2000={old}")

if date_issues:
    save_csv(pd.DataFrame(date_issues), "date_consistency_report_listings.csv")

# Q6 — Top counties by median list price
if "CountyOrParish" in df.columns and "ListPrice" in df.columns:
    county = (
        df[df["ListPrice"] > 0]
        .groupby("CountyOrParish")["ListPrice"]
        .agg(MedianListPrice="median", MeanListPrice="mean", Listings="count")
        .reset_index()
        .sort_values("MedianListPrice", ascending=False)
    )
    county["MedianListPrice"] = county["MedianListPrice"].round(0)
    county["MeanListPrice"]   = county["MeanListPrice"].round(0)
    save_csv(county, "county_median_list_price_listings.csv")

    top10 = county.head(10)
    print(f"\nQ6 — Top 10 Counties by Median List Price:")
    print(top10[["CountyOrParish", "MedianListPrice", "Listings"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Blues_r", len(top10))
    ax.barh(top10["CountyOrParish"][::-1], top10["MedianListPrice"][::-1], color=colors)
    ax.set_xlabel("Median List Price ($)")
    ax.set_title("Top 10 Counties by Median List Price (Listings — Residential)", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for i, (val, name) in enumerate(zip(top10["MedianListPrice"][::-1],
                                        top10["CountyOrParish"][::-1])):
        ax.text(val * 1.005, i, f"${val:,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig("top10_counties_median_list_price_listings.png")
    print("     Saved: top10_counties_median_list_price_listings.png")

# Q7 — MlsStatus breakdown chart
if "MlsStatus" in df.columns:
    mls_v = df["MlsStatus"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_mls = sns.color_palette("Set2", len(mls_v))
    ax.bar(mls_v.index, mls_v.values, color=colors_mls, edgecolor="white")
    ax.set_title("MLS Status Distribution (Listings — Residential)", fontsize=12)
    ax.set_xlabel("MlsStatus")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for j, (label, val) in enumerate(zip(mls_v.index, mls_v.values)):
        ax.text(j, val + max(mls_v) * 0.01, f"{val:,}", ha="center", fontsize=8)
    plt.tight_layout()
    save_fig("mlsstatus_distribution_listings.png")
    print("\nSaved: mlsstatus_distribution_listings.png")

# Q8 — Year built distribution
if "YearBuilt" in df.columns:
    yb = df["YearBuilt"].dropna()
    yb = yb[(yb >= 1900) & (yb <= 2026)]
    yb_counts = yb.value_counts().sort_index().reset_index()
    yb_counts.columns = ["YearBuilt", "Count"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(yb_counts["YearBuilt"], yb_counts["Count"],
                    color="#2171B5", alpha=0.7)
    ax.set_title("Listings by Year Built (Residential)", fontsize=12)
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Number of Listings")
    plt.tight_layout()
    save_fig("year_built_distribution_listings.png")
    print("Saved: year_built_distribution_listings.png")

# Save answers summary
answers_df = pd.Series(answers).rename_axis("Metric").reset_index(name="Value")
save_csv(answers_df, "analytical_answers_summary_listings.csv")


# ════════════════════════════════════════════════════════════════════════════════
# 8. SAVE VALIDATED DATASET
# ════════════════════════════════════════════════════════════════════════════════
validated_path = os.path.join(OUTPUT_DIR, "listings_residential_validated.csv")
df.to_csv(validated_path, index=False)
deliverables.append(validated_path)
print(f"\nValidated dataset saved: listings_residential_validated.csv ({df.shape[0]:,} rows x {df.shape[1]} cols)")


# ════════════════════════════════════════════════════════════════════════════════
# 9. DELIVERABLES SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"ALL DELIVERABLES SAVED TO: outputs/")
print(f"{'='*60}")
# deduplicate deliverables list
seen = set()
unique_deliverables = []
for p in deliverables:
    if p not in seen:
        seen.add(p)
        unique_deliverables.append(p)
for i, path in enumerate(unique_deliverables, 1):
    print(f"  {i:>2}. {os.path.basename(path)}")
print(f"\nTotal: {len(unique_deliverables)} files")
print("Listings EDA complete.\n")
