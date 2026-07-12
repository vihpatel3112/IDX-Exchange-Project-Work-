"""
Weeks 2-3 EDA: Sold Residential Dataset
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Accept _v2 if present, otherwise fall back to _updated
CANDIDATES = [
    os.path.join(BASE_DIR, "combined_sold_residential_updated_v2.csv"),
    os.path.join(BASE_DIR, "combined_sold_residential_updated.csv"),
]
DATA_FILE = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA_FILE is None:
    sys.exit("ERROR: No combined sold residential CSV found. Run week1_aggregate.py first.")

deliverables = []   # tracks every file saved

# ── Helper ─────────────────────────────────────────────────────────────────────
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
print(f"File: {os.path.basename(DATA_FILE)}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ════════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURE INSPECTION
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 2 — Structure inspection")
print(f"{'='*60}")

dtypes_df = df.dtypes.reset_index()
dtypes_df.columns = ["Column", "DType"]
dtypes_df["NonNull"] = df.notna().sum().values
dtypes_df["NullPct"] = (df.isna().mean() * 100).round(2).values

save_csv(dtypes_df, "structure_dtypes.csv")
print(f"Columns: {df.shape[1]}  |  Rows: {df.shape[0]:,}")
print(f"Saved: structure_dtypes.csv")

# Sample of dtypes
print("\nColumn dtypes (first 15):")
print(dtypes_df.head(15).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 3. PROPERTYTYPE VALIDATION
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3 — PropertyType validation")
print(f"{'='*60}")

if "PropertyType" not in df.columns:
    print("WARNING: 'PropertyType' column not found — skipping validation.")
    pt_counts = pd.DataFrame()
else:
    pt_counts = (
        df["PropertyType"]
        .value_counts(dropna=False)
        .rename_axis("PropertyType")
        .reset_index(name="Count")
    )
    pt_counts["Share%"] = (pt_counts["Count"] / len(df) * 100).round(2)
    save_csv(pt_counts, "propertytype_distribution.csv")
    print(pt_counts.to_string(index=False))

    only_residential = (df["PropertyType"].dropna() == "Residential").all()
    print(f"\nAll rows are Residential: {only_residential}")
    if not only_residential:
        non_res = df[df["PropertyType"] != "Residential"].shape[0]
        print(f"Non-residential rows found: {non_res:,} — these should be investigated.")


# ════════════════════════════════════════════════════════════════════════════════
# 4. MISSING VALUES REPORT
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 4 — Missing values report")
print(f"{'='*60}")

missing = pd.DataFrame({
    "Column":       df.columns.tolist(),
    "MissingCount": df.isna().sum().values,
    "MissingPct":   (df.isna().mean() * 100).round(2).values,
})
missing.sort_values("MissingPct", ascending=False, inplace=True)
missing["Flag90"] = missing["MissingPct"] > 90

save_csv(missing, "missing_values_report.csv")
flagged = missing[missing["Flag90"]]
print(f"Columns >90% missing ({len(flagged)}):")
if len(flagged):
    print(flagged[["Column", "MissingPct"]].to_string(index=False))
else:
    print("  None — all columns are reasonably populated.")

print(f"\nTop 10 most missing:")
print(missing.head(10)[["Column", "MissingPct"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 5. NUMERIC DISTRIBUTION SUMMARIES
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 5 — Numeric distribution summaries")
print(f"{'='*60}")

KEY_NUMERIC = ["ClosePrice", "LivingArea", "DaysOnMarket", "ListPrice", "YearBuilt",
               "OriginalListPrice", "BedroomsTotal", "BathroomsTotalInteger"]
available = [c for c in KEY_NUMERIC if c in df.columns]
missing_cols = [c for c in KEY_NUMERIC if c not in df.columns]
if missing_cols:
    print(f"WARNING: columns not found, skipped: {missing_cols}")

stats_rows = []
for col in available:
    s = df[col].dropna()
    stats_rows.append({
        "Column":    col,
        "Count":     int(s.count()),
        "Mean":      round(s.mean(), 2),
        "Median":    round(s.median(), 2),
        "Std":       round(s.std(), 2),
        "Min":       round(s.min(), 2),
        "P25":       round(s.quantile(0.25), 2),
        "P75":       round(s.quantile(0.75), 2),
        "P95":       round(s.quantile(0.95), 2),
        "Max":       round(s.max(), 2),
        "Skewness":  round(s.skew(), 4),
    })

stats_df = pd.DataFrame(stats_rows)
save_csv(stats_df, "numeric_distribution_summary.csv")
print(stats_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# 6. HISTOGRAMS & BOXPLOTS
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 6 — Generating charts")
print(f"{'='*60}")

PLOT_COLS = [c for c in ["ClosePrice", "LivingArea", "DaysOnMarket", "ListPrice", "YearBuilt"]
             if c in df.columns]

sns.set_theme(style="whitegrid", palette="muted")

# --- Histograms (grid) --------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(PLOT_COLS):
    data = df[col].dropna()
    # cap extreme outliers for readability (99th pct)
    cap = data.quantile(0.99)
    data_capped = data[data <= cap]
    axes[i].hist(data_capped, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[i].set_title(f"{col}\n(capped at 99th pct = {cap:,.0f})", fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("Count", fontsize=8)
    axes[i].tick_params(labelsize=7)
    axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

for j in range(len(PLOT_COLS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribution of Key Numeric Fields (Sold Residential)", fontsize=13, y=1.01)
plt.tight_layout()
save_fig("histograms_key_fields.png")
print("Saved: histograms_key_fields.png")

# --- Boxplots (grid) ----------------------------------------------------------
fig, axes = plt.subplots(1, len(PLOT_COLS), figsize=(4 * len(PLOT_COLS), 6))
if len(PLOT_COLS) == 1:
    axes = [axes]
for i, col in enumerate(PLOT_COLS):
    data = df[col].dropna()
    cap = data.quantile(0.99)
    data_capped = data[data <= cap]
    axes[i].boxplot(data_capped, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", color="#2d4a7a"),
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel(col, fontsize=8)
    axes[i].tick_params(labelsize=7)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.suptitle("Boxplots of Key Numeric Fields (Sold Residential, capped 99th pct)", fontsize=12)
plt.tight_layout()
save_fig("boxplots_key_fields.png")
print("Saved: boxplots_key_fields.png")


# ════════════════════════════════════════════════════════════════════════════════
# 7. ANALYTICAL QUESTIONS
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 7 — Answering analytical questions")
print(f"{'='*60}")

answers = {}

# Q1 — Residential vs other share
if not pt_counts.empty:
    res_row = pt_counts[pt_counts["PropertyType"] == "Residential"]
    res_share = res_row["Share%"].values[0] if len(res_row) else 0
    answers["Residential share %"] = round(res_share, 2)
    answers["Non-residential rows"] = int(len(df) - (res_row["Count"].values[0] if len(res_row) else 0))
    print(f"\nQ1 — Residential share: {res_share:.2f}%")

# Q2 — Median and average close price
if "ClosePrice" in df.columns:
    cp = df["ClosePrice"].dropna()
    median_cp = cp.median()
    mean_cp   = cp.mean()
    answers["Median ClosePrice"] = round(median_cp, 2)
    answers["Mean ClosePrice"]   = round(mean_cp, 2)
    print(f"\nQ2 — Close Price:")
    print(f"     Median : ${median_cp:,.0f}")
    print(f"     Average: ${mean_cp:,.0f}")

# Q3 — Days on Market
if "DaysOnMarket" in df.columns:
    dom = df["DaysOnMarket"].dropna()
    dom_median = dom.median()
    dom_mean   = dom.mean()
    dom_p75    = dom.quantile(0.75)
    dom_p95    = dom.quantile(0.95)
    answers["Median DaysOnMarket"] = round(dom_median, 1)
    answers["Mean DaysOnMarket"]   = round(dom_mean, 1)
    answers["P75 DaysOnMarket"]    = round(dom_p75, 1)
    answers["P95 DaysOnMarket"]    = round(dom_p95, 1)
    print(f"\nQ3 — Days on Market:")
    print(f"     Median : {dom_median:.0f} days")
    print(f"     Average: {dom_mean:.1f} days")
    print(f"     75th pct: {dom_p75:.0f} days  |  95th pct: {dom_p95:.0f} days")

    # DOM histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    dom_capped = dom[dom <= dom.quantile(0.98)]
    ax.hist(dom_capped, bins=60, color="#DD8452", edgecolor="white", linewidth=0.3)
    ax.axvline(dom_median, color="navy", linewidth=2, linestyle="--", label=f"Median = {dom_median:.0f}")
    ax.axvline(dom_mean,   color="red",  linewidth=1.5, linestyle=":",  label=f"Mean = {dom_mean:.1f}")
    ax.set_title("Days on Market Distribution (Sold Residential)", fontsize=12)
    ax.set_xlabel("Days on Market")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    save_fig("days_on_market_histogram.png")
    print("     Saved: days_on_market_histogram.png")

# Q4 — % sold above vs below list price
if "ClosePrice" in df.columns and "ListPrice" in df.columns:
    valid = df[["ClosePrice", "ListPrice"]].dropna()
    valid = valid[(valid["ListPrice"] > 0) & (valid["ClosePrice"] > 0)]
    valid["SaleToList"] = valid["ClosePrice"] / valid["ListPrice"]
    pct_above = (valid["SaleToList"] > 1).mean() * 100
    pct_below = (valid["SaleToList"] < 1).mean() * 100
    pct_exact = (valid["SaleToList"] == 1).mean() * 100
    median_ratio = valid["SaleToList"].median() * 100
    answers["% Sold Above List"]  = round(pct_above, 2)
    answers["% Sold Below List"]  = round(pct_below, 2)
    answers["% Sold At List"]     = round(pct_exact, 2)
    answers["Median Sale-to-List %"] = round(median_ratio, 2)
    print(f"\nQ4 — Sale vs List Price ({len(valid):,} records):")
    print(f"     Sold above list: {pct_above:.1f}%")
    print(f"     Sold at list   : {pct_exact:.1f}%")
    print(f"     Sold below list: {pct_below:.1f}%")
    print(f"     Median sale-to-list ratio: {median_ratio:.2f}%")

    # Sale-to-list histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ratio_capped = valid["SaleToList"]
    ratio_capped = ratio_capped[(ratio_capped >= 0.7) & (ratio_capped <= 1.3)]
    ax.hist(ratio_capped, bins=80, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="red", linewidth=2, linestyle="--", label="List Price = Close Price")
    ax.axvline(median_ratio/100, color="navy", linewidth=1.5, linestyle=":", label=f"Median = {median_ratio:.2f}%")
    ax.set_title("Sale-to-List Price Ratio (Sold Residential)", fontsize=12)
    ax.set_xlabel("ClosePrice / ListPrice")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    save_fig("sale_to_list_ratio.png")
    print("     Saved: sale_to_list_ratio.png")

# Q5 — Date consistency issues
print(f"\nQ5 — Date consistency checks:")
date_cols = [c for c in ["ListingContractDate", "CloseDate", "PurchaseContractDate",
                          "ContractStatusChangeDate"] if c in df.columns]
date_issues = []
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    bad = df[col].isna().sum()
    future = (df[col] > pd.Timestamp("today")).sum()
    too_old = (df[col] < pd.Timestamp("2000-01-01")).sum()
    date_issues.append({"DateColumn": col, "UnparsedRows": int(bad),
                        "FutureDate": int(future), "Before2000": int(too_old)})
    print(f"     {col}: unparsed={bad}, future={future}, before-2000={too_old}")

if date_issues:
    date_issues_df = pd.DataFrame(date_issues)
    save_csv(date_issues_df, "date_consistency_report.csv")

# Close before listing check
if "ListingContractDate" in df.columns and "CloseDate" in df.columns:
    inverted = (df["CloseDate"] < df["ListingContractDate"]).sum()
    answers["CloseDate before ListingDate (bad)"] = int(inverted)
    print(f"     CloseDate < ListingContractDate (inconsistent): {inverted:,}")

# Q6 — Top counties by median close price
if "CountyOrParish" in df.columns and "ClosePrice" in df.columns:
    county = (
        df.groupby("CountyOrParish")["ClosePrice"]
        .agg(MedianClosePrice="median", MeanClosePrice="mean", Transactions="count")
        .reset_index()
        .sort_values("MedianClosePrice", ascending=False)
    )
    county["MedianClosePrice"] = county["MedianClosePrice"].round(0)
    county["MeanClosePrice"]   = county["MeanClosePrice"].round(0)
    save_csv(county, "county_median_close_price.csv")

    top10 = county.head(10)
    print(f"\nQ6 — Top 10 Counties by Median Close Price:")
    print(top10[["CountyOrParish", "MedianClosePrice", "Transactions"]].to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Blues_r", len(top10))
    ax.barh(top10["CountyOrParish"][::-1], top10["MedianClosePrice"][::-1], color=colors)
    ax.set_xlabel("Median Close Price ($)")
    ax.set_title("Top 10 Counties by Median Close Price (Sold Residential)", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for i, (val, name) in enumerate(zip(top10["MedianClosePrice"][::-1],
                                        top10["CountyOrParish"][::-1])):
        ax.text(val * 1.005, i, f"${val:,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig("top10_counties_median_price.png")
    print("     Saved: top10_counties_median_price.png")

    # Bottom 10 counties
    bottom10 = county.tail(10)
    save_csv(bottom10, "county_bottom10_median_price.csv")

# Save answers summary
answers_df = pd.Series(answers).rename_axis("Metric").reset_index(name="Value")
save_csv(answers_df, "analytical_answers_summary.csv")


# ════════════════════════════════════════════════════════════════════════════════
# 8. YEAR BUILT TREND (bonus)
# ════════════════════════════════════════════════════════════════════════════════
if "YearBuilt" in df.columns:
    yb = df["YearBuilt"].dropna()
    yb = yb[(yb >= 1900) & (yb <= 2026)]
    yb_counts = yb.value_counts().sort_index().reset_index()
    yb_counts.columns = ["YearBuilt", "Count"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(yb_counts["YearBuilt"], yb_counts["Count"], color="#4C72B0", alpha=0.7)
    ax.set_title("Properties Sold by Year Built (Sold Residential)", fontsize=12)
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Number of Transactions")
    plt.tight_layout()
    save_fig("year_built_distribution.png")


# ════════════════════════════════════════════════════════════════════════════════
# 9. SAVE VALIDATED DATASET
# ════════════════════════════════════════════════════════════════════════════════
validated_path = os.path.join(OUTPUT_DIR, "sold_residential_validated.csv")
df.to_csv(validated_path, index=False)
deliverables.append(validated_path)
print(f"\nValidated dataset saved: sold_residential_validated.csv ({df.shape[0]:,} rows x {df.shape[1]} cols)")


# ════════════════════════════════════════════════════════════════════════════════
# 10. DELIVERABLES SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"ALL DELIVERABLES SAVED TO: outputs/")
print(f"{'='*60}")
for i, path in enumerate(deliverables, 1):
    print(f"  {i:>2}. {os.path.basename(path)}")
print(f"\nTotal: {len(deliverables)} files")
print("EDA complete.\n")
