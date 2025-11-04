import pandas as pd
import numpy as np

# ---------------------------
# Example: load your data
# ---------------------------
# df = pd.read_csv("claims.csv")               # real use
# For demonstration assume df exists

# --- 0. Ensure date columns are datetime ---
df['Paid Date'] = pd.to_datetime(df['Paid Date'], errors='coerce')
df['Incurred Date'] = pd.to_datetime(df['Incurred Date'], errors='coerce')

# --- 1. Extract year & month (choose Paid Date; change if needed) ---
df['Claim_Year'] = df['Paid Date'].dt.year
df['Claim_Month'] = df['Paid Date'].dt.month

# If you have claims spanning multiple years and you want to limit to a single year,
# you can filter here, e.g.: df = df[df['Claim_Year']==2024]

# --- 2. Define which columns are "static member-level" vs "claim-level" ---
static_cols = [
    'Client Name', 'Client Identifier', 'Scheme Category/ Section Name',
    'Scheme Category/ Section Name Identifier', 'status of member',
    'claimant unique ID', 'Claimant year of birth', 'claimant Gender',
    'short post code', 'Unique Member Reference', 'Contract Start Date',
    'Contract End Date', 'Provider Type'
]

# Ensure these are present in df (ignore missing ones)
static_cols = [c for c in static_cols if c in df.columns]

# Claim-level columns (everything else that varies with a claim)
claim_level_cols = [c for c in df.columns if c not in static_cols + ['Claim_Year', 'Claim_Month']]

# --- 3. Build canonical member-year static table: take first non-null value per group ---
def first_non_null(series):
    non_nulls = series.dropna().unique()
    return non_nulls[0] if len(non_nulls) > 0 else np.nan

member_info = (
    df
    .groupby(['Unique Member Reference', 'Claim_Year'], dropna=False)
    [static_cols]
    .agg(lambda s: first_non_null(s))
    .reset_index()
)

# If 'Unique Member Reference' may be missing but another combination is needed,
# you could group by Client Identifier or claimant unique ID instead.

# --- 4. Prepare all 12 months per member-year (cross join) ---
all_months = pd.DataFrame({'Claim_Month': range(1, 13)})

expanded = member_info.merge(all_months, how='cross')  # requires pandas >= 1.2

# --- 5. Merge claim transactions onto expanded template ---
# Keep claim-level columns only from original df to avoid overwriting static columns
claims_min = df[['Unique Member Reference', 'Claim_Year', 'Claim_Month'] + claim_level_cols].copy()

merged = expanded.merge(
    claims_min,
    on=['Unique Member Reference', 'Claim_Year', 'Claim_Month'],
    how='left',
    suffixes=('', '_claim')
)

# --- 6. If some static columns were present in claim rows with different names,
# prefer the member_info (already present). If you'd like to fill any remaining static NaNs
# from claim rows, do this carefully (example below) -- often not necessary.
for col in static_cols:
    if col not in merged.columns:
        continue
    # If there are NaNs in the static column (rare), attempt to fill from any matching claim rows:
    if merged[col].isna().any():
        claim_col_same = col + '_claim'
        if claim_col_same in merged.columns:
            merged[col] = merged[col].fillna(merged[claim_col_same])
            merged.drop(columns=[claim_col_same], inplace=True)

# --- 7. Final tidy up: sort, reorder columns ---
# Put member columns first, then year-month, then static columns, then claim-level columns
first_cols = ['Unique Member Reference', 'Claim_Year', 'Claim_Month'] + static_cols
remaining_cols = [c for c in merged.columns if c not in first_cols]
final_cols = first_cols + remaining_cols

merged = merged[final_cols].sort_values(['Unique Member Reference', 'Claim_Year', 'Claim_Month']).reset_index(drop=True)

# --- 8. Export or inspect ---
# merged.to_csv("claims_12_months_with_static.csv", index=False)
print(merged.head(30))
