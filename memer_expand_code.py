# MOnthly Data Creation Code
import pandas as pd
import numpy as np

# --- 1. Load Data ---
# assuming df already loaded

# --- 2. Convert date columns to datetime ---
df['Paid Date'] = pd.to_datetime(df['Paid Date'], errors='coerce')
df['Incurred Date'] = pd.to_datetime(df['Incurred Date'], errors='coerce')
df['Contract Start Date'] = pd.to_datetime(df['Contract Start Date'], errors='coerce')
df['Contract End Date'] = pd.to_datetime(df['Contract End Date'], errors='coerce')

# --- 3. Extract year and month ---
df['Claim_Year'] = df['Paid Date'].dt.year
df['Claim_Month'] = df['Paid Date'].dt.month

# --- 4. Define static columns ---
static_cols = [
    'Client Name', 'Client Identifier', 'Scheme Category/ Section Name',
    'Scheme Category/ Section Name Identifier', 'Status of Member',
    'Claimant Unique ID', 'Claimant Year of Birth', 'Claimant Gender',
    'Short Post Code', 'Unique Member Reference', 'Contract Start Date',
    'Contract End Date', 'Provider Type'
]
static_cols = [c for c in static_cols if c in df.columns]

# --- 5. Build unique claimant-year static info ---
def first_non_null(series):
    non_null = series.dropna().unique()
    return non_null[0] if len(non_null) > 0 else np.nan

claimant_info = (
    df.groupby(['Claimant Unique ID'], as_index=False)[static_cols]
      .agg(lambda s: first_non_null(s))
)

# --- 6. Add contract-based year expansion ---
rows = []
for _, r in claimant_info.iterrows():
    member_id = r['Claimant Unique ID']
    start = r['Contract Start Date']
    end = r['Contract End Date']

    # Handle invalid or missing contract dates
    if pd.isna(start) or pd.isna(end) or start > end:
        continue

    start_year = start.year
    end_year = end.year

    # Create full range of years from start to end
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            row = r.copy()
            row['Claim_Year'] = y
            row['Claim_Month'] = m
            rows.append(row)

# Dense month-year dataset based on contract window
expanded = pd.DataFrame(rows)

# --- 7. Merge back claim-level data ---
claim_level_cols = [
    c for c in df.columns if c not in static_cols + ['Claim_Year', 'Claim_Month']
]
claims_min = df[['Claimant Unique ID', 'Claim_Year', 'Claim_Month'] + claim_level_cols].copy()

merged = expanded.merge(
    claims_min,
    on=['Claimant Unique ID', 'Claim_Year', 'Claim_Month'],
    how='left'
)

# --- 8. Sort and reset index ---
merged = merged.sort_values(['Claimant Unique ID', 'Claim_Year', 'Claim_Month']).reset_index(drop=True)

# --- 9. Fill Claim Amount for missing months with 0 ---
merged['Claim Amount'] = merged['Claim Amount'].fillna(0)

# --- 10. Save or inspect ---
print("✅ Done! Expanded monthly dataset created using contract years.")
print("Output shape:", merged.shape)
(merged.head(20))
