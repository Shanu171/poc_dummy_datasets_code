import pandas as pd
import numpy as np

# --- 1. Load Data ---


# --- 2. Convert date columns to datetime ---
df['Paid Date'] = pd.to_datetime(df['Paid Date'], errors='coerce')
df['Incurred Date'] = pd.to_datetime(df['Incurred Date'], errors='coerce')

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
    df.groupby(['Claimant Unique ID', 'Claim_Year'], as_index=False)[static_cols]
      .agg(lambda s: first_non_null(s))
)

# --- 6. Create 12 months for each claimant-year combination ---
all_months = pd.DataFrame({'Claim_Month': range(1, 13)})
expanded = claimant_info.merge(all_months, how='cross')

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

# --- 9. (Optional) Fill Claim Amount/ID for missing months with NaN or 0 ---
merged['Claim Amount'] = merged['Claim Amount'].fillna(0)

# --- 10. Save the result ---
#output_path = "/mnt/data/uk_pmi_claims_200k_12month_expanded.csv"
#merged.to_csv(output_path, index=False)

print("✅ Done! Expanded monthly dataset created.")
print("Output shape:", merged.shape)
(merged.head(20))
