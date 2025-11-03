import pandas as pd
import numpy as np

# --- 1. Load your claim data ---
# df = pd.read_csv("claims.csv")  # Example if reading from CSV
# For demo, assume df already loaded

# --- 2. Convert date columns to datetime ---
df['Paid Date'] = pd.to_datetime(df['Paid Date'], errors='coerce')
df['Incurred Date'] = pd.to_datetime(df['Incurred Date'], errors='coerce')

# --- 3. Extract year and month for grouping ---
df['Claim_Year'] = df['Paid Date'].dt.year
df['Claim_Month'] = df['Paid Date'].dt.month

# --- 4. Define member ID and year combinations ---
member_years = df[['Unique Member Reference', 'Client Name', 'Claim_Year']].drop_duplicates()

# --- 5. Create a template with all 12 months ---
all_months = pd.DataFrame({'Claim_Month': range(1, 13)})
expanded = (
    member_years
    .merge(all_months, how='cross')
)

# --- 6. Merge back the original claims ---
merged = expanded.merge(
    df,
    on=['Unique Member Reference', 'Client Name', 'Claim_Year', 'Claim_Month'],
    how='left',
    suffixes=('', '_orig')
)

# --- 7. Sort for neatness ---
merged = merged.sort_values(['Unique Member Reference', 'Claim_Year', 'Claim_Month'])

# --- 8. Optional: Fill static member info forward if missing ---
static_cols = [
    'Client Identifier', 'Scheme Category/ Section Name', 'Scheme Category/ Section Name Identifier',
    'status of member', 'claimant unique ID', 'Claimant year of birth', 'claimant Gender',
    'short post code', 'Contract Start Date', 'Contract End Date', 'Provider Type'
]

for col in static_cols:
    if col in merged.columns:
        merged[col] = merged.groupby(['Unique Member Reference', 'Claim_Year'])[col].ffill().bfill()

# --- 9. For missing claim months, Claim Amount etc. will stay NaN ---
# Optionally replace with 0 if you prefer:
# merged['Claim Amount'] = merged['Claim Amount'].fillna(0)

# --- 10. Reset index ---
merged.reset_index(drop=True, inplace=True)

# --- 11. Save or inspect ---
# merged.to_csv("claims_filled_12_months.csv", index=False)
print(merged.head(24))
