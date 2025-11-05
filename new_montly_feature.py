import pandas as pd
import numpy as np
from scipy.stats import skew, entropy

# --- 1️⃣ Input: monthly dataset ---
monthly_df = df_claim.copy()

# --- 2️⃣ Ensure required columns exist ---
required_cols = ["Claimant Unique ID", "Claim_Year", "Claim_Month", "Claim Amount"]
missing = [c for c in required_cols if c not in monthly_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --- 3️⃣ Fill NA claim amounts with 0 (non-claim months) ---
monthly_df["Claim Amount"] = monthly_df["Claim Amount"].fillna(0)

# --- 4️⃣ Compute quarter number (Q1–Q4) for seasonal grouping ---
monthly_df["Quarter"] = ((monthly_df["Claim_Month"] - 1) // 3) + 1

# --- 5️⃣ Define a function to compute yearly-level features ---
def yearly_features(g):
    x = g["Claim Amount"].values
    probs = np.abs(x) / (x.sum() + 1e-6)

    # Quarterly sums
    qsum = g.groupby("Quarter")["Claim Amount"].sum().to_dict()

    return pd.Series({
        "total_claim_amount": x.sum(),
        "mean_monthly_amount": x.mean(),
        "std_monthly_amount": x.std(),
        "coef_var": x.std() / (x.mean() + 1e-6),
        "skew_monthly_amount": skew(x),
        "claim_month_entropy": entropy(probs),
        "months_with_claims": np.sum(x > 0),
        "max_monthly_amount": x.max(),
        "min_monthly_amount": x.min(),
        "Q1_amount": qsum.get(1, 0),
        "Q2_amount": qsum.get(2, 0),
        "Q3_amount": qsum.get(3, 0),
        "Q4_amount": qsum.get(4, 0),
    })

# --- 6️⃣ Apply yearly aggregation ---
yearly = (
    monthly_df
    .groupby(["Claimant Unique ID", "Claim_Year"])
    .apply(yearly_features)
    .reset_index()
)

# --- 7️⃣ Add lag and trend features (temporal structure) ---
yearly = yearly.sort_values(["Claimant Unique ID", "Claim_Year"]).reset_index(drop=True)
yearly["lag1_total"] = yearly.groupby("Claimant Unique ID")["total_claim_amount"].shift(1)
yearly["yoy_change"] = (
    (yearly["total_claim_amount"] - yearly["lag1_total"]) / (yearly["lag1_total"] + 1e-6)
)
yearly["lag1_total"].fillna(0, inplace=True)
yearly["yoy_change"].fillna(0, inplace=True)

# --- 8️⃣ Optional: Merge static member-level features ---
static_cols = [
    "Client Name", "Client Identifier", "Claimant Gender", "Claimant Year of Birth",
    "Scheme Category/ Section Name", "Provider Type"
]
static_cols = [c for c in static_cols if c in monthly_df.columns]
if static_cols:
    static_features = (
        monthly_df.groupby("Claimant Unique ID")[static_cols].agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else np.nan)
        .reset_index()
    )
    yearly = yearly.merge(static_features, on="Claimant Unique ID", how="left")

# --- 9️⃣ Optional: Derive Age (if Year of Birth available) ---
if "Claimant Year of Birth" in yearly.columns:
    yearly["age"] = yearly["Claim_Year"] - yearly["Claimant Year of Birth"]

# --- 🔟 Sort and finalize ---
yearly = yearly.sort_values(["Claimant Unique ID", "Claim_Year"]).reset_index(drop=True)

# --- ✅ Done ---
print("✅ Yearly aggregated dataset created successfully.")
print("Shape:", yearly.shape)
print(yearly.head(10))
