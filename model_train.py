"""
PMI Claims Prediction Pipeline (2020–2025)
-------------------------------------------
Implements:
  1️⃣ Single-model regression (Tweedie) → LightGBM, XGBoost, MLP
  2️⃣ Two-part model (Prob. × Severity) → LightGBM, XGBoost, MLP

Join keys:
  - Claims: 'Claimant Unique ID'
  - Membership: 'Unique ID'

Train: 2020–2024
Predict: 2025
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error, roc_auc_score
import joblib, warnings, json
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
CLAIMS_PATH = "/mnt/data/uk_pmi_claims_200k.csv"
MEMBERS_PATH = "/mnt/data/uk_pmi_membership_120k.csv"

JOIN_CLAIM_COL = "Claimant Unique ID"
JOIN_MEMBER_COL = "Unique ID"
AMOUNT_COL = "Amount Paid"
INCURRED_DATE_COL = "Incurred Date"
CLAIM_TYPE_COL = "Claim Type"
LOS_COL = "Calculated Length of Service"
CONDITION_COL = "Condition Code"
PROVIDER_COL = "Provider Type"
POSTCODE_COL = "short post code"

DOB_COL = "Claimant year of birth"
GENDER_COL = "claimant Gender"
JOIN_DATE_COL = "Contract Start Date"
SCHEME_COL = "Scheme Category/ Section Name"
STATUS_COL = "status of member"

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
PRED_YEAR = 2025

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
print("📥 Loading data ...")
claims = pd.read_csv(CLAIMS_PATH, low_memory=False)
members = pd.read_csv(MEMBERS_PATH, low_memory=False)
claims[JOIN_CLAIM_COL] = claims[JOIN_CLAIM_COL].astype(str)
members[JOIN_MEMBER_COL] = members[JOIN_MEMBER_COL].astype(str)

print(f"Claims shape: {claims.shape}, Members shape: {members.shape}")

# -------------------------------------------------------------------
# BASIC CLEANING & AGGREGATION
# -------------------------------------------------------------------
claims["incurred_dt"] = pd.to_datetime(claims[INCURRED_DATE_COL], errors="coerce")
claims["claim_year"] = claims["incurred_dt"].dt.year
claims["amount_paid"] = pd.to_numeric(claims[AMOUNT_COL], errors="coerce").fillna(0)
claims["los"] = pd.to_numeric(claims.get(LOS_COL, 0), errors="coerce").fillna(0)
claims["is_inpatient"] = claims[CLAIM_TYPE_COL].str.lower().str.contains("inpat", na=False).astype(int)
claims["is_outpatient"] = claims[CLAIM_TYPE_COL].str.lower().str.contains("outpat", na=False).astype(int)

# --- Aggregate by member-year ---
agg = claims.groupby([JOIN_CLAIM_COL, "claim_year"]).agg(
    total_claim_amount=("amount_paid", "sum"),
    claim_count=("amount_paid", "count"),
    inpatient_count=("is_inpatient", "sum"),
    outpatient_count=("is_outpatient", "sum"),
    total_los=("los", "sum"),
    avg_los=("los", "mean"),
    unique_conditions=(CONDITION_COL, pd.Series.nunique),
    unique_providers=(PROVIDER_COL, pd.Series.nunique),
).reset_index().rename(columns={JOIN_CLAIM_COL: "Unique ID", "claim_year": "year"})
agg = agg.fillna(0)

# -------------------------------------------------------------------
# CREATE MEMBER-YEAR SCAFFOLD (zero-claim years)
# -------------------------------------------------------------------
all_members = members[JOIN_MEMBER_COL].unique()
years = list(range(2020, 2026))
scaffold = pd.MultiIndex.from_product([all_members, years], names=["Unique ID", "year"]).to_frame(index=False)
df = scaffold.merge(agg, how="left", on=["Unique ID", "year"]).fillna(0)

# -------------------------------------------------------------------
# ADD DEMOGRAPHICS
# -------------------------------------------------------------------
members["birth_year"] = pd.to_numeric(members[DOB_COL], errors="coerce")
members["join_year"] = pd.to_datetime(members[JOIN_DATE_COL], errors="coerce").dt.year
members = members.rename(columns={JOIN_MEMBER_COL: "Unique ID"})
df = df.merge(members[["Unique ID", "birth_year", GENDER_COL, SCHEME_COL, POSTCODE_COL, STATUS_COL, "join_year"]],
              on="Unique ID", how="left")

df["age"] = df["year"] - df["birth_year"]
df["tenure_years"] = df["year"] - df["join_year"]
df["age"] = df["age"].clip(lower=0)
df["tenure_years"] = df["tenure_years"].clip(lower=0)

# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
df = df.sort_values(["Unique ID", "year"])
grp = df.groupby("Unique ID")

# Lag features
for lag in [1, 2, 3]:
    df[f"total_claim_amount_lag{lag}"] = grp["total_claim_amount"].shift(lag)
    df[f"claim_count_lag{lag}"] = grp["claim_count"].shift(lag)
    df[f"inpatient_count_lag{lag}"] = grp["inpatient_count"].shift(lag)
    df[f"total_los_lag{lag}"] = grp["total_los"].shift(lag)
df.fillna(0, inplace=True)

# Rolling features (3 years)
df["rolling_3yr_avg_cost"] = grp["total_claim_amount"].rolling(3, 1).mean().reset_index(level=0, drop=True)
df["rolling_3yr_max_cost"] = grp["total_claim_amount"].rolling(3, 1).max().reset_index(level=0, drop=True)
df["rolling_3yr_total_claims"] = grp["claim_count"].rolling(3, 1).sum().reset_index(level=0, drop=True)
df["std_dev_cost_3yr"] = grp["total_claim_amount"].rolling(3, 1).std().reset_index(level=0, drop=True).fillna(0)

# Years since last claim
def years_since_last_claim(x):
    last = None
    out = []
    for _, amt in enumerate(x):
        if amt > 0:
            last = 0
            out.append(0)
        else:
            if last is None:
                out.append(999)
            else:
                last += 1
                out.append(last)
    return out
df["years_since_last_claim"] = grp["total_claim_amount"].transform(years_since_last_claim)

# Claim-free years in last 3
df["claim_free_yrs_last3"] = grp["total_claim_amount"].rolling(3, 1).apply(lambda x: (x == 0).sum()).reset_index(level=0, drop=True)

# Target (next year's total claim)
df["target_next_year"] = grp["total_claim_amount"].shift(-1)
df.fillna(0, inplace=True)

# -------------------------------------------------------------------
# MODELING DATA (train 2020–2024)
# -------------------------------------------------------------------
model_df = df[(df["year"] >= 2020) & (df["year"] <= 2024)].copy()
model_df = model_df[model_df["target_next_year"].notnull()]

feature_cols = [
    "age","tenure_years",
    "total_claim_amount_lag1","total_claim_amount_lag2","total_claim_amount_lag3",
    "claim_count_lag1","inpatient_count_lag1","total_los_lag1",
    "rolling_3yr_avg_cost","rolling_3yr_max_cost","rolling_3yr_total_claims",
    "std_dev_cost_3yr","years_since_last_claim","claim_free_yrs_last3"
]

# Add categorical features
for col in [GENDER_COL, SCHEME_COL, STATUS_COL, POSTCODE_COL]:
    if col in model_df.columns:
        model_df[col] = model_df[col].astype(str)
        feature_cols.append(col)

# Label encode
cat_cols = [c for c in feature_cols if model_df[c].dtype == "object"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    model_df[c] = le.fit_transform(model_df[c])
    encoders[c] = le

X = model_df[feature_cols].fillna(0)
y = model_df["target_next_year"]
y_log = np.log1p(y)

# Split: 2020–2023 train, 2024 validate
train_idx = model_df["year"] < 2024
X_train, X_val = X[train_idx], X[~train_idx]
y_train, y_val = y[train_idx], y[~train_idx]
y_log_train, y_log_val = y_log[train_idx], y_log[~train_idx]
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

metrics = {}

# -------------------------------------------------------------------
# APPROACH 1: Single-Model (Tweedie)
# -------------------------------------------------------------------
print("\n⚙️ Training Single-Model Regressors...")

# LightGBM
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)
lgb_params = {
    "objective": "tweedie", "tweedie_variance_power": 1.4,
    "learning_rate": 0.05, "num_leaves": 64, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbosity": -1, "seed": 42
}
lgb_model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_val],
                      valid_names=["train","val"], early_stopping_rounds=100, verbose_eval=100)
pred_lgb = np.maximum(0, lgb_model.predict(X_val))
metrics["LGBM_Tweedie"] = float(rmsle(y_val, pred_lgb))

# XGBoost
try:
    xgb_model = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.4,
                                 learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_estimators=300)
except:
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror",
                                 learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_estimators=300)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred_xgb = np.maximum(0, xgb_model.predict(X_val))
metrics["XGB_Tweedie"] = float(rmsle(y_val, pred_xgb))

# MLP (Neural Network)
mlp = MLPRegressor(hidden_layer_sizes=(128,64), random_state=42, max_iter=500)
mlp.fit(X_train, y_log_train)
pred_mlp = np.expm1(mlp.predict(X_val))
metrics["MLP_LogMSE"] = float(rmsle(y_val, pred_mlp))

# -------------------------------------------------------------------
# APPROACH 2: Two-Part Model
# -------------------------------------------------------------------
print("\n⚙️ Training Two-Part Models (Prob × Severity)...")
has_claim_train = (y_train > 0).astype(int)
has_claim_val = (y_val > 0).astype(int)

# LightGBM Two-part
clf = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=31)
clf.fit(X_train, has_claim_train)
reg = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.4)
reg.fit(X_train[has_claim_train==1], y_train[has_claim_train==1])
pred_two_lgb = clf.predict_proba(X_val)[:,1] * np.maximum(0, reg.predict(X_val))
metrics["LGBM_TwoPart"] = float(rmsle(y_val, pred_two_lgb))

# XGBoost Two-part
clf_x = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
clf_x.fit(X_train, has_claim_train)
reg_x = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.4)
reg_x.fit(X_train[has_claim_train==1], y_train[has_claim_train==1])
pred_two_xgb = clf_x.predict_proba(X_val)[:,1] * np.maximum(0, reg_x.predict(X_val))
metrics["XGB_TwoPart"] = float(rmsle(y_val, pred_two_xgb))

# MLP Two-part
clf_nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300)
clf_nn.fit(X_train, has_claim_train)
reg_nn = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=400)
reg_nn.fit(X_train[has_claim_train==1], y_log_train[has_claim_train==1])
pred_two_mlp = clf_nn.predict_proba(X_val)[:,1] * np.expm1(reg_nn.predict(X_val))
metrics["MLP_TwoPart"] = float(rmsle(y_val, pred_two_mlp))

# -------------------------------------------------------------------
# INFERENCE FOR 2025
# -------------------------------------------------------------------
print("\n🔮 Predicting 2025 ...")
score_df = df[df["year"] == 2024].copy()
for c in feature_cols:
    if c not in score_df:
        score_df[c] = 0
score_df[feature_cols] = score_df[feature_cols].fillna(0)
for c, le in encoders.items():
    score_df[c] = le.transform(score_df[c].astype(str))

# Predict
score_df["pred_LGBM_Tweedie_2025"] = np.maximum(0, lgb_model.predict(score_df[feature_cols]))
score_df["pred_XGB_Tweedie_2025"] = np.maximum(0, xgb_model.predict(score_df[feature_cols]))
score_df["pred_MLP_LogMSE_2025"] = np.maximum(0, np.expm1(mlp.predict(score_df[feature_cols])))

score_df["pred_LGBM_TwoPart_2025"] = clf.predict_proba(score_df[feature_cols])[:,1] * np.maximum(0, reg.predict(score_df[feature_cols]))
score_df["pred_XGB_TwoPart_2025"] = clf_x.predict_proba(score_df[feature_cols])[:,1] * np.maximum(0, reg_x.predict(score_df[feature_cols]))
score_df["pred_MLP_TwoPart_2025"] = clf_nn.predict_proba(score_df[feature_cols])[:,1] * np.maximum(0, np.expm1(reg_nn.predict(score_df[feature_cols])))

pred_cols = [c for c in score_df.columns if "pred_" in c]
score_df_out = score_df[["Unique ID","year"] + pred_cols]
score_df_out.rename(columns={"year":"as_of_year"}, inplace=True)
score_df_out.to_csv("predictions_2025_all_models.csv", index=False)

# -------------------------------------------------------------------
# SAVE MODELS & METRICS
# -------------------------------------------------------------------
models = {
    "lgbm_tweedie": lgb_model, "xgb_tweedie": xgb_model, "mlp": mlp,
    "lgbm_two_part": (clf, reg), "xgb_two_part": (clf_x, reg_x), "mlp_two_part": (clf_nn, reg_nn)
}
joblib.dump(models, "trained_models_2020_2025.pkl")
with open("metrics_summary.json","w") as f: json.dump(metrics, f, indent=2)

print("\n✅ Training complete.")
print("Validation RMSLE scores:")
for k,v in metrics.items():
    print(f"{k}: {v:.4f}")
print("\nPredictions saved to: predictions_2025_all_models.csv")
