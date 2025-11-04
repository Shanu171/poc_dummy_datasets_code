#!/usr/bin/env python3
"""
Inference script for 2025 predictions.

- Recomputes features as of year 2024 from raw CSVs.
- Loads models from a pickled file (default: trained_models_2020_2025.pkl).
- Attempts to reuse saved encoders if present; otherwise fits best-effort encoders on current data.
- Outputs: CSV with per-member predictions for each model.

Usage:
  python inference_2025.py --claims /path/to/claims.csv --members /path/to/members.csv --models trained_models_2020_2025.pkl
"""

import argparse, os, joblib, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

def safe_read_csv(path):
    return pd.read_csv(path, low_memory=False)

# ---------- Configurable column names (adjust if your file differs) ----------
DEFAULTS = {
    "join_claim_col": "Claimant Unique ID",   # claims table
    "join_mem_col":   "Unique ID",            # membership table
    "amount_col":     "Amount Paid",
    "incurred_col":   "Incurred Date",
    "claim_type":     "Claim Type",
    "los_col":        "Calculated Length of Service",
    "condition_col":  "Condition Code",
    "provider_col":   "Provider Type",
    "postcode_col":   "short post code",
    "dob_col":        "Claimant year of birth",
    "gender_col":     "claimant Gender",
    "join_date_col":  "Contract Start Date",
    "scheme_col":     "Scheme Category/ Section Name",
    "status_col":     "status of member"
}

# ---------------- Feature engineering helpers ----------------
def build_member_year_features(claims_df, members_df, as_of_year=2024):
    """
    Build per-member features computed as of 'as_of_year' (e.g., 2024).
    Returns a DataFrame with one row per member and the features used for modeling.
    """
    j_claim = cfg["join_claim_col"]
    j_mem = cfg["join_mem_col"]
    amt = cfg["amount_col"]
    inc = cfg["incurred_col"]
    ct = cfg["claim_type"]
    losc = cfg["los_col"]
    cond = cfg["condition_col"]
    prov = cfg["provider_col"]
    postc = cfg["postcode_col"]

    # normalize ids
    claims_df[j_claim] = claims_df[j_claim].astype(str)
    members_df[j_mem] = members_df[j_mem].astype(str)

    # parse dates & numeric fields
    claims_df["incurred_dt"] = pd.to_datetime(claims_df.get(inc, pd.NaT), errors="coerce")
    claims_df["year"] = claims_df["incurred_dt"].dt.year
    claims_df["amount_paid"] = pd.to_numeric(claims_df.get(amt, 0), errors="coerce").fillna(0)
    claims_df["los"] = pd.to_numeric(claims_df.get(losc, 0), errors="coerce").fillna(0)
    claims_df["is_inpatient"] = claims_df.get(ct, "").astype(str).str.lower().str.contains("inpat", na=False).astype(int)
    claims_df["is_outpatient"] = claims_df.get(ct, "").astype(str).str.lower().str.contains("outpat", na=False).astype(int)

    # aggregate per member-year
    agg = claims_df.groupby([j_claim, "year"], as_index=False).agg(
        total_claim_amount=("amount_paid","sum"),
        claim_count=("amount_paid","count"),
        inpatient_count=("is_inpatient","sum"),
        outpatient_count=("is_outpatient","sum"),
        total_los=("los","sum"),
        avg_los=("los","mean"),
        unique_conditions=(cond, pd.Series.nunique),
        unique_providers=(prov, pd.Series.nunique),
    )
    agg = agg.rename(columns={j_claim: j_mem})

    # build scaffold for years from 2020..as_of_year
    all_members = members_df[j_mem].unique()
    years = list(range(2020, as_of_year+1))
    scaffold = pd.MultiIndex.from_product([all_members, years], names=[j_mem, "year"]).to_frame(index=False)
    df = scaffold.merge(agg, how="left", on=[j_mem, "year"]).fillna(0)

    # merge member static info
    members_df["birth_year"] = pd.to_numeric(members_df.get(cfg["dob_col"], np.nan), errors="coerce")
    members_df["join_year"] = pd.to_datetime(members_df.get(cfg["join_date_col"], pd.NaT), errors="coerce").dt.year
    mem_sel = members_df[[j_mem, "birth_year", cfg["gender_col"], cfg["scheme_col"], cfg["postcode_col"], cfg["status_col"], "join_year"]].copy()
    mem_sel = mem_sel.rename(columns={j_mem: j_mem})
    df = df.merge(mem_sel, on=j_mem, how="left")

    # compute age & tenure per year
    df["age"] = df["year"] - df["birth_year"]
    df["tenure_years"] = df["year"] - df["join_year"]
    df["age"] = df["age"].clip(lower=0).fillna(0)
    df["tenure_years"] = df["tenure_years"].clip(lower=0).fillna(0)

    # sort and compute lags/rolling (3-year)
    df = df.sort_values([j_mem, "year"])
    g = df.groupby(j_mem)
    for lag in [1,2,3]:
        df[f"total_claim_amount_lag{lag}"] = g["total_claim_amount"].shift(lag).fillna(0)
        df[f"claim_count_lag{lag}"] = g["claim_count"].shift(lag).fillna(0)
        df[f"inpatient_count_lag{lag}"] = g["inpatient_count"].shift(lag).fillna(0)
        df[f"total_los_lag{lag}"] = g["total_los"].shift(lag).fillna(0)

    df["rolling_3yr_avg_cost"] = g["total_claim_amount"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df["rolling_3yr_max_cost"] = g["total_claim_amount"].rolling(3, min_periods=1).max().reset_index(level=0, drop=True).fillna(0)
    df["rolling_3yr_total_claims"] = g["claim_count"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0)
    df["std_dev_cost_3yr"] = g["total_claim_amount"].rolling(3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    # years_since_last_claim
    def years_since(s):
        last = None
        out = []
        for val in s:
            if val>0:
                last = 0
                out.append(0)
            else:
                if last is None:
                    out.append(999)
                else:
                    last += 1
                    out.append(last)
        return out
    df["years_since_last_claim"] = g["total_claim_amount"].transform(years_since)

    # claim_free_yrs_last3
    df["claim_free_yrs_last3"] = g["total_claim_amount"].rolling(3,1).apply(lambda x: (x==0).sum()).reset_index(level=0, drop=True).fillna(0)

    # target_next_year (not used in inference)
    df["target_next_year"] = g["total_claim_amount"].shift(-1)

    return df

# ----------------- Main -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--claims", required=True)
    p.add_argument("--members", required=True)
    p.add_argument("--models", default="trained_models_2020_2025.pkl", help="Pickle file with trained models")
    p.add_argument("--encoders", default="encoders.pkl", help="Optional: pickled encoders (LabelEncoder dict)")
    p.add_argument("--out", default="predictions_2025_inference.csv")
    p.add_argument("--as_of_year", type=int, default=2024)
    args = p.parse_args()

    cfg = DEFAULTS.copy()
    # allow overrides if needed via env / future args (kept simple now)
    cfg.update({})  # no overrides currently

    CLAIMS_PATH = args.claims
    MEMBERS_PATH = args.members
    MODELS_PKL = args.models
    ENCODERS_PKL = args.encoders
    OUT_CSV = args.out
    AS_OF = args.as_of_year
    print("Loading data...")
    claims = safe_read_csv(CLAIMS_PATH)
    members = safe_read_csv(MEMBERS_PATH)

    # Build features up to AS_OF (2024)
    print(f"Building features up to year {AS_OF} ...")
    feature_df = build_member_year_features(claims, members, as_of_year=AS_OF)

    # Keep only rows for as_of_year (we will predict next year)
    score_df = feature_df[feature_df["year"] == AS_OF].copy().reset_index(drop=True)

    # Define feature columns (must match your training script's feature list)
    feature_cols = [
        "age","tenure_years",
        "total_claim_amount_lag1","total_claim_amount_lag2","total_claim_amount_lag3",
        "claim_count_lag1","inpatient_count_lag1","total_los_lag1",
        "rolling_3yr_avg_cost","rolling_3yr_max_cost","rolling_3yr_total_claims",
        "std_dev_cost_3yr","years_since_last_claim","claim_free_yrs_last3"
    ]
    # add categorical candidate columns if present
    for c in [cfg["gender_col"], cfg["scheme_col"], cfg["status_col"], cfg["postcode_col"]]:
        if c in score_df.columns:
            feature_cols.append(c)

    # fill NA
    for c in feature_cols:
        if c not in score_df.columns:
            score_df[c] = 0
    X_score = score_df[feature_cols].fillna(0).copy()

    # Load models
    if not os.path.exists(MODELS_PKL):
        raise FileNotFoundError(f"Model file not found: {MODELS_PKL}")
    print("Loading models from:", MODELS_PKL)
    models_obj = joblib.load(MODELS_PKL)

    # Try load encoders (either inside models or separate file)
    encoders = {}
    if isinstance(models_obj, dict) and "encoders" in models_obj:
        encoders = models_obj.get("encoders", {})
        print("Using encoders embedded in model file.")
    elif os.path.exists(ENCODERS_PKL):
        try:
            encoders = joblib.load(ENCODERS_PKL)
            print("Loaded encoders from", ENCODERS_PKL)
        except Exception:
            encoders = {}
    else:
        print("No saved encoders found — will create label encoders on the fly (best-effort).")

    # Apply encoders: for any categorical column, if encoder available map, else fit on score data
    from sklearn.preprocessing import LabelEncoder
    for c in [cfg["gender_col"], cfg["scheme_col"], cfg["status_col"], cfg["postcode_col"]]:
        if c in X_score.columns:
            X_score[c] = X_score[c].astype(str).fillna("nan")
            if c in encoders:
                le = encoders[c]
                # map unseen labels to -1
                known = {k: v for k, v in zip(le.classes_, le.transform(le.classes_))}
                X_score[c] = X_score[c].map(lambda x: known.get(x, -1)).astype(int)
            else:
                le = LabelEncoder()
                X_score[c] = le.fit_transform(X_score[c])
                encoders[c] = le

    # Now produce predictions for each model found in models_obj
    preds = pd.DataFrame()
    preds["Unique ID"] = score_df["Unique ID"]
    preds["as_of_year"] = AS_OF

    # helper to safe predict
    def safe_predict(model, X):
        try:
            return np.maximum(0, model.predict(X))
        except Exception as e:
            # some xgboost models expect DMatrix; try xgboost API
            try:
                import xgboost as xgb
                if hasattr(model, "get_booster"):
                    dm = xgb.DMatrix(X)
                    return np.maximum(0, model.get_booster().predict(dm))
            except Exception:
                print("Prediction failed for model:", e)
            # fallback zeros
            return np.zeros(len(X))

    # iterate keys
    for key in models_obj:
        # handle common cases saved by training script
        if key == "lgbm_tweedie":
            model = models_obj[key]
            preds["pred_lgbm_tweedie_2025"] = safe_predict(model, X_score)
        elif key == "xgb_tweedie":
            model = models_obj[key]
            preds["pred_xgb_tweedie_2025"] = safe_predict(model, X_score)
        elif key == "mlp":
            model = models_obj[key]
            # training MLP may have been trained on log(y); attempt both strategies
            try:
                yhat = model.predict(X_score)
                # if negative or obviously log-space, try expm1 if it makes sense
                if (yhat < 0).sum() > 0 or np.median(yhat) < 1:
                    preds["pred_mlp_2025"] = np.maximum(0, np.expm1(yhat))
                else:
                    preds["pred_mlp_2025"] = np.maximum(0, yhat)
            except Exception:
                preds["pred_mlp_2025"] = np.zeros(len(X_score))
        elif key == "lgbm_two_part":
            clf, reg = models_obj[key]
            p_any = clf.predict_proba(X_score)[:,1]
            sev = safe_predict(reg, X_score)
            preds["pred_lgbm_twopart_2025"] = p_any * sev
        elif key == "xgb_two_part":
            clf, reg = models_obj[key]
            p_any = clf.predict_proba(X_score)[:,1]
            sev = safe_predict(reg, X_score)
            preds["pred_xgb_twopart_2025"] = p_any * sev
        elif key == "mlp_two_part":
            clf, reg = models_obj[key]
            p_any = clf.predict_proba(X_score)[:,1]
            # reg might be log-output; try expm1 if values small
            sev_raw = reg.predict(X_score)
            sev = np.maximum(0, np.expm1(sev_raw)) if np.median(sev_raw) < 2 else np.maximum(0, sev_raw)
            preds["pred_mlp_twopart_2025"] = p_any * sev
        else:
            # if models_obj is dict with other names or nested tuples
            model = models_obj[key]
            # try to detect type
            try:
                if hasattr(model, "predict"):
                    preds[f"pred_{key}_2025"] = safe_predict(model, X_score)
                elif isinstance(model, (list, tuple)) and len(model)==2:
                    clf, reg = model
                    p_any = clf.predict_proba(X_score)[:,1]
                    sev = safe_predict(reg, X_score)
                    preds[f"pred_{key}_2025"] = p_any * sev
            except Exception as e:
                print("Skipping unknown model key:", key, "error:", e)

    # Save encoders used (optional)
    try:
        joblib.dump(encoders, "inference_encoders_used.pkl")
    except Exception:
        pass

    # Write predictions
    preds.to_csv(OUT_CSV, index=False)
    print("Saved predictions to:", OUT_CSV)
