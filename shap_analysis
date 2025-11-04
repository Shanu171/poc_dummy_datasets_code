"""
explain_models_shap.py
Generates feature importances + SHAP explainability artifacts for models trained earlier.
Assumes:
 - models saved in 'trained_models_2020_2025.pkl'
 - training/validation feature matrix available as CSV or can be recomputed
 - feature column list available or embedded in the saved object
"""

import os, joblib, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# -----------------------
# CONFIG
# -----------------------
MODELS_PKL = "trained_models_2020_2025.pkl"   # from training script
FEATURES_JSON = "features_list.json"         # optional: if you saved features earlier
TRAIN_X_CSV = "X_train.csv"                  # optional: if you exported X_train; otherwise recompute before running
VAL_X_CSV = "X_val.csv"
SAMPLE_SIZE = 5000       # number of rows to sample for SHAP summary (reduce if memory limited)
OUT_DIR = "explainability_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# LOAD MODELS
# -----------------------
print("Loading models...")
models_obj = joblib.load(MODELS_PKL)

# models_obj expected keys (from earlier script):
# e.g., "lgbm_tweedie", "xgb_tweedie", "mlp", "lgbm_two_part"->(clf, reg), etc.
# Adapt to the keys you actually saved.
model_keys = list(models_obj.keys())
print("Found models:", model_keys)

# -----------------------
# LOAD DATA (X_val or X_train sample)
# -----------------------
# Prefer a saved validation dataset; else ask user to recompute. We'll try to load X_val.csv first.
X = None
if os.path.exists(VAL_X_CSV):
    print("Loading validation features from", VAL_X_CSV)
    X = pd.read_csv(VAL_X_CSV, index_col=None)
elif os.path.exists(TRAIN_X_CSV):
    print("Loading training features from", TRAIN_X_CSV)
    X = pd.read_csv(TRAIN_X_CSV, index_col=None)

if X is None:
    raise FileNotFoundError("No X_val.csv or X_train.csv found. Re-run the training script to export X_train/X_val, or adapt this script to reconstruct the feature matrix.")

# ensure features list order
if os.path.exists(FEATURES_JSON):
    feat_list = json.load(open(FEATURES_JSON))
    X = X[feat_list]
    print("Using feature list from", FEATURES_JSON)
else:
    feat_list = X.columns.tolist()
    print("Using feature columns inferred from X file.")

# sample for SHAP to limit memory/time
if len(X) > SAMPLE_SIZE:
    X_sample = X.sample(SAMPLE_SIZE, random_state=42)
else:
    X_sample = X.copy()

# -----------------------
# HELPERS
# -----------------------
def save_feature_importances(df_imp, name):
    out_csv = os.path.join(OUT_DIR, f"feature_importances_{name}.csv")
    df_imp.to_csv(out_csv, index=False)
    print("Saved feature importances to", out_csv)

def save_shap_summary_plot(explainer, shap_values, Xsample, model_name):
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, Xsample, show=False)
    out_png = os.path.join(OUT_DIR, f"shap_summary_{model_name}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved SHAP summary plot to", out_png)

def save_shap_values_csv(shap_values, Xsample, model_name, max_cols=2000):
    # shap_values shape: (n_samples, n_features) or list for multi-output
    sv = np.array(shap_values)
    if sv.ndim == 3:  # tree explainer sometimes returns (nsamples, nclasses, nfeatures)
        sv = sv[:,0,:]
    sv_df = pd.DataFrame(sv, columns=Xsample.columns)
    sv_df = pd.concat([Xsample.reset_index(drop=True), sv_df.add_prefix("shap_")], axis=1)
    out_csv = os.path.join(OUT_DIR, f"shap_values_{model_name}_sample.csv")
    sv_df.to_csv(out_csv, index=False)
    print("Saved SHAP values sample to", out_csv)

# -----------------------
# EXPLAIN TREE MODELS (LGB & XGB)
# -----------------------
# Look for common keys and explain them
tree_models = {}
if "lgbm_tweedie" in models_obj:
    tree_models["lgbm_tweedie"] = models_obj["lgbm_tweedie"]
if "xgb_tweedie" in models_obj:
    tree_models["xgb_tweedie"] = models_obj["xgb_tweedie"]
# also check two-part regressors (reg part)
if "lgbm_two_part" in models_obj:
    clf, reg = models_obj["lgbm_two_part"]
    tree_models["lgbm_twopart_reg"] = reg
if "xgb_two_part" in models_obj:
    clf_x, reg_x = models_obj["xgb_two_part"]
    tree_models["xgb_twopart_reg"] = reg_x

for name, model in tree_models.items():
    try:
        print(f"\nExplaining tree model: {name}")
        # feature importances (gain or importance)
        if hasattr(model, "feature_importance"):
            imp_gain = model.feature_importance(importance_type="gain")
            imp_df = pd.DataFrame({"feature": feat_list, "importance_gain": imp_gain})
            imp_df = imp_df.sort_values("importance_gain", ascending=False)
            save_feature_importances(imp_df, name)
        elif hasattr(model, "get_booster"):
            # xgboost model
            imp = model.get_booster().get_score(importance_type="gain")
            imp_df = pd.DataFrame([
                {"feature": k, "importance_gain": imp.get(k, 0.0)} for k in feat_list
            ])
            imp_df = imp_df.sort_values("importance_gain", ascending=False)
            save_feature_importances(imp_df, name)
        # SHAP TreeExplainer (fast for tree models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        save_shap_summary_plot(explainer, shap_values, X_sample, name)
        save_shap_values_csv(shap_values, X_sample, name)
    except Exception as e:
        print(f"Failed to explain {name} via SHAP: {e}")

# -----------------------
# EXPLAIN MLP (Neural Net) via Permutation Importance
# -----------------------
if "mlp" in models_obj:
    print("\nExplaining MLP model via permutation importance (this may take a while)...")
    mlp_model = models_obj["mlp"]
    # We need a holdout X_val and y_val saved earlier; assume X (validation) and y_val loaded alongside X.
    # If y_val available in a CSV file named 'y_val.csv' load it, else skip permutation.
    y_val_path = "y_val.csv"
    if os.path.exists(y_val_path):
        y_val = pd.read_csv(y_val_path).iloc[:,0].values
        # compute permutation importance
        r = permutation_importance(mlp_model, X, y_val, n_repeats=10, random_state=42, n_jobs=4)
        perm_df = pd.DataFrame({"feature": feat_list, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        perm_df = perm_df.sort_values("importance_mean", ascending=False)
        out_perm = os.path.join(OUT_DIR, "permutation_importance_mlp.csv")
        perm_df.to_csv(out_perm, index=False)
        print("Saved permutation importances for MLP to", out_perm)
    else:
        print("y_val.csv not found — skipping permutation importance for MLP. If you want this, save y_val to y_val.csv and re-run.")

# -----------------------
# DONE
# -----------------------
print("\nAll done. Check folder:", OUT_DIR)
