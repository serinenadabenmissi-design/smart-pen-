import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

# ---------- CONFIG ----------
CSV_PATH = "sessions_summary.csv"
RANDOM_STATE = 42

N_ITER_SEARCH = 40     # reduce/increase for speed/quality
INNER_CV = 3
TOP_K = 7              # number of top features to keep for final model

MODEL_OUT = "xgb_best.joblib"
SCALER_OUT = "scaler_selected.joblib"
FEATURES_OUT = "selected_features.txt"

# ---------- LOAD ----------
df = pd.read_csv(CSV_PATH)
df = df.dropna(how="all").reset_index(drop=True)
df = df[df["label"].isin(["bad", "medium", "perfect"])].reset_index(drop=True)

# ---------- COLUMN MAPPING ----------
def find_col(df_columns, candidates):
    for name in candidates:
        if name in df_columns:
            return name
    return None

candidates = {
    "speed": ["speed", "speed_score", "avg_speed", "mean_speed"],
    "vibration": ["vibration", "vibration_score", "vibe", "vibration_mean"],
    "pause_ratio": ["pause_ratio", "pause", "pause_ratio_score"],
    "smoothness": ["smoothness", "smoothness_score"],
    "gyro_std": ["gyro_std", "gyro_std_score", "gyro_std_dev"],
    "gyro_max": ["gyro_max", "gyro_max_score"],
    "gyro_range": ["gyro_range", "gyro_range_score"],
    "jerk_mean": ["jerk_mean", "jerk_avg", "jerk_mean_score"],
    "jerk_max": ["jerk_max", "jerk_peak"],
    "energy": ["energy", "energy_sum", "signal_energy"]
}
col_map = {logical: find_col(df.columns, cand) for logical, cand in candidates.items()}

# ---------- BUILD FEATURES ----------
X_df = pd.DataFrame(index=df.index)
def safe_col(name):
    return col_map.get(name)

for logical in ["speed", "vibration", "pause_ratio", "smoothness", "gyro_std", "gyro_max",
                "gyro_range", "jerk_mean", "jerk_max", "energy"]:
    col = safe_col(logical)
    if col is not None:
        X_df[logical] = df[col]

# derived
if safe_col("energy") is not None and safe_col("speed") is not None:
    X_df["gyro_energy_ratio"] = df[safe_col("energy")] / (df[safe_col("speed")] + 1)
if safe_col("jerk_mean") is not None and safe_col("speed") is not None:
    X_df["jerk_speed_ratio"] = df[safe_col("jerk_mean")] / (df[safe_col("speed")] + 1)
if safe_col("vibration") is not None and safe_col("gyro_std") is not None:
    X_df["stability_score"] = 1.0 / (df[safe_col("vibration")] + df[safe_col("gyro_std")] + 1.0)
if safe_col("smoothness") is not None and safe_col("pause_ratio") is not None:
    X_df["control_score"] = df[safe_col("smoothness")] / (df[safe_col("pause_ratio")] + 0.01)
if safe_col("gyro_max") is not None and safe_col("gyro_std") is not None:
    X_df["peak_ratio"] = df[safe_col("gyro_max")] / (df[safe_col("gyro_std")] + 1)

X_df = X_df.loc[:, X_df.notna().any(axis=0)]
if X_df.shape[1] < 3:
    raise RuntimeError("Too few features available to train. Check CSV and mapping.")
X_df = X_df.fillna(X_df.median())

# ---------- TARGET ----------
label_map = {"bad": 0, "medium": 1, "perfect": 2}
y = df["label"].map(label_map).values
feature_names = list(X_df.columns)

# optional log transform for skewed
skews = X_df.skew().abs()
for c in skews[skews > 1.0].index:
    if (X_df[c] > 0).all():
        X_df[c] = np.log1p(X_df[c])

# Fit a scaler on full features (only for search stage)
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X_df.values)

# ---------- RANDOMIZED SEARCH (optimize f1_macro) ----------
xgb_base = XGBClassifier(objective="multi:softprob", use_label_encoder=False,
                         eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

param_dist = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.19),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": uniform(0.0, 0.5),
    "min_child_weight": randint(1, 8),
    "reg_alpha": uniform(0.0, 1.0),
    "reg_lambda": uniform(0.5, 2.0),
}

inner_cv = StratifiedKFold(n_splits=INNER_CV, shuffle=True, random_state=RANDOM_STATE)

rand_search = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_dist,
                                 n_iter=N_ITER_SEARCH, scoring="f1_macro", cv=inner_cv,
                                 random_state=RANDOM_STATE, verbose=1, n_jobs=-1, refit=True)

rand_search.fit(X_full, y)
best_search = rand_search.best_estimator_

# ---------- SELECT TOP-K FEATURES (using importances from best_search) ----------
importances = best_search.feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
selected = [f for f, _ in feat_imp[:min(TOP_K, len(feat_imp))]]

# Prepare X_selected and scaler for final training
X_selected_df = X_df[selected]
scaler_selected = StandardScaler()
X_selected = scaler_selected.fit_transform(X_selected_df.values)

# ---------- TRAIN FINAL MODEL ON SELECTED FEATURES (use best params) ----------
# use best params from rand_search to initialize final model
best_params = rand_search.best_params_.copy()
# create final model with same hyperparams (safely pass only params that XGB supports)
final_model = XGBClassifier(objective="multi:softprob", use_label_encoder=False,
                            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                            **best_params)
# fit on full selected data
final_model.fit(X_selected, y)

# ---------- OPTIONAL: quick CV on final model ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(final_model, X_selected, y, cv=cv, scoring="accuracy", n_jobs=-1)

# ---------- SAVE ARTIFACTS ----------
dump(final_model, MODEL_OUT)
dump(scaler_selected, SCALER_OUT)
with open(FEATURES_OUT, "w") as f:
    for feat in selected:
        f.write(feat + "\n")

# ---------- PRINT SUMMARY ----------
print("Saved artifacts:")
print(" - model:", MODEL_OUT)
print(" - scaler:", SCALER_OUT)
print(" - features:", FEATURES_OUT)
print("Selected features (top-{}): {}".format(len(selected), selected))
print("Final CV accuracy (selected features): {:.2f}% (std {:.2f})".format(cv_scores.mean()*100, cv_scores.std()*100))