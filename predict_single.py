# predict_single_wrapper.py

import numpy as np
import pandas as pd
from joblib import load
import json
import os

LABEL_MAP_INV = {0: "bad", 1: "medium", 2: "perfect"}

MODEL_IN = "xgb_best.joblib"
SCALER_IN = "scaler_selected.joblib"
FEATURES_IN = "selected_features.txt"
TRANSFORM_IN = "transforms.json"

# Charger artefacts une seule fois
def load_artifacts():
    model = load(MODEL_IN)
    scaler = load(SCALER_IN)
    with open(FEATURES_IN, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
    log_cols = []
    if os.path.exists(TRANSFORM_IN):
        try:
            with open(TRANSFORM_IN, "r") as f:
                t = json.load(f)
                log_cols = t.get("log1p_columns", []) or []
        except Exception:
            log_cols = []
    return model, scaler, selected_features, log_cols

model, scaler, selected_features, log_cols = load_artifacts()

# Function timporti f Streamlit
def predict_sample(input_dict):
    # Build feature vector
    vec = []
    for feat in selected_features:
        if feat not in input_dict:
            raise ValueError(f"Missing feature: {feat}")
        val = float(input_dict[feat])
        if feat in log_cols:
            if val < 0:
                raise ValueError(f"Cannot apply log1p on negative value for feature {feat}")
            val = np.log1p(val)
        vec.append(val)
    X = np.array(vec).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return LABEL_MAP_INV[int(pred)], {LABEL_MAP_INV[i]: float(proba[i]) for i in range(len(proba))}
